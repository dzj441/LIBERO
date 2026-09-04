"""Contracts for the matched Arrange Table goal-specification variants."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from libero.libero.agent_env.arrange_table import (
    ArrangeTableTextGoalEvaluator,
    arrange_table_private_evaluator,
)
from libero.libero.agent_env import task_references
from libero.libero.agent_env.task_references import (
    load_task_reference_rgb,
    resolve_task_reference_path,
)
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from scripts.launch_agent_episode import _task_instruction


TASK_NAME = "arrange_table"
VISUAL_GOAL_INSTRUCTION = (
    "Arrange the table according to the provided goal image."
)
TEXT_GOAL_INSTRUCTION = (
    "Arrange the table. For a clean table, the butter should be placed inside "
    "the basket, and each cup should be placed on a plate."
)


def test_arrange_table_exposes_two_matched_goal_variants():
    suite = get_benchmark("libero_arrange_table")()

    assert suite.get_num_tasks() == 2
    visual_task = suite.get_task(0)
    text_task = suite.get_task(1)
    assert visual_task.name == TASK_NAME
    assert text_task.name == "arrange_table_text_goal"
    assert visual_task.language == VISUAL_GOAL_INSTRUCTION
    assert text_task.language == TEXT_GOAL_INSTRUCTION
    for task_id, task in enumerate((visual_task, text_task)):
        assert task.problem_folder == "libero_arrange_table"
        assert task.bddl_file == f"{TASK_NAME}.bddl"
        assert task.init_states_file == f"{TASK_NAME}.pruned_init"
        assert Path(suite.get_task_bddl_file_path(task_id)).is_file()
        assert len(suite.get_task_init_states(task_id)) > 0

    with pytest.raises(ValueError, match="task_order_index=0"):
        get_benchmark("libero_arrange_table")(task_order_index=1)


def test_arrange_table_instructions_are_exact():
    assert _task_instruction("libero_arrange_table", 0) == VISUAL_GOAL_INSTRUCTION
    assert _task_instruction("libero_arrange_table", 1) == TEXT_GOAL_INSTRUCTION


def test_arrange_table_bddl_is_parseable_and_has_finite_init_state():
    suite = get_benchmark("libero_arrange_table")()
    task = suite.get_task(0)
    parsed = robosuite_parse_problem(suite.get_task_bddl_file_path(0))
    assert parsed["goal_state"] == [
        ["on", "porcelain_mug_1", "plate_1"],
        ["on", "white_yellow_mug_1", "plate_2"],
        ["in", "butter_1", "basket_1_contain_region"],
    ]
    assert parsed["obj_of_interest"] == [
        "porcelain_mug_1",
        "white_yellow_mug_1",
        "butter_1",
        "plate_1",
        "plate_2",
        "basket_1",
    ]
    assert parsed["initial_state"] == [
        ["on", "plate_1", "living_room_table_plate_left_region"],
        ["on", "plate_2", "living_room_table_plate_right_region"],
        ["on", "basket_1", "living_room_table_basket_init_region"],
        [
            "on",
            "porcelain_mug_1",
            "living_room_table_porcelain_mug_init_region",
        ],
        [
            "on",
            "white_yellow_mug_1",
            "living_room_table_white_yellow_mug_init_region",
        ],
        ["on", "butter_1", "living_room_table_butter_init_region"],
    ]
    init_states = suite.get_task_init_states(0)
    assert np.asarray(init_states).shape == (50, 97)
    assert np.isfinite(np.asarray(init_states)).all()


def test_task_reference_allowlist_returns_only_arrange_table_goal():
    reference_path = resolve_task_reference_path("libero_arrange_table", 0)
    assert reference_path is not None
    assert reference_path.name == "goal_rgb.png"
    assert reference_path.parent.name == "libero_arrange_table"
    assert resolve_task_reference_path("libero_10", 4) is None
    assert resolve_task_reference_path("libero_arrange_table", 1) is None

    image = load_task_reference_rgb("libero_arrange_table", 0)
    assert image is not None
    assert image.dtype == np.uint8
    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.shape[0] > 0 and image.shape[1] > 0
    assert image.flags.c_contiguous
    with Image.open(reference_path) as pil_image:
        pil_image.load()
        assert pil_image.mode == "RGB"
        assert pil_image.info == {}


def test_task_reference_rejects_symlink_components(monkeypatch, tmp_path):
    root = tmp_path / "references"
    root.mkdir()
    real_directory = tmp_path / "real_arrange_table"
    real_directory.mkdir()
    (real_directory / "goal_rgb.png").write_bytes(
        Path(
            "libero/libero/assets/task_references/libero_arrange_table/goal_rgb.png"
        ).read_bytes()
    )
    (root / "libero_arrange_table").symlink_to(real_directory, target_is_directory=True)
    monkeypatch.setattr(task_references, "_REFERENCE_ROOT", root)

    with pytest.raises(ValueError, match="must not contain symlinks"):
        resolve_task_reference_path("libero_arrange_table", 0)


class _FakeDomain:
    def __init__(self, true_predicates):
        self.true_predicates = {tuple(predicate) for predicate in true_predicates}

    def _eval_predicate(self, predicate):
        return tuple(predicate) in self.true_predicates


class _FakeEnv:
    def __init__(self, true_predicates):
        self.env = _FakeDomain(true_predicates)


@pytest.mark.parametrize(
    "cup_predicates",
    [
        (
            ("on", "porcelain_mug_1", "plate_1"),
            ("on", "white_yellow_mug_1", "plate_2"),
        ),
        (
            ("on", "porcelain_mug_1", "plate_2"),
            ("on", "white_yellow_mug_1", "plate_1"),
        ),
    ],
)
def test_text_goal_checker_accepts_either_one_to_one_cup_assignment(
    cup_predicates,
):
    evaluator = ArrangeTableTextGoalEvaluator(
        _FakeEnv(
            (
                ("in", "butter_1", "basket_1_contain_region"),
                *cup_predicates,
            )
        )
    )

    result = evaluator.result()

    assert result["success"] is True
    assert sum(result["cup_assignment_results"]) == 1


def test_text_goal_checker_rejects_missing_butter_or_non_distinct_plates():
    same_plate = ArrangeTableTextGoalEvaluator(
        _FakeEnv(
            (
                ("in", "butter_1", "basket_1_contain_region"),
                ("on", "porcelain_mug_1", "plate_1"),
                ("on", "white_yellow_mug_1", "plate_1"),
            )
        )
    )
    missing_butter = ArrangeTableTextGoalEvaluator(
        _FakeEnv(
            (
                ("on", "porcelain_mug_1", "plate_1"),
                ("on", "white_yellow_mug_1", "plate_2"),
            )
        )
    )

    assert same_plate.result()["success"] is False
    assert missing_butter.result()["success"] is False


def test_private_checker_is_selected_only_for_text_goal_variant():
    env = _FakeEnv(())

    assert arrange_table_private_evaluator(
        env, suite="libero_arrange_table", task_id=0
    ) is None
    assert isinstance(
        arrange_table_private_evaluator(
            env, suite="libero_arrange_table", task_id=1
        ),
        ArrangeTableTextGoalEvaluator,
    )
    assert arrange_table_private_evaluator(
        env, suite="libero_object", task_id=1
    ) is None
