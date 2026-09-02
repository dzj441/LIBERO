"""Contract tests for the single-task Arrange Table goal-image variant."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from libero.libero.agent_env import task_references
from libero.libero.agent_env.task_references import (
    load_task_reference_rgb,
    resolve_task_reference_path,
)
from libero.libero.benchmark import get_benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from scripts.launch_agent_episode import _task_instruction


SOURCE_TASK = (
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_"
    "yellow_and_white_mug_on_the_right_plate"
)


def test_arrange_table_is_a_single_reused_libero10_task():
    suite = get_benchmark("libero_arrange_table")()

    assert suite.get_num_tasks() == 1
    task = suite.get_task(0)
    assert task.name == SOURCE_TASK
    assert task.problem_folder == "libero_10"
    assert task.bddl_file == f"{SOURCE_TASK}.bddl"
    assert task.init_states_file == f"{SOURCE_TASK}.pruned_init"
    assert task.language == "Arrange Table"
    assert Path(suite.get_task_bddl_file_path(0)).is_file()
    assert len(suite.get_task_init_states(0)) > 0

    with pytest.raises(ValueError, match="task_order_index=0"):
        get_benchmark("libero_arrange_table")(task_order_index=1)


def test_arrange_table_instruction_is_exact():
    assert _task_instruction("libero_arrange_table", 0) == "Arrange Table"


def test_arrange_table_bddl_is_parseable_and_has_finite_init_state():
    suite = get_benchmark("libero_arrange_table")()
    task = suite.get_task(0)
    parsed = robosuite_parse_problem(suite.get_task_bddl_file_path(0))
    assert parsed["goal_state"]
    init_states = suite.get_task_init_states(0)
    assert np.isfinite(np.asarray(init_states[0])).all()


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
