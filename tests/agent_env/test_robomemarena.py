from types import SimpleNamespace
import subprocess

import h5py
import numpy as np
import pytest

from libero.libero.agent_env.robomemarena import (
    RoboMemArenaOrderedStageEvaluator,
    RoboMemArenaTask4Evaluator,
    get_robomemarena_task_spec,
    robomemarena_bddl_path,
    robomemarena_source_fingerprint,
    robomemarena_task_variant,
    task4_init_state_id_from_recorded_instruction,
)
from libero.libero.agent_env.robomemarena_demo import (
    load_robomemarena_full_trajectory,
)
from libero.libero.agent_env.robomemarena_vendor.stage import reference_stage


class _NamedModel:
    def __init__(self, sites, bodies):
        self.sites = sites
        self.bodies = bodies

    def site_name2id(self, name):
        if name not in self.sites:
            raise KeyError(name)
        return self.sites.index(name)

    def body_name2id(self, name):
        if name not in self.bodies:
            raise KeyError(name)
        return self.bodies.index(name)


def _fake_env():
    sites = [
        "wooden_cabinet_1_top_region",
        "wooden_cabinet_1_middle_region",
        "wooden_cabinet_1_bottom_region",
    ]
    bodies = ["butter_1", "cream_cheese_1"]
    data = SimpleNamespace(
        site_xpos=np.asarray(
            [[0.0, 0.0, 0.30], [0.0, 0.0, 0.20], [0.0, 0.0, 0.10]],
            dtype=np.float64,
        ),
        body_xpos=np.asarray(
            [[0.50, 0.50, 0.50], [0.0, 0.0, 0.30]],
            dtype=np.float64,
        ),
        time=0.0,
    )
    sim = SimpleNamespace(model=_NamedModel(sites, bodies), data=data)
    return SimpleNamespace(sim=sim)


def test_task4_contract_names_eight_required_and_one_optional_stage():
    spec = get_robomemarena_task_spec(4)
    assert "all drawers in order" in spec.instruction
    assert len(spec.required_stage_names) == 8
    assert spec.required_stage_names[-2:] == (
        "07_Open_Occupied_Drawer_Again",
        "08_Put_Butter_Occupied_Drawer",
    )
    assert spec.optional_stage_names == (
        "09_Close_Occupied_Drawer_Final",
    )


def test_task4_all_occupied_drawer_variants_are_frozen_and_addressable():
    expected = {0: "top", 1: "middle", 2: "bottom"}
    for init_state_id, drawer in expected.items():
        path = robomemarena_bddl_path(
            None, task_id=4, init_state_id=init_state_id
        )
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert (
            f"(In cream_cheese_1 wooden_cabinet_1_{drawer}_region)"
            in text
        )
        assert (
            f"(In butter_1 wooden_cabinet_1_{drawer}_region)" in text
        )
        assert robomemarena_task_variant(
            task_id=4, init_state_id=init_state_id
        ) == drawer

    with pytest.raises(ValueError, match=r"0 \(top\), 1 \(middle\), or 2"):
        robomemarena_task_variant(task_id=4, init_state_id=3)
    with pytest.raises(ValueError, match="supports init_state_id 0 only"):
        robomemarena_task_variant(task_id=7, init_state_id=1)


@pytest.mark.parametrize(
    ("drawer", "expected_id"),
    (("top", 0), ("middle", 1), ("bottom", 2)),
)
def test_task4_hdf5_instruction_recovers_omitted_scene_variant(
    drawer, expected_id
):
    instruction = (
        "open all drawers, then open the "
        f"{drawer} drawer again, place the butter"
    )
    assert task4_init_state_id_from_recorded_instruction(instruction) == (
        expected_id
    )


def test_all_26_tasks_have_frozen_instructions_bddl_and_stage_contracts():
    for task_id in range(1, 27):
        spec = get_robomemarena_task_spec(task_id)
        assert spec.task_id == task_id
        assert spec.instruction.endswith(".")
        assert spec.bddl_relative_path.endswith(".bddl")
        assert spec.required_stage_names


def test_task7_checker_requires_drainer_after_exactly_two_pours():
    spec = get_robomemarena_task_spec(7)

    assert spec.required_stage_names == (
        "01_Lift_Tomato_Sauce",
        "02_Pour_One",
        "03_Pour_Two",
        "04_Place_Tomato_Sauce_Bowl_Drainer",
    )
    before_drainer = {
        name: not name.endswith("Bowl_Drainer")
        for name in spec.required_stage_names
    }
    assert not reference_stage._stage_success_from_stage_done(
        7, before_drainer
    )


def test_task7_drainer_checker_accepts_both_legal_compartments():
    model = _NamedModel(
        [
            "bowl_drainer_1_left_region",
            "bowl_drainer_1_right_region",
        ],
        ["tomato_sauce_1"],
    )
    model.site_size = np.asarray(
        [[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]], dtype=np.float64
    )
    data = SimpleNamespace(
        site_xpos=np.asarray(
            [[0.0, -0.20, 0.10], [0.0, 0.20, 0.10]],
            dtype=np.float64,
        ),
        site_xmat=np.asarray([np.eye(3), np.eye(3)], dtype=np.float64),
        body_xpos=np.asarray([[0.0, -0.20, 0.10]], dtype=np.float64),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
    check = reference_stage._terminal_task_specs(7)[0].check_fn

    assert check(env, {}, 0)
    data.body_xpos[0] = np.asarray([0.0, 0.20, 0.10])
    assert check(env, {}, 0)
    data.body_xpos[0] = np.asarray([0.0, 0.0, 0.10])
    assert not check(env, {}, 0)


def test_task14_accepts_either_non_top_drawer_as_another_drawer():
    sites = [
        "wooden_cabinet_1_top_region",
        "wooden_cabinet_1_middle_region",
        "wooden_cabinet_1_bottom_region",
    ]
    bodies = ["cookies_1", "chocolate_pudding_1"]
    model = _NamedModel(sites, bodies)
    data = SimpleNamespace(
        site_xpos=np.asarray(
            [[0.0, 0.0, 0.30], [0.0, 0.0, 0.20], [0.0, 0.0, 0.10]],
            dtype=np.float64,
        ),
        body_xpos=np.asarray(
            [[0.0, -0.05, 0.30], [0.0, -0.05, 0.20]],
            dtype=np.float64,
        ),
    )
    env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
    checks = reference_stage._terminal_task_specs(14)

    assert [stage.name for stage in reference_stage._task_specs(14)][-3:] == [
        "04_Open_Other_Drawer",
        "05_Place_Chocolate_Other_Drawer",
        "06_Close_Other_Drawer",
    ]
    assert all(stage.check_fn(env, {}, 0) for stage in checks)

    data.body_xpos[1] = np.asarray([0.0, -0.05, 0.10])
    assert all(stage.check_fn(env, {}, 0) for stage in checks)

    data.body_xpos[1] = np.asarray([0.0, -0.05, 0.30])
    assert not all(stage.check_fn(env, {}, 0) for stage in checks)


def test_task21_instruction_unambiguously_refers_to_butter_current_location():
    spec = get_robomemarena_task_spec(21)

    assert "where the butter is placed" in spec.instruction
    assert "where the butter was placed" not in spec.instruction


def test_internal_source_fingerprint_needs_no_external_checkout():
    fingerprint = robomemarena_source_fingerprint(task_id=25)

    assert fingerprint["source_kind"] == "vendored_compatibility_subset"
    assert len(fingerprint["source_commit"]) == 40
    assert len(fingerprint["bddl_sha256"]) == 64
    assert len(fingerprint["runtime_stage_reference_sha256"]) == 64


def test_task4_private_checker_requires_order_and_uses_eight_stage_success():
    env = _fake_env()
    evaluator = RoboMemArenaTask4Evaluator(env)
    evaluator.reset()

    # Opening the middle drawer early must not skip the required top stage.
    env.sim.data.site_xpos[1, 1] = -0.12
    evaluator.observe({})
    assert evaluator.result()["completed_stage_names"] == []

    transitions = (
        (0, -0.12),
        (0, -0.01),
        (1, -0.12),
        (1, -0.01),
        (2, -0.12),
        (2, -0.01),
        (0, -0.12),
    )
    for site_index, y_position in transitions:
        env.sim.data.site_xpos[site_index, 1] = y_position
        env.sim.data.time += 0.05
        evaluator.observe({})

    top = env.sim.data.site_xpos[0]
    env.sim.data.body_xpos[0] = top + np.asarray([0.02, 0.02, 0.02])
    evaluator.observe({})
    result = evaluator.result()
    assert result["success"] is True
    assert result["completed_required_stage_count"] == 8
    assert result["stage_score_percent"] == 100.0
    assert result["optional_stage_names"] == [
        "09_Close_Occupied_Drawer_Final"
    ]

    env.sim.data.site_xpos[0, 1] = -0.01
    evaluator.observe({})
    assert evaluator.result()["completed_stage_names"][-1] == (
        "09_Close_Occupied_Drawer_Final"
    )


def test_task4_checker_targets_the_drawer_that_was_initially_occupied():
    env = _fake_env()
    env.sim.data.body_xpos[1] = env.sim.data.site_xpos[2] + np.asarray(
        [0.01, 0.01, 0.01]
    )
    evaluator = RoboMemArenaTask4Evaluator(env)
    evaluator.reset()

    for site_index, y_position in (
        (0, -0.12),
        (0, -0.01),
        (1, -0.12),
        (1, -0.01),
        (2, -0.12),
        (2, -0.01),
        (2, -0.12),
    ):
        env.sim.data.site_xpos[site_index, 1] = y_position
        evaluator.observe({})

    env.sim.data.body_xpos[0] = env.sim.data.site_xpos[2] + np.asarray(
        [0.02, 0.02, 0.02]
    )
    evaluator.observe({})

    assert evaluator.result()["success"] is True


def test_ordered_checker_revalidates_terminal_state(monkeypatch):
    env = _fake_env()
    terminal = {"valid": True}
    stage_names = get_robomemarena_task_spec(1).required_stage_names
    monkeypatch.setattr(
        reference_stage,
        "_task_specs",
        lambda task_id: [
            reference_stage.StageSpec(
                name, lambda env, state, start: True
            )
            for name in stage_names
        ],
    )
    monkeypatch.setattr(
        reference_stage,
        "_terminal_task_specs",
        lambda task_id: [
            reference_stage.StageSpec(
                "Terminal_Target",
                lambda env, state, start: terminal["valid"],
            )
        ],
    )
    monkeypatch.setattr(
        reference_stage,
        "_build_initial_state",
        lambda env: {"step_idx": 0, "tilt_angles": []},
    )
    monkeypatch.setattr(
        reference_stage,
        "_update_state",
        lambda obs, state: state.update(step_idx=state["step_idx"] + 1),
    )

    evaluator = RoboMemArenaOrderedStageEvaluator(env, task_id=1)
    evaluator.reset()
    evaluator.observe({})
    evaluator.observe({})
    assert evaluator.result()["success"] is True

    terminal["valid"] = False
    result = evaluator.result()
    assert result["success"] is False
    assert result["failure_reason"] == "terminal_state_invalid"


def test_source_fingerprint_requires_clean_versioned_task_inputs(tmp_path):
    root = tmp_path / "RoboMemArena"
    paths = (
        root / "evaluation_benchmark/bddl/4_drawer_butter.bddl",
        root
        / "evaluation_benchmark/libero_fork/libero/assets/"
        "articulated_objects/wooden_cabinet_tall_bottom.xml",
        root / "evaluation_benchmark/scripts/task2_26_reference_stage.py",
    )
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q", root), check=True)
    subprocess.run(
        ("git", "-C", root, "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", root, "config", "user.name", "Test"),
        check=True,
    )
    subprocess.run(("git", "-C", root, "add", "."), check=True)
    subprocess.run(("git", "-C", root, "commit", "-qm", "fixture"), check=True)

    fingerprint = robomemarena_source_fingerprint(root, task_id=4)

    assert len(fingerprint["source_commit"]) == 40
    assert len(fingerprint["bddl_sha256"]) == 64
    assert len(fingerprint["cabinet_asset_sha256"]) == 64
    assert len(fingerprint["runtime_stage_reference_sha256"]) == 64
    assert len(fingerprint["external_upstream_stage_reference_sha256"]) == 64

    paths[0].write_text("modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked modifications"):
        robomemarena_source_fingerprint(root, task_id=4)


def _write_full_trajectory(path, *, action_value=0.1, gripper_value=None):
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["language_instruction"] = "open and close all drawers"
        actions = np.full((2, 7), action_value, dtype=np.float64)
        if gripper_value is not None:
            actions[:, 6] = gripper_value
        demo.create_dataset("actions", data=actions)
        observations = demo.create_group("obs")
        observations.create_dataset(
            "agentview_rgb", data=np.zeros((2, 4, 4, 3), dtype=np.uint8)
        )
        observations.create_dataset(
            "eye_in_hand_rgb", data=np.zeros((2, 4, 4, 3), dtype=np.uint8)
        )
        observations.create_dataset("ee_pos", data=np.zeros((2, 3)))
        observations.create_dataset("ee_ori", data=np.zeros((2, 3)))
        observations.create_dataset("gripper_states", data=np.zeros((2, 2)))
        observations.create_dataset("joint_states", data=np.zeros((2, 7)))


def test_full_trajectory_loader_recovers_seed_and_validates_alignment(tmp_path):
    dataset = tmp_path / "example_full_seed123_task4.hdf5"
    _write_full_trajectory(dataset)

    trajectory = load_robomemarena_full_trajectory(
        dataset, expected_task_id=4
    )

    assert trajectory.task_id == 4
    assert trajectory.seed == 123
    assert trajectory.actions.shape == (2, 7)
    assert trajectory.gripper_action_clipped_to_contract is False
    assert trajectory.raw_gripper_action_range == (0.1, 0.1)
    assert trajectory.observation_count == 2
    assert trajectory.recorded_instruction == "open and close all drawers"


def test_full_trajectory_loader_rejects_out_of_range_actions(tmp_path):
    dataset = tmp_path / "example_full_seed123_task4.hdf5"
    _write_full_trajectory(dataset, action_value=1.1)

    with pytest.raises(ValueError, match="finite normalized OSC"):
        load_robomemarena_full_trajectory(dataset, expected_task_id=4)


def test_full_trajectory_loader_normalizes_upstream_gripper_plus_two(tmp_path):
    dataset = tmp_path / "example_full_seed123_task10.hdf5"
    _write_full_trajectory(dataset, gripper_value=2.0)

    trajectory = load_robomemarena_full_trajectory(
        dataset, expected_task_id=10
    )

    assert trajectory.gripper_action_clipped_to_contract is True
    assert trajectory.raw_gripper_action_range == (2.0, 2.0)
    np.testing.assert_array_equal(trajectory.actions[:, 6], [1.0, 1.0])
