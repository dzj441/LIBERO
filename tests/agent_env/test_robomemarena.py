from types import SimpleNamespace
import subprocess

import h5py
import numpy as np
import pytest

from libero.libero.agent_env.robomemarena import (
    RoboMemArenaTask4Evaluator,
    get_robomemarena_task_spec,
    robomemarena_source_fingerprint,
)
from libero.libero.agent_env.robomemarena_demo import (
    load_robomemarena_full_trajectory,
)


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
    bodies = ["butter_1"]
    data = SimpleNamespace(
        site_xpos=np.asarray(
            [[0.0, 0.0, 0.30], [0.0, 0.0, 0.20], [0.0, 0.0, 0.10]],
            dtype=np.float64,
        ),
        body_xpos=np.asarray([[0.50, 0.50, 0.50]], dtype=np.float64),
        time=0.0,
    )
    sim = SimpleNamespace(model=_NamedModel(sites, bodies), data=data)
    return SimpleNamespace(sim=sim)


def test_task4_contract_names_eight_required_and_one_optional_stage():
    spec = get_robomemarena_task_spec(4)
    assert "all drawers in order" in spec.instruction
    assert len(spec.required_stage_names) == 8
    assert spec.optional_stage_names == ("09_Close_Top_Drawer_Final",)


def test_all_26_tasks_have_frozen_instructions_bddl_and_stage_contracts():
    for task_id in range(1, 27):
        spec = get_robomemarena_task_spec(task_id)
        assert spec.task_id == task_id
        assert spec.instruction.endswith(".")
        assert spec.bddl_relative_path.endswith(".bddl")
        assert spec.required_stage_names


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
    assert result["optional_stage_names"] == ["09_Close_Top_Drawer_Final"]

    env.sim.data.site_xpos[0, 1] = -0.01
    evaluator.observe({})
    assert evaluator.result()["completed_stage_names"][-1] == (
        "09_Close_Top_Drawer_Final"
    )


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
