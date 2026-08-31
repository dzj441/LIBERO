from types import SimpleNamespace
import subprocess

import numpy as np
import pytest

from libero.libero.agent_env.robomemarena import (
    RoboMemArenaTask4Evaluator,
    get_robomemarena_task_spec,
    robomemarena_source_fingerprint,
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
    assert spec.optional_stage_names == ("09_close_top_drawer_final",)


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
    assert result["optional_stage_names"] == ["09_close_top_drawer_final"]

    env.sim.data.site_xpos[0, 1] = -0.01
    evaluator.observe({})
    assert evaluator.result()["completed_stage_names"][-1] == (
        "09_close_top_drawer_final"
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
    assert len(fingerprint["stage_reference_sha256"]) == 64

    paths[0].write_text("modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked modifications"):
        robomemarena_source_fingerprint(root, task_id=4)
