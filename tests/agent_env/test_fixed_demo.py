import hashlib
import json
from pathlib import Path

import pytest

from libero.libero.agent_env.artifacts import write_public_observation
from libero.libero.agent_env.fixed_demo import (
    P4_MASTER_SCHEMA_VERSION,
    contact_sheet_indices,
    project_fixed_demo_bundle,
    source_action_semantics,
    validate_fixed_demo_bundle,
    validate_p4_replay_master,
)
from libero.libero.agent_env.profiles import project_public_observation
from test_profiles import _master


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path, media_type: str):
    return {
        "path": str(path.relative_to(root)),
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _inventory(root: Path, directory: Path):
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _test_master(root: Path) -> Path:
    frames = []
    for index in range(2):
        frame_id = f"frame_{index:06d}"
        internal = _master(frame_index=index)
        internal["observation_id"] = frame_id
        frame_root = root / "frames" / frame_id
        observation = write_public_observation(
            project_public_observation(internal, "level4"), frame_root
        )
        frames.append(
            {
                "frame_index": index,
                "observation_id": frame_id,
                "observation": str(observation.relative_to(root)),
                "source_action_index": None if index == 0 else index - 1,
                "files": _inventory(root, frame_root),
            }
        )

    trajectory = root / "source_trajectory.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "transition_index": 0,
                "observation_before": "frames/frame_000000/observation.json",
                "source_action": {
                    "schema_version": "libero.normalized_osc_pose_action.v1",
                    "normalized_vector_7d": [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0],
                },
                "observation_after": "frames/frame_000001/observation.json",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": P4_MASTER_SCHEMA_VERSION,
        "task": {
            "instruction": "pick up the alphabet soup and place it in the basket",
            "problem_name": "private_problem_name",
            "environment_name": "private_environment_name",
        },
        "source": {
            "dataset_path": "/private/source/demo.hdf5",
            "demo_key": "demo_0",
        },
        "capture": {
            "profile": "level4",
            "frame_count": 2,
            "transition_count": 1,
            "frames": frames,
            "trajectory": _artifact(trajectory, root, "application/x-ndjson"),
        },
        "action_semantics": source_action_semantics(),
        "verification": {
            "verified_success": True,
            "first_success_step": 0,
        },
    }
    (root / "p4_master_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_project_level4_bundle_keeps_actions_and_drops_private_provenance(tmp_path):
    master = _test_master(tmp_path / "master")
    validate_p4_replay_master(master)
    bundle = tmp_path / "bundle"
    project_fixed_demo_bundle(
        master_root=master,
        destination=bundle,
        profile="level4",
        expected_task_instruction=(
            "pick up the alphabet soup and place it in the basket"
        ),
    )
    manifest = validate_fixed_demo_bundle(
        bundle,
        expected_profile="level4",
        expected_task_instruction=(
            "pick up the alphabet soup and place it in the basket"
        ),
    )
    assert manifest["demonstration"]["actions_present"] is True
    assert manifest["demonstration"]["frame_count"] == 2
    assert len((bundle / "trajectory.jsonl").read_text().splitlines()) == 1
    assert (bundle / "frames/frame_000000/annotations").is_dir()
    assert not (bundle / "frames/frame_000001/annotations").exists()
    rendered = "\n".join(
        path.read_text(errors="ignore") for path in bundle.rglob("*.json*")
    )
    assert "/private/source" not in rendered
    assert "first_success_step" not in rendered


@pytest.mark.parametrize("profile_number", (1, 2, 3, 4))
def test_file_level_projection_matches_every_profile_allowlist(
    tmp_path, profile_number
):
    master = _test_master(tmp_path / "master")
    bundle = tmp_path / "bundle"
    profile = f"level{profile_number}"
    project_fixed_demo_bundle(
        master_root=master,
        destination=bundle,
        profile=profile,
        expected_task_instruction=(
            "pick up the alphabet soup and place it in the basket"
        ),
    )
    validate_fixed_demo_bundle(
        bundle,
        expected_profile=profile,
        expected_task_instruction=(
            "pick up the alphabet soup and place it in the basket"
        ),
    )
    initial = json.loads(
        (bundle / "frames/frame_000000/observation.json").read_text()
    )
    subsequent = json.loads(
        (bundle / "frames/frame_000001/observation.json").read_text()
    )
    assert ("proprioception" in initial) is (profile_number >= 3)
    assert ("annotations" in initial) is (profile_number >= 2)
    assert "annotations" not in subsequent
    for camera in ("head", "wrist"):
        assert ("depth" in initial["cameras"][camera]) is (profile_number >= 4)
        assert (
            bundle / "frames" / "frame_000000" / camera / "depth_m.npy"
        ).exists() is (profile_number >= 4)

    rendered = "\n".join(
        path.read_text(errors="ignore") for path in bundle.rglob("*.json*")
    )
    for forbidden in (
        "/private/source",
        "dataset_path",
        "bddl_file",
        "mujoco_state",
        "raw_segmentation",
        "object_ground_truth_pose",
        "first_success_step",
        "checker",
    ):
        assert forbidden not in rendered


def test_contact_sheet_sampling_is_deterministic_and_keeps_endpoints():
    assert contact_sheet_indices(3) == [0, 1, 2]
    sampled = contact_sheet_indices(149)
    assert len(sampled) == 12
    assert sampled[0] == 0
    assert sampled[-1] == 148
    assert sampled == sorted(set(sampled))
