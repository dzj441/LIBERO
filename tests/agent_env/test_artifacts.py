import json

import numpy as np
import pytest

from libero.libero.agent_env.artifacts import (
    replace_current_public_observation,
    write_public_observation,
)
from libero.libero.agent_env.profiles import project_public_observation
from test_profiles import _master


def test_materialized_level4_frame_uses_files_for_dense_arrays(tmp_path):
    public = project_public_observation(_master(), "level4")
    json_path = write_public_observation(public, tmp_path)
    metadata = json.loads(json_path.read_text())

    assert (tmp_path / metadata["cameras"]["head"]["rgb"]["file"]).is_file()
    depth = metadata["cameras"]["head"]["depth"]
    assert np.load(tmp_path / depth["metric_file"], allow_pickle=False).dtype == np.float32
    assert (tmp_path / depth["preview_file"]).is_file()
    role = metadata["annotations"]["cameras"]["head"]["manipulated_object"]
    assert (tmp_path / role["mask_file"]).is_file()
    assert "mask" not in role


def test_materialized_level1_has_no_hidden_level4_files(tmp_path):
    public = project_public_observation(_master(), "level1")
    json_path = write_public_observation(public, tmp_path)
    metadata = json.loads(json_path.read_text())

    assert "depth" not in metadata["cameras"]["head"]
    assert not (tmp_path / "head" / "depth_m.npy").exists()
    assert not (tmp_path / "annotations").exists()


def test_current_only_replacement_removes_stale_modalities(tmp_path):
    current = tmp_path / "current"
    replace_current_public_observation(
        project_public_observation(_master(), "level4"), current
    )
    assert (current / "annotations").is_dir()
    assert (current / "head" / "depth_m.npy").is_file()

    replace_current_public_observation(
        project_public_observation(_master(frame_index=1), "level1"), current
    )
    assert not (current / "annotations").exists()
    assert not (current / "head" / "depth_m.npy").exists()
    assert (current / "head" / "rgb.png").is_file()


@pytest.mark.parametrize("profile_number", (1, 2, 3, 4))
@pytest.mark.parametrize("frame_index", (0, 1))
def test_materialized_profile_matrix_has_exact_file_surface(
    tmp_path, profile_number, frame_index
):
    profile = f"level{profile_number}"
    root = tmp_path / f"{profile}-frame-{frame_index}"
    public = project_public_observation(_master(frame_index=frame_index), profile)
    write_public_observation(public, root)

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_files = {
        "observation.json",
        "head/rgb.png",
        "wrist/rgb.png",
    }
    if profile_number >= 4:
        for camera in ("head", "wrist"):
            expected_files.update(
                {
                    f"{camera}/depth_m.npy",
                    f"{camera}/depth_valid_mask.png",
                    f"{camera}/depth_visualization.png",
                }
            )
    if profile_number >= 2 and frame_index == 0:
        for camera in ("head", "wrist"):
            expected_files.update(
                {
                    f"annotations/{camera}/manipulated_object_mask.png",
                    f"annotations/{camera}/goal_fixture_mask.png",
                    f"annotations/{camera}/annotations_overlay.png",
                }
            )
    assert actual_files == expected_files

    metadata = json.loads((root / "observation.json").read_text())
    assert ("proprioception" in metadata) is (profile_number >= 3)
    assert ("annotations" in metadata) is (
        profile_number >= 2 and frame_index == 0
    )
    for camera in ("head", "wrist"):
        assert ("depth" in metadata["cameras"][camera]) is (profile_number >= 4)

    rendered = (root / "observation.json").read_text()
    for forbidden in (
        "private",
        "reward",
        "checker",
        "actor_pose",
        "contact_points",
        "instance_id",
        "raw_segmentation",
    ):
        assert forbidden not in rendered
