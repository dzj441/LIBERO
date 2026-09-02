import json

import numpy as np
import pytest
from PIL import Image

from libero.libero.agent_env.artifacts import (
    depth_preview_with_metadata,
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
    entity = metadata["annotations"]["cameras"]["head"]["task_entities"][
        "entity_000"
    ]
    assert (tmp_path / entity["mask_file"]).is_file()
    assert "mask" not in entity


def test_depth_preview_falls_back_to_finite_extrema_for_dominant_surface():
    depth = np.full((10, 10), 0.1, dtype=np.float32)
    depth[0, 0] = 0.5
    valid = np.ones(depth.shape, dtype=np.bool_)

    preview, metadata = depth_preview_with_metadata(depth, valid)

    assert metadata["preview_range_source"] == "finite_min_max_fallback"
    assert metadata["preview_mapping"] == "inverse_depth_linear_near_white"
    assert metadata["preview_near_m"] == pytest.approx(0.1)
    assert metadata["preview_far_m"] == pytest.approx(0.5)
    assert preview[0, 0] == 0
    assert preview[1, 1] == 255


def test_depth_preview_uses_percentile_range_for_normal_distribution():
    depth = np.linspace(0.1, 0.8, 100, dtype=np.float32).reshape(10, 10)
    valid = np.ones(depth.shape, dtype=np.bool_)

    preview, metadata = depth_preview_with_metadata(depth, valid)

    assert metadata["preview_range_source"] == "inverse_depth_percentile_2_98"
    assert metadata["preview_mapping"] == "inverse_depth_linear_near_white"
    assert metadata["preview_near_m"] < metadata["preview_far_m"]
    assert preview[0, 0] == 255
    assert preview[-1, -1] == 0


def test_depth_preview_keeps_invalid_and_nonfinite_pixels_black():
    depth = np.array(
        [[0.1, np.nan], [np.inf, 0.0]], dtype=np.float32
    )
    valid = np.ones(depth.shape, dtype=np.bool_)

    preview, metadata = depth_preview_with_metadata(depth, valid)

    assert preview[0, 0] == 255
    np.testing.assert_array_equal(
        preview, np.array([[255, 0], [0, 0]], dtype=np.uint8)
    )
    assert metadata["preview_near_m"] == pytest.approx(0.1)
    assert metadata["preview_far_m"] == pytest.approx(0.1)
    assert metadata["preview_range_source"] == "degenerate"


def test_materialized_depth_preserves_raw_array_and_publishes_preview_metadata(
    tmp_path,
):
    public = project_public_observation(_master(), "level4")
    raw_depth = np.array(
        [[0.1, 0.5], [np.nan, np.inf]], dtype=np.float32
    )
    public["cameras"]["head"]["depth_m"] = raw_depth.copy()
    public["cameras"]["head"]["depth_valid_mask"] = np.ones(
        raw_depth.shape, dtype=np.bool_
    )

    json_path = write_public_observation(public, tmp_path)
    metadata = json.loads(json_path.read_text())
    depth_metadata = metadata["cameras"]["head"]["depth"]

    np.testing.assert_array_equal(
        np.load(tmp_path / depth_metadata["metric_file"], allow_pickle=False),
        raw_depth,
    )
    assert depth_metadata["preview_mapping"] == "inverse_depth_linear_near_white"
    assert depth_metadata["preview_range_source"] == "inverse_depth_percentile_2_98"
    assert (
        0.1
        <= depth_metadata["preview_near_m"]
        < depth_metadata["preview_far_m"]
        <= 0.5
    )
    valid_preview = np.asarray(
        Image.open(tmp_path / depth_metadata["valid_mask_file"])
    )
    np.testing.assert_array_equal(
        valid_preview, np.array([[255, 255], [0, 0]], dtype=np.uint8)
    )
    preview = np.asarray(Image.open(tmp_path / depth_metadata["preview_file"]))
    assert np.all(preview[1, :] == 0)


def test_materialized_level1_has_no_hidden_level4_files(tmp_path):
    public = project_public_observation(_master(), "level1")
    json_path = write_public_observation(public, tmp_path)
    metadata = json.loads(json_path.read_text())

    assert "depth" not in metadata["cameras"]["head"]
    assert not (tmp_path / "head" / "depth_m.npy").exists()
    assert not (tmp_path / "annotations").exists()


def test_materialized_task_reference_uses_a_plain_png_file(tmp_path):
    public = project_public_observation(
        _master(task_reference=True), "level1"
    )
    json_path = write_public_observation(public, tmp_path)
    metadata = json.loads(json_path.read_text())
    reference = metadata["task_reference"]

    assert reference["semantics"] == "desired_object_arrangement"
    assert set(reference["rgb"]) == {"file", "media_type", "shape"}
    assert reference["rgb"]["file"] == "task_reference/rgb.png"
    reference_path = tmp_path / reference["rgb"]["file"]
    assert reference_path.is_file()
    assert reference_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(reference_path).info == {}
    assert reference["rgb"]["shape"] == [2, 3, 3]


def test_current_only_replacement_removes_stale_task_reference(tmp_path):
    current = tmp_path / "current"
    replace_current_public_observation(
        project_public_observation(_master(task_reference=True), "level1"),
        current,
    )
    assert (current / "task_reference" / "rgb.png").is_file()

    replace_current_public_observation(
        project_public_observation(_master(frame_index=1), "level1"),
        current,
    )
    assert not (current / "task_reference").exists()
    assert "task_reference" not in json.loads(
        (current / "observation.json").read_text()
    )


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
                    f"annotations/{camera}/entity_000_mask.png",
                    f"annotations/{camera}/entity_001_mask.png",
                    f"annotations/{camera}/entity_002_mask.png",
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
