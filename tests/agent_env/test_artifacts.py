import json

import numpy as np

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
