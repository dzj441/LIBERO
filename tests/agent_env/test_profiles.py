from copy import deepcopy

import numpy as np
import pytest

from libero.libero.agent_env.profiles import (
    ObservationProfile,
    project_public_observation,
)
from libero.libero.agent_env.annotation_contract import (
    TASK_ENTITY_ANNOTATION_SCHEMA_VERSION,
)


def _master(frame_index=0, *, task_reference=False):
    annotation = {
        "visible": True,
        "visible_pixel_count": 4,
        "bbox_xyxy": [0, 0, 2, 2],
        "mask": np.ones((2, 2), dtype=np.bool_),
        "private_instance_id": 41,
    }
    camera = {
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth_m": np.ones((2, 2), dtype=np.float32),
        "depth_valid_mask": np.ones((2, 2), dtype=np.bool_),
        "intrinsic_matrix_3x3": np.eye(3),
        "matrix_T_robot_base_from_camera_opencv_4x4": np.eye(4),
        "private_camera_target": [1, 2, 3],
    }
    master = {
        "observation_id": f"obs_{frame_index:06d}",
        "frame_index": frame_index,
        "sim_time_s": 1.25,
        "coordinate_conventions": {
            "robot_state_frame": "robot_base",
            "eef_delta_frame": "robot_base",
            "eef_rotation_delta": "rotation_vector_rad_left_applied",
            "quaternion_order": "xyzw",
            "camera_frame": "OpenCV: +X right, +Y down, +Z forward",
            "camera_extrinsic": "T_robot_base_from_camera",
            "bbox_xyxy": "exclusive",
            "length_unit": "metre",
            "private_planner_frame": "secret",
        },
        "state": {
            "arm_joint_position_rad_7d": np.zeros(7),
            "gripper_finger_joint_position_m_2d": np.zeros(2),
            "gripper_width_m": 0.04,
            "eef_pose_robot_base_xyzw_7d": np.zeros(7),
            "private_actor_pose": np.ones(7),
        },
        "proprioception": {
            "arm_joint_velocity_rad_s_7d": np.zeros(7),
            "gripper_finger_joint_velocity_m_s_2d": np.zeros(2),
            "gripper_width_velocity_m_s": 0.0,
            "commanded_arm_joint_torque_nm_7d": np.zeros(7),
            "commanded_arm_joint_torque_available": True,
            "eef_force_sensor_n_3d": np.zeros(3),
            "eef_torque_sensor_nm_3d": np.zeros(3),
            "eef_twist_robot_base_6d": np.zeros(6),
            "force_torque_frame": "eef_ft_sensor",
            "private_contact_points": [[0, 0, 0]],
        },
        "cameras": {"head": deepcopy(camera), "wrist": deepcopy(camera)},
        "annotations": {
            "schema_version": TASK_ENTITY_ANNOTATION_SCHEMA_VERSION,
            "schedule": "initial_observation_only",
            "cameras": {
                name: {
                    "task_entities": {
                        f"entity_{index:03d}": deepcopy(annotation)
                        for index in range(3)
                    }
                }
                for name in ("head", "wrist")
            },
            "private_raw_segmentation": np.ones((2, 2), dtype=np.int32),
        },
        "reward": 1.0,
        "checker_details": {"secret": True},
    }
    if task_reference:
        master["task_reference"] = {
            "semantics": "desired_object_arrangement",
            "rgb": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        }
    return master


@pytest.mark.parametrize("profile", list(ObservationProfile))
def test_all_profiles_keep_base_rgb_and_state(profile):
    public = project_public_observation(_master(), profile)
    assert set(public["cameras"]) == {"head", "wrist"}
    assert set(public["cameras"]["head"]) >= {"rgb"}
    assert set(public["state"]) == {
        "arm_joint_position_rad_7d",
        "gripper_finger_joint_position_m_2d",
        "gripper_width_m",
        "eef_pose_robot_base_xyzw_7d",
    }


def test_profile_modalities_are_strict_supersets():
    p1 = project_public_observation(_master(), "level1")
    p2 = project_public_observation(_master(), "level2")
    p3 = project_public_observation(_master(), "level3")
    p4 = project_public_observation(_master(), "level4")

    assert "annotations" not in p1
    assert "annotations" in p2
    assert "proprioception" not in p2
    assert "proprioception" in p3
    assert "depth_m" not in p3["cameras"]["head"]
    assert "depth_m" in p4["cameras"]["head"]
    assert "intrinsic_matrix_3x3" in p4["cameras"]["wrist"]


def test_annotations_are_initial_observation_only():
    public = project_public_observation(_master(frame_index=1), "level4")
    assert "annotations" not in public


@pytest.mark.parametrize("profile", list(ObservationProfile))
def test_all_profiles_preserve_task_reference(profile):
    public = project_public_observation(
        _master(task_reference=True), profile
    )

    assert public["task_reference"]["semantics"] == "desired_object_arrangement"
    np.testing.assert_array_equal(
        public["task_reference"]["rgb"],
        _master(task_reference=True)["task_reference"]["rgb"],
    )


def test_task_reference_is_persistent_across_frames():
    first = project_public_observation(
        _master(frame_index=0, task_reference=True), "level1"
    )
    later = project_public_observation(
        _master(frame_index=1, task_reference=True), "level1"
    )

    assert first["task_reference"]["semantics"] == "desired_object_arrangement"
    assert later["task_reference"]["semantics"] == "desired_object_arrangement"
    np.testing.assert_array_equal(
        first["task_reference"]["rgb"], later["task_reference"]["rgb"]
    )


def test_legacy_task_reference_semantics_are_accepted_and_preserved():
    master = _master(task_reference=True)
    master["task_reference"]["semantics"] = "desired_final_state"

    public = project_public_observation(master, "level1")

    assert public["task_reference"]["semantics"] == "desired_final_state"
    np.testing.assert_array_equal(
        public["task_reference"]["rgb"], master["task_reference"]["rgb"]
    )


@pytest.mark.parametrize(
    "reference",
    (
        {"semantics": "desired_object_arrangement", "rgb": [[[]]]},
        {
            "semantics": "desired_object_arrangement",
            "rgb": np.zeros((2, 2, 3), dtype=np.float32),
        },
        {
            "semantics": "desired_object_arrangement",
            "rgb": np.zeros((2, 2), dtype=np.uint8),
        },
        {
            "semantics": "desired_object_arrangement",
            "rgb": np.zeros((2, 2, 1), dtype=np.uint8),
        },
        {
            "semantics": "desired_object_arrangement",
            "rgb": np.zeros((0, 2, 3), dtype=np.uint8),
        },
        {
            "semantics": "desired_object_arrangement",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "extra": "not allowed",
        },
        {
            "semantics": "not_the_contract",
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
        },
    ),
)
def test_invalid_task_reference_is_rejected(reference):
    master = _master()
    master["task_reference"] = reference
    with pytest.raises((TypeError, ValueError)):
        project_public_observation(master, "level1")


def test_regular_master_has_no_task_reference():
    for profile in ObservationProfile:
        assert "task_reference" not in project_public_observation(
            _master(), profile
        )


def test_task_entities_are_anonymous_variable_cardinality_and_camera_aligned():
    public = project_public_observation(_master(), "level2")
    annotations = public["annotations"]
    assert annotations["schema_version"] == "libero.task_entities.v1"
    for camera_name in ("head", "wrist"):
        assert tuple(
            annotations["cameras"][camera_name]["task_entities"]
        ) == ("entity_000", "entity_001", "entity_002")
    rendered = repr(annotations)
    assert "manipulated_object" not in rendered
    assert "goal_fixture" not in rendered


def test_level2_initial_observation_fails_closed_without_annotations():
    master = _master()
    del master["annotations"]
    with pytest.raises(KeyError, match="missing annotations"):
        project_public_observation(master, "level2")


def test_nested_private_fields_and_checker_data_do_not_leak():
    public = project_public_observation(_master(), "level4")
    rendered = repr(public)
    for forbidden in (
        "private",
        "reward",
        "checker",
        "actor_pose",
        "contact_points",
        "instance_id",
    ):
        assert forbidden not in rendered
