"""Observation-profile definitions and strict public projection.

The projector is deliberately an allowlist.  Adding a value to the internal
master observation never makes it public until it is explicitly selected here.
"""

from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
from typing import Any, Mapping


COORDINATE_CONVENTION_FIELDS = (
    "robot_state_frame",
    "eef_delta_frame",
    "eef_rotation_delta",
    "quaternion_order",
    "camera_frame",
    "camera_extrinsic",
    "bbox_xyxy",
    "length_unit",
)
STATE_FIELDS = (
    "arm_joint_position_rad_7d",
    "gripper_finger_joint_position_m_2d",
    "gripper_width_m",
    "eef_pose_robot_base_xyzw_7d",
)
PROPRIOCEPTION_FIELDS = (
    "arm_joint_velocity_rad_s_7d",
    "gripper_finger_joint_velocity_m_s_2d",
    "gripper_width_velocity_m_s",
    "commanded_arm_joint_torque_nm_7d",
    "commanded_arm_joint_torque_available",
    "eef_force_sensor_n_3d",
    "eef_torque_sensor_nm_3d",
    "eef_twist_robot_base_6d",
    "force_torque_frame",
)


class ObservationProfile(IntEnum):
    """Agent-visible observation levels."""

    LEVEL1 = 1
    LEVEL2 = 2
    LEVEL3 = 3
    LEVEL4 = 4

    @classmethod
    def parse(cls, value: "ObservationProfile | int | str") -> "ObservationProfile":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        normalized = str(value).strip().lower().replace("_", "")
        aliases = {
            "1": cls.LEVEL1,
            "p1": cls.LEVEL1,
            "level1": cls.LEVEL1,
            "2": cls.LEVEL2,
            "p2": cls.LEVEL2,
            "level2": cls.LEVEL2,
            "3": cls.LEVEL3,
            "p3": cls.LEVEL3,
            "level3": cls.LEVEL3,
            "4": cls.LEVEL4,
            "p4": cls.LEVEL4,
            "level4": cls.LEVEL4,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown observation profile: {value!r}") from exc

    @property
    def public_name(self) -> str:
        return f"level{int(self)}"


def profile_capabilities(profile: ObservationProfile | int | str) -> dict[str, Any]:
    """Return declarative, non-task-specific capabilities for a profile."""

    profile = ObservationProfile.parse(profile)
    return {
        "head_rgb": True,
        "wrist_rgb": True,
        "kinematic_state": True,
        "initial_bbox_and_mask": profile >= ObservationProfile.LEVEL2,
        "dynamic_proprioception": profile >= ObservationProfile.LEVEL3,
        "metric_depth": profile >= ObservationProfile.LEVEL4,
        "camera_calibration": profile >= ObservationProfile.LEVEL4,
        "annotation_schedule": (
            "initial_observation_only"
            if profile >= ObservationProfile.LEVEL2
            else "disabled"
        ),
    }


def project_public_observation(
    master: Mapping[str, Any],
    profile: ObservationProfile | int | str,
) -> dict[str, Any]:
    """Project an internal P4 master frame through a strict public allowlist.

    ``master`` is host-private and may grow over time.  This function copies
    only the fields defined by the public contract.  In particular, reward,
    checker state, object poses, raw segmentation ids, and task metadata are
    never forwarded implicitly.
    """

    profile = ObservationProfile.parse(profile)
    required = {
        "observation_id",
        "frame_index",
        "sim_time_s",
        "state",
        "cameras",
    }
    missing = required.difference(master)
    if missing:
        raise KeyError(f"master observation missing required fields: {sorted(missing)}")

    public: dict[str, Any] = {
        "schema_version": "libero.agent_observation.v1",
        "observation_id": str(master["observation_id"]),
        "frame_index": int(master["frame_index"]),
        "sim_time_s": float(master["sim_time_s"]),
        "profile": profile.public_name,
        "capabilities": profile_capabilities(profile),
        "coordinate_conventions": _copy_required_fields(
            master["coordinate_conventions"], COORDINATE_CONVENTION_FIELDS
        ),
        "state": _copy_required_fields(master["state"], STATE_FIELDS),
        "cameras": {},
    }

    for public_camera_name in ("head", "wrist"):
        source_camera = master["cameras"][public_camera_name]
        public_camera: dict[str, Any] = {"rgb": deepcopy(source_camera["rgb"])}
        if profile >= ObservationProfile.LEVEL4:
            public_camera.update(
                {
                    "depth_m": deepcopy(source_camera["depth_m"]),
                    "depth_valid_mask": deepcopy(source_camera["depth_valid_mask"]),
                    "intrinsic_matrix_3x3": deepcopy(
                        source_camera["intrinsic_matrix_3x3"]
                    ),
                    "matrix_T_robot_base_from_camera_opencv_4x4": deepcopy(
                        source_camera[
                            "matrix_T_robot_base_from_camera_opencv_4x4"
                        ]
                    ),
                }
            )
        public["cameras"][public_camera_name] = public_camera

    if profile >= ObservationProfile.LEVEL3:
        public["proprioception"] = _copy_required_fields(
            master["proprioception"], PROPRIOCEPTION_FIELDS
        )

    if profile >= ObservationProfile.LEVEL2 and public["frame_index"] == 0:
        if "annotations" not in master:
            raise KeyError("Level 2+ initial observation is missing annotations")
        annotations = master["annotations"]
        public["annotations"] = {
            "schedule": str(annotations["schedule"]),
            "cameras": {},
        }
        for camera_name in ("head", "wrist"):
            public_roles: dict[str, Any] = {}
            for role in ("manipulated_object", "goal_fixture"):
                public_roles[role] = _copy_required_fields(
                    annotations["cameras"][camera_name][role],
                    ("visible", "visible_pixel_count", "bbox_xyxy", "mask"),
                )
            public["annotations"]["cameras"][camera_name] = public_roles

    return public


def _copy_required_fields(
    source: Mapping[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    missing = set(fields).difference(source)
    if missing:
        raise KeyError(f"source missing public contract fields: {sorted(missing)}")
    return {field: deepcopy(source[field]) for field in fields}
