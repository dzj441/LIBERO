"""Collect a host-private P4 master frame from a live LIBERO environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import robosuite.macros as robosuite_macros
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from .control import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix


CAMERA_SOURCES = {
    "head": "agentview",
    "wrist": "robot0_eye_in_hand",
}


@dataclass(frozen=True)
class AnnotationRoles:
    """Host-private instance names used to produce public semantic roles."""

    manipulated_object: str
    goal_fixture: str


def infer_annotation_roles(parsed_problem: Mapping[str, Any]) -> AnnotationRoles:
    """Infer pick/place annotation roles, failing closed when ambiguous."""

    object_instances = {
        instance
        for instances in parsed_problem.get("objects", {}).values()
        for instance in instances
    }
    fixture_instances = {
        instance
        for instances in parsed_problem.get("fixtures", {}).values()
        for instance in instances
    }
    known_instances = object_instances | fixture_instances
    regions = parsed_problem.get("regions", {})
    goals = parsed_problem.get("goal_state", [])

    manipulated: str | None = None
    goal_fixture: str | None = None
    for goal in goals:
        if len(goal) < 2:
            continue
        if manipulated is None and goal[1] in object_instances:
            manipulated = goal[1]
        for argument in goal[2:]:
            candidate = argument if argument in known_instances else None
            if argument in regions:
                region_target = regions[argument].get("target")
                if region_target in known_instances:
                    candidate = region_target
            if candidate is not None and candidate != manipulated:
                goal_fixture = candidate
                break
        if manipulated is not None and goal_fixture is not None:
            break

    objects_of_interest = list(parsed_problem.get("obj_of_interest", []))
    if manipulated is None and objects_of_interest:
        candidate = objects_of_interest[0]
        if candidate in object_instances:
            manipulated = candidate
    if goal_fixture is None:
        candidates = [name for name in objects_of_interest if name != manipulated]
        if len(candidates) == 1 and candidates[0] in known_instances:
            goal_fixture = candidates[0]

    if manipulated is None or goal_fixture is None:
        raise ValueError(
            "could not infer manipulated_object and goal_fixture from this task; "
            "pass AnnotationRoles explicitly"
        )
    return AnnotationRoles(manipulated, goal_fixture)


class MasterObservationCollector:
    """Collect all candidate public signals without checker or object poses."""

    def __init__(
        self,
        env: Any,
        camera_height: int,
        camera_width: int,
        annotation_roles: AnnotationRoles | None = None,
    ) -> None:
        self.env = env
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.annotation_roles = annotation_roles or infer_annotation_roles(
            env.env.parsed_problem
        )

    def collect(self, raw_observation: Mapping[str, Any], frame_index: int) -> dict[str, Any]:
        robot = self.env.robots[0]
        controller = robot.controller
        controller.update(force=True)

        rotation_world_from_base = quaternion_xyzw_to_matrix(robot.base_ori)
        position_world_from_base = np.asarray(robot.base_pos, dtype=np.float64)
        matrix_world_from_base = np.eye(4)
        matrix_world_from_base[:3, :3] = rotation_world_from_base
        matrix_world_from_base[:3, 3] = position_world_from_base
        matrix_base_from_world = np.linalg.inv(matrix_world_from_base)

        eef_position_base = rotation_world_from_base.T @ (
            np.asarray(controller.ee_pos, dtype=np.float64) - position_world_from_base
        )
        eef_rotation_base = rotation_world_from_base.T @ np.asarray(
            controller.ee_ori_mat, dtype=np.float64
        )
        eef_quaternion_base = matrix_to_quaternion_xyzw(eef_rotation_base)

        gripper_qpos = np.asarray(
            [robot.sim.data.qpos[index] for index in robot._ref_gripper_joint_pos_indexes],
            dtype=np.float64,
        )
        gripper_qvel = np.asarray(
            [robot.sim.data.qvel[index] for index in robot._ref_gripper_joint_vel_indexes],
            dtype=np.float64,
        )

        torques = None if robot.torques is None else np.asarray(robot.torques, dtype=np.float64)
        controller_linear_velocity_base = rotation_world_from_base.T @ np.asarray(
            controller.ee_pos_vel, dtype=np.float64
        )
        controller_angular_velocity_base = rotation_world_from_base.T @ np.asarray(
            controller.ee_ori_vel, dtype=np.float64
        )

        master: dict[str, Any] = {
            "observation_id": f"obs_{frame_index:06d}",
            "frame_index": int(frame_index),
            "sim_time_s": float(self.env.sim.data.time),
            "coordinate_conventions": {
                "robot_state_frame": "robot_base",
                "eef_delta_frame": "robot_base",
                "eef_rotation_delta": "rotation_vector_rad_left_applied",
                "quaternion_order": "xyzw",
                "camera_frame": "OpenCV: +X right, +Y down, +Z forward",
                "camera_extrinsic": "T_robot_base_from_camera",
                "bbox_xyxy": "[x_min, y_min, x_max_exclusive, y_max_exclusive]",
                "length_unit": "metre",
            },
            "state": {
                "arm_joint_position_rad_7d": np.asarray(
                    robot._joint_positions, dtype=np.float64
                ).copy(),
                "gripper_finger_joint_position_m_2d": gripper_qpos.copy(),
                "gripper_width_m": float(gripper_qpos[0] - gripper_qpos[1]),
                "eef_pose_robot_base_xyzw_7d": np.concatenate(
                    (eef_position_base, eef_quaternion_base)
                ),
            },
            "proprioception": {
                "arm_joint_velocity_rad_s_7d": np.asarray(
                    robot._joint_velocities, dtype=np.float64
                ).copy(),
                "gripper_finger_joint_velocity_m_s_2d": gripper_qvel.copy(),
                "gripper_width_velocity_m_s": float(gripper_qvel[0] - gripper_qvel[1]),
                "commanded_arm_joint_torque_nm_7d": (
                    None if torques is None else torques.copy()
                ),
                "commanded_arm_joint_torque_available": torques is not None,
                "eef_force_sensor_n_3d": np.asarray(robot.ee_force, dtype=np.float64).copy(),
                "eef_torque_sensor_nm_3d": np.asarray(robot.ee_torque, dtype=np.float64).copy(),
                "eef_twist_robot_base_6d": np.concatenate(
                    (controller_linear_velocity_base, controller_angular_velocity_base)
                ),
                "force_torque_frame": "eef_ft_sensor",
            },
            "cameras": {},
        }

        for public_name, source_name in CAMERA_SOURCES.items():
            rgb_key = f"{source_name}_image"
            depth_key = f"{source_name}_depth"
            if rgb_key not in raw_observation:
                raise KeyError(f"raw observation missing {rgb_key}")
            rgb = _to_opencv_image(np.asarray(raw_observation[rgb_key]))
            if depth_key not in raw_observation:
                raise KeyError(f"raw observation missing {depth_key}")
            normalized_depth = _to_opencv_image(
                np.asarray(raw_observation[depth_key], dtype=np.float32)
            )
            depth_m = get_real_depth_map(self.env.sim, normalized_depth)
            if depth_m.ndim == 3 and depth_m.shape[-1] == 1:
                depth_m = depth_m[..., 0]
            depth_m = np.asarray(depth_m, dtype=np.float32)
            valid = np.isfinite(depth_m) & (depth_m > 0.0)

            matrix_world_from_camera = get_camera_extrinsic_matrix(
                self.env.sim, source_name
            )
            matrix_base_from_camera = matrix_base_from_world @ matrix_world_from_camera
            master["cameras"][public_name] = {
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "depth_m": np.ascontiguousarray(depth_m, dtype=np.float32),
                "depth_valid_mask": np.ascontiguousarray(valid, dtype=np.bool_),
                "intrinsic_matrix_3x3": get_camera_intrinsic_matrix(
                    self.env.sim,
                    source_name,
                    self.camera_height,
                    self.camera_width,
                ).astype(np.float64),
                "matrix_T_robot_base_from_camera_opencv_4x4": np.asarray(
                    matrix_base_from_camera, dtype=np.float64
                ),
            }

        # Segmentation is consumed only to produce the two public semantic
        # roles, and only for the initial frame.  Raw IDs and instance names do
        # not enter the master frame.
        if frame_index == 0:
            master["annotations"] = {
                "schedule": "initial_observation_only",
                "cameras": {
                    public_name: self._camera_annotations(raw_observation, source_name)
                    for public_name, source_name in CAMERA_SOURCES.items()
                },
            }
        return master

    def _camera_annotations(
        self, raw_observation: Mapping[str, Any], source_name: str
    ) -> dict[str, Any]:
        key = f"{source_name}_segmentation_instance"
        if key not in raw_observation:
            raise KeyError(f"raw observation missing {key}")
        segmentation = _to_opencv_image(np.asarray(raw_observation[key]))
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]

        instance_names = list(self.env.env.model.instances_to_ids.keys())
        role_instances = {
            "manipulated_object": self.annotation_roles.manipulated_object,
            "goal_fixture": self.annotation_roles.goal_fixture,
        }
        annotations: dict[str, Any] = {}
        for public_role, private_instance_name in role_instances.items():
            if private_instance_name not in instance_names:
                raise ValueError(
                    f"annotation instance {private_instance_name!r} is absent from model"
                )
            segmentation_id = instance_names.index(private_instance_name) + 1
            mask = np.ascontiguousarray(segmentation == segmentation_id)
            annotations[public_role] = _annotation_from_mask(mask)
        return annotations


def _annotation_from_mask(mask: np.ndarray) -> dict[str, Any]:
    y_coordinates, x_coordinates = np.nonzero(mask)
    if len(x_coordinates) == 0:
        bbox = None
    else:
        bbox = [
            int(x_coordinates.min()),
            int(y_coordinates.min()),
            int(x_coordinates.max()) + 1,
            int(y_coordinates.max()) + 1,
        ]
    return {
        "visible": bool(len(x_coordinates)),
        "visible_pixel_count": int(len(x_coordinates)),
        "bbox_xyxy": bbox,
        "mask": np.ascontiguousarray(mask, dtype=np.bool_),
    }


def _to_opencv_image(array: np.ndarray) -> np.ndarray:
    """Convert robosuite's configured row convention to top-left OpenCV rows."""

    if robosuite_macros.IMAGE_CONVENTION == "opengl":
        array = array[::-1]
    elif robosuite_macros.IMAGE_CONVENTION != "opencv":
        raise ValueError(
            f"unsupported robosuite IMAGE_CONVENTION={robosuite_macros.IMAGE_CONVENTION!r}"
        )
    return np.ascontiguousarray(array)
