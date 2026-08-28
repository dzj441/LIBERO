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

from .annotation_contract import (
    TASK_ENTITY_ANNOTATION_SCHEMA_VERSION,
    task_entity_id,
)
from .control import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix


CAMERA_SOURCES = {
    "head": "agentview",
    "wrist": "robot0_eye_in_hand",
}


@dataclass(frozen=True)
class TaskEntitySelection:
    """Host-private LIBERO instances selected for anonymous public masks."""

    instance_names: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(str(name) for name in self.instance_names)
        if not normalized:
            raise ValueError("at least one task entity is required")
        if any(not name for name in normalized):
            raise ValueError("task-entity instance names must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("task-entity instance names must be unique")
        object.__setattr__(self, "instance_names", normalized)

    def anonymous_instances(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (task_entity_id(index), private_name)
            for index, private_name in enumerate(self.instance_names)
        )


def infer_task_entities(parsed_problem: Mapping[str, Any]) -> TaskEntitySelection:
    """Select every BDDL object of interest without assigning task roles."""

    known_instances = {
        instance
        for group in ("objects", "fixtures")
        for instances in parsed_problem.get(group, {}).values()
        for instance in instances
    }
    regions = parsed_problem.get("regions", {})
    selected: list[str] = []
    for value in parsed_problem.get("obj_of_interest", []):
        reference = str(value)
        instance = reference
        if reference in regions:
            instance = str(regions[reference].get("target", ""))
        if instance not in known_instances:
            raise ValueError(
                f"task entity {reference!r} does not resolve to a known object "
                "or fixture instance"
            )
        if instance not in selected:
            selected.append(instance)
    if not selected:
        raise ValueError("task has no valid obj_of_interest entries to annotate")
    return TaskEntitySelection(tuple(selected))


class MasterObservationCollector:
    """Collect all candidate public signals without checker or object poses."""

    def __init__(
        self,
        env: Any,
        camera_height: int,
        camera_width: int,
        task_entities: TaskEntitySelection | None = None,
    ) -> None:
        self.env = env
        self.camera_height = int(camera_height)
        self.camera_width = int(camera_width)
        self.task_entities = task_entities or infer_task_entities(
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

        # Segmentation is consumed only to produce anonymous task-entity masks
        # on the initial frame. Raw IDs, private instance names, semantic class
        # names, and manipulated/goal role bindings do not enter the master.
        if frame_index == 0:
            master["annotations"] = {
                "schema_version": TASK_ENTITY_ANNOTATION_SCHEMA_VERSION,
                "schedule": "initial_observation_only",
                "cameras": {
                    public_name: {
                        "task_entities": self._camera_task_entities(
                            raw_observation, source_name
                        )
                    }
                    for public_name, source_name in CAMERA_SOURCES.items()
                },
            }
        return master

    def _camera_task_entities(
        self, raw_observation: Mapping[str, Any], source_name: str
    ) -> dict[str, Any]:
        key = f"{source_name}_segmentation_instance"
        if key not in raw_observation:
            raise KeyError(f"raw observation missing {key}")
        segmentation = _to_opencv_image(np.asarray(raw_observation[key]))
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]

        instance_names = list(self.env.env.model.instances_to_ids.keys())
        annotations: dict[str, Any] = {}
        for public_id, private_instance_name in self.task_entities.anonymous_instances():
            if private_instance_name not in instance_names:
                raise ValueError(
                    f"annotation instance {private_instance_name!r} is absent from model"
                )
            segmentation_id = instance_names.index(private_instance_name) + 1
            mask = np.ascontiguousarray(segmentation == segmentation_id)
            annotations[public_id] = _annotation_from_mask(mask)
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
