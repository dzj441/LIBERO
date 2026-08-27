#!/usr/bin/env python3
"""Replay a verified P4 EEF trajectory through the high-level Agent executor.

This evaluator-private diagnostic deliberately uses the demonstration's exact
initial MuJoCo state.  Unlike native action replay, each recorded post-action
EEF pose is treated as an absolute waypoint and reached through
``BaseFrameOSCExecutor``.  The resulting report isolates whether the public
metric EEF interface preserves contact-sensitive task behavior.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# GL selection must happen before importing LIBERO / robosuite.
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from libero.libero.agent_env.control import (  # noqa: E402
    BaseFrameOSCExecutor,
    EEFCommand,
    OSCControlConfig,
    matrix_to_rotation_vector,
    quaternion_xyzw_to_matrix,
)
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402
from libero.libero.utils.demonstration_replay import (  # noqa: E402
    DEFAULT_CAMERA_NAMES,
    _compose_camera_frame,
    ending_true_streak,
    load_demonstration_episode,
    maximum_true_streak,
    normalize_demo_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay P4 EEF poses through the high-level Agent executor"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--p4-master", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bddl-file")
    parser.add_argument("--bddl-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--stable-success-steps", type=int, default=10)
    parser.add_argument("--post-action-settle-steps", type=int, default=2)
    parser.add_argument(
        "--waypoint-stride",
        type=int,
        default=1,
        help="Track every Nth P4 frame; the final frame is always included",
    )
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    return parser.parse_args()


def _load_target_frames(master: Path) -> list[dict[str, Any]]:
    manifest_path = master / "p4_master_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = manifest.get("capture", {})
    count = int(capture.get("frame_count", 0))
    if count < 2:
        raise ValueError(f"P4 master must contain at least two frames: {master}")
    frames = []
    for index in range(count):
        path = master / "frames" / f"frame_{index:06d}" / "observation.json"
        observation = json.loads(path.read_text(encoding="utf-8"))
        frames.append(observation)
    return frames


def _controller_pose_base(env: Any) -> tuple[np.ndarray, np.ndarray]:
    robot = env.robots[0]
    controller = robot.controller
    controller.update(force=True)
    rotation_world_from_base = quaternion_xyzw_to_matrix(robot.base_ori)
    position_world_from_base = np.asarray(robot.base_pos, dtype=np.float64)
    position_base = rotation_world_from_base.T @ (
        np.asarray(controller.ee_pos, dtype=np.float64) - position_world_from_base
    )
    rotation_base = rotation_world_from_base.T @ np.asarray(
        controller.ee_ori_mat, dtype=np.float64
    )
    return position_base, rotation_base


def _gripper_width(env: Any) -> float:
    robot = env.robots[0]
    indexes = list(robot._ref_gripper_joint_pos_indexes)
    positions = np.asarray(
        [robot.sim.data.qpos[index] for index in indexes], dtype=np.float64
    )
    return float(positions[0] - positions[1])


def _gripper_limits(env: Any) -> tuple[float, float]:
    robot = env.robots[0]
    actuator_indexes = [
        robot.sim.model.actuator_name2id(name) for name in robot.gripper.actuators
    ]
    ranges = np.asarray(
        robot.sim.model.actuator_ctrlrange[actuator_indexes], dtype=np.float64
    )
    return float(ranges[0, 0] - ranges[1, 1]), float(
        ranges[0, 1] - ranges[1, 0]
    )


def _joint_snapshot(env: Any) -> dict[str, list[float]]:
    model = env.sim.model
    data = env.sim.data
    result: dict[str, list[float]] = {}
    for joint_id in range(model.njnt):
        name = model.joint_id2name(joint_id)
        if name is None or not any(
            token in name.lower() for token in ("drawer", "cabinet")
        ):
            continue
        start = int(model.jnt_qposadr[joint_id])
        end = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < model.njnt
            else int(model.nq)
        )
        result[name] = np.asarray(data.qpos[start:end], dtype=np.float64).tolist()
    return result


def _target_pose(frame: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    pose = np.asarray(frame["state"]["eef_pose_robot_base_xyzw_7d"], dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"invalid EEF pose shape: {pose.shape}")
    width = float(frame["state"]["gripper_width_m"])
    return pose[:3], quaternion_xyzw_to_matrix(pose[3:]), width


def main() -> int:
    args = parse_args()
    if args.video_stride < 1:
        raise ValueError("--video-stride must be positive")
    if args.stable_success_steps < 1:
        raise ValueError("--stable-success-steps must be positive")
    if args.waypoint_stride < 1:
        raise ValueError("--waypoint-stride must be positive")

    dataset = Path(args.dataset).expanduser().resolve()
    master = Path(args.p4_master).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    episode = load_demonstration_episode(
        dataset,
        normalize_demo_key(args.episode),
        bddl_file=args.bddl_file,
        bddl_root=args.bddl_root,
    )
    target_frames = _load_target_frames(master)
    if len(target_frames) != len(episode.actions) + 1:
        raise ValueError(
            "P4 frame count must equal native action count + 1: "
            f"{len(target_frames)} versus {len(episode.actions)}"
        )

    env = ControlEnv(
        bddl_file_name=str(episode.bddl_file),
        robots=list(episode.robots),
        controller=episode.controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=list(DEFAULT_CAMERA_NAMES),
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        camera_depths=False,
        camera_segmentations=None,
        use_object_obs=False,
        render_gpu_device_id=args.render_gpu_device_id,
        ignore_done=True,
        control_freq=episode.control_freq,
        horizon=100000,
    )

    video_frames: list[np.ndarray] = []
    native_cycle_index = 0

    def record_control_step(observation: Mapping[str, Any]) -> None:
        nonlocal native_cycle_index
        if native_cycle_index % args.video_stride == 0:
            video_frames.append(_compose_camera_frame(observation))
        native_cycle_index += 1

    config = OSCControlConfig(
        post_action_settle_steps=args.post_action_settle_steps
    )
    executor = BaseFrameOSCExecutor(
        env, config=config, control_step_callback=record_control_step
    )
    transition_records: list[dict[str, Any]] = []
    success_trace: list[bool] = []
    gl_info: dict[str, str | None] = {
        "vendor": None,
        "renderer": None,
        "version": None,
    }
    try:
        np.random.seed(args.seed)
        env.reset()
        observation = env.set_init_state(episode.init_state)
        settle_action = np.zeros(7, dtype=np.float64)
        settle_action[-1] = episode.actions[0, -1]
        for _ in range(args.settle_steps):
            observation, _, _, _ = env.step(settle_action)
        video_frames.append(_compose_camera_frame(observation))

        from OpenGL import GL

        def decode_gl_string(name: int) -> str | None:
            value = GL.glGetString(name)
            return value.decode("utf-8") if isinstance(value, bytes) else value

        gl_info = {
            "vendor": decode_gl_string(GL.GL_VENDOR),
            "renderer": decode_gl_string(GL.GL_RENDERER),
            "version": decode_gl_string(GL.GL_VERSION),
        }

        frame0_position, frame0_rotation, frame0_width = _target_pose(target_frames[0])
        actual_position, actual_rotation = _controller_pose_base(env)
        initial_alignment = {
            "position_error_m": float(np.linalg.norm(frame0_position - actual_position)),
            "orientation_error_rad": float(
                np.linalg.norm(
                    matrix_to_rotation_vector(frame0_rotation @ actual_rotation.T)
                )
            ),
            "gripper_width_error_m": float(frame0_width - _gripper_width(env)),
            "articulation_joints": _joint_snapshot(env),
        }

        minimum_width, maximum_width = _gripper_limits(env)
        target_frame_indices = list(
            range(args.waypoint_stride, len(target_frames), args.waypoint_stride)
        )
        if not target_frame_indices or target_frame_indices[-1] != len(target_frames) - 1:
            target_frame_indices.append(len(target_frames) - 1)
        for transition_index, target_frame_index in enumerate(target_frame_indices):
            target_frame = target_frames[target_frame_index]
            target_position, target_rotation, target_width = _target_pose(target_frame)
            current_position, current_rotation = _controller_pose_base(env)
            current_width = _gripper_width(env)
            bounded_target_width = float(
                np.clip(target_width, minimum_width, maximum_width)
            )
            command = EEFCommand.create(
                delta_position_m=target_position - current_position,
                delta_rotation_rotvec_rad=matrix_to_rotation_vector(
                    target_rotation @ current_rotation.T
                ),
                delta_gripper_width_m=bounded_target_width - current_width,
            )
            observation, execution = executor.execute(command)
            actual_position, actual_rotation = _controller_pose_base(env)
            success = bool(env.check_success())
            success_trace.append(success)
            transition_records.append(
                {
                    "transition_index": transition_index,
                    "target_frame_index": target_frame_index,
                    "command": {
                        "delta_position_m": command.delta_position_m.tolist(),
                        "delta_rotation_rotvec_rad": (
                            command.delta_rotation_rotvec_rad.tolist()
                        ),
                        "delta_gripper_width_m": command.delta_gripper_width_m,
                    },
                    "execution": execution.to_public_dict(),
                    "target_error_after_execution": {
                        "position_m": float(
                            np.linalg.norm(target_position - actual_position)
                        ),
                        "orientation_rad": float(
                            np.linalg.norm(
                                matrix_to_rotation_vector(
                                    target_rotation @ actual_rotation.T
                                )
                            )
                        ),
                        "gripper_width_m": float(
                            bounded_target_width - _gripper_width(env)
                        ),
                    },
                    "checker_success": success,
                    "articulation_joints": _joint_snapshot(env),
                }
            )

        stable_trace: list[bool] = []
        hold_action = np.zeros(7, dtype=np.float64)
        for _ in range(args.stable_success_steps):
            observation, _, _, _ = env.step(hold_action)
            record_control_step(observation)
            stable_trace.append(bool(env.check_success()))
    finally:
        env.close()

    video_path = output / "waypoint_replay.mp4"
    import imageio.v2 as imageio

    imageio.mimsave(
        video_path,
        video_frames,
        fps=args.video_fps / args.video_stride,
        quality=7,
    )
    final_success = bool(success_trace and success_trace[-1])
    final_stable_streak = ending_true_streak(stable_trace)
    report = {
        "schema_version": "libero.eef_waypoint_replay_diagnostic.v1",
        "mode": "high_level_metric_eef_waypoints",
        "dataset": str(dataset),
        "episode": episode.demo_key,
        "p4_master": str(master),
        "waypoint_count": len(transition_records),
        "initial_alignment": initial_alignment,
        "verified_success": bool(
            final_success and final_stable_streak >= args.stable_success_steps
        ),
        "final_success": final_success,
        "first_success_waypoint": next(
            (index for index, value in enumerate(success_trace) if value), None
        ),
        "maximum_success_streak": maximum_true_streak(success_trace),
        "post_replay_stable_success_streak": final_stable_streak,
        "required_stable_success_steps": args.stable_success_steps,
        "total_native_control_cycles": native_cycle_index,
        "articulation_joints_final": (
            transition_records[-1]["articulation_joints"]
            if transition_records
            else initial_alignment["articulation_joints"]
        ),
        "control_config": {
            "waypoint_stride": args.waypoint_stride,
            "post_action_settle_steps": config.post_action_settle_steps,
            "position_tolerance_m": config.position_tolerance_m,
            "orientation_tolerance_rad": config.orientation_tolerance_rad,
            "max_translation_substep_m": config.max_translation_substep_m,
            "max_rotation_substep_rad": config.max_rotation_substep_rad,
            "max_motion_control_steps": config.max_motion_control_steps,
        },
        "render": {
            "backend": os.environ.get("MUJOCO_GL"),
            "gpu_device_id": args.render_gpu_device_id,
            "gl": gl_info,
            "video": str(video_path),
        },
        "transitions": transition_records,
    }
    report_path = output / "waypoint_replay_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in report.items() if key != "transitions"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[report] {report_path}")
    print(f"[video] {video_path}")
    return 0 if report["verified_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
