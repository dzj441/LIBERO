#!/usr/bin/env python3
"""Host an ordered LIBERO curriculum behind one Unix socket."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import signal
import sys
import traceback
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.agent_env import make_libero_agent_env  # noqa: E402
from libero.libero.agent_env.control import ActionInterface  # noqa: E402
from libero.libero.agent_env.fixed_demo import (  # noqa: E402
    project_fixed_demo_bundle,
)
from libero.libero.agent_env.private_recording import (  # noqa: E402
    PrivateRolloutVideoRecorder,
)
from libero.libero.agent_env.runtime_contract import (  # noqa: E402
    build_curriculum_server_ready_contract,
)
from libero.libero.agent_env.service import (  # noqa: E402
    AgentEpisodeService,
    MultiEpisodeService,
    _write_json_atomic,
)
from scripts.libero_agent_server import EpisodeUnixServer  # noqa: E402


LOGGER = logging.getLogger("libero-curriculum-server")
CONFIG_SCHEMA_VERSION = "libero.agent_curriculum_server_config.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    run_directory = args.run_directory.resolve()
    socket_path = (
        args.socket.resolve()
        if args.socket.is_absolute()
        else (workspace / args.socket).resolve()
    )
    if workspace not in socket_path.parents:
        raise ValueError("control socket must be inside the agent workspace")
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        socket_path.unlink()

    config = _read_config(args.config)
    episodes = config["episodes"]
    action_interface = ActionInterface.parse(config["action_interface"])
    recorders: dict[int, PrivateRolloutVideoRecorder] = {}

    def before_episode_start(
        episode_index: int, episode_directory: Path
    ) -> Mapping[str, Any]:
        episode = episodes[episode_index]
        return _publish_episode_inputs(
            episode=episode,
            workspace=workspace,
            episode_directory=episode_directory,
            profile=config["profile"],
        )

    def service_factory(
        episode_index: int, episode_directory: Path
    ) -> AgentEpisodeService:
        episode = episodes[episode_index]
        recorder = PrivateRolloutVideoRecorder(
            episode_directory / "continuous_video.mp4"
        )
        recorders[episode_index] = recorder
        try:
            agent_env = make_libero_agent_env(
                suite=episode["suite"],
                task_id=episode["task_id"],
                init_state_id=episode["init_state_id"],
                profile=config["profile"],
                seed=episode["seed"],
                camera_height=config["resolution"],
                camera_width=config["resolution"],
                render_gpu_device_id=config["render_gpu_device_id"],
                initial_settle_control_steps=config[
                    "initial_settle_control_steps"
                ],
                max_agent_steps=config["max_agent_steps_per_episode"],
                private_control_step_callback=recorder.append_raw_observation,
            )
        except BaseException:
            recorder.close()
            recorders.pop(episode_index, None)
            raise
        if agent_env.task_instruction != episode["task_instruction"]:
            agent_env.close()
            recorder.close()
            recorders.pop(episode_index, None)
            raise RuntimeError("curriculum task instruction differs from config")
        _write_json_atomic(
            episode_directory / "episode_manifest.json",
            {
                "schema_version": "libero.agent_curriculum_episode.v1",
                "episode_index": episode_index,
                "episode_count": len(episodes),
                **episode,
                "fixed_demo_master": None,
            },
        )
        return AgentEpisodeService(
            agent_env,
            workspace_directory=workspace,
            current_observation_directory=(
                workspace / "benchmark_inputs" / "current_observation"
            ),
            private_run_directory=episode_directory,
            action_interface=action_interface,
        )

    def after_episode_close(episode_index: int) -> None:
        recorder = recorders.pop(episode_index, None)
        if recorder is not None:
            recorder.close()

    service = MultiEpisodeService(
        task_instructions=[episode["task_instruction"] for episode in episodes],
        service_factory=service_factory,
        private_run_directory=run_directory,
        before_episode_start=before_episode_start,
        after_episode_close=after_episode_close,
    )
    ready_contract = build_curriculum_server_ready_contract(
        episodes=episodes,
        profile=config["profile"],
        resolution=config["resolution"],
        render_gpu_device_id=config["render_gpu_device_id"],
        initial_settle_control_steps=config["initial_settle_control_steps"],
        max_agent_steps=config["max_agent_steps_per_episode"],
        action_interface=action_interface,
    )
    server: EpisodeUnixServer | None = None
    interrupted_reason = "server_stopped_before_curriculum_finish"
    try:
        original_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            server = EpisodeUnixServer(".libero/control.sock", service)
        finally:
            os.chdir(original_cwd)
        os.chmod(socket_path, 0o600)
        server.timeout = 0.5

        def request_stop(signum: int, _frame: Any) -> None:
            nonlocal interrupted_reason
            interrupted_reason = f"server_received_signal_{signum}_before_finish"
            if server is not None:
                server.stop_requested = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        _write_json_atomic(run_directory / "server_ready.json", ready_contract)
        LOGGER.info(
            "ready protocol=libero.agent_unix_socket.v1 episodes=%d socket=%s",
            len(episodes),
            socket_path,
        )
        while not server.stop_requested:
            if os.getppid() != args.launcher_pid:
                interrupted_reason = "launcher_process_exited_before_finish"
                break
            server.handle_request()
        if service.finished:
            return 0
        service.finalize_aborted(interrupted_reason)
        return 2
    finally:
        if server is not None:
            server.server_close()
        if socket_path.exists() or socket_path.is_symlink():
            socket_path.unlink()
        service.close()
        for recorder in recorders.values():
            recorder.close()


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported curriculum server config")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 2:
        raise ValueError("curriculum server requires at least two episodes")
    required_global = {
        "profile",
        "resolution",
        "render_gpu_device_id",
        "initial_settle_control_steps",
        "max_agent_steps_per_episode",
        "action_interface",
    }
    if not required_global.issubset(value):
        raise ValueError("curriculum server config is missing global fields")
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("each curriculum episode must be an object")
        required = {
            "suite",
            "task_id",
            "init_state_id",
            "seed",
            "task_instruction",
            "icl_condition",
            "fixed_demo_master_manifest_sha256",
        }
        if not required.issubset(episode):
            raise ValueError("curriculum episode is missing required fields")
        if episode["icl_condition"] not in {"none", "fixed_demo"}:
            raise ValueError("unsupported curriculum ICL condition")
        master = episode.get("fixed_demo_master")
        if episode["icl_condition"] == "fixed_demo" and not master:
            raise ValueError("fixed_demo curriculum episode requires a master")
        if episode["icl_condition"] == "none" and master is not None:
            raise ValueError("non-ICL curriculum episode cannot name a master")
    return value


def _publish_episode_inputs(
    *,
    episode: Mapping[str, Any],
    workspace: Path,
    episode_directory: Path,
    profile: str,
) -> dict[str, Any]:
    destination = workspace / "benchmark_inputs" / "expert_demo"
    if episode["icl_condition"] == "none":
        _remove_directory(destination)
        return {"fixed_demo_available": False, "expert_demo": None}

    staging = destination.parent / (
        f".{destination.name}.episode-next-{secrets.token_hex(6)}"
    )
    receipt = project_fixed_demo_bundle(
        master_root=episode["fixed_demo_master"],
        destination=staging,
        profile=profile,
        expected_task_instruction=episode["task_instruction"],
    )
    try:
        _replace_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    private_receipt = dict(receipt)
    private_receipt["agent_bundle"] = "benchmark_inputs/expert_demo"
    _write_json_atomic(
        episode_directory / "icl_projection_receipt.json", private_receipt
    )
    return {
        "fixed_demo_available": True,
        "expert_demo": "benchmark_inputs/expert_demo/",
    }


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("public expert-demo destination must not be a symlink")
    backup = destination.parent / (
        f".{destination.name}.previous-{secrets.token_hex(6)}"
    )
    moved_previous = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_previous = True
        os.replace(source, destination)
    except BaseException:
        if moved_previous and not destination.exists():
            os.replace(backup, destination)
            moved_previous = False
        raise
    finally:
        if moved_previous and backup.exists():
            shutil.rmtree(backup)


def _remove_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("public expert-demo destination must not be a symlink")
    if path.exists():
        shutil.rmtree(path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        raise SystemExit(main())
    except Exception:
        LOGGER.error("curriculum server startup failed\n%s", traceback.format_exc())
        raise
