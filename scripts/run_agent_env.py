#!/usr/bin/env python3
"""Persistent JSON-lines bridge between a coding-agent shell and LIBERO."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# These must be selected before importing robosuite through LIBERO.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

from libero.libero.agent_env import (  # noqa: E402
    make_libero_agent_env,
)
from libero.libero.agent_env.control import ActionInterface  # noqa: E402
from libero.libero.agent_env.service import AgentEpisodeService  # noqa: E402


def emit(payload: dict[str, Any]) -> None:
    # Robosuite and LIBERO print informational lines to stdout during startup.
    # A fixed prefix lets a shell client select protocol records reliably.
    print(
        "JSON_RESULT " + json.dumps(payload, default=_jsonable, sort_keys=True),
        flush=True,
    )


class AgentEnvBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        agent_env = make_libero_agent_env(
            suite=args.suite,
            task_id=args.task_id,
            init_state_id=args.init_state_id,
            profile=args.profile,
            seed=args.seed,
            camera_height=args.resolution,
            camera_width=args.resolution,
            render_gpu_device_id=args.render_gpu_device_id,
            initial_settle_control_steps=args.initial_settle_control_steps,
            max_agent_steps=args.max_agent_steps,
        )
        self.service = AgentEpisodeService(
            agent_env,
            workspace_directory=Path.cwd(),
            current_observation_directory=args.observation_dir,
            action_interface=args.action_interface,
        )

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.service.handle(request)

    def close(self) -> None:
        self.service.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--profile", default="level4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument(
        "--action-interface",
        choices=tuple(interface.value for interface in ActionInterface),
        default=ActionInterface.METRIC_OSC_STEP.value,
    )
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=Path("benchmark_inputs/current_observation"),
    )
    return parser.parse_args()


def main() -> None:
    bridge: AgentEnvBridge | None = None
    try:
        bridge = AgentEnvBridge(parse_args())
        emit(
            {
                "event": "ready",
                "protocol": "libero.agent_env.jsonl.v1",
                "next": {"command": "start"},
                "subsequent_request_binding": {
                    "field": "observation_id",
                    "value": "latest_successful_response",
                    "required_for": [
                        bridge.service.action_interface.wire_command,
                        "finish",
                    ],
                },
            }
        )
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("each request must be a JSON object")
                response = bridge.handle(request)
                emit(response)
                if request.get("command") == "finish":
                    return
            except Exception as exc:
                # Do not expose a traceback, simulator path, or checker detail
                # through the public bridge. Evaluator-private logging can wrap
                # this process separately.
                emit({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if bridge is not None:
            bridge.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
