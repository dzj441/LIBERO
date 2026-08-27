"""One-episode service shared by the stdin and Unix-socket transports."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .artifacts import replace_current_public_observation
from .control import ActionInterface
from .environment import LiberoAgentEnv


class AgentEpisodeService:
    """Expose one selected robot action command and keep private audit data."""

    def __init__(
        self,
        agent_env: LiberoAgentEnv,
        *,
        workspace_directory: str | Path,
        current_observation_directory: str | Path,
        private_run_directory: str | Path | None = None,
        action_interface: ActionInterface | str = ActionInterface.METRIC_OSC_STEP,
    ) -> None:
        self.agent_env = agent_env
        self.action_interface = ActionInterface.parse(action_interface)
        self.workspace_directory = Path(workspace_directory).resolve()
        self.current_observation_directory = Path(
            current_observation_directory
        ).resolve()
        self.private_run_directory = (
            None
            if private_run_directory is None
            else Path(private_run_directory).resolve()
        )
        self.state = "ready"
        self.latest_observation_id: str | None = None
        self._event_index = 0
        if self.private_run_directory is not None:
            self.private_run_directory.mkdir(parents=True, exist_ok=True)
            (self.private_run_directory / "private_observations").mkdir(
                parents=True, exist_ok=True
            )

    @property
    def finished(self) -> bool:
        return self.state == "finished"

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "start":
            response = self._start()
        elif (
            command == "osc_step"
            and self.action_interface is ActionInterface.METRIC_OSC_STEP
        ):
            response = self._osc_step(request)
        elif (
            command == "osc_sequence"
            and self.action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
        ):
            response = self._osc_sequence(request)
        elif command == "finish":
            response = self._finish(request)
        else:
            raise ValueError(
                "unknown command; expected start, "
                f"{self.action_interface.wire_command}, or finish"
            )
        self._record_event(request=request, response=response)
        return response

    def record_error(self, request: object, exc: BaseException) -> None:
        self._record_event(
            request=request,
            response={
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )

    def finalize_aborted(self, reason: str) -> None:
        if self.finished:
            return
        self.state = "aborted"
        self._write_result(
            {
                "schema_version": "libero.agent_run_result.v1",
                "status": "aborted",
                "reason": str(reason),
                "finished_at": _utc_now(),
            }
        )

    def close(self) -> None:
        self.agent_env.close()

    def _start(self) -> dict[str, Any]:
        if self.state != "ready":
            raise RuntimeError("start may be called exactly once")
        result = self.agent_env.start_episode()
        observation = result["observation"]
        observation_file = self._publish(observation)
        self.state = "active"
        return {
            "ok": True,
            "observation_id": observation["observation_id"],
            "observation_file": observation_file,
            "execution": {},
        }

    def _osc_step(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("osc_step requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.step_osc_target(
            delta_position_m=request.get("delta_position_m", (0.0, 0.0, 0.0)),
            delta_rotation_rotvec_rad=request.get(
                "delta_rotation_rotvec_rad", (0.0, 0.0, 0.0)
            ),
            delta_gripper_width_m=request.get("delta_gripper_width_m", 0.0),
        )
        observation = result["observation"]
        observation_file = self._publish(observation)
        return {
            "ok": True,
            "observation_id": observation["observation_id"],
            "observation_file": observation_file,
            "execution": result["execution"],
        }

    def _osc_sequence(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("osc_sequence requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.step_osc_sequence(actions=request.get("actions", ()))
        observation = result["observation"]
        observation_file = self._publish(observation)
        return {
            "ok": True,
            "observation_id": observation["observation_id"],
            "observation_file": observation_file,
            "execution": result["execution"],
        }

    def _finish(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if self.state != "active":
            raise RuntimeError("finish requires one active episode")
        self._require_latest_observation(request)
        result = self.agent_env.finish_episode()
        self.state = "finished"
        response = {
            "ok": True,
            "status": "finished",
            "success": bool(result["success"]),
            "accepted_agent_steps": int(result["accepted_agent_steps"]),
        }
        self._write_result(
            {
                "schema_version": "libero.agent_run_result.v1",
                **deepcopy(response),
                "finished_at": _utc_now(),
            }
        )
        return response

    def _publish(self, observation: Mapping[str, Any]) -> str:
        json_path = replace_current_public_observation(
            observation, self.current_observation_directory
        ).resolve()
        try:
            relative_path = json_path.relative_to(self.workspace_directory)
        except ValueError as exc:
            raise ValueError(
                "current observation directory must be inside the agent workspace"
            ) from exc

        if self.private_run_directory is not None:
            observation_id = str(observation["observation_id"])
            destination = (
                self.private_run_directory / "private_observations" / observation_id
            )
            if destination.exists():
                raise RuntimeError(f"duplicate observation id: {observation_id}")
            shutil.copytree(self.current_observation_directory, destination)
        self.latest_observation_id = str(observation["observation_id"])
        return relative_path.as_posix()

    def _require_latest_observation(self, request: Mapping[str, Any]) -> None:
        requested = request.get("observation_id")
        if not isinstance(requested, str) or not requested:
            raise ValueError("observation_id must identify the latest observation")
        if requested != self.latest_observation_id:
            raise ValueError(
                "observation_id is stale; latest observation is "
                f"{self.latest_observation_id!r}"
            )

    def _record_event(self, request: object, response: Mapping[str, Any]) -> None:
        if self.private_run_directory is None:
            return
        event = {
            "schema_version": "libero.agent_action_event.v1",
            "event_index": self._event_index,
            "recorded_at": _utc_now(),
            "request": _jsonable(request),
            "response": _jsonable(response),
        }
        self._event_index += 1
        path = self.private_run_directory / "actions.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_result(self, result: Mapping[str, Any]) -> None:
        if self.private_run_directory is None:
            return
        _write_json_atomic(self.private_run_directory / "result.json", result)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
