"""Evaluator-private commitments for one Agent-controlled LIBERO run."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    ActionInterface,
)
from .profiles import ObservationProfile


SERVER_READY_SCHEMA_VERSION = "libero.agent_server_ready.v1"


def build_server_ready_contract(
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    task_instruction: str,
    profile: ObservationProfile | int | str,
    seed: int,
    resolution: int,
    render_gpu_device_id: int,
    initial_settle_control_steps: int,
    max_agent_steps: int | None,
    action_interface: ActionInterface | str,
) -> dict[str, Any]:
    """Build the exact host/server contract checked before Codex is launched."""

    profile = ObservationProfile.parse(profile)
    action_interface = ActionInterface.parse(action_interface)
    normalized_instruction = " ".join(str(task_instruction).split())
    return {
        "schema_version": SERVER_READY_SCHEMA_VERSION,
        "transport": "unix_socket",
        "protocol": "libero.agent_unix_socket.v1",
        "suite": str(suite),
        "task_id": int(task_id),
        "init_state_id": int(init_state_id),
        "task_instruction_sha256": sha256_text(normalized_instruction),
        "observation_profile": profile.public_name,
        "seed": int(seed),
        "resolution": int(resolution),
        "render_gpu_device_id": int(render_gpu_device_id),
        "initial_settle_control_steps": int(initial_settle_control_steps),
        "max_agent_steps": (
            None if max_agent_steps is None else int(max_agent_steps)
        ),
        "action_interface": action_interface.value,
        "accepted_operations": [
            "start",
            action_interface.wire_command,
            "finish",
        ],
        "max_native_osc_micro_steps_per_submission": (
            MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
            else None
        ),
        "observation_retention": "current_only",
        "observation_publication": "atomic_replace_before_response",
    }


def validate_server_ready_contract(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject any startup configuration drift before an Agent can act."""

    if dict(actual) == dict(expected):
        return
    keys = sorted(set(actual) | set(expected))
    differing = [key for key in keys if actual.get(key) != expected.get(key)]
    raise RuntimeError(
        "LIBERO server ready contract differs from launcher request: "
        + ", ".join(differing)
    )


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
