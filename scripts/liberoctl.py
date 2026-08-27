#!/usr/bin/env python3
"""Minimal Unix-socket client exposed to a LIBERO coding agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any, Sequence


MAX_RESPONSE_BYTES = 1024 * 1024
MAX_ACTION_FILE_BYTES = 256 * 1024
METRIC_OSC_STEP = "metric_osc_step"
NATIVE_OSC_SEQUENCE = "native_osc_sequence"
ACTION_INTERFACES = frozenset({METRIC_OSC_STEP, NATIVE_OSC_SEQUENCE})


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    action_interface: str | None = None,
) -> argparse.Namespace:
    action_interface = action_interface or configured_action_interface()
    if action_interface not in ACTION_INTERFACES:
        raise ValueError(f"unsupported LIBERO action interface: {action_interface!r}")
    parser = argparse.ArgumentParser(prog="liberoctl")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("start")
    if action_interface == METRIC_OSC_STEP:
        step = subparsers.add_parser("osc-step")
        step.add_argument(
            "--position",
            type=float,
            nargs=3,
            metavar=("DX", "DY", "DZ"),
            default=(0.0, 0.0, 0.0),
        )
        step.add_argument(
            "--rotation",
            type=float,
            nargs=3,
            metavar=("RX", "RY", "RZ"),
            default=(0.0, 0.0, 0.0),
        )
        step.add_argument("--gripper-delta-m", type=float, default=0.0)
    else:
        sequence = subparsers.add_parser("osc-sequence")
        sequence.add_argument("--actions-file", type=Path, required=True)
    subparsers.add_parser("finish")
    return parser.parse_args(argv)


def request_for_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.operation == "start":
        return {"command": "start"}
    if args.operation == "finish":
        return {"command": "finish"}
    if args.operation == "osc-sequence":
        return {
            "command": "osc_sequence",
            "actions": load_actions_file(args.actions_file),
        }
    return {
        "command": "osc_step",
        "delta_position_m": list(args.position),
        "delta_rotation_rotvec_rad": list(args.rotation),
        "delta_gripper_width_m": args.gripper_delta_m,
    }


def configured_action_interface() -> str:
    return os.environ.get("LIBERO_ACTION_INTERFACE", METRIC_OSC_STEP)


def load_actions_file(path: str | Path) -> list[Any]:
    path = Path(path)
    if path.stat().st_size > MAX_ACTION_FILE_BYTES:
        raise ValueError("OSC action file exceeds client size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("OSC action file must contain one JSON array")
    return value


def control_socket_path() -> Path:
    configured = os.environ.get("LIBERO_CONTROL_SOCKET")
    workspace = Path(__file__).resolve().parent.parent
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else workspace / path
    return workspace / ".libero" / "control.sock"


def send_request(socket_path: str | Path, request: dict[str, Any]) -> dict[str, Any]:
    encoded = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
    chunks: list[bytes] = []
    size = 0
    socket_path = Path(socket_path).resolve()
    original_cwd = os.open(".", os.O_RDONLY)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        try:
            os.chdir(socket_path.parent)
            # AF_UNIX limits the bytes in the address argument.  Connecting by
            # basename keeps a persistent, deeply nested workspace usable.
            connection.connect(socket_path.name)
        finally:
            os.fchdir(original_cwd)
            os.close(original_cwd)
        connection.sendall(encoded)
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise RuntimeError("server response exceeds protocol size limit")
            if b"\n" in chunk:
                break
    payload = b"".join(chunks).split(b"\n", 1)[0]
    if not payload:
        raise RuntimeError("LIBERO server closed without a response")
    response = json.loads(payload.decode("utf-8"))
    if not isinstance(response, dict):
        raise RuntimeError("LIBERO server returned a non-object response")
    return response


def main() -> int:
    try:
        response = send_request(control_socket_path(), request_for_args(parse_args()))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
