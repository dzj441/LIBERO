from __future__ import annotations

import json
from pathlib import Path

import pytest

from libero.libero.agent_env.run_viewer import (
    RunRepository,
    ViewerDataError,
    _robot_command,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _event(timestamp: str, ordinal: int, item: dict) -> dict:
    return {
        "timestamp": timestamp,
        "ordinal": ordinal,
        "type": "event_msg",
        "payload": {"type": "item_completed", "item": item},
    }


@pytest.mark.parametrize(
    "shell_command, wire_command",
    (
        ("liberoctl step --position 0 0 0", "step"),
        ("liberoctl osc-step --position 0 0 0", "osc_step"),
        ("liberoctl osc-sequence --actions-file scratch/a.json", "osc_sequence"),
    ),
)
def test_robot_command_normalizes_legacy_and_ab_interfaces(
    shell_command, wire_command
):
    assert _robot_command(shell_command) == wire_command


def _make_run(tmp_path: Path) -> tuple[RunRepository, Path, Path]:
    runs = tmp_path / "runs"
    run = runs / "example"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "TASK_PROMPT.txt").write_text(
        "Pick up the object.\n\nUse liberoctl.", encoding="utf-8"
    )
    current = workspace / "benchmark_inputs" / "current_observation" / "head"
    current.mkdir(parents=True)
    (current / "rgb.png").write_bytes(b"latest")

    _write_json(
        run / "run_manifest.json",
        {
            "run_id": "example",
            "suite": "libero_object",
            "task_id": 0,
            "profile": "level4",
            "icl_condition": "fixed_demo",
            "workspace": str(workspace),
        },
    )
    _write_json(
        run / "result.json",
        {"status": "finished", "success": True, "accepted_agent_steps": 1},
    )
    _write_json(
        run / "codex_session_metadata.json",
        {"session_id": "session-1", "cwd": str(workspace), "episode_resumable": False},
    )
    session = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "ordinal": 0,
            "type": "session_meta",
            "payload": {
                "session_id": "session-1",
                "cli_version": "0.test",
                "cwd": str(workspace),
                "base_instructions": {"text": "base instructions"},
            },
        },
        {
            "timestamp": "2026-01-01T00:00:00.100Z",
            "ordinal": 1,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Pick up the object."}],
            },
        },
        _event(
            "2026-01-01T00:00:01Z",
            2,
            {
                "type": "AgentMessage",
                "id": "message-1",
                "content": [{"type": "Text", "text": "I will inspect first."}],
            },
        ),
        _event(
            "2026-01-01T00:00:02Z",
            3,
            {
                "type": "Reasoning",
                "id": "reasoning-1",
                "summary_text": [],
                "raw_content": ["hidden reasoning must not appear"],
            },
        ),
        _event(
            "2026-01-01T00:00:03Z",
            4,
            {
                "type": "CommandExecution",
                "id": "command-1",
                "command": ["/bin/bash", "-lc", "liberoctl start"],
                "status": "completed",
                "stdout": '{"ok":true}',
                "exit_code": 0,
            },
        ),
        _event(
            "2026-01-01T00:00:04Z",
            5,
            {
                "type": "ImageView",
                "id": "image-1",
                "path": (current / "rgb.png").as_uri(),
            },
        ),
        _event(
            "2026-01-01T00:00:05Z",
            6,
            {
                "type": "CommandExecution",
                "id": "command-2",
                "command": [
                    "/bin/bash",
                    "-lc",
                    "liberoctl osc-step --position 0 0 0 --rotation 0 0 0 --gripper-delta-m 0",
                ],
                "status": "completed",
                "stdout": '{"ok":true}',
                "exit_code": 0,
            },
        ),
    ]
    _write_jsonl(run / "codex_session.jsonl", session)
    _write_jsonl(
        run / "actions.jsonl",
        [
            {
                "request": {"command": "start"},
                "response": {"ok": True, "observation_id": "obs_000000"},
                "recorded_at": "2026-01-01T00:00:02.9Z",
            },
            {
                "request": {
                    "command": "osc_step",
                    "delta_position_m": [0, 0, 0],
                    "delta_rotation_rotvec_rad": [0, 0, 0],
                    "delta_gripper_width_m": 0,
                },
                "response": {"ok": True, "observation_id": "obs_000001"},
                "recorded_at": "2026-01-01T00:00:04.9Z",
            },
        ],
    )
    for index in range(2):
        observation_id = f"obs_{index:06d}"
        root = run / "private_observations" / observation_id
        (root / "head").mkdir(parents=True)
        (root / "head" / "rgb.png").write_bytes(f"frame-{index}".encode())
        _write_json(
            root / "observation.json",
            {
                "observation_id": observation_id,
                "frame_index": index,
                "profile": "level4",
                "cameras": {
                    "head": {
                        "rgb": {"file": "head/rgb.png"},
                    }
                },
                "state": {"gripper_width_m": 0.08},
                "proprioception": {"eef_force_sensor_n_3d": [0, 0, 0]},
            },
        )
    (run / "continuous_video.mp4").write_bytes(b"video")
    return RunRepository(runs), run, workspace


def test_viewer_aligns_codex_session_with_actions(tmp_path: Path) -> None:
    repository, _, _ = _make_run(tmp_path)

    detail = repository.detail("example")

    assert detail["alignment"] == {
        "action_records": 2,
        "session_robot_commands": 2,
        "matched_robot_commands": 2,
    }
    assert [step["command"] for step in detail["steps"]] == ["start", "osc_step"]
    assert detail["steps"][1]["output_observation"]["observation_id"] == "obs_000001"
    assert detail["summary"]["task_instruction"] == "Pick up the object."


def test_current_observation_image_view_maps_to_historical_frame(tmp_path: Path) -> None:
    repository, run, _ = _make_run(tmp_path)

    detail = repository.detail("example")
    views = [
        item
        for item in detail["steps"][1]["agent_activity"]
        if item["kind"] == "image_view"
    ]

    assert len(views) == 1
    assert views[0]["artifact"] == "run/private_observations/obs_000000/head/rgb.png"
    assert repository.resolve_artifact("example", views[0]["artifact"]) == (
        run / "private_observations" / "obs_000000" / "head" / "rgb.png"
    )


def test_viewer_never_exposes_raw_reasoning(tmp_path: Path) -> None:
    repository, _, _ = _make_run(tmp_path)

    serialized = json.dumps(repository.detail("example"))

    assert "hidden reasoning must not appear" not in serialized
    assert "I will inspect first" in serialized


def test_artifact_endpoint_uses_normalized_allowlist(tmp_path: Path) -> None:
    repository, run, workspace = _make_run(tmp_path)
    (workspace / "secret.txt").write_text("secret", encoding="utf-8")

    assert repository.resolve_artifact("example", "run/continuous_video.mp4") == (
        run / "continuous_video.mp4"
    )
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact("example", "workspace/secret.txt")
    with pytest.raises(ViewerDataError):
        repository.resolve_artifact("example", "../secret.txt")
