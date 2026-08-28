from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from libero.libero.agent_env.run_viewer import (
    RunRepository,
    ViewerDataError,
    _mcp_robot_command,
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


@pytest.mark.parametrize(
    "item, wire_command",
    (
        ({"server": "libero", "tool": "start_episode"}, "start"),
        ({"server_name": "libero", "tool_name": "osc_sequence"}, "osc_sequence"),
        ({"server": "libero", "name": "finish_episode"}, "finish"),
        ({"name": "mcp__libero__osc_sequence"}, "osc_sequence"),
        ({"server": "another", "tool": "start_episode"}, None),
    ),
)
def test_mcp_robot_command_only_accepts_libero_server(item, wire_command):
    assert _mcp_robot_command(item) == wire_command


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


def test_viewer_aligns_libero_mcp_calls_with_actions(tmp_path: Path) -> None:
    _repository, run, _workspace = _make_run(tmp_path)
    records = [
        json.loads(line)
        for line in (run / "codex_session.jsonl").read_text().splitlines()
    ]
    for record in records:
        payload = record.get("payload", {})
        item = payload.get("item", {}) if isinstance(payload, dict) else {}
        if item.get("id") == "command-1":
            payload["item"] = {
                "type": "McpToolCall",
                "id": "mcp-1",
                "server": "libero",
                "tool": "start_episode",
                "arguments": {},
                "status": "completed",
            }
        elif item.get("id") == "command-2":
            payload["item"] = {
                "type": "McpToolCall",
                "id": "mcp-2",
                "server": "libero",
                "tool": "osc_sequence",
                "arguments": {"actions": [[0, 0, 0, 0, 0, 0, -1]]},
                "status": "completed",
            }
    _write_jsonl(run / "codex_session.jsonl", records)
    actions = [
        json.loads(line) for line in (run / "actions.jsonl").read_text().splitlines()
    ]
    actions[1]["request"] = {
        "command": "osc_sequence",
        "actions": [[0, 0, 0, 0, 0, 0, -1]],
    }
    _write_jsonl(run / "actions.jsonl", actions)

    detail = RunRepository(run.parent).detail("example")

    assert detail["alignment"] == {
        "action_records": 2,
        "session_robot_commands": 2,
        "matched_robot_commands": 2,
    }
    assert detail["steps"][0]["agent_activity"][-1]["kind"] == "mcp_tool_call"
    assert detail["steps"][1]["agent_activity"][-1]["title"] == (
        "libero.osc_sequence"
    )


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


def test_session_coverage_accounts_for_hidden_and_visible_records(tmp_path: Path) -> None:
    repository, _, _ = _make_run(tmp_path)

    coverage = repository.detail("example")["session"]["coverage"]

    assert coverage["public_trace_complete"] is True
    assert coverage["viewer_complete"] is True
    assert coverage["unsupported_types"] == {}
    assert coverage["classification_counts"]["deliberately_hidden"] == 1
    assert coverage["hidden_field_counts"]["raw_content"] == 1
    assert coverage["artifact_coverage"] == {
        "image_view_events": 1,
        "image_view_artifacts_available": 1,
        "image_view_artifacts_missing": 0,
    }


def test_lifecycle_runtime_and_injected_context_are_visible(tmp_path: Path) -> None:
    _repository, run, workspace = _make_run(tmp_path)
    session_path = run / "codex_session.jsonl"
    records = [json.loads(line) for line in session_path.read_text().splitlines()]
    records.extend(
        [
            {
                "timestamp": "2026-01-01T00:00:06Z",
                "ordinal": 7,
                "type": "turn_context",
                "payload": {
                    "model": "gpt-test",
                    "effort": "high",
                    "cwd": str(workspace),
                    "sandbox_policy": {"type": "danger-full-access"},
                    "approval_policy": "never",
                },
            },
            {
                "timestamp": "2026-01-01T00:00:06.1Z",
                "ordinal": 8,
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<environment_context><cwd>/tmp</cwd></environment_context>",
                        }
                    ],
                },
            },
            {
                "timestamp": "2026-01-01T00:00:06.2Z",
                "ordinal": 9,
                "type": "event_msg",
                "payload": {
                    "type": "task_started",
                    "model_context_window": 1234,
                },
            },
            {
                "timestamp": "2026-01-01T00:00:06.3Z",
                "ordinal": 10,
                "type": "event_msg",
                "payload": {
                    "type": "turn_aborted",
                    "reason": "interrupted",
                },
            },
        ]
    )
    _write_jsonl(session_path, records)

    detail = RunRepository(run.parent).detail("example")
    session = detail["session"]

    assert session["task_user_messages"] == ["Pick up the object."]
    assert session["runtime_user_messages"] == [
        "<environment_context><cwd>/tmp</cwd></environment_context>"
    ]
    assert session["runtime_settings"]["model"] == "gpt-test"
    assert session["runtime_settings"]["reasoning_effort"] == "high"
    assert session["runtime_settings"]["context_window"] == 1234
    assert [item["kind"] for item in detail["tail_activity"]][-2:] == [
        "task_started",
        "turn_aborted",
    ]


def test_archived_viewed_image_survives_deleted_workspace(tmp_path: Path) -> None:
    _repository, run, workspace = _make_run(tmp_path)
    scratch = workspace / "scratch" / "crop.png"
    scratch.parent.mkdir()
    scratch.write_bytes(b"derived-image")
    records = [
        json.loads(line)
        for line in (run / "codex_session.jsonl").read_text().splitlines()
    ]
    records.insert(
        -1,
        _event(
            "2026-01-01T00:00:04.5Z",
            99,
            {"type": "ImageView", "id": "image-derived", "path": scratch.as_uri()},
        ),
    )
    _write_jsonl(run / "codex_session.jsonl", records)
    archived = run / "viewed_artifacts" / "crop.png"
    archived.parent.mkdir()
    shutil.copy2(scratch, archived)
    _write_json(
        run / "viewed_artifacts_manifest.json",
        {
            "artifacts": [
                {
                    "source_path": scratch.as_uri(),
                    "source_absolute_path": str(scratch.resolve()),
                    "archived_file": "viewed_artifacts/crop.png",
                    "status": "archived",
                }
            ]
        },
    )
    shutil.rmtree(workspace)

    repository = RunRepository(run.parent)
    detail = repository.detail("example")
    derived = [
        item
        for step in detail["steps"]
        for item in step["agent_activity"]
        if item.get("title") == scratch.as_uri()
    ]

    assert len(derived) == 1
    assert derived[0]["artifact"] == "run/viewed_artifacts/crop.png"
    assert repository.resolve_artifact("example", derived[0]["artifact"]) == archived
