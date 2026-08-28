import json
from pathlib import Path

from libero.libero.agent_env.control import ActionInterface
from scripts.launch_agent_episode import (
    _allocate_workspace,
    _archive_viewed_artifacts,
    _prepare_workspace,
    build_codex_command,
    build_task_prompt,
)


def test_prompt_is_nonstrategic_and_documents_delta_gripper_workflow():
    prompt = build_task_prompt(
        "pick up the alphabet soup and place it in the basket"
    )
    assert prompt.splitlines()[0] == (
        "pick up the alphabet soup and place it in the basket"
    )
    assert "liberoctl start" in prompt
    assert "liberoctl osc-step" in prompt
    assert "metric Cartesian target delta" in prompt
    assert "OSC_POSE controller" in prompt
    assert "--gripper-delta-m DG" in prompt
    assert "positive opens, negative closes" in prompt
    assert "liberoctl finish" in prompt
    assert "waypoint" not in prompt.lower()
    assert "expert_demo" not in prompt


def test_fixed_demo_prompt_only_adds_separate_episode_notice():
    prompt = build_task_prompt(
        "pick up the alphabet soup and place it in the basket",
        icl_condition="fixed_demo",
    )
    assert "benchmark_inputs/expert_demo/" in prompt
    assert "separate episode of the same task" in prompt
    assert "object or goal poses may differ" in prompt
    assert "native per-control-cycle OSC_POSE actions" in prompt
    assert "measured EEF state observations" in prompt
    assert "EEF poses are observations, not actions" in prompt
    assert "A commanded target is not guaranteed" not in prompt
    assert "imitate" not in prompt.lower()


def test_native_sequence_prompt_documents_exact_bounded_micro_action_contract():
    prompt = build_task_prompt(
        "open the top drawer and put the bowl inside",
        icl_condition="fixed_demo",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
    )
    assert "liberoctl osc-sequence --actions-file PATH" in prompt
    assert "1 to 20 normalized 7D OSC_POSE micro actions" in prompt
    assert "[dx, dy, dz, rx, ry, rz, gripper]" in prompt
    assert "within [-1, 1]" in prompt
    assert "at most 50 accepted submissions" in prompt
    assert "same component semantics" in prompt
    assert "liberoctl osc-step" not in prompt


def test_mcp_prompt_names_only_the_three_robot_tools():
    prompt = build_task_prompt(
        "open the top drawer and put the bowl inside",
        icl_condition="fixed_demo",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
        control_transport="mcp",
    )
    assert "`start_episode` robot tool" in prompt
    assert "`osc_sequence` robot tool" in prompt
    assert "`finish_episode` robot tool" in prompt
    assert "liberoctl" not in prompt
    assert "1 to 20 normalized 7D OSC_POSE micro actions" in prompt


def test_native_workspace_exposes_only_the_selected_wire_operation(tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _prepare_workspace(
        source_root,
        workspace,
        "prompt",
        "test-run",
        icl_condition="none",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
    )
    episode = json.loads((workspace / ".libero/episode.json").read_text())
    assert episode["operations"] == ["start", "osc_sequence", "finish"]
    assert episode["action_interface"] == "native_osc_sequence"
    assert episode["max_native_osc_micro_steps_per_submission"] == 20
    assert (workspace / "bin/liberoctl").stat().st_mode & 0o111
    assert "seed" not in episode
    assert "server_ready" not in episode
    assert not (workspace / "run_manifest.json").exists()


def test_mcp_workspace_exposes_adapter_without_liberoctl(tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _prepare_workspace(
        source_root,
        workspace,
        "prompt",
        "test-run",
        icl_condition="none",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
        control_transport="mcp",
    )
    episode = json.loads((workspace / ".libero/episode.json").read_text())
    assert episode["operations"] == [
        "start_episode",
        "osc_sequence",
        "finish_episode",
    ]
    assert episode["control_transport"] == "mcp"
    assert (workspace / "bin/libero_mcp_server").stat().st_mode & 0o111
    assert not (workspace / "bin/liberoctl").exists()


def test_codex_command_is_persistent_one_shot_and_noninteractive():
    command = build_codex_command(
        codex_bin="codex",
        prompt="task prompt",
        model="gpt-test",
        effort="high",
    )
    assert command[:2] == ["codex", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--dangerously-bypass-hook-trust" in command
    assert "--skip-git-repo-check" in command
    assert "--ephemeral" not in command
    assert "--no-alt-screen" not in command
    assert command[-1] == "task prompt"


def test_codex_command_injects_required_workspace_local_mcp(tmp_path):
    server = tmp_path / "bin/libero_mcp_server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\n", encoding="utf-8")
    command = build_codex_command(
        codex_bin="codex",
        prompt="task prompt",
        workspace=tmp_path,
        control_transport="mcp",
    )
    rendered = "\n".join(command)
    assert f'mcp_servers.libero.command="{server}"' in rendered
    assert "mcp_servers.libero.required=true" in rendered
    assert 'mcp_servers.libero.default_tools_approval_mode="auto"' in rendered
    assert '"start_episode", "osc_sequence", "finish_episode"' in rendered


def test_system_temp_workspace_is_random_and_left_for_system_cleanup(tmp_path):
    workspace, ephemeral = _allocate_workspace(
        canonical_root=tmp_path / "repo",
        requested_root=tmp_path / "temporary_workspaces",
        run_id="run-id-must-not-be-the-directory-name",
        keep_workspace=False,
    )

    assert ephemeral is True
    assert workspace.name.startswith("libero-agent-workspace-")
    assert "run-id" not in workspace.name
    assert workspace.is_dir()


def test_archive_viewed_artifacts_skips_current_observation_and_keeps_scratch(
    tmp_path,
):
    workspace = tmp_path / "libero-agent-workspace-example"
    run = tmp_path / "run"
    current = workspace / "benchmark_inputs/current_observation/head/rgb.png"
    scratch = workspace / "scratch/crop.png"
    current.parent.mkdir(parents=True)
    scratch.parent.mkdir(parents=True)
    current.write_bytes(b"current")
    scratch.write_bytes(b"crop")
    run.mkdir()
    session = run / "codex_session.jsonl"
    records = []
    for ordinal, path in enumerate((current, scratch)):
        records.append(
            {
                "timestamp": f"2026-01-01T00:00:0{ordinal}Z",
                "ordinal": ordinal,
                "type": "event_msg",
                "payload": {
                    "type": "item_completed",
                    "item": {"type": "ImageView", "path": path.as_uri()},
                },
            }
        )
    session.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = _archive_viewed_artifacts(
        session_path=session,
        workspace=workspace,
        run_directory=run,
    )

    by_path = {entry["source_path"]: entry for entry in manifest["artifacts"]}
    assert by_path[current.as_uri()]["status"] == "historical_observation_archive"
    scratch_entry = by_path[scratch.as_uri()]
    assert scratch_entry["status"] == "archived"
    assert (run / scratch_entry["archived_file"]).read_bytes() == b"crop"
