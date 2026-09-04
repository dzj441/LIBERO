import json
from pathlib import Path
import sys

from libero.libero.agent_env.control import ActionInterface
from scripts.launch_agent_episode import (
    _allocate_workspace,
    _archive_experience_context_contract,
    _archive_viewed_artifacts,
    _codex_infrastructure_error_from_session,
    _prepare_workspace,
    build_codex_command,
    build_task_prompt,
    parse_args,
    _task_instruction,
)
from scripts.launch_agent_curriculum import parse_args as parse_curriculum_args
from scripts.run_agent_experiment_matrix import (
    build_launch_command,
    parse_args as parse_matrix_args,
)


def test_codex_usage_limit_is_classified_as_infrastructure_error(tmp_path):
    session = tmp_path / "codex_session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "error": {
                        "message": "You've hit your usage limit.",
                        "codex_error_info": "usage_limit_exceeded",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _codex_infrastructure_error_from_session(session) == (
        "Codex usage limit reached before episode completion"
    )


def test_codex_agent_error_is_not_misclassified_as_infrastructure(tmp_path):
    session = tmp_path / "codex_session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "error": {
                        "message": "Agent command failed validation.",
                        "codex_error_info": "other",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _codex_infrastructure_error_from_session(session) is None


def test_codex_model_capacity_is_classified_as_infrastructure_error(tmp_path):
    session = tmp_path / "codex_session.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "error": {
                        "message": "Selected model is at capacity.",
                        "codex_error_info": "server_overloaded",
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _codex_infrastructure_error_from_session(session) == (
        "Codex service connection failed before episode completion"
    )


def test_launcher_defaults_to_mcp_native_osc(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch_agent_episode.py"])
    args = parse_args()
    assert args.control_transport == "mcp"
    assert args.action_interface == ActionInterface.NATIVE_OSC_SEQUENCE.value
    assert args.codex_execution_mode == "exec"


def test_all_agent_entrypoints_default_to_luna_max(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["launch_agent_episode.py"])
    episode_args = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_curriculum.py",
            "--curriculum-plan",
            str(tmp_path / "plan.json"),
        ],
    )
    curriculum_args = parse_curriculum_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_experiment_matrix.py",
            "--matrix",
            str(tmp_path / "matrix.json"),
        ],
    )
    matrix_args = parse_matrix_args()

    for args in (episode_args, curriculum_args, matrix_args):
        assert args.codex_model == "gpt-5.6-luna"
        assert args.codex_effort == "max"


def test_all_agent_entrypoints_preserve_explicit_codex_override(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_episode.py",
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    episode_args = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_curriculum.py",
            "--curriculum-plan",
            str(tmp_path / "plan.json"),
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    curriculum_args = parse_curriculum_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_experiment_matrix.py",
            "--matrix",
            str(tmp_path / "matrix.json"),
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    matrix_args = parse_matrix_args()

    for args in (episode_args, curriculum_args, matrix_args):
        assert args.codex_model == "gpt-5.6-sol"
        assert args.codex_effort == "high"


def _assert_codex_runtime_flags(command, *, model, effort):
    assert command[command.index("--model") + 1] == model
    assert f'model_reasoning_effort="{effort}"' in command


def _single_episode_matrix_run():
    return {
        "run_id": "runtime_defaults",
        "mode": "single_episode",
        "profile": "level4",
        "episode": {
            "suite": "libero_object",
            "task_id": 0,
            "init_state_id": 0,
            "seed": 0,
            "max_agent_steps": 50,
            "icl_condition": "none",
            "fixed_demo_master": None,
            "experience_context_spec": None,
        },
    }


def _matrix_codex_command(args, tmp_path):
    launch_command = build_launch_command(
        _single_episode_matrix_run(),
        batch_root=tmp_path / "batch",
        launcher_root=tmp_path,
        artifact_root=tmp_path,
        render_gpu_device_id=0,
        resolution=256,
        initial_settle_control_steps=10,
        codex_bin=args.codex_bin,
        codex_model=args.codex_model,
        codex_effort=args.codex_effort,
        https_proxy=args.https_proxy,
    )
    assert launch_command[launch_command.index("--codex-model") + 1] == (
        args.codex_model
    )
    assert launch_command[launch_command.index("--codex-effort") + 1] == (
        args.codex_effort
    )
    return build_codex_command(
        codex_bin=args.codex_bin,
        prompt="task prompt",
        model=launch_command[launch_command.index("--codex-model") + 1],
        effort=launch_command[launch_command.index("--codex-effort") + 1],
    )


def test_default_codex_settings_reach_final_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["launch_agent_episode.py"])
    episode_args = parse_args()
    episode_command = build_codex_command(
        codex_bin="codex",
        prompt="task prompt",
        model=episode_args.codex_model,
        effort=episode_args.codex_effort,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_curriculum.py",
            "--curriculum-plan",
            str(tmp_path / "plan.json"),
        ],
    )
    curriculum_args = parse_curriculum_args()
    curriculum_command = build_codex_command(
        codex_bin=curriculum_args.codex_bin,
        prompt="task prompt",
        model=curriculum_args.codex_model,
        effort=curriculum_args.codex_effort,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_experiment_matrix.py",
            "--matrix",
            str(tmp_path / "matrix.json"),
        ],
    )
    matrix_command = _matrix_codex_command(parse_matrix_args(), tmp_path)

    for command in (episode_command, curriculum_command, matrix_command):
        _assert_codex_runtime_flags(
            command,
            model="gpt-5.6-luna",
            effort="max",
        )


def test_explicit_codex_settings_reach_final_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_episode.py",
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    episode_args = parse_args()
    episode_command = build_codex_command(
        codex_bin=episode_args.codex_bin,
        prompt="task prompt",
        model=episode_args.codex_model,
        effort=episode_args.codex_effort,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_agent_curriculum.py",
            "--curriculum-plan",
            str(tmp_path / "plan.json"),
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    curriculum_args = parse_curriculum_args()
    curriculum_command = build_codex_command(
        codex_bin=curriculum_args.codex_bin,
        prompt="task prompt",
        model=curriculum_args.codex_model,
        effort=curriculum_args.codex_effort,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_experiment_matrix.py",
            "--matrix",
            str(tmp_path / "matrix.json"),
            "--codex-model",
            "gpt-5.6-sol",
            "--codex-effort",
            "high",
        ],
    )
    matrix_command = _matrix_codex_command(parse_matrix_args(), tmp_path)

    for command in (episode_command, curriculum_command, matrix_command):
        _assert_codex_runtime_flags(
            command,
            model="gpt-5.6-sol",
            effort="high",
        )


def test_robomemarena_task_instruction_resolves_without_private_stage_hints():
    instruction = _task_instruction("robomemarena", 4)
    assert instruction == (
        "Open and close all drawers in order to check. Put butter into the "
        "drawer that already contains an object."
    )
    assert "top" not in instruction.lower()


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


def test_experience_context_prompt_only_announces_public_bundle():
    prompt = build_task_prompt(
        "open the top drawer and put the bowl inside",
        icl_condition="experience_context",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
        control_transport="mcp",
    )
    assert "benchmark_inputs/experience_context/" in prompt
    assert "source task, outcome, and available modality" in prompt
    assert "matched" not in prompt.lower()
    assert "irrelevant" not in prompt.lower()
    assert "relation" not in prompt.lower()
    assert "expert_demo" not in prompt


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
        icl_condition="none",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
    )
    episode = json.loads((workspace / ".libero/episode.json").read_text())
    assert episode["operations"] == ["start", "osc_sequence", "finish"]
    assert episode["action_interface"] == "native_osc_sequence"
    assert episode["max_native_osc_micro_steps_per_submission"] == 20
    assert (workspace / "bin/liberoctl").stat().st_mode & 0o111
    assert "seed" not in episode
    assert "run_id" not in episode
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
    assert "run_id" not in episode
    assert (workspace / "bin/libero_mcp_server").stat().st_mode & 0o111
    assert not (workspace / "bin/liberoctl").exists()


def test_experience_context_workspace_names_only_the_public_context_root(tmp_path):
    source_root = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _prepare_workspace(
        source_root,
        workspace,
        "prompt",
        icl_condition="experience_context",
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
        control_transport="mcp",
    )
    episode = json.loads((workspace / ".libero/episode.json").read_text())
    assert episode["icl_condition"] == "experience_context"
    assert episode["experience_context"] == (
        "benchmark_inputs/experience_context"
    )
    assert episode["expert_demo"] is None


def test_archives_context_manifests_without_large_payload(tmp_path):
    bundle = tmp_path / "bundle"
    item = bundle / "experiences" / "experience_000"
    item.mkdir(parents=True)
    (item / "manifest.json").write_text('{"item": true}\n', encoding="utf-8")
    (item / "large.mp4").write_bytes(b"large payload")
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "experiences": [
                    {
                        "experience_id": "experience_000",
                        "manifest": {
                            "path": "experiences/experience_000/manifest.json"
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "snapshot"
    _archive_experience_context_contract(bundle, destination)
    assert (destination / "manifest.json").is_file()
    assert (destination / "experience_000.json").is_file()
    assert not (destination / "large.mp4").exists()


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


def test_interactive_codex_command_opens_tui_without_sending_prompt(tmp_path):
    server = tmp_path / "bin/libero_mcp_server"
    server.parent.mkdir()
    server.write_text("#!/bin/sh\n", encoding="utf-8")

    command = build_codex_command(
        codex_bin="codex",
        prompt="operator must paste this prompt",
        model="gpt-test",
        effort="high",
        workspace=tmp_path,
        control_transport="mcp",
        execution_mode="interactive",
    )

    assert command[0] == "codex"
    assert "exec" not in command
    assert "--no-alt-screen" in command
    assert "--skip-git-repo-check" not in command
    assert "operator must paste this prompt" not in command
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--dangerously-bypass-hook-trust" in command
    assert f'mcp_servers.libero.command="{server}"' in "\n".join(command)


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
