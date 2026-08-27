import json
from pathlib import Path

from libero.libero.agent_env.control import ActionInterface
from scripts.launch_agent_episode import (
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
