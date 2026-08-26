from scripts.launch_agent_episode import build_codex_command, build_task_prompt


def test_prompt_is_nonstrategic_and_documents_delta_gripper_workflow():
    prompt = build_task_prompt(
        "pick up the alphabet soup and place it in the basket"
    )
    assert prompt.splitlines()[0] == (
        "pick up the alphabet soup and place it in the basket"
    )
    assert "liberoctl start" in prompt
    assert "liberoctl step" in prompt
    assert "--gripper-delta-m DG" in prompt
    assert "positive opens, negative closes" in prompt
    assert "liberoctl finish" in prompt
    assert "waypoint" not in prompt.lower()


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
