import json

from libero.libero.agent_env.control import ActionInterface
from libero.libero.agent_env.profiles import project_public_observation
from libero.libero.agent_env.service import AgentEpisodeService
from test_profiles import _master


class _FakeAgentEnv:
    def __init__(self):
        self.closed = False

    def start_episode(self):
        return {
            "task_instruction": "private duplicate instruction",
            "observation": project_public_observation(_master(), "level2"),
        }

    def step_osc_target(self, **arguments):
        assert arguments == {
            "delta_position_m": [0.01, 0.0, 0.0],
            "delta_rotation_rotvec_rad": [0.0, 0.0, 0.1],
            "delta_gripper_width_m": -0.005,
        }
        return {
            "accepted_agent_step": 1,
            "execution": {"command_completed": True},
            "observation": project_public_observation(
                _master(frame_index=1), "level2"
            ),
        }

    def step_osc_sequence(self, **arguments):
        assert arguments == {
            "actions": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0]],
        }
        return {
            "accepted_agent_step": 1,
            "execution": {
                "command_completed": True,
                "micro_step_count": 1,
            },
            "observation": project_public_observation(
                _master(frame_index=1), "level2"
            ),
        }

    def finish_episode(self):
        return {"success": True, "accepted_agent_steps": 1}

    def close(self):
        self.closed = True


def test_service_exposes_only_three_operations_and_current_observation(tmp_path):
    workspace = tmp_path / "workspace"
    run_directory = tmp_path / "private"
    service = AgentEpisodeService(
        _FakeAgentEnv(),
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
        private_run_directory=run_directory,
    )

    started = service.handle({"command": "start"})
    assert started == {
        "ok": True,
        "observation_id": "obs_000000",
        "observation_file": "benchmark_inputs/current_observation/observation.json",
        "execution": {},
    }
    current = workspace / started["observation_file"]
    assert current.is_file()
    assert (current.parent / "annotations").is_dir()

    stepped = service.handle(
        {
            "command": "osc_step",
            "delta_position_m": [0.01, 0.0, 0.0],
            "delta_rotation_rotvec_rad": [0.0, 0.0, 0.1],
            "delta_gripper_width_m": -0.005,
        }
    )
    assert stepped["observation_id"] == "obs_000001"
    assert not (current.parent / "annotations").exists()
    assert (run_directory / "private_observations" / "obs_000000").is_dir()
    assert (run_directory / "private_observations" / "obs_000001").is_dir()

    finished = service.handle({"command": "finish"})
    assert finished["success"] is True
    assert json.loads((run_directory / "result.json").read_text())["status"] == "finished"
    assert len((run_directory / "actions.jsonl").read_text().splitlines()) == 3


def test_service_rejects_legacy_step_command(tmp_path):
    workspace = tmp_path / "workspace"
    service = AgentEpisodeService(
        _FakeAgentEnv(),
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
    )

    service.handle({"command": "start"})
    try:
        service.handle({"command": "step"})
    except ValueError as exc:
        assert "osc_step" in str(exc)
    else:
        raise AssertionError("legacy step command should be rejected")


def test_native_sequence_service_exposes_only_selected_action_interface(tmp_path):
    workspace = tmp_path / "workspace"
    service = AgentEpisodeService(
        _FakeAgentEnv(),
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
        action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
    )

    service.handle({"command": "start"})
    try:
        service.handle({"command": "osc_step"})
    except ValueError as exc:
        assert "osc_sequence" in str(exc)
        assert "osc_step" not in str(exc)
    else:
        raise AssertionError("inactive metric interface should be rejected")

    stepped = service.handle(
        {
            "command": "osc_sequence",
            "actions": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0]],
        }
    )
    assert stepped["observation_id"] == "obs_000001"
    assert stepped["execution"]["micro_step_count"] == 1


def test_service_marks_unfinished_episode_aborted(tmp_path):
    workspace = tmp_path / "workspace"
    run_directory = tmp_path / "private"
    service = AgentEpisodeService(
        _FakeAgentEnv(),
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
        private_run_directory=run_directory,
    )
    service.handle({"command": "start"})
    service.finalize_aborted("codex_process_exited_before_finish")
    result = json.loads((run_directory / "result.json").read_text())
    assert result["status"] == "aborted"
    assert result["reason"] == "codex_process_exited_before_finish"
