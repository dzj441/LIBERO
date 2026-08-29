import json

from libero.libero.agent_env.control import ActionInterface
from libero.libero.agent_env.profiles import project_public_observation
from libero.libero.agent_env.service import AgentEpisodeService, MultiEpisodeService
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
            "observation_id": "obs_000000",
            "delta_position_m": [0.01, 0.0, 0.0],
            "delta_rotation_rotvec_rad": [0.0, 0.0, 0.1],
            "delta_gripper_width_m": -0.005,
        }
    )
    assert stepped["observation_id"] == "obs_000001"
    assert not (current.parent / "annotations").exists()
    assert (run_directory / "private_observations" / "obs_000000").is_dir()
    assert (run_directory / "private_observations" / "obs_000001").is_dir()

    finished = service.handle(
        {"command": "finish", "observation_id": "obs_000001"}
    )
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
            "observation_id": "obs_000000",
            "actions": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0]],
        }
    )
    assert stepped["observation_id"] == "obs_000001"
    assert stepped["execution"]["micro_step_count"] == 1


def test_service_rejects_missing_or_stale_observation_without_acting(tmp_path):
    workspace = tmp_path / "workspace"
    service = AgentEpisodeService(
        _FakeAgentEnv(),
        workspace_directory=workspace,
        current_observation_directory=(
            workspace / "benchmark_inputs" / "current_observation"
        ),
    )
    service.handle({"command": "start"})

    for request in (
        {"command": "osc_step"},
        {"command": "osc_step", "observation_id": "obs_stale"},
    ):
        try:
            service.handle(request)
        except ValueError as exc:
            assert "observation_id" in str(exc)
        else:
            raise AssertionError("unbound action should be rejected")

    assert service.latest_observation_id == "obs_000000"
    assert service.agent_env.closed is False


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


def test_multi_episode_service_sequences_children_and_records_final_target(tmp_path):
    workspace = tmp_path / "workspace"
    run_directory = tmp_path / "private"
    environments = [_FakeAgentEnv(), _FakeAgentEnv(), _FakeAgentEnv()]
    prepared: list[int] = []
    closed: list[int] = []

    def before_start(index, episode_directory):
        prepared.append(index)
        return {
            "fixed_demo_available": index != 1,
            "expert_demo": (
                "benchmark_inputs/expert_demo" if index != 1 else None
            ),
        }

    def factory(index, episode_directory):
        return AgentEpisodeService(
            environments[index],
            workspace_directory=workspace,
            current_observation_directory=(
                workspace / "benchmark_inputs" / "current_observation"
            ),
            private_run_directory=episode_directory,
            action_interface=ActionInterface.NATIVE_OSC_SEQUENCE,
        )

    service = MultiEpisodeService(
        task_instructions=("first task", "second task", "combined task"),
        service_factory=factory,
        private_run_directory=run_directory,
        before_episode_start=before_start,
        after_episode_close=closed.append,
    )

    for episode_index in range(3):
        started = service.handle({"command": "start"})
        assert started["episode_index"] == episode_index
        assert started["episode_count"] == 3
        assert started["task_instruction"] == (
            "first task",
            "second task",
            "combined task",
        )[episode_index]
        assert started["fixed_demo_available"] == (
            episode_index != 1
        )
        stepped = service.handle(
            {
                "command": "osc_sequence",
                "observation_id": "obs_000000",
                "actions": [[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0]],
            }
        )
        assert stepped["episode_index"] == episode_index
        finished = service.handle(
            {"command": "finish", "observation_id": "obs_000001"}
        )
        assert finished["next_episode_available"] == (episode_index < 2)
        assert finished["curriculum_complete"] == (episode_index == 2)

    assert service.finished is True
    assert prepared == [0, 1, 2]
    assert closed == [0, 1, 2]
    assert all(environment.closed for environment in environments)
    result = json.loads((run_directory / "result.json").read_text())
    assert result["status"] == "finished"
    assert result["success"] is True
    assert result["all_episodes_success"] is True
    assert result["completed_episode_count"] == 3
    assert result["accepted_agent_steps"] == 3
    events = [
        json.loads(line)
        for line in (run_directory / "actions.jsonl").read_text().splitlines()
    ]
    assert [event["episode_index"] for event in events] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
    ]
    for episode_index in range(3):
        episode_directory = (
            run_directory / "episodes" / f"episode_{episode_index:03d}"
        )
        assert (episode_directory / "result.json").is_file()
        assert len((episode_directory / "actions.jsonl").read_text().splitlines()) == 3


def test_multi_episode_service_requires_finish_before_next_start(tmp_path):
    workspace = tmp_path / "workspace"

    def factory(_index, episode_directory):
        return AgentEpisodeService(
            _FakeAgentEnv(),
            workspace_directory=workspace,
            current_observation_directory=(
                workspace / "benchmark_inputs" / "current_observation"
            ),
            private_run_directory=episode_directory,
        )

    service = MultiEpisodeService(
        task_instructions=("first", "second"),
        service_factory=factory,
        private_run_directory=tmp_path / "private",
    )
    service.handle({"command": "start"})
    try:
        service.handle({"command": "start"})
    except RuntimeError as exc:
        assert "finish the active episode" in str(exc)
    else:
        raise AssertionError("a second start must be rejected while active")
