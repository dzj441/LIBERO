from types import SimpleNamespace

import numpy as np
import pytest

import libero.libero.agent_env.environment as environment_module
from libero.libero.agent_env.environment import LiberoAgentEnv


class _FakeNativeSequenceExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, actions):
        self.calls += 1
        return {"raw": self.calls}, SimpleNamespace(
            to_public_dict=lambda: {
                "command_completed": True,
                "micro_step_count": len(actions),
            }
        )


def _native_only_agent_env(max_agent_steps=None):
    agent_env = LiberoAgentEnv.__new__(LiberoAgentEnv)
    agent_env._started = True
    agent_env._finished = False
    agent_env._agent_step_index = 0
    agent_env._latest_raw_observation = {"raw": 0}
    agent_env.max_agent_steps = max_agent_steps
    agent_env.native_sequence_executor = _FakeNativeSequenceExecutor()
    agent_env._public_observation = lambda frame_index: {
        "observation_id": f"obs_{frame_index:06d}",
    }
    return agent_env


def test_native_sequence_submission_limit_is_hard_capped_at_one_hundred():
    agent_env = _native_only_agent_env(max_agent_steps=None)
    for index in range(100):
        result = agent_env.step_osc_sequence([[0.0] * 7])
        assert result["accepted_agent_step"] == index + 1

    with pytest.raises(RuntimeError, match=r"agent step limit reached \(100\)"):
        agent_env.step_osc_sequence([[0.0] * 7])
    assert agent_env.native_sequence_executor.calls == 100


def test_native_sequence_honors_stricter_configured_submission_limit():
    agent_env = _native_only_agent_env(max_agent_steps=2)
    agent_env.step_osc_sequence([[0.0] * 7])
    agent_env.step_osc_sequence([[0.0] * 7])
    with pytest.raises(RuntimeError, match=r"agent step limit reached \(2\)"):
        agent_env.step_osc_sequence([[0.0] * 7])


class _BrokenBddlDiagnosticEnv:
    @staticmethod
    def check_success():
        raise KeyError("missing_region")


class _SuccessfulPrivateEvaluator:
    @staticmethod
    def result():
        return {"success": True, "stage_score_percent": 100.0}


def test_private_checker_remains_authoritative_when_bddl_diagnostic_breaks():
    agent_env = LiberoAgentEnv.__new__(LiberoAgentEnv)
    agent_env._started = True
    agent_env._finished = False
    agent_env._agent_step_index = 3
    agent_env.env = _BrokenBddlDiagnosticEnv()
    agent_env.private_episode_evaluator = _SuccessfulPrivateEvaluator()

    result = agent_env.finish_episode()

    assert result["success"] is True
    assert result["accepted_agent_steps"] == 3
    private = result["private_evaluation"]
    assert private["bddl_final_goal_success"] is None
    assert private["bddl_final_goal_diagnostic_error"] == {
        "error_type": "KeyError",
        "message": "'missing_region'",
    }


def test_constructor_forwards_task_reference_to_observation_collector(monkeypatch):
    captured = {}

    class _FakeCollector:
        def __init__(
            self,
            env,
            camera_height,
            camera_width,
            task_entities=None,
            task_reference_rgb=None,
        ):
            captured["task_reference_rgb"] = task_reference_rgb

    class _FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(environment_module, "MasterObservationCollector", _FakeCollector)
    monkeypatch.setattr(environment_module, "BaseFrameOSCExecutor", _FakeExecutor)
    monkeypatch.setattr(environment_module, "NativeOSCSequenceExecutor", _FakeExecutor)

    reference = np.zeros((2, 3, 3), dtype=np.uint8)
    LiberoAgentEnv(
        object(),
        profile="level1",
        camera_height=2,
        camera_width=3,
        task_instruction="Arrange Table",
        task_reference_rgb=reference,
    )

    assert captured["task_reference_rgb"] is reference
