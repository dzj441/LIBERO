from types import SimpleNamespace

import pytest

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


def test_native_sequence_submission_limit_is_hard_capped_at_fifty():
    agent_env = _native_only_agent_env(max_agent_steps=None)
    for index in range(50):
        result = agent_env.step_osc_sequence([[0.0] * 7])
        assert result["accepted_agent_step"] == index + 1

    with pytest.raises(RuntimeError, match=r"agent step limit reached \(50\)"):
        agent_env.step_osc_sequence([[0.0] * 7])
    assert agent_env.native_sequence_executor.calls == 50


def test_native_sequence_honors_stricter_configured_submission_limit():
    agent_env = _native_only_agent_env(max_agent_steps=2)
    agent_env.step_osc_sequence([[0.0] * 7])
    agent_env.step_osc_sequence([[0.0] * 7])
    with pytest.raises(RuntimeError, match=r"agent step limit reached \(2\)"):
        agent_env.step_osc_sequence([[0.0] * 7])
