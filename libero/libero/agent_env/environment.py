"""Stateful AgentEnv lifecycle around a single live LIBERO simulation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .control import (
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
    BaseFrameOSCExecutor,
    EEFCommand,
    NativeOSCSequenceExecutor,
    OSCControlConfig,
)
from .observation import MasterObservationCollector, TaskEntitySelection
from .profiles import ObservationProfile, project_public_observation


class LiberoAgentEnv:
    """Expose one stateful Agent action at a time without private task state.

    Reward and task checker outputs are intentionally withheld until
    ``finish_episode``. Every accepted metric target or native OSC sequence
    returns the actual post-execution observation from the same trajectory.
    """

    def __init__(
        self,
        env: Any,
        profile: ObservationProfile | int | str,
        camera_height: int,
        camera_width: int,
        *,
        task_instruction: str,
        initial_state: np.ndarray | None = None,
        task_entities: TaskEntitySelection | None = None,
        control_config: OSCControlConfig | None = None,
        initial_settle_control_steps: int = 10,
        max_agent_steps: int | None = None,
        private_control_step_callback: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> None:
        if initial_settle_control_steps < 0:
            raise ValueError("initial_settle_control_steps must be non-negative")
        if max_agent_steps is not None and max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive when provided")
        self.env = env
        self.profile = ObservationProfile.parse(profile)
        self.task_instruction = str(task_instruction)
        self.initial_state = (
            None if initial_state is None else np.asarray(initial_state).copy()
        )
        self.initial_settle_control_steps = int(initial_settle_control_steps)
        self.max_agent_steps = max_agent_steps
        self.collector = MasterObservationCollector(
            env,
            camera_height=camera_height,
            camera_width=camera_width,
            task_entities=task_entities,
        )
        self.private_control_step_callback = private_control_step_callback
        self.executor = BaseFrameOSCExecutor(
            env,
            control_config,
            control_step_callback=private_control_step_callback,
        )
        self.native_sequence_executor = NativeOSCSequenceExecutor(
            env,
            control_step_callback=private_control_step_callback,
        )
        self._started = False
        self._finished = False
        self._agent_step_index = 0
        self._latest_raw_observation: dict[str, Any] | None = None

    def start_episode(self) -> dict[str, Any]:
        if self._started and not self._finished:
            raise RuntimeError("episode is already active")
        raw_observation = self.env.reset()
        if self.initial_state is not None:
            raw_observation = self.env.set_init_state(self.initial_state)

        hold_action = np.zeros(7, dtype=np.float64)
        for _ in range(self.initial_settle_control_steps):
            raw_observation, _reward, _done, _info = self.env.step(hold_action)
            if self.private_control_step_callback is not None:
                self.private_control_step_callback(raw_observation)

        self._started = True
        self._finished = False
        self._agent_step_index = 0
        self._latest_raw_observation = raw_observation
        return {
            "task_instruction": self.task_instruction,
            "observation": self._public_observation(frame_index=0),
        }

    def step_osc_target(
        self,
        delta_position_m: Sequence[float] = (0.0, 0.0, 0.0),
        delta_rotation_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        delta_gripper_width_m: float = 0.0,
    ) -> dict[str, Any]:
        self._require_active()
        self._require_agent_step_budget(self.max_agent_steps)
        command = EEFCommand.create(
            delta_position_m=delta_position_m,
            delta_rotation_rotvec_rad=delta_rotation_rotvec_rad,
            delta_gripper_width_m=delta_gripper_width_m,
        )
        raw_observation, execution = self.executor.execute(command)
        self._latest_raw_observation = raw_observation
        self._agent_step_index += 1
        return {
            "accepted_agent_step": self._agent_step_index,
            "execution": execution.to_public_dict(),
            "observation": self._public_observation(
                frame_index=self._agent_step_index
            ),
        }

    def step_osc_sequence(
        self,
        actions: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        """Execute 1--20 exact normalized OSC actions as one Agent submission."""

        self._require_active()
        configured_limit = self.max_agent_steps
        sequence_limit = (
            MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
            if configured_limit is None
            else min(configured_limit, MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS)
        )
        self._require_agent_step_budget(sequence_limit)
        raw_observation, execution = self.native_sequence_executor.execute(actions)
        self._latest_raw_observation = raw_observation
        self._agent_step_index += 1
        return {
            "accepted_agent_step": self._agent_step_index,
            "execution": execution.to_public_dict(),
            "observation": self._public_observation(
                frame_index=self._agent_step_index
            ),
        }

    def finish_episode(self) -> dict[str, Any]:
        self._require_active()
        success = bool(self.env.check_success())
        self._finished = True
        return {
            "success": success,
            "accepted_agent_steps": self._agent_step_index,
        }

    def close(self) -> None:
        self.env.close()

    def _public_observation(self, frame_index: int) -> dict[str, Any]:
        if self._latest_raw_observation is None:
            raise RuntimeError("no observation is available")
        master = self.collector.collect(self._latest_raw_observation, frame_index)
        return project_public_observation(master, self.profile)

    def _require_active(self) -> None:
        if not self._started:
            raise RuntimeError("start_episode must be called first")
        if self._finished:
            raise RuntimeError("episode has already finished")

    def _require_agent_step_budget(self, limit: int | None) -> None:
        if limit is not None and self._agent_step_index >= limit:
            raise RuntimeError(f"agent step limit reached ({limit})")
