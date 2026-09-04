"""Factory for the official LIBERO suites and deterministic init states."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

import libero.libero as libero_package
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from .arrange_table import arrange_table_private_evaluator
from .control import MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS, OSCControlConfig
from .environment import LiberoAgentEnv
from .observation import TaskEntitySelection
from .profiles import ObservationProfile
from .task_references import load_task_reference_rgb


def make_libero_agent_env(
    *,
    suite: str = "libero_object",
    task_id: int = 0,
    init_state_id: int = 0,
    profile: ObservationProfile | int | str = ObservationProfile.LEVEL4,
    seed: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    task_entities: TaskEntitySelection | None = None,
    control_config: OSCControlConfig | None = None,
    initial_settle_control_steps: int = 10,
    max_agent_steps: int | None = None,
    native_sequence_submission_limit: int | None = (
        MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
    ),
    private_control_step_callback: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    render_gpu_device_id: int = -1,
    bddl_root: str | os.PathLike[str] | None = None,
    init_states_root: str | os.PathLike[str] | None = None,
    **env_kwargs: Any,
) -> LiberoAgentEnv:
    """Create an agent-safe environment for one official task and init state."""

    benchmark_class = get_benchmark(suite)
    task_suite = benchmark_class()
    if not 0 <= task_id < task_suite.get_num_tasks():
        raise ValueError(
            f"task_id must be in [0, {task_suite.get_num_tasks()}), got {task_id}"
        )
    task = task_suite.get_task(task_id)
    package_root = Path(libero_package.__file__).resolve().parent
    bddl_root = Path(bddl_root) if bddl_root is not None else package_root / "bddl_files"
    init_states_root = (
        Path(init_states_root)
        if init_states_root is not None
        else package_root / "init_files"
    )
    bddl_path = bddl_root / task.problem_folder / task.bddl_file
    init_state_path = init_states_root / task.problem_folder / task.init_states_file
    init_states = _load_trusted_init_states(os.fspath(init_state_path))
    if not 0 <= init_state_id < len(init_states):
        raise ValueError(
            f"init_state_id must be in [0, {len(init_states)}), got {init_state_id}"
        )

    reserved = {
        "bddl_file_name",
        "camera_names",
        "camera_heights",
        "camera_widths",
        "camera_depths",
        "camera_segmentations",
        "use_object_obs",
        "ignore_done",
        "initialization_noise",
        "render_gpu_device_id",
        "horizon",
    }
    conflicts = reserved.intersection(env_kwargs)
    if conflicts:
        raise ValueError(f"factory-managed env kwargs cannot be overridden: {sorted(conflicts)}")

    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        use_object_obs=False,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=render_gpu_device_id,
        horizon=10000,
        **env_kwargs,
    )
    env.seed(seed)
    instruction = " ".join(task.language.split())
    private_episode_evaluator = arrange_table_private_evaluator(
        env,
        suite=suite,
        task_id=task_id,
    )
    return LiberoAgentEnv(
        env,
        profile=profile,
        camera_height=camera_height,
        camera_width=camera_width,
        task_instruction=instruction,
        task_reference_rgb=load_task_reference_rgb(suite, task_id),
        initial_state=np.asarray(init_states[init_state_id]),
        task_entities=task_entities,
        control_config=control_config,
        initial_settle_control_steps=initial_settle_control_steps,
        max_agent_steps=max_agent_steps,
        native_sequence_submission_limit=native_sequence_submission_limit,
        private_control_step_callback=private_control_step_callback,
        private_episode_evaluator=private_episode_evaluator,
    )


def _load_trusted_init_states(path: str) -> Any:
    """Load the official local init-state tensor across PyTorch versions."""

    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)
