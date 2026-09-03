#!/usr/bin/env python3
"""Generate deterministic Arrange Table init states and its verified goal image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image
import torch

from libero.libero.envs import OffScreenRenderEnv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_BDDL = (
    REPOSITORY_ROOT
    / "libero"
    / "libero"
    / "bddl_files"
    / "libero_arrange_table"
    / "arrange_table.bddl"
)
INIT_STATES = (
    REPOSITORY_ROOT
    / "libero"
    / "libero"
    / "init_files"
    / "libero_arrange_table"
    / "arrange_table.pruned_init"
)
GOAL_IMAGE = (
    REPOSITORY_ROOT
    / "libero"
    / "libero"
    / "assets"
    / "task_references"
    / "libero_arrange_table"
    / "goal_rgb.png"
)

GOAL_INITIAL_STATE = """  (:init
    (On plate_1 living_room_table_plate_left_region)
    (On plate_2 living_room_table_plate_right_region)
    (On basket_1 living_room_table_basket_init_region)
    (On porcelain_mug_1 plate_1)
    (On white_yellow_mug_1 plate_2)
    (In butter_1 basket_1_contain_region)
  )
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-init-states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=20)
    return parser.parse_args()


def _make_env(bddl_path: Path, args: argparse.Namespace) -> OffScreenRenderEnv:
    return OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_depths=False,
        use_object_obs=False,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=args.render_gpu_device_id,
        horizon=10000,
    )


def _predicate_results(env: OffScreenRenderEnv) -> list[bool]:
    return [
        bool(env.env._eval_predicate(predicate))
        for predicate in env.env.parsed_problem["goal_state"]
    ]


def _settle(env: OffScreenRenderEnv, steps: int) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] | None = None
    hold = np.zeros(7, dtype=np.float64)
    for _ in range(steps):
        observation, _reward, _done, _info = env.step(hold)
    if observation is None:
        observation = env.env._get_observations()
    return observation


def _write_init_states(args: argparse.Namespace) -> tuple[int, int]:
    if args.num_init_states <= 0:
        raise ValueError("--num-init-states must be positive")
    env = _make_env(TASK_BDDL, args)
    states: list[np.ndarray] = []
    try:
        env.seed(args.seed)
        for state_index in range(args.num_init_states):
            env.reset()
            _settle(env, args.settle_steps)
            predicate_results = _predicate_results(env)
            if any(predicate_results) or env.check_success():
                raise RuntimeError(
                    "generated initial state unexpectedly satisfies a goal "
                    f"predicate at index {state_index}: {predicate_results}"
                )
            states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
    finally:
        env.close()

    stacked = np.stack(states)
    INIT_STATES.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=INIT_STATES.parent,
        prefix=f".{INIT_STATES.name}.",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(stacked, temporary_path)
        temporary_path.replace(INIT_STATES)
    finally:
        temporary_path.unlink(missing_ok=True)
    return stacked.shape


def _goal_bddl_text() -> str:
    source = TASK_BDDL.read_text(encoding="utf-8")
    init_start = source.index("  (:init\n")
    goal_start = source.index("  (:goal\n", init_start)
    return source[:init_start] + GOAL_INITIAL_STATE + "\n" + source[goal_start:]


def _write_goal_image(args: argparse.Namespace) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix="libero-arrange-table-goal-") as temp:
        goal_bddl = Path(temp) / "arrange_table_goal.bddl"
        goal_bddl.write_text(_goal_bddl_text(), encoding="utf-8")
        env = _make_env(goal_bddl, args)
        try:
            env.seed(args.seed)
            observation = env.reset()
            if args.settle_steps:
                observation = _settle(env, args.settle_steps)
            predicate_results = _predicate_results(env)
            if not all(predicate_results) or not env.check_success():
                raise RuntimeError(
                    "goal-state scene does not satisfy all checker predicates: "
                    f"{predicate_results}"
                )
            # robosuite camera arrays use a bottom-left image origin.
            rgb = np.ascontiguousarray(
                np.flipud(np.asarray(observation["agentview_image"], dtype=np.uint8))
            )
        finally:
            env.close()

    GOAL_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=GOAL_IMAGE.parent,
        prefix=f".{GOAL_IMAGE.name}.",
        suffix=".png",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        Image.fromarray(rgb, mode="RGB").save(temporary_path, format="PNG")
        temporary_path.replace(GOAL_IMAGE)
    finally:
        temporary_path.unlink(missing_ok=True)
    return rgb.shape


def main() -> None:
    args = parse_args()
    init_shape = _write_init_states(args)
    goal_shape = _write_goal_image(args)
    print(
        json.dumps(
            {
                "bddl": os.fspath(TASK_BDDL),
                "init_states": os.fspath(INIT_STATES),
                "init_states_shape": init_shape,
                "goal_image": os.fspath(GOAL_IMAGE),
                "goal_image_shape": goal_shape,
                "initial_goal_predicates_all_false": True,
                "goal_checker_all_true": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
