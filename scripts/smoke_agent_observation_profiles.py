#!/usr/bin/env python3
"""Create and inspect one public LIBERO observation-profile frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Select the offscreen backend before importing robosuite through LIBERO.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.agent_env import (  # noqa: E402
    make_libero_agent_env,
    write_public_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--profile", default="level4")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    agent_env = make_libero_agent_env(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        profile=args.profile,
        seed=args.seed,
        camera_height=args.resolution,
        camera_width=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
    )
    try:
        result = agent_env.start_episode()
        json_path = write_public_observation(
            result["observation"], args.output_dir / "obs_000000"
        )
        print(
            json.dumps(
                {
                    "task_instruction": result["task_instruction"],
                    "profile": result["observation"]["profile"],
                    "observation_json": str(json_path.resolve()),
                },
                indent=2,
            )
        )
    finally:
        agent_env.close()


if __name__ == "__main__":
    main()
