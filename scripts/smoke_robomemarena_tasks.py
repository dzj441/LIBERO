#!/usr/bin/env python3
"""Reset and step one representative task from each RoboMemArena category."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any


os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


REPRESENTATIVE_TASKS = {
    1: "multi_object_sequence",
    4: "multi_object_occlusion",
    10: "multi_object_counting",
    25: "multi_object_transferring",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(REPRESENTATIVE_TASKS),
    )
    parser.add_argument("--seed", type=int, default=1830315042)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    source_root = Path(__file__).resolve().parents[1]

    from scripts.robomemarena_bootstrap import activate_robomemarena_core

    merged_fork = activate_robomemarena_core(source_root=source_root)

    from libero.libero.agent_env.robomemarena import (
        get_robomemarena_task_spec,
        make_robomemarena_agent_env,
        robomemarena_source_fingerprint,
    )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_id in args.task_ids:
        started_at = time.monotonic()
        env = None
        try:
            spec = get_robomemarena_task_spec(task_id)
            env = make_robomemarena_agent_env(
                task_id=task_id,
                profile="level4",
                seed=args.seed,
                camera_height=256,
                camera_width=256,
                render_gpu_device_id=args.render_gpu_device_id,
                initial_settle_control_steps=1,
                max_agent_steps=2,
            )
            initial = env.start_episode()
            stepped = env.step_osc_sequence([[0.0] * 7])
            finish = env.finish_episode()
            initial_frame = initial["observation"]
            stepped_frame = stepped["observation"]
            record = {
                "task_id": task_id,
                "category": REPRESENTATIVE_TASKS.get(task_id, "custom"),
                "instruction": spec.instruction,
                "source_kind": robomemarena_source_fingerprint(
                    task_id=task_id
                )["source_kind"],
                "initial_observation_id": initial_frame["observation_id"],
                "stepped_observation_id": stepped_frame["observation_id"],
                "initial_keys": sorted(initial_frame),
                "camera_names": sorted(initial_frame["cameras"]),
                "execution_control_steps": stepped["execution"][
                    "control_steps"
                ],
                "finish_success": finish["success"],
                "private_stage_count": finish["private_evaluation"][
                    "required_stage_count"
                ],
                "elapsed_s": time.monotonic() - started_at,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        except Exception as exc:
            failure = {
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(json.dumps(failure, sort_keys=True), flush=True)
        finally:
            if env is not None:
                env.close()

    report = {
        "schema_version": "libero.robomemarena_environment_smoke.v1",
        "merged_fork_is_system_temporary": str(merged_fork).startswith(
            "/tmp/"
        ),
        "external_checkout_used": False,
        "seed": args.seed,
        "records": records,
        "failures": failures,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
