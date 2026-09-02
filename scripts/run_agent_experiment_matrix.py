#!/usr/bin/env python3
"""Run a fully enumerated Agent experiment matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from libero.libero.agent_env.experiments import (  # noqa: E402
    load_experiment_matrix,
    summarize_experiment_runs,
    write_experiment_summary,
)
from libero.libero.agent_env.codex_defaults import (  # noqa: E402
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument(
        "--launcher-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Clean LIBERO checkout whose launchers and runtime code are evaluated",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=SOURCE_ROOT,
        help="Root used to resolve relative fixed-demo master paths",
    )
    parser.add_argument("--only-run", action="append", default=[])
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--codex-effort", default=DEFAULT_CODEX_EFFORT)
    parser.add_argument("--https-proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_experiment_matrix(args.matrix)
    launcher_root = args.launcher_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    if not (launcher_root / "scripts" / "launch_agent_episode.py").is_file():
        raise FileNotFoundError("launcher root does not contain LIBERO launchers")
    batch_root = (
        args.batch_root or SOURCE_ROOT / "agent_runs" / matrix["name"]
    ).expanduser().resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    selected = _select_runs(matrix["runs"], args.only_run)
    _freeze_matrix(batch_root / "experiment_matrix_resolved.json", matrix)

    return_code = 0
    for ordinal, run in enumerate(selected, start=1):
        run_directory = batch_root / run["run_id"]
        existing = _read_json(run_directory / "result.json")
        if existing.get("status") == "finished":
            print(
                f"[{ordinal}/{len(selected)}] skip finished {run['run_id']}",
                flush=True,
            )
            continue
        if run_directory.exists():
            raise FileExistsError(
                f"refusing to overwrite incomplete run: {run_directory}"
            )

        command = build_launch_command(
            run,
            batch_root=batch_root,
            launcher_root=launcher_root,
            artifact_root=artifact_root,
            render_gpu_device_id=args.render_gpu_device_id,
            resolution=args.resolution,
            initial_settle_control_steps=args.initial_settle_control_steps,
            codex_bin=args.codex_bin,
            codex_model=args.codex_model,
            codex_effort=args.codex_effort,
            https_proxy=args.https_proxy,
        )
        print(
            f"[{ordinal}/{len(selected)}] start {run['run_id']} "
            f"({run['condition']}, {run['replicate_id']})",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue

        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (os.fspath(launcher_root), existing_pythonpath)
            if item
        )
        completed = subprocess.run(
            command,
            cwd=launcher_root,
            env=environment,
            check=False,
        )
        summary = summarize_experiment_runs(matrix, batch_root)
        write_experiment_summary(summary, batch_root)
        if completed.returncode != 0:
            return_code = completed.returncode
            print(
                f"[{ordinal}/{len(selected)}] failed {run['run_id']} "
                f"with exit code {completed.returncode}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                break

    if not args.dry_run:
        summary = summarize_experiment_runs(matrix, batch_root)
        paths = write_experiment_summary(summary, batch_root)
        print(f"summary: {paths['markdown']}", flush=True)
    return return_code


def build_launch_command(
    run: Mapping[str, Any],
    *,
    batch_root: Path,
    launcher_root: Path = SOURCE_ROOT,
    artifact_root: Path = SOURCE_ROOT,
    render_gpu_device_id: int,
    resolution: int,
    initial_settle_control_steps: int,
    codex_bin: str,
    codex_model: str,
    codex_effort: str,
    https_proxy: str,
) -> list[str]:
    common = [
        "--run-id",
        str(run["run_id"]),
        "--run-root",
        os.fspath(batch_root),
        "--render-gpu-device-id",
        str(render_gpu_device_id),
        "--resolution",
        str(resolution),
        "--initial-settle-control-steps",
        str(initial_settle_control_steps),
        "--codex-bin",
        codex_bin,
        "--codex-model",
        codex_model,
        "--codex-effort",
        codex_effort,
        "--https-proxy",
        https_proxy,
    ]
    if run["mode"] == "single_episode":
        episode = run["episode"]
        command = [
            sys.executable,
            os.fspath(launcher_root / "scripts" / "launch_agent_episode.py"),
            "--suite",
            episode["suite"],
            "--task-id",
            str(episode["task_id"]),
            "--init-state-id",
            str(episode["init_state_id"]),
            "--seed",
            str(episode["seed"]),
            "--profile",
            run["profile"],
            "--max-agent-steps",
            str(episode["max_agent_steps"]),
            "--icl",
            episode["icl_condition"],
        ]
        if episode["fixed_demo_master"] is not None:
            command.extend(
                [
                    "--fixed-demo-master",
                    os.fspath(
                        _resolve_master(
                            episode["fixed_demo_master"], artifact_root
                        )
                    ),
                ]
            )
        if episode["experience_context_spec"] is not None:
            command.extend(
                [
                    "--experience-context-spec",
                    os.fspath(
                        _resolve_master(
                            episode["experience_context_spec"], artifact_root
                        )
                    ),
                ]
            )
        return command + common

    plan_directory = batch_root / "experiment_plans"
    plan_directory.mkdir(parents=True, exist_ok=True)
    plan_path = plan_directory / f"{run['run_id']}.json"
    plan = {
        "schema_version": "libero.agent_curriculum_plan.v1",
        "name": f"{run['run_id']} curriculum",
        "profile": run["profile"],
        "episodes": [
            _materialize_episode(episode, artifact_root)
            for episode in run["episodes"]
        ],
        "primary_metric_episode_index": len(run["episodes"]) - 1,
    }
    _write_json_atomic(plan_path, plan)
    return [
        sys.executable,
        os.fspath(launcher_root / "scripts" / "launch_agent_curriculum.py"),
        "--curriculum-plan",
        os.fspath(plan_path),
        "--experience-guidance",
        run["experience_guidance"],
        *common,
    ]


def _select_runs(
    runs: list[dict[str, Any]], requested: list[str]
) -> list[dict[str, Any]]:
    if not requested:
        return runs
    selected_ids = set(requested)
    known_ids = {run["run_id"] for run in runs}
    missing = selected_ids - known_ids
    if missing:
        raise ValueError(f"unknown --only-run values: {sorted(missing)}")
    return [run for run in runs if run["run_id"] in selected_ids]


def _materialize_episode(
    episode: Mapping[str, Any], artifact_root: Path
) -> dict[str, Any]:
    value = dict(episode)
    master = value.get("fixed_demo_master")
    if isinstance(master, str):
        value["fixed_demo_master"] = os.fspath(
            _resolve_master(master, artifact_root)
        )
    return value


def _resolve_master(value: str, artifact_root: Path) -> Path:
    candidate = Path(value).expanduser()
    return (
        candidate if candidate.is_absolute() else artifact_root / candidate
    ).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _freeze_matrix(path: Path, matrix: Mapping[str, Any]) -> None:
    existing = _read_json(path)
    if existing:
        if existing != matrix:
            raise ValueError(
                "batch root already contains a different resolved experiment matrix"
            )
        return
    _write_json_atomic(path, matrix)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
