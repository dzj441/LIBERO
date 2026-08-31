#!/usr/bin/env python3
"""Run one Codex session across an ordered LIBERO episode curriculum."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from libero.libero.agent_env.control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
    ActionInterface,
)
from libero.libero.agent_env.fixed_demo import (
    file_sha256,
    validate_p4_replay_master,
)
from libero.libero.agent_env.runtime_contract import (
    build_curriculum_server_ready_contract,
    canonical_json_sha256,
    sha256_text,
)
from scripts.launch_agent_episode import (
    MCP_TOOL_NAMES,
    _allocate_workspace,
    _archive_viewed_artifacts,
    _canonical_repository_root,
    _copy_codex_session,
    _create_new_directory,
    _git_value,
    _new_run_id,
    _prepare_workspace,
    _read_json,
    _server_environment,
    _session_files,
    _task_instruction,
    _terminate_process,
    _terminate_process_group,
    _utc_now,
    _validate_run_id,
    _wait_for_server_ready,
    _write_json_atomic,
    build_codex_command,
)


PLAN_SCHEMA_VERSION = "libero.agent_curriculum_plan.v1"
SERVER_CONFIG_SCHEMA_VERSION = "libero.agent_curriculum_server_config.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=50)
    parser.add_argument("--nvidia-runtime-root", type=Path)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-effort")
    parser.add_argument("--https-proxy", default="http://127.0.0.1:7890")
    parser.add_argument(
        "--experience-guidance",
        choices=("implicit", "explicit"),
        default="implicit",
        help="Whether the prompt identifies the first episodes as preparation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_agent_steps > MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS:
        raise ValueError(
            "native_osc_sequence permits at most "
            f"{MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS} accepted submissions per episode"
        )
    source_root = Path(__file__).resolve().parents[1]
    canonical_root = _canonical_repository_root(source_root)
    plan_path = args.curriculum_plan.expanduser().resolve()
    plan = load_curriculum_plan(plan_path, source_root=source_root)
    profile = plan["profile"]
    action_interface = ActionInterface.NATIVE_OSC_SEQUENCE
    run_root = (args.run_root or canonical_root / "agent_runs").resolve()
    nvidia_runtime_root = (
        args.nvidia_runtime_root or canonical_root / "runtime" / "nvidia"
    ).resolve()
    run_id = args.run_id or _new_run_id()
    _validate_run_id(run_id)
    run_directory = run_root / run_id
    _create_new_directory(run_directory)
    try:
        workspace, system_temp_workspace = _allocate_workspace(
            canonical_root=canonical_root,
            requested_root=args.workspace_root,
            run_id=run_id,
            keep_workspace=args.keep_workspace,
        )
    except Exception:
        run_directory.rmdir()
        raise

    episodes = _enrich_episodes(
        plan["episodes"],
        source_root=source_root,
        default_max_agent_steps=args.max_agent_steps,
    )
    prompt = build_curriculum_prompt(
        episode_count=len(episodes),
        fixed_demo_possible=any(
            episode["icl_condition"] == "fixed_demo" for episode in episodes
        ),
        experience_guidance=args.experience_guidance,
    )
    _prepare_workspace(
        source_root,
        workspace,
        prompt,
        icl_condition="none",
        action_interface=action_interface,
        control_transport="mcp",
    )
    _write_json_atomic(
        workspace / ".libero" / "episode.json",
        {
            "schema_version": "libero.agent_curriculum_workspace.v1",
            "episode_resumable": False,
            "run_mode": "multi_episode_curriculum",
            "episode_count": len(episodes),
            "operations": list(MCP_TOOL_NAMES),
            "action_interface": action_interface.value,
            "control_transport": "mcp",
            "max_native_osc_micro_steps_per_submission": (
                MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            ),
            "default_max_agent_steps_per_episode": args.max_agent_steps,
            "episode_max_agent_steps": [
                episode["max_agent_steps"] for episode in episodes
            ],
            "experience_guidance": args.experience_guidance,
            "observation_retention": "current_only",
            "next_task_disclosure": "start_response_only",
            "expert_demo": "published_per_episode_when_available",
        },
    )
    shutil.copy2(workspace / "TASK_PROMPT.txt", run_directory / "agent_prompt.txt")
    shutil.copy2(
        workspace / ".libero" / "episode.json",
        run_directory / "agent_workspace_contract.json",
    )

    server_config = {
        "schema_version": SERVER_CONFIG_SCHEMA_VERSION,
        "profile": profile,
        "resolution": args.resolution,
        "render_gpu_device_id": args.render_gpu_device_id,
        "initial_settle_control_steps": args.initial_settle_control_steps,
        "max_agent_steps_per_episode": args.max_agent_steps,
        "action_interface": action_interface.value,
        "episodes": episodes,
    }
    server_config_path = run_directory / "curriculum_server_config.json"
    _write_json_atomic(server_config_path, server_config)
    expected_ready = build_curriculum_server_ready_contract(
        episodes=episodes,
        profile=profile,
        resolution=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        initial_settle_control_steps=args.initial_settle_control_steps,
        max_agent_steps=args.max_agent_steps,
        action_interface=action_interface,
    )
    socket_path = workspace / ".libero" / "control.sock"
    server_ready_path = run_directory / "server_ready.json"
    server_environment, driver_version = _server_environment(
        source_root, nvidia_runtime_root
    )
    source_commit = _git_value(source_root, "rev-parse", "HEAD")
    source_status = _git_value(
        source_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        required=False,
    ) or ""
    prompt_sha256 = file_sha256(workspace / "TASK_PROMPT.txt")
    workspace_contract_sha256 = file_sha256(workspace / ".libero" / "episode.json")
    configuration_fingerprint = {
        "plan": _public_plan_commitment(plan, episodes),
        "profile": profile,
        "resolution": args.resolution,
        "render_gpu_device_id": args.render_gpu_device_id,
        "initial_settle_control_steps": args.initial_settle_control_steps,
        "default_max_agent_steps_per_episode": args.max_agent_steps,
        "episode_max_agent_steps": [
            episode["max_agent_steps"] for episode in episodes
        ],
        "experience_guidance": args.experience_guidance,
        "action_interface": action_interface.value,
        "control_transport": "mcp",
        "source_commit": source_commit,
        "operator_prompt_sha256": prompt_sha256,
        "workspace_contract_sha256": workspace_contract_sha256,
        "server_ready_contract_sha256": canonical_json_sha256(expected_ready),
    }
    manifest = {
        "schema_version": "libero.agent_curriculum_run_manifest.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        "curriculum_name": plan["name"],
        "run_mode": "multi_episode_curriculum",
        "episode_count": len(episodes),
        "primary_metric_episode_index": plan["primary_metric_episode_index"],
        "episodes": _private_manifest_episodes(episodes),
        "profile": profile,
        "action_interface": action_interface.value,
        "control_transport": "mcp",
        "default_max_agent_steps_per_episode": args.max_agent_steps,
        "episode_max_agent_steps": [
            episode["max_agent_steps"] for episode in episodes
        ],
        "experience_guidance": args.experience_guidance,
        "source_checkout": os.fspath(source_root),
        "source_commit": source_commit,
        "source_branch": _git_value(
            source_root, "branch", "--show-current", required=False
        ),
        "source_worktree_dirty": bool(source_status),
        "workspace": os.fspath(workspace),
        "workspace_lifecycle": (
            "system_temporary" if system_temp_workspace else "persistent_debug"
        ),
        "workspace_retained": True,
        "workspace_cleanup_owner": (
            "operating_system" if system_temp_workspace else "evaluator"
        ),
        "episode_resumable": False,
        "codex_execution_mode": "exec",
        "transport": "unix_socket",
        "agent_control_adapter": "mcp_stdio",
        "observation_retention": "current_only",
        "render_backend": "egl",
        "nvidia_userspace_driver": driver_version,
        "server_ready_verified": False,
        "codex_binary": args.codex_bin,
        "codex_model_requested": args.codex_model,
        "codex_effort_requested": args.codex_effort,
        "integrity": {
            "algorithm": "sha256",
            "configuration_sha256": canonical_json_sha256(
                configuration_fingerprint
            ),
            "operator_prompt_sha256": prompt_sha256,
            "workspace_contract_sha256": workspace_contract_sha256,
            "expected_server_ready_contract_sha256": canonical_json_sha256(
                expected_ready
            ),
            "source_worktree_status_sha256": sha256_text(source_status),
        },
    }
    _write_json_atomic(run_directory / "run_manifest.json", manifest)

    server_command = [
        sys.executable,
        "-u",
        os.fspath(source_root / "scripts" / "libero_curriculum_server.py"),
        "--config",
        os.fspath(server_config_path),
        "--workspace",
        os.fspath(workspace),
        "--socket",
        ".libero/control.sock",
        "--run-directory",
        os.fspath(run_directory),
        "--launcher-pid",
        str(os.getpid()),
    ]
    return _run_processes(
        args=args,
        source_root=source_root,
        workspace=workspace,
        run_directory=run_directory,
        socket_path=socket_path,
        server_ready_path=server_ready_path,
        server_environment=server_environment,
        server_command=server_command,
        expected_ready=expected_ready,
        manifest=manifest,
        prompt=prompt,
    )


def load_curriculum_plan(path: Path, *, source_root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported curriculum plan schema")
    episodes = value.get("episodes")
    if not isinstance(episodes, list) or len(episodes) < 2:
        raise ValueError("curriculum plan requires at least two episodes")
    primary = value.get("primary_metric_episode_index", len(episodes) - 1)
    if not isinstance(primary, int) or not 0 <= primary < len(episodes):
        raise ValueError("primary_metric_episode_index is outside the curriculum")
    if primary != len(episodes) - 1:
        raise ValueError("primary_metric_episode_index must identify the final episode")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("curriculum plan requires a name")
    profile = value.get("profile", "level4")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "name": " ".join(name.split()),
        "profile": str(profile),
        "episodes": episodes,
        "primary_metric_episode_index": primary,
        "source_root": os.fspath(source_root),
    }


def _enrich_episodes(
    raw_episodes: list[dict[str, Any]],
    *,
    source_root: Path,
    default_max_agent_steps: int = 50,
) -> list[dict[str, Any]]:
    episodes = []
    for episode_index, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict):
            raise ValueError("each curriculum episode must be an object")
        required = {"suite", "task_id", "init_state_id", "seed", "icl_condition"}
        if not required.issubset(raw):
            raise ValueError("curriculum episode is missing required fields")
        icl_condition = str(raw["icl_condition"])
        if icl_condition not in {"none", "fixed_demo"}:
            raise ValueError("unsupported curriculum ICL condition")
        master_value = raw.get("fixed_demo_master")
        if icl_condition == "fixed_demo" and not isinstance(master_value, str):
            raise ValueError("fixed_demo curriculum episode requires a master")
        if icl_condition == "none" and master_value is not None:
            raise ValueError("non-ICL curriculum episode cannot name a master")
        suite = str(raw["suite"])
        task_id = int(raw["task_id"])
        task_instruction = _task_instruction(suite, task_id)
        max_agent_steps = int(raw.get("max_agent_steps", default_max_agent_steps))
        if not 1 <= max_agent_steps <= MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS:
            raise ValueError(
                f"episode {episode_index} max_agent_steps must be between 1 and "
                f"{MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS}"
            )
        master = None
        master_hash = None
        if master_value is not None:
            candidate = Path(master_value).expanduser()
            master = (candidate if candidate.is_absolute() else source_root / candidate).resolve()
            manifest_path = master / "p4_master_manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"episode {episode_index} fixed-demo master is missing"
                )
            master_manifest = validate_p4_replay_master(master)
            master_instruction = " ".join(
                str(master_manifest["task"]["instruction"]).split()
            )
            if master_instruction != task_instruction:
                raise ValueError(
                    f"episode {episode_index} fixed-demo task does not match"
                )
            master_hash = file_sha256(manifest_path)
        episodes.append(
            {
                "episode_index": episode_index,
                "suite": suite,
                "task_id": task_id,
                "init_state_id": int(raw["init_state_id"]),
                "seed": int(raw["seed"]),
                "task_instruction": task_instruction,
                "max_agent_steps": max_agent_steps,
                "icl_condition": icl_condition,
                "fixed_demo_master": (
                    None if master is None else os.fspath(master)
                ),
                "fixed_demo_master_manifest_sha256": master_hash,
            }
        )
    return episodes


def build_curriculum_prompt(
    *,
    episode_count: int,
    fixed_demo_possible: bool,
    experience_guidance: str = "implicit",
) -> str:
    if experience_guidance not in {"implicit", "explicit"}:
        raise ValueError("experience_guidance must be implicit or explicit")
    demo_notice = ""
    if fixed_demo_possible:
        demo_notice = (
            " If `fixed_demo_available` is true, a verified successful "
            "demonstration from a separate episode of the same task is available "
            "at `benchmark_inputs/expert_demo/`; its scene and object poses may "
            "differ. It contains native per-control-cycle OSC_POSE actions and "
            "measured EEF states, where EEF poses are observations rather than "
            "actions."
        )
    guidance_notice = ""
    if experience_guidance == "explicit":
        guidance_notice = (
            "\nThe episodes before the final episode are preparatory experiences. "
            "Use any relevant experience from them when attempting the final "
            "episode.\n"
        )
    return f"""Complete {episode_count} prepared LIBERO episodes in order within this one session.
{guidance_notice}

For each episode:
1. Call the `start_episode` robot tool when no episode is active. Its result gives the current `task_instruction`, episode index, initial observation, `max_agent_steps` budget, and whether a fixed demonstration is available.{demo_notice}
2. Complete that current task with the `osc_sequence` robot tool. Its `actions` argument contains 1 to {MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} normalized 7D OSC_POSE micro actions in `[dx, dy, dz, rx, ry, rz, gripper]` order. Every component must be within [-1, 1]. Translation 1.0 corresponds to 0.05 m, rotation 1.0 to a 0.5 rad rotation-vector component, gripper -1 opens, and +1 closes. Each sequence call counts as one Agent step.
3. After each action, inspect `benchmark_inputs/current_observation/observation.json` and any referenced files before acting again.
4. Call `finish_episode` once for the current task. If it returns `next_episode_available=true`, begin the next episode with `start_episode`. The run is complete only when `curriculum_complete=true`.
"""


def _prepare_codex_environment(
    *, workspace: Path, https_proxy: str
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (os.fspath(workspace / "bin"), environment.get("PATH", ""))
    )
    environment["LIBERO_CONTROL_SOCKET"] = ".libero/control.sock"
    environment["LIBERO_AGENT_WORKSPACE"] = os.fspath(workspace)
    environment["LIBERO_ACTION_INTERFACE"] = (
        ActionInterface.NATIVE_OSC_SEQUENCE.value
    )
    environment["HTTPS_PROXY"] = https_proxy
    return environment


def _run_processes(
    *,
    args: argparse.Namespace,
    source_root: Path,
    workspace: Path,
    run_directory: Path,
    socket_path: Path,
    server_ready_path: Path,
    server_environment: dict[str, str],
    server_command: list[str],
    expected_ready: dict[str, Any],
    manifest: dict[str, Any],
    prompt: str,
) -> int:
    server_log_path = run_directory / "server.log"
    codex_started_at = time.time()
    known_sessions = _session_files()
    server_log = server_log_path.open("w", encoding="utf-8")
    server_process: subprocess.Popen[Any] | None = None
    codex_process: subprocess.Popen[Any] | None = None
    codex_return_code: int | None = None
    server_return_code: int | None = None
    infrastructure_error: str | None = None
    session_archive_error: str | None = None
    caught_exception: BaseException | None = None
    try:
        server_process = subprocess.Popen(
            server_command,
            cwd=source_root,
            env=server_environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        actual_ready = _wait_for_server_ready(
            socket_path,
            server_ready_path,
            server_process,
            expected_contract=expected_ready,
            timeout_s=args.server_ready_timeout_s,
            server_log_path=server_log_path,
        )
        manifest["server_ready_verified"] = True
        manifest["integrity"]["actual_server_ready_contract_sha256"] = (
            canonical_json_sha256(actual_ready)
        )
        _write_json_atomic(run_directory / "run_manifest.json", manifest)

        codex_command = build_codex_command(
            codex_bin=args.codex_bin,
            prompt=prompt,
            model=args.codex_model,
            effort=args.codex_effort,
            workspace=workspace,
            control_transport="mcp",
        )
        print(f"run_id: {run_directory.name}", flush=True)
        print(f"workspace: {workspace}", flush=True)
        print(f"private_run: {run_directory}", flush=True)
        print("starting Codex CLI for curriculum...", flush=True)
        codex_process = subprocess.Popen(
            codex_command,
            cwd=workspace,
            env=_prepare_codex_environment(
                workspace=workspace, https_proxy=args.https_proxy
            ),
            stdin=subprocess.DEVNULL,
        )
        while codex_process.poll() is None:
            server_return_code = server_process.poll()
            if server_return_code is not None:
                result = _read_json(run_directory / "result.json")
                if result.get("status") != "finished":
                    infrastructure_error = (
                        "LIBERO curriculum server exited before final finish "
                        f"with code {server_return_code}"
                    )
                    _terminate_process(codex_process)
                    break
            time.sleep(0.25)
        codex_return_code = codex_process.wait()
        if server_process.poll() is None:
            result = _read_json(run_directory / "result.json")
            if result.get("status") == "finished":
                try:
                    server_process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    _terminate_process_group(server_process)
            else:
                _terminate_process_group(server_process)
        server_return_code = server_process.wait()
    except BaseException as exc:
        infrastructure_error = f"{type(exc).__name__}: {exc}"
        caught_exception = exc
        if codex_process is not None and codex_process.poll() is None:
            _terminate_process(codex_process)
        if server_process is not None and server_process.poll() is None:
            _terminate_process_group(server_process)
        codex_return_code = None if codex_process is None else codex_process.poll()
        server_return_code = None if server_process is None else server_process.poll()
    finally:
        server_log.close()
        try:
            archived_session = _copy_codex_session(
                workspace=workspace,
                run_directory=run_directory,
                known_sessions=known_sessions,
                started_at=codex_started_at,
            )
            if archived_session is not None:
                _archive_viewed_artifacts(
                    session_path=archived_session,
                    workspace=workspace,
                    run_directory=run_directory,
                )
        except BaseException as exc:
            session_archive_error = f"{type(exc).__name__}: {exc}"
            if infrastructure_error is None:
                infrastructure_error = (
                    f"session archival failed: {session_archive_error}"
                )
            if caught_exception is None:
                caught_exception = exc

    result_path = run_directory / "result.json"
    result = _read_json(result_path)
    if caught_exception is not None and not result:
        result = {
            "schema_version": "libero.agent_curriculum_result.v1",
            "status": "infrastructure_error",
            "reason": infrastructure_error,
            "finished_at": _utc_now(),
        }
    if result.get("status") == "aborted" and codex_return_code is not None:
        result["reason"] = "codex_process_exited_before_curriculum_finish"
    result.update(
        {
            "codex_exit_code": codex_return_code,
            "server_exit_code": server_return_code,
            "launcher_finished_at": _utc_now(),
            "infrastructure_error": infrastructure_error,
            "session_archive_error": session_archive_error,
        }
    )
    _write_json_atomic(result_path, result)
    if caught_exception is not None:
        print(infrastructure_error, file=sys.stderr, flush=True)
        return 2
    if infrastructure_error is not None or result.get("status") != "finished":
        return 2
    return 0 if codex_return_code == 0 else 2


def _public_plan_commitment(
    plan: dict[str, Any], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "name": plan["name"],
        "profile": plan["profile"],
        "primary_metric_episode_index": plan["primary_metric_episode_index"],
        "episodes": [
            {
                key: episode[key]
                for key in (
                    "episode_index",
                    "suite",
                    "task_id",
                    "init_state_id",
                    "seed",
                    "task_instruction",
                    "max_agent_steps",
                    "icl_condition",
                    "fixed_demo_master_manifest_sha256",
                )
            }
            for episode in episodes
        ],
    }


def _private_manifest_episodes(
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **episode,
            "fixed_demo_master": episode["fixed_demo_master"],
        }
        for episode in episodes
    ]


if __name__ == "__main__":
    raise SystemExit(main())
