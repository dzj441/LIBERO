#!/usr/bin/env python3
"""Run one Codex-controlled LIBERO episode in an isolated workspace."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from libero.libero.benchmark import get_benchmark
from libero.libero.agent_env.control import (
    MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION,
    MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS,
    ActionInterface,
)
from libero.libero.agent_env.fixed_demo import file_sha256, project_fixed_demo_bundle
from libero.libero.agent_env.runtime_contract import (
    build_server_ready_contract,
    canonical_json_sha256,
    sha256_text,
    validate_server_ready_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--profile", default="level4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--initial-settle-control-steps", type=int, default=10)
    parser.add_argument("--max-agent-steps", type=int, default=50)
    parser.add_argument(
        "--action-interface",
        choices=tuple(interface.value for interface in ActionInterface),
        default=ActionInterface.METRIC_OSC_STEP.value,
        help="Mutually exclusive public robot-control condition",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--workspace-root",
        type=Path,
        help="Parent for the isolated workspace (defaults to the system temp disk)",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Use a named persistent debug workspace instead of a random temp path",
    )
    parser.add_argument("--nvidia-runtime-root", type=Path)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model")
    parser.add_argument("--codex-effort")
    parser.add_argument("--https-proxy", default="http://127.0.0.1:7890")
    parser.add_argument(
        "--icl",
        choices=("none", "fixed_demo"),
        default="none",
        help="Static in-context demonstration condition",
    )
    parser.add_argument(
        "--fixed-demo-master",
        type=Path,
        help="Evaluator-private verified P4 replay master for --icl fixed_demo",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action_interface = ActionInterface.parse(args.action_interface)
    if args.icl == "fixed_demo" and args.fixed_demo_master is None:
        raise ValueError("--icl fixed_demo requires --fixed-demo-master")
    if args.icl == "none" and args.fixed_demo_master is not None:
        raise ValueError("--fixed-demo-master is valid only with --icl fixed_demo")
    if (
        action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
        and args.max_agent_steps > MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS
    ):
        raise ValueError(
            "native_osc_sequence permits at most "
            f"{MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS} accepted submissions"
        )
    source_root = Path(__file__).resolve().parents[1]
    canonical_root = _canonical_repository_root(source_root)
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
    task_instruction = _task_instruction(args.suite, args.task_id)
    prompt = build_task_prompt(
        task_instruction,
        icl_condition=args.icl,
        action_interface=action_interface,
    )
    _prepare_workspace(
        source_root,
        workspace,
        prompt,
        run_id,
        icl_condition=args.icl,
        action_interface=action_interface,
    )
    shutil.copy2(workspace / "TASK_PROMPT.txt", run_directory / "agent_prompt.txt")
    shutil.copy2(
        workspace / ".libero" / "episode.json",
        run_directory / "agent_workspace_contract.json",
    )
    icl_projection_receipt = None
    if args.icl == "fixed_demo":
        icl_projection_receipt = project_fixed_demo_bundle(
            master_root=args.fixed_demo_master,
            destination=workspace / "benchmark_inputs" / "expert_demo",
            profile=args.profile,
            expected_task_instruction=task_instruction,
        )
        _write_json_atomic(
            run_directory / "icl_projection_receipt.json",
            icl_projection_receipt,
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
    expected_server_ready = build_server_ready_contract(
        suite=args.suite,
        task_id=args.task_id,
        init_state_id=args.init_state_id,
        task_instruction=task_instruction,
        profile=args.profile,
        seed=args.seed,
        resolution=args.resolution,
        render_gpu_device_id=args.render_gpu_device_id,
        initial_settle_control_steps=args.initial_settle_control_steps,
        max_agent_steps=args.max_agent_steps,
        action_interface=action_interface,
    )
    run_configuration = {
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "profile": expected_server_ready["observation_profile"],
        "icl_condition": args.icl,
        "fixed_demo_available": args.icl == "fixed_demo",
        "seed": args.seed,
        "resolution": args.resolution,
        "render_gpu_device_id": args.render_gpu_device_id,
        "initial_settle_control_steps": args.initial_settle_control_steps,
        "max_agent_steps": args.max_agent_steps,
        "action_interface": action_interface.value,
        "codex_binary": args.codex_bin,
        "codex_model_requested": args.codex_model,
        "codex_effort_requested": args.codex_effort,
        "max_native_osc_micro_steps_per_submission": (
            MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
            if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
            else None
        ),
    }
    prompt_sha256 = file_sha256(workspace / "TASK_PROMPT.txt")
    workspace_contract_sha256 = file_sha256(workspace / ".libero" / "episode.json")
    fixed_demo_manifest_sha256 = (
        None
        if args.icl == "none"
        else file_sha256(
            workspace / "benchmark_inputs" / "expert_demo" / "manifest.json"
        )
    )
    configuration_fingerprint = {
        **run_configuration,
        "source_commit": source_commit,
        "render_backend": "egl",
        "nvidia_userspace_driver": driver_version,
        "operator_prompt_sha256": prompt_sha256,
        "workspace_contract_sha256": workspace_contract_sha256,
        "fixed_demo_manifest_sha256": fixed_demo_manifest_sha256,
        "server_ready_contract_sha256": canonical_json_sha256(
            expected_server_ready
        ),
    }
    manifest = {
        "schema_version": "libero.agent_run_manifest.v1",
        "run_id": run_id,
        "created_at": _utc_now(),
        **run_configuration,
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
        "observation_retention": "current_only",
        "render_backend": "egl",
        "nvidia_userspace_driver": driver_version,
        "server_ready_verified": False,
        "integrity": {
            "algorithm": "sha256",
            "configuration_sha256": canonical_json_sha256(
                configuration_fingerprint
            ),
            "operator_prompt_sha256": prompt_sha256,
            "workspace_contract_sha256": workspace_contract_sha256,
            "fixed_demo_manifest_sha256": fixed_demo_manifest_sha256,
            "expected_server_ready_contract_sha256": canonical_json_sha256(
                expected_server_ready
            ),
            "source_worktree_status_sha256": sha256_text(source_status),
        },
    }
    _write_json_atomic(run_directory / "run_manifest.json", manifest)

    server_command = [
        sys.executable,
        "-u",
        os.fspath(source_root / "scripts" / "libero_agent_server.py"),
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--init-state-id",
        str(args.init_state_id),
        "--profile",
        args.profile,
        "--seed",
        str(args.seed),
        "--resolution",
        str(args.resolution),
        "--render-gpu-device-id",
        str(args.render_gpu_device_id),
        "--initial-settle-control-steps",
        str(args.initial_settle_control_steps),
        "--max-agent-steps",
        str(args.max_agent_steps),
        "--action-interface",
        action_interface.value,
        "--workspace",
        os.fspath(workspace),
        "--socket",
        ".libero/control.sock",
        "--run-directory",
        os.fspath(run_directory),
        "--launcher-pid",
        str(os.getpid()),
    ]
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
        actual_server_ready = _wait_for_server_ready(
            socket_path,
            server_ready_path,
            server_process,
            expected_contract=expected_server_ready,
            timeout_s=args.server_ready_timeout_s,
            server_log_path=server_log_path,
        )
        manifest["server_ready_verified"] = True
        manifest["integrity"]["actual_server_ready_contract_sha256"] = (
            canonical_json_sha256(actual_server_ready)
        )
        _write_json_atomic(run_directory / "run_manifest.json", manifest)

        codex_environment = os.environ.copy()
        codex_environment["PATH"] = os.pathsep.join(
            (os.fspath(workspace / "bin"), codex_environment.get("PATH", ""))
        )
        codex_environment["LIBERO_CONTROL_SOCKET"] = ".libero/control.sock"
        codex_environment["LIBERO_ACTION_INTERFACE"] = action_interface.value
        codex_environment["HTTPS_PROXY"] = args.https_proxy
        codex_command = build_codex_command(
            codex_bin=args.codex_bin,
            prompt=prompt,
            model=args.codex_model,
            effort=args.codex_effort,
        )

        print(f"run_id: {run_id}", flush=True)
        print(f"workspace: {workspace}", flush=True)
        print(f"private_run: {run_directory}", flush=True)
        print("starting Codex CLI...", flush=True)
        codex_process = subprocess.Popen(
            codex_command,
            cwd=workspace,
            env=codex_environment,
            stdin=subprocess.DEVNULL,
        )

        while codex_process.poll() is None:
            server_return_code = server_process.poll()
            if server_return_code is not None:
                result = _read_json(run_directory / "result.json")
                if result.get("status") != "finished":
                    infrastructure_error = (
                        "LIBERO server exited before finish "
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
        codex_return_code = (
            None if codex_process is None else codex_process.poll()
        )
        server_return_code = (
            None if server_process is None else server_process.poll()
        )
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
                infrastructure_error = f"session archival failed: {session_archive_error}"
            if caught_exception is None:
                caught_exception = exc
    result_path = run_directory / "result.json"
    result = _read_json(result_path)
    if caught_exception is not None and not result:
        result = {
            "schema_version": "libero.agent_run_result.v1",
            "status": "infrastructure_error",
            "reason": infrastructure_error,
            "finished_at": _utc_now(),
        }
    if result.get("status") == "aborted" and codex_return_code is not None:
        result["reason"] = "codex_process_exited_before_finish"
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


def build_task_prompt(
    task_instruction: str,
    *,
    icl_condition: str = "none",
    action_interface: ActionInterface | str = ActionInterface.METRIC_OSC_STEP,
) -> str:
    instruction = " ".join(str(task_instruction).split())
    action_interface = ActionInterface.parse(action_interface)
    if icl_condition not in {"none", "fixed_demo"}:
        raise ValueError(f"unsupported ICL condition: {icl_condition!r}")
    icl_notice = ""
    if icl_condition == "fixed_demo":
        compatibility_notice = ""
        if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE:
            compatibility_notice = (
                " Each source action vector has the same component semantics as "
                "one `osc-sequence` micro action."
            )
        icl_notice = (
            "\nA verified successful demonstration from a separate episode of "
            "the same task is available at `benchmark_inputs/expert_demo/`. "
            "The current scene configuration and object or goal poses may differ. "
            "The demonstration records the expert's native per-control-cycle "
            "OSC_POSE actions and measured EEF state observations. The measured "
            "EEF poses are observations, not actions."
            f"{compatibility_notice}\n"
        )
    if action_interface is ActionInterface.METRIC_OSC_STEP:
        control_instruction = (
            "2. Control the robot with `liberoctl osc-step --position DX DY DZ "
            "--rotation RX RY RZ --gripper-delta-m DG`. Each command specifies "
            "a metric Cartesian target delta executed through LIBERO's OSC_POSE "
            "controller. Position deltas are robot-base-frame metres. Rotation "
            "deltas are robot-base-frame rotation vectors in radians. DG is the "
            "change in total jaw opening width in metres: positive opens, negative "
            "closes, and zero preserves the current gripper target and grip force. "
            "A target outside the physical gripper-width range is rejected."
        )
    else:
        control_instruction = (
            "2. Control the robot with `liberoctl osc-sequence --actions-file "
            "PATH`. PATH must contain a JSON array of 1 to "
            f"{MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION} normalized 7D OSC_POSE "
            "micro actions in `[dx, dy, dz, rx, ry, rz, gripper]` order. Every "
            "component must be within [-1, 1]. Each vector executes one LIBERO "
            "policy interval; translation 1.0 corresponds to 0.05 m, rotation "
            "1.0 to a 0.5 rad rotation-vector component, gripper -1 opens, and "
            "+1 closes. One sequence submission counts as one Agent action, with "
            f"at most {MAX_NATIVE_OSC_SEQUENCE_SUBMISSIONS} accepted submissions."
        )
    return f"""{instruction}
{icl_notice}

A LIBERO episode has been prepared for you.

1. Run `liberoctl start` exactly once to begin and receive the initial observation.
{control_instruction}
3. Wait for each step to complete, then inspect `benchmark_inputs/current_observation/observation.json` and any referenced files before issuing another step.
4. When you have completed the task, run `liberoctl finish` exactly once. Only finish reports official task success.
"""


def build_codex_command(
    *,
    codex_bin: str,
    prompt: str,
    model: str | None = None,
    effort: str | None = None,
) -> list[str]:
    """Build a persistent, one-shot Codex CLI invocation for one episode."""

    command = [
        codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--dangerously-bypass-hook-trust",
        "--skip-git-repo-check",
        "--color",
        "never",
    ]
    if model:
        command.extend(("--model", model))
    if effort:
        command.extend(("--config", f'model_reasoning_effort="{effort}"'))
    command.append(prompt)
    return command


def _task_instruction(suite: str, task_id: int) -> str:
    benchmark_class = get_benchmark(suite)
    task_suite = benchmark_class()
    if not 0 <= task_id < task_suite.get_num_tasks():
        raise ValueError(
            f"task_id must be in [0, {task_suite.get_num_tasks()}), got {task_id}"
        )
    return " ".join(task_suite.get_task(task_id).language.split())


def _prepare_workspace(
    source_root: Path,
    workspace: Path,
    prompt: str,
    run_id: str,
    *,
    icl_condition: str,
    action_interface: ActionInterface | str,
) -> None:
    action_interface = ActionInterface.parse(action_interface)
    (workspace / ".libero").mkdir(mode=0o700)
    (workspace / "benchmark_inputs").mkdir()
    (workspace / "scratch").mkdir()
    binary_directory = workspace / "bin"
    binary_directory.mkdir()
    client = binary_directory / "liberoctl"
    shutil.copy2(source_root / "scripts" / "liberoctl.py", client)
    client.chmod(0o755)
    (workspace / "TASK_PROMPT.txt").write_text(prompt, encoding="utf-8")
    _write_json_atomic(
        workspace / ".libero" / "episode.json",
        {
            "schema_version": "libero.agent_workspace.v1",
            "run_id": run_id,
            "episode_resumable": False,
            "operations": [
                "start",
                action_interface.wire_command,
                "finish",
            ],
            "action_interface": action_interface.value,
            "max_native_osc_micro_steps_per_submission": (
                MAX_NATIVE_OSC_MICRO_STEPS_PER_SUBMISSION
                if action_interface is ActionInterface.NATIVE_OSC_SEQUENCE
                else None
            ),
            "observation_retention": "current_only",
            "icl_condition": icl_condition,
            "expert_demo": (
                "benchmark_inputs/expert_demo"
                if icl_condition == "fixed_demo"
                else None
            ),
        },
    )


def _server_environment(
    source_root: Path, nvidia_runtime_root: Path
) -> tuple[dict[str, str], str]:
    environment = os.environ.copy()
    driver_version = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ),
        text=True,
    ).splitlines()[0].strip()
    driver_directory = nvidia_runtime_root / driver_version
    library_directory = driver_directory / "runtime-libs-full"
    egl_manifest = driver_directory / "10_nvidia.local.json"
    if not library_directory.is_dir() or not egl_manifest.is_file():
        raise FileNotFoundError(
            f"matching EGL userspace stack is unavailable for NVIDIA {driver_version}"
        )
    environment["MUJOCO_GL"] = "egl"
    environment["PYOPENGL_PLATFORM"] = "egl"
    environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = os.fspath(egl_manifest)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        (os.fspath(library_directory), environment.get("LD_LIBRARY_PATH", ""))
    ).rstrip(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(source_root), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment, driver_version


def _canonical_repository_root(source_root: Path) -> Path:
    common_git = Path(
        subprocess.check_output(
            (
                "git",
                "-C",
                os.fspath(source_root),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            text=True,
        ).strip()
    ).resolve()
    return common_git.parent if common_git.name == ".git" else source_root


def _wait_for_server_ready(
    socket_path: Path,
    ready_path: Path,
    process: subprocess.Popen[Any],
    *,
    expected_contract: Mapping[str, Any],
    timeout_s: float,
    server_log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"LIBERO server exited during startup with code {return_code}; "
                f"see {server_log_path}"
            )
        try:
            socket_ready = stat.S_ISSOCK(socket_path.stat().st_mode)
        except FileNotFoundError:
            socket_ready = False
        if socket_ready and ready_path.is_file():
            actual = _read_json(ready_path)
            validate_server_ready_contract(actual, expected_contract)
            return actual
        time.sleep(0.2)
    raise TimeoutError(
        "LIBERO server did not publish a verified ready contract within "
        f"{timeout_s}s"
    )


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _validate_run_id(run_id: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if not run_id or len(run_id) > 96 or any(character not in allowed for character in run_id):
        raise ValueError("run_id must contain only letters, digits, '-' and '_'")


def _create_new_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()


def _allocate_workspace(
    *,
    canonical_root: Path,
    requested_root: Path | None,
    run_id: str,
    keep_workspace: bool,
) -> tuple[Path, bool]:
    if keep_workspace:
        root = (
            requested_root
            or canonical_root.parent / "agent_workspaces" / "libero"
        ).expanduser().resolve()
        workspace = root / run_id
        _create_new_directory(workspace)
        return workspace, False
    parent = requested_root.expanduser().resolve() if requested_root else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(
            prefix="libero-agent-workspace-",
            dir=os.fspath(parent) if parent is not None else None,
        )
    ).resolve()
    return workspace, True


def _terminate_process(process: subprocess.Popen[Any], timeout_s: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _terminate_process_group(
    process: subprocess.Popen[Any], timeout_s: float = 15.0
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _session_files() -> set[Path]:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions_root = codex_home / "sessions"
    return set(sessions_root.rglob("*.jsonl")) if sessions_root.is_dir() else set()


def _copy_codex_session(
    *,
    workspace: Path,
    run_directory: Path,
    known_sessions: set[Path],
    started_at: float,
) -> Path | None:
    candidates = []
    for path in _session_files():
        try:
            if path not in known_sessions or path.stat().st_mtime >= started_at - 2.0:
                metadata = _session_metadata(path)
                if Path(metadata.get("cwd", "")).resolve() == workspace:
                    candidates.append((path.stat().st_mtime, path, metadata))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    _modified, source, metadata = max(candidates, key=lambda item: item[0])
    shutil.copy2(source, run_directory / "codex_session.jsonl")
    _write_json_atomic(
        run_directory / "codex_session_metadata.json",
        {
            "schema_version": "libero.codex_session_reference.v1",
            "session_id": metadata.get("session_id") or metadata.get("id"),
            "cwd": metadata.get("cwd"),
            "source_file": os.fspath(source),
            "episode_resumable": False,
        },
    )
    return run_directory / "codex_session.jsonl"


def _session_image_view_paths(session_path: Path) -> list[str]:
    paths: list[str] = []
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return paths
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            continue
        payload = record.get("payload")
        if (
            record.get("type") != "event_msg"
            or not isinstance(payload, dict)
            or payload.get("type") != "item_completed"
        ):
            continue
        item = payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "ImageView":
            continue
        path = item.get("path")
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
    return paths


def _image_view_file(value: str, *, base: Path | None = None) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    if not parsed.scheme:
        path = Path(value).expanduser()
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve()
    return None


def _is_current_observation_path(source: Path, workspace: Path) -> bool:
    try:
        relative = source.relative_to(workspace)
    except ValueError:
        return False
    markers = (
        Path("benchmark_inputs/current_observation"),
        Path("benchmark_inputs/live_observation/current"),
    )
    return any(
        relative.parts[: len(marker.parts)] == marker.parts for marker in markers
    )


def _archive_viewed_artifacts(
    *, session_path: Path, workspace: Path, run_directory: Path
) -> dict[str, Any]:
    """Preserve files explicitly viewed by Codex before workspace cleanup."""

    archive_root = run_directory / "viewed_artifacts"
    entries: list[dict[str, Any]] = []
    archived_by_digest: dict[tuple[str, str], Path] = {}
    for source_value in _session_image_view_paths(session_path):
        source = _image_view_file(source_value, base=workspace)
        entry: dict[str, Any] = {
            "source_path": source_value,
            "source_absolute_path": os.fspath(source) if source is not None else None,
        }
        if source is None:
            entry["status"] = "unsupported_uri"
            entries.append(entry)
            continue
        try:
            relative = source.relative_to(workspace)
        except ValueError:
            relative = None
        if relative is not None:
            entry["workspace_relative_path"] = relative.as_posix()
        if _is_current_observation_path(source, workspace):
            entry["status"] = "historical_observation_archive"
            entries.append(entry)
            continue
        try:
            source.relative_to(run_directory)
        except ValueError:
            inside_run = False
        else:
            inside_run = True
        if inside_run:
            entry["status"] = "already_in_private_run"
            entry["archived_file"] = source.relative_to(run_directory).as_posix()
            entries.append(entry)
            continue
        if not source.is_file():
            entry["status"] = "source_missing_at_archive_time"
            entries.append(entry)
            continue
        digest = file_sha256(source)
        suffix = source.suffix.lower()
        if not suffix or len(suffix) > 12 or not suffix[1:].isalnum():
            suffix = ".bin"
        key = (digest, suffix)
        destination = archived_by_digest.get(key)
        if destination is None:
            archive_root.mkdir(parents=True, exist_ok=True)
            destination = archive_root / f"{digest}{suffix}"
            if not destination.is_file():
                shutil.copy2(source, destination)
            archived_by_digest[key] = destination
        entry.update(
            {
                "status": "archived",
                "archived_file": destination.relative_to(run_directory).as_posix(),
                "sha256": digest,
                "size_bytes": destination.stat().st_size,
            }
        )
        entries.append(entry)
    manifest = {
        "schema_version": "libero.viewed_artifacts_archive.v1",
        "created_at": _utc_now(),
        "artifact_count": sum(
            entry.get("status") in {"archived", "already_in_private_run"}
            for entry in entries
        ),
        "artifacts": entries,
    }
    _write_json_atomic(run_directory / "viewed_artifacts_manifest.json", manifest)
    return manifest


def _session_metadata(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        first_record = json.loads(stream.readline())
    if first_record.get("type") != "session_meta":
        raise ValueError("not a Codex session log")
    payload = first_record.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid Codex session metadata")
    return payload


def _git_value(source_root: Path, *arguments: str, required: bool = True) -> str | None:
    try:
        return subprocess.check_output(
            ("git", "-C", os.fspath(source_root), *arguments),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        if required:
            raise
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
