"""Optional RoboMemArena task source for the agent benchmark.

The external checkout supplies simulation assets and BDDL only.  Public
observations, control transport, audit logs, and success reporting remain owned
by this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from libero.libero.envs import OffScreenRenderEnv

from .control import OSCControlConfig
from .environment import LiberoAgentEnv
from .observation import TaskEntitySelection
from .profiles import ObservationProfile


ROBOMEMARENA_SUITE = "robomemarena"
ROBOMEMARENA_SOURCE_SCHEMA_VERSION = "libero.robomemarena_source.v1"


@dataclass(frozen=True)
class RoboMemArenaTaskSpec:
    task_id: int
    instruction: str
    bddl_relative_path: str
    required_stage_names: tuple[str, ...]
    optional_stage_names: tuple[str, ...] = ()


TASK_SPECS = {
    4: RoboMemArenaTaskSpec(
        task_id=4,
        instruction=(
            "Open and close all drawers in order to check. Put butter into "
            "the drawer that already contains an object."
        ),
        bddl_relative_path="evaluation_benchmark/bddl/4_drawer_butter.bddl",
        required_stage_names=(
            "01_open_top_drawer",
            "02_close_top_drawer",
            "03_open_middle_drawer",
            "04_close_middle_drawer",
            "05_open_bottom_drawer",
            "06_close_bottom_drawer",
            "07_open_top_drawer_again",
            "08_put_butter_in_top_drawer",
        ),
        optional_stage_names=("09_close_top_drawer_final",),
    )
}


def get_robomemarena_task_spec(task_id: int) -> RoboMemArenaTaskSpec:
    try:
        return TASK_SPECS[int(task_id)]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(TASK_SPECS))
        raise ValueError(
            f"unsupported RoboMemArena task_id {task_id}; supported: {supported}"
        ) from exc


def robomemarena_source_fingerprint(
    checkout_root: str | os.PathLike[str], *, task_id: int
) -> dict[str, Any]:
    """Validate and fingerprint the evaluator-private external task source."""

    root = Path(checkout_root).expanduser().resolve()
    spec = get_robomemarena_task_spec(task_id)
    required_files = {
        "bddl_sha256": root / spec.bddl_relative_path,
        "cabinet_asset_sha256": (
            root
            / "evaluation_benchmark/libero_fork/libero/assets/"
            "articulated_objects/wooden_cabinet_tall_bottom.xml"
        ),
        "stage_reference_sha256": (
            root
            / "evaluation_benchmark/scripts/task2_26_reference_stage.py"
        ),
    }
    missing = [str(path) for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "RoboMemArena checkout is incomplete: " + ", ".join(missing)
        )
    commit = _git_output(root, "rev-parse", "HEAD")
    tracked_status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "RoboMemArena checkout has tracked modifications; use a clean, "
            "versioned task source"
        )
    return {
        "schema_version": ROBOMEMARENA_SOURCE_SCHEMA_VERSION,
        "task_id": spec.task_id,
        "source_commit": commit,
        **{
            label: _file_sha256(path)
            for label, path in required_files.items()
        },
    }


class RoboMemArenaTask4Evaluator:
    """Evaluator-private ordered-stage monitor for RoboMemArena Task 4."""

    OPEN_THRESHOLD_M = 0.10
    CLOSED_THRESHOLD_M = 0.08
    BUTTER_HORIZONTAL_THRESHOLD_M = 0.25
    BUTTER_HEIGHT_THRESHOLD_M = 0.15

    def __init__(self, env: Any) -> None:
        self.env = env
        self.spec = get_robomemarena_task_spec(4)
        self._initial_sites: dict[str, np.ndarray] = {}
        self._completed: list[str] = []
        self._events: list[dict[str, Any]] = []
        self._control_step = 0
        self._active = False
        self._checks: tuple[tuple[str, Callable[[], bool]], ...] = (
            (self.spec.required_stage_names[0], lambda: self._drawer_open("top")),
            (self.spec.required_stage_names[1], lambda: self._drawer_closed("top")),
            (
                self.spec.required_stage_names[2],
                lambda: self._drawer_open("middle"),
            ),
            (
                self.spec.required_stage_names[3],
                lambda: self._drawer_closed("middle"),
            ),
            (
                self.spec.required_stage_names[4],
                lambda: self._drawer_open("bottom"),
            ),
            (
                self.spec.required_stage_names[5],
                lambda: self._drawer_closed("bottom"),
            ),
            (self.spec.required_stage_names[6], lambda: self._drawer_open("top")),
            (self.spec.required_stage_names[7], self._butter_in_top_drawer),
            (self.spec.optional_stage_names[0], lambda: self._drawer_closed("top")),
        )

    def reset(self) -> None:
        self._initial_sites = {
            drawer: self._site_position(
                f"wooden_cabinet_1_{drawer}_region"
            )
            for drawer in ("top", "middle", "bottom")
        }
        self._completed = []
        self._events = []
        self._control_step = 0
        self._active = True

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        if not self._active:
            return
        self._control_step += 1
        stage_index = len(self._completed)
        if stage_index >= len(self._checks):
            return
        name, check = self._checks[stage_index]
        if check():
            self._completed.append(name)
            self._events.append(
                {
                    "stage_index": stage_index,
                    "stage_name": name,
                    "control_step": self._control_step,
                    "sim_time_s": float(self.env.sim.data.time),
                }
            )

    def result(self) -> dict[str, Any]:
        required = self.spec.required_stage_names
        required_completed = sum(name in self._completed for name in required)
        success = required_completed == len(required)
        return {
            "schema_version": "libero.robomemarena_private_evaluation.v1",
            "task_id": self.spec.task_id,
            "success": success,
            "required_stage_count": len(required),
            "completed_required_stage_count": required_completed,
            "stage_score_percent": (
                100.0 * required_completed / len(required)
            ),
            "ordered_stage_names": [
                *required,
                *self.spec.optional_stage_names,
            ],
            "completed_stage_names": list(self._completed),
            "optional_stage_names": list(self.spec.optional_stage_names),
            "control_steps_observed": self._control_step,
            "stage_events": list(self._events),
        }

    def _drawer_open(self, drawer: str) -> bool:
        return self._drawer_displacement(drawer) > self.OPEN_THRESHOLD_M

    def _drawer_closed(self, drawer: str) -> bool:
        return self._drawer_displacement(drawer) < self.CLOSED_THRESHOLD_M

    def _drawer_displacement(self, drawer: str) -> float:
        current = self._site_position(
            f"wooden_cabinet_1_{drawer}_region"
        )
        initial = self._initial_sites[drawer]
        return abs(float(current[1] - initial[1]))

    def _butter_in_top_drawer(self) -> bool:
        butter = self._body_position("butter_1")
        region = self._site_position("wooden_cabinet_1_top_region")
        horizontal = float(np.linalg.norm(butter[:2] - region[:2]))
        height = abs(float(butter[2] - region[2]))
        return (
            horizontal < self.BUTTER_HORIZONTAL_THRESHOLD_M
            and height < self.BUTTER_HEIGHT_THRESHOLD_M
        )

    def _site_position(self, name: str) -> np.ndarray:
        for candidate in (name, f"{name}_main"):
            try:
                site_id = self.env.sim.model.site_name2id(candidate)
            except Exception:
                continue
            return np.asarray(
                self.env.sim.data.site_xpos[site_id], dtype=np.float64
            ).copy()
        raise RuntimeError(f"private checker could not resolve site {name!r}")

    def _body_position(self, name: str) -> np.ndarray:
        for candidate in (name, f"{name}_main"):
            try:
                body_id = self.env.sim.model.body_name2id(candidate)
            except Exception:
                continue
            return np.asarray(
                self.env.sim.data.body_xpos[body_id], dtype=np.float64
            ).copy()
        raise RuntimeError(f"private checker could not resolve body {name!r}")


def make_robomemarena_agent_env(
    *,
    checkout_root: str | os.PathLike[str],
    task_id: int = 4,
    init_state_id: int = 0,
    profile: ObservationProfile | int | str = ObservationProfile.LEVEL4,
    seed: int = 0,
    camera_height: int = 256,
    camera_width: int = 256,
    task_entities: TaskEntitySelection | None = None,
    control_config: OSCControlConfig | None = None,
    initial_settle_control_steps: int = 10,
    max_agent_steps: int | None = None,
    private_control_step_callback: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    render_gpu_device_id: int = -1,
    **env_kwargs: Any,
) -> LiberoAgentEnv:
    """Create one task using RoboMemArena physics and our public contract."""

    spec = get_robomemarena_task_spec(task_id)
    if init_state_id != 0:
        raise ValueError(
            "RoboMemArena adapter currently supports deterministic reset "
            "init_state_id 0 only"
        )
    root = Path(checkout_root).expanduser().resolve()
    robomemarena_source_fingerprint(root, task_id=task_id)
    bddl_path = root / spec.bddl_relative_path
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
        "control_freq",
    }
    conflicts = reserved.intersection(env_kwargs)
    if conflicts:
        raise ValueError(
            "factory-managed env kwargs cannot be overridden: "
            f"{sorted(conflicts)}"
        )
    env = OffScreenRenderEnv(
        bddl_file_name=os.fspath(bddl_path),
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=camera_height,
        camera_widths=camera_width,
        camera_depths=True,
        camera_segmentations="instance",
        # RoboMemArena's fork currently assumes these private sensors exist.
        # They stay inside the server; MasterObservationCollector allowlists the
        # public state and never serializes object pose/relative-pose fields.
        use_object_obs=True,
        ignore_done=True,
        initialization_noise=None,
        render_gpu_device_id=render_gpu_device_id,
        horizon=10000,
        control_freq=20,
        **env_kwargs,
    )
    env.seed(seed)
    evaluator = RoboMemArenaTask4Evaluator(env)
    return LiberoAgentEnv(
        env,
        profile=profile,
        camera_height=camera_height,
        camera_width=camera_width,
        task_instruction=spec.instruction,
        initial_state=None,
        task_entities=task_entities,
        control_config=control_config,
        initial_settle_control_steps=initial_settle_control_steps,
        max_agent_steps=max_agent_steps,
        private_control_step_callback=private_control_step_callback,
        private_episode_evaluator=evaluator,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", os.fspath(root), *arguments),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
