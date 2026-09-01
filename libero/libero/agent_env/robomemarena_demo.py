"""Host-private loading for RoboMemArena full-trajectory demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np


_DATASET_NAME = re.compile(r"_seed(?P<seed>\d+)_task(?P<task_id>\d+)\.hdf5$")
_REQUIRED_OBSERVATIONS = {
    "agentview_rgb": ("uint8", 4),
    "eye_in_hand_rgb": ("uint8", 4),
    "ee_pos": ("float", 2),
    "ee_ori": ("float", 2),
    "gripper_states": ("float", 2),
    "joint_states": ("float", 2),
}


@dataclass(frozen=True)
class RoboMemArenaFullTrajectory:
    dataset_path: Path
    demo_key: str
    task_id: int
    seed: int
    actions: np.ndarray
    recorded_instruction: str
    observation_count: int


def load_robomemarena_full_trajectory(
    dataset_path: str | Path, *, expected_task_id: int
) -> RoboMemArenaFullTrajectory:
    """Validate one ModelScope full trajectory without trusting its outcome."""

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"RoboMemArena HDF5 is missing or unsafe: {path}")
    match = _DATASET_NAME.search(path.name)
    if match is None:
        raise ValueError(f"RoboMemArena HDF5 name has no seed/task identity: {path.name}")
    task_id = int(match.group("task_id"))
    seed = int(match.group("seed"))
    if task_id != int(expected_task_id):
        raise ValueError(
            f"RoboMemArena HDF5 task {task_id} differs from expected task "
            f"{expected_task_id}"
        )

    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"data"} or "demo_0" not in handle["data"]:
            raise ValueError("RoboMemArena HDF5 must contain data/demo_0")
        demo = handle["data/demo_0"]
        if "actions" not in demo or "obs" not in demo:
            raise ValueError("RoboMemArena HDF5 lacks actions or observations")
        actions = np.asarray(demo["actions"], dtype=np.float64)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or len(actions) == 0
            or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1.0 + 1.0e-9)
        ):
            raise ValueError(
                "RoboMemArena actions must be finite normalized OSC vectors "
                f"with shape (T, 7), got {actions.shape}"
            )
        observations = demo["obs"]
        missing = sorted(set(_REQUIRED_OBSERVATIONS).difference(observations))
        if missing:
            raise ValueError(
                "RoboMemArena HDF5 lacks required observations: " + ", ".join(missing)
            )
        for name, (kind, rank) in _REQUIRED_OBSERVATIONS.items():
            dataset = observations[name]
            if len(dataset) != len(actions) or dataset.ndim != rank:
                raise ValueError(
                    f"RoboMemArena observation {name} is not aligned to actions"
                )
            if kind == "uint8" and dataset.dtype != np.dtype(np.uint8):
                raise ValueError(f"RoboMemArena observation {name} must be uint8")
            if kind == "float" and dataset.dtype.kind != "f":
                raise ValueError(f"RoboMemArena observation {name} must be floating point")
        recorded_instruction = " ".join(
            str(demo.attrs.get("language_instruction", "")).split()
        )
        if not recorded_instruction:
            raise ValueError("RoboMemArena HDF5 has no language_instruction")

    return RoboMemArenaFullTrajectory(
        dataset_path=path,
        demo_key="demo_0",
        task_id=task_id,
        seed=seed,
        actions=actions,
        recorded_instruction=recorded_instruction,
        observation_count=len(actions),
    )
