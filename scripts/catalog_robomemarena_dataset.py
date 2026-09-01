#!/usr/bin/env python3
"""Audit RoboMemArena HDF5 trajectories and build presentation videos."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np


DATASET_IDENTITY = re.compile(r"_seed(?P<seed>\d+)_task(?P<task_id>\d+)\.hdf5$")
REQUIRED_OBSERVATIONS: dict[str, tuple[str, int]] = {
    "agentview_rgb": ("uint8", 4),
    "eye_in_hand_rgb": ("uint8", 4),
    "ee_pos": ("float", 2),
    "ee_ori": ("float", 2),
    "gripper_states": ("float", 2),
    "joint_states": ("float", 2),
}

CATEGORY_BY_REPOSITORY = {
    "RoboMemArena-Multi-Object-Sequence": "multi_object_sequence",
    "RoboMemArena-Multi-Object-Occlusion": "multi_object_occlusion",
    "RoboMemArena-Multi-Object-Counting": "multi_object_counting",
    "RoboMemArena-Multi-Object-Transferring": "multi_object_transferring",
}

# Public instructions from RoboMemArena's official evaluation runner. They are
# intentionally distinct from the sometimes abbreviated HDF5 attribute.
TASK_INSTRUCTIONS = {
    1: "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
    2: "Pick and place butter into the basket, then pick and place popcorn into the same basket.",
    3: "Pick and place cream into the basket, then pick and place chocolate into the same basket.",
    4: "Open and close all drawers in order to check. Put butter into the drawer that already contains an object.",
    5: "Open and close all drawers in order to check. Put butter into the empty drawer.",
    6: "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.",
    7: "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.",
    8: "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    9: "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    10: "Pour wine into the mug twice.",
    11: "Put cookies into the top drawer and put butter into another drawer.",
    12: "Put cookies into the middle drawer and then put chocolate into the same drawer.",
    13: "Put cookies into the middle drawer and then put butter into the same drawer.",
    14: "Put cookies into the top drawer and put chocolate into another drawer.",
    15: "Pick and place butter into the frypan, then pour milk over it twice.",
    16: "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.",
    17: "Put butter into the middle drawer and then put chocolate into the same drawer.",
    18: "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.",
    19: "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2.",
    20: "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
    21: "Put butter into the microwave and then put chocolate into the location where the butter was placed.",
    22: "Pour tomato sauce over cookies twice, then put the cookies into the microwave.",
    23: "Put cream into the microwave and then put popcorn into the location where the cream was placed.",
    24: "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.",
    25: "Pick and place butter and cream from plate1 to plate2, respectively.",
    26: "Pick and place chocolate and cream from plate1 to plate2, respectively.",
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    relative_path: str
    repository: str
    category: str
    task_id: int
    seed: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories-per-task", type=int, default=3)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--overwrite-videos", action="store_true")
    return parser.parse_args()


def discover_candidates(data_root: Path) -> dict[int, list[Candidate]]:
    grouped: dict[int, list[Candidate]] = defaultdict(list)
    for repository, category in CATEGORY_BY_REPOSITORY.items():
        repository_root = data_root / repository
        if not repository_root.is_dir():
            raise FileNotFoundError(f"missing official repository: {repository_root}")
        for path in repository_root.glob("*/*/full_trajectory/*.hdf5"):
            if path.is_symlink() or path.stat().st_size <= 1_000_000:
                continue
            match = DATASET_IDENTITY.search(path.name)
            if match is None:
                continue
            task_id = int(match.group("task_id"))
            seed = int(match.group("seed"))
            grouped[task_id].append(
                Candidate(
                    path=path.resolve(),
                    relative_path=path.relative_to(data_root).as_posix(),
                    repository=repository,
                    category=category,
                    task_id=task_id,
                    seed=seed,
                )
            )
    for values in grouped.values():
        values.sort(key=lambda item: (item.seed, item.relative_path))
    return dict(grouped)


def validate_candidate(candidate: Candidate) -> dict[str, Any]:
    with h5py.File(candidate.path, "r") as handle:
        if set(handle.keys()) != {"data"} or "demo_0" not in handle["data"]:
            raise ValueError(f"{candidate.path}: expected data/demo_0")
        demo = handle["data/demo_0"]
        if "actions" not in demo or "obs" not in demo:
            raise ValueError(f"{candidate.path}: missing actions or obs")
        actions = demo["actions"]
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
            raise ValueError(f"{candidate.path}: invalid action shape {actions.shape}")
        action_values = np.asarray(actions, dtype=np.float64)
        if not np.isfinite(action_values).all():
            raise ValueError(f"{candidate.path}: non-finite action")
        max_absolute_action = float(np.max(np.abs(action_values)))
        max_absolute_motion_action = float(
            np.max(np.abs(action_values[:, :6]))
        )
        if max_absolute_motion_action > 1.0 + 1.0e-9:
            raise ValueError(
                f"{candidate.path}: normalized OSC motion exceeds one: "
                f"{max_absolute_motion_action}"
            )
        # RoboMemArena Task 10 uses +2 as an additional close command. The
        # underlying robosuite gripper channel is sign-based, so a public
        # action bundle must map this to +1 rather than rejecting the otherwise
        # valid official trajectory.
        gripper_values = action_values[:, 6]
        if np.max(np.abs(gripper_values)) > 2.0 + 1.0e-9:
            raise ValueError(
                f"{candidate.path}: unsupported gripper action values "
                f"{np.unique(gripper_values).tolist()}"
            )

        observations = demo["obs"]
        missing = sorted(set(REQUIRED_OBSERVATIONS).difference(observations))
        if missing:
            raise ValueError(f"{candidate.path}: missing observations {missing}")
        observation_shapes: dict[str, list[int]] = {}
        for name, (kind, rank) in REQUIRED_OBSERVATIONS.items():
            dataset = observations[name]
            if len(dataset) != len(actions) or dataset.ndim != rank:
                raise ValueError(
                    f"{candidate.path}: {name} is not action-aligned: "
                    f"{dataset.shape} vs {actions.shape}"
                )
            if kind == "uint8" and dataset.dtype != np.dtype(np.uint8):
                raise ValueError(f"{candidate.path}: {name} is not uint8")
            if kind == "float" and dataset.dtype.kind != "f":
                raise ValueError(f"{candidate.path}: {name} is not floating point")
            observation_shapes[name] = [int(value) for value in dataset.shape]

        for name in ("agentview_rgb", "eye_in_hand_rgb"):
            shape = observations[name].shape
            if shape[-1] != 3 or shape[1:] != observations["agentview_rgb"].shape[1:]:
                raise ValueError(f"{candidate.path}: incompatible RGB shape {shape}")

        instruction = " ".join(
            str(demo.attrs.get("language_instruction", "")).split()
        )
        if not instruction:
            raise ValueError(f"{candidate.path}: missing language_instruction")
        num_samples = int(demo.attrs.get("num_samples", len(actions)))
        if num_samples != len(actions):
            raise ValueError(
                f"{candidate.path}: num_samples={num_samples}, actions={len(actions)}"
            )

        return {
            "relative_path": candidate.relative_path,
            "repository": candidate.repository,
            "category": candidate.category,
            "task_id": candidate.task_id,
            "seed": candidate.seed,
            "bytes": candidate.path.stat().st_size,
            "frame_count": len(actions),
            "action_shape": [int(value) for value in actions.shape],
            "action_dtype": str(actions.dtype),
            "max_absolute_action": max_absolute_action,
            "max_absolute_motion_action": max_absolute_motion_action,
            "gripper_action_values": [
                float(value) for value in np.unique(gripper_values)
            ],
            "requires_gripper_sign_normalization": bool(
                np.any(np.abs(gripper_values) > 1.0)
            ),
            "recorded_instruction": instruction,
            "observation_shapes": observation_shapes,
        }


def video_filename(instruction: str) -> str:
    safe = instruction.strip().rstrip(".").replace("/", "-").replace("\x00", "")
    return f"{safe}.mp4"


def export_video(
    candidate: Candidate,
    destination: Path,
    *,
    fps: float,
    overwrite: bool,
) -> None:
    if destination.is_file() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(candidate.path, "r") as handle:
        observations = handle["data/demo_0/obs"]
        head = observations["agentview_rgb"]
        wrist = observations["eye_in_hand_rgb"]
        writer = imageio.get_writer(
            destination,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        try:
            for frame_index in range(len(head)):
                # RoboMemArena's DataRecorder already writes both RGB streams
                # in top-left image coordinates. Applying robosuite's live
                # observation flip again would turn the archived images
                # upside down.
                head_frame = np.ascontiguousarray(head[frame_index])
                wrist_frame = np.ascontiguousarray(wrist[frame_index])
                frame = np.concatenate((head_frame, wrist_frame), axis=1)
                cv2.putText(
                    frame,
                    "agentview",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "wrist",
                    (head_frame.shape[1] + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.append_data(frame)
        finally:
            writer.close()


def write_markdown(records: Iterable[dict[str, Any]], destination: Path) -> None:
    lines = [
        "# RoboMemArena task catalog",
        "",
        "Each task retains three official full trajectories. The presentation "
        "video uses the lowest available selected seed and shows agentview on "
        "the left and wrist RGB on the right.",
        "",
        "| Task | Category | Instruction | Seeds | Frames | Video |",
        "|---:|---|---|---|---:|---|",
    ]
    for record in records:
        seeds = ", ".join(str(value) for value in record["selected_seeds"])
        lines.append(
            "| {task_id} | {category} | {instruction} | {seeds} | "
            "{frames} | {video} |".format(
                task_id=record["task_id"],
                category=record["category"],
                instruction=record["instruction"],
                seeds=seeds,
                frames=record["representative_frame_count"],
                video=record["video_file"],
            )
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_arguments()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.trajectories_per_task <= 0:
        raise ValueError("--trajectories-per-task must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = discover_candidates(data_root)
    expected_tasks = set(TASK_INSTRUCTIONS)
    if set(grouped) != expected_tasks:
        raise RuntimeError(
            f"materialized task set differs: got {sorted(grouped)}, "
            f"expected {sorted(expected_tasks)}"
        )

    validated: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    for task_id in sorted(TASK_INSTRUCTIONS):
        candidates = grouped[task_id]
        if len(candidates) < args.trajectories_per_task:
            raise RuntimeError(
                f"task {task_id} has {len(candidates)} materialized trajectories; "
                f"need {args.trajectories_per_task}"
            )
        selected = candidates[: args.trajectories_per_task]
        selected_records = [validate_candidate(item) for item in selected]
        validated.extend(selected_records)

        representative = selected[0]
        instruction = TASK_INSTRUCTIONS[task_id]
        filename = video_filename(instruction)
        if not args.skip_videos:
            export_video(
                representative,
                output_dir / filename,
                fps=args.fps,
                overwrite=args.overwrite_videos,
            )
        catalog.append(
            {
                "task_id": task_id,
                "category": representative.category,
                "instruction": instruction,
                "selected_seeds": [item.seed for item in selected],
                "selected_trajectories": [
                    item.relative_path for item in selected
                ],
                "recorded_instructions": [
                    item["recorded_instruction"] for item in selected_records
                ],
                "representative_seed": representative.seed,
                "representative_trajectory": representative.relative_path,
                "representative_frame_count": selected_records[0]["frame_count"],
                "video_file": filename,
                "video_fps": args.fps,
                "video_layout": "agentview_left__wrist_right",
                "video_image_orientation": (
                    "hdf5_stored_top_left_no_additional_flip"
                ),
            }
        )

    audit = {
        "schema_version": "libero.robomemarena_dataset_audit.v1",
        "data_root": str(data_root),
        "task_count": len(catalog),
        "validated_trajectory_count": len(validated),
        "trajectories_per_task": args.trajectories_per_task,
        "all_valid": True,
        "trajectories": validated,
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "task_catalog.json").write_text(
        json.dumps(
            {
                "schema_version": "libero.robomemarena_task_catalog.v1",
                "data_root": str(data_root),
                "tasks": catalog,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_markdown(catalog, output_dir / "README.md")
    print(
        json.dumps(
            {
                "task_count": len(catalog),
                "validated_trajectory_count": len(validated),
                "output_dir": str(output_dir),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
