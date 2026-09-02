"""Materialize one public observation as ordinary agent-readable files."""

from __future__ import annotations

import colorsys
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .annotation_contract import validate_task_entity_mapping
from .profiles import validate_task_reference


def write_public_observation(
    observation: Mapping[str, Any], output_directory: str | Path
) -> Path:
    """Write a projected frame and return its ``observation.json`` path.

    The input must already be public. This writer knows only anonymous task
    entities and cannot serialize raw segmentation IDs or private names.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = deepcopy(dict(observation))

    if "task_reference" in metadata:
        task_reference = metadata["task_reference"]
        validate_task_reference(task_reference)
        reference_rgb = np.array(task_reference["rgb"], copy=True)
        reference_directory = output_directory / "task_reference"
        reference_directory.mkdir(parents=True, exist_ok=True)
        reference_path = reference_directory / "rgb.png"
        Image.fromarray(reference_rgb).save(reference_path)
        metadata["task_reference"] = {
            "semantics": task_reference["semantics"],
            "rgb": {
                "file": str(reference_path.relative_to(output_directory)),
                "media_type": "image/png",
                "shape": list(reference_rgb.shape),
            },
        }

    for camera_name in ("head", "wrist"):
        camera_directory = output_directory / camera_name
        camera_directory.mkdir(parents=True, exist_ok=True)
        camera = metadata["cameras"][camera_name]

        rgb = np.asarray(camera.pop("rgb"), dtype=np.uint8)
        rgb_path = camera_directory / "rgb.png"
        Image.fromarray(rgb).save(rgb_path)
        camera["rgb"] = {
            "file": str(rgb_path.relative_to(output_directory)),
            "media_type": "image/png",
            "shape": list(rgb.shape),
        }

        if "depth_m" in camera:
            depth = np.asarray(camera.pop("depth_m"), dtype=np.float32)
            valid = np.asarray(camera.pop("depth_valid_mask"), dtype=np.bool_)
            depth_path = camera_directory / "depth_m.npy"
            valid_path = camera_directory / "depth_valid_mask.png"
            preview_path = camera_directory / "depth_visualization.png"
            np.save(depth_path, depth, allow_pickle=False)
            Image.fromarray(valid.astype(np.uint8) * 255).save(valid_path)
            Image.fromarray(_depth_preview(depth, valid)).save(preview_path)
            camera["depth"] = {
                "metric_file": str(depth_path.relative_to(output_directory)),
                "preview_file": str(preview_path.relative_to(output_directory)),
                "valid_mask_file": str(valid_path.relative_to(output_directory)),
                "dtype": "float32",
                "shape": list(depth.shape),
                "unit": "metre",
            }

    if "annotations" in metadata:
        annotations_root = output_directory / "annotations"
        for camera_name in ("head", "wrist"):
            camera_annotations = metadata["annotations"]["cameras"][camera_name]
            task_entities = camera_annotations["task_entities"]
            entity_ids = validate_task_entity_mapping(task_entities)
            camera_directory = annotations_root / camera_name
            camera_directory.mkdir(parents=True, exist_ok=True)
            rgb = np.asarray(
                observation["cameras"][camera_name]["rgb"], dtype=np.uint8
            )
            overlay = Image.fromarray(rgb).convert("RGBA")
            for index, entity_id in enumerate(entity_ids):
                annotation = task_entities[entity_id]
                mask = np.asarray(annotation.pop("mask"), dtype=np.bool_)
                mask_path = camera_directory / f"{entity_id}_mask.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                annotation["mask_file"] = str(mask_path.relative_to(output_directory))
                overlay = _add_annotation_overlay(
                    overlay,
                    mask,
                    annotation["bbox_xyxy"],
                    _task_entity_color(index),
                )
            overlay_path = camera_directory / "annotations_overlay.png"
            overlay.convert("RGB").save(overlay_path)
            camera_annotations["overlay_file"] = str(
                overlay_path.relative_to(output_directory)
            )

    json_path = output_directory / "observation.json"
    json_path.write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return json_path


def replace_current_public_observation(
    observation: Mapping[str, Any], current_directory: str | Path
) -> Path:
    """Replace a current-only directory from a fully written staging tree."""

    current_directory = Path(current_directory)
    if not current_directory.name or current_directory == current_directory.parent:
        raise ValueError("current_directory must name a dedicated child directory")
    if current_directory.is_symlink():
        raise ValueError("current_directory must not be a symbolic link")
    parent = current_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".{current_directory.name}.next-", dir=parent)
    )
    backup_directory: Path | None = None
    try:
        write_public_observation(observation, temporary_directory)
        if current_directory.exists():
            backup_directory = Path(
                tempfile.mkdtemp(
                    prefix=f".{current_directory.name}.previous-", dir=parent
                )
            )
            backup_directory.rmdir()
            os.replace(current_directory, backup_directory)
        os.replace(temporary_directory, current_directory)
        if backup_directory is not None:
            shutil.rmtree(backup_directory)
        return current_directory / "observation.json"
    except Exception:
        if not current_directory.exists() and backup_directory is not None:
            os.replace(backup_directory, current_directory)
            backup_directory = None
        raise
    finally:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        if backup_directory is not None and backup_directory.exists():
            shutil.rmtree(backup_directory)


def _depth_preview(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    preview = np.zeros(depth.shape, dtype=np.uint8)
    values = depth[valid]
    if values.size == 0:
        return preview
    lower, upper = np.percentile(values, [2.0, 98.0])
    if upper <= lower:
        upper = lower + 1.0e-6
    normalized = np.clip((depth - lower) / (upper - lower), 0.0, 1.0)
    # Near is bright so foreground geometry remains easy to inspect.
    preview[valid] = np.round((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return preview


def _task_entity_color(index: int) -> tuple[int, int, int]:
    """Generate a stable, role-neutral overlay color for an anonymous entity."""

    hue = (0.03 + index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return tuple(int(round(channel * 255.0)) for channel in (red, green, blue))


def _add_annotation_overlay(
    image: Image.Image,
    mask: np.ndarray,
    bbox_xyxy: list[int] | None,
    color: tuple[int, int, int],
) -> Image.Image:
    tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
    tint[mask, :3] = color
    tint[mask, 3] = 80
    image = Image.alpha_composite(image, Image.fromarray(tint, mode="RGBA"))
    if bbox_xyxy is not None:
        x_min, y_min, x_max, y_max = bbox_xyxy
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x_min, y_min, max(x_min, x_max - 1), max(y_min, y_max - 1)],
            outline=color + (255,),
            width=2,
        )
    return image


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
