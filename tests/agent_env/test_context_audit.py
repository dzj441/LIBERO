import json
from pathlib import Path

import pytest

from libero.libero.agent_env.context_audit import (
    audit_experience_context_run,
    compare_action_trajectories,
)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_action_comparison_detects_exact_subsequence_and_adaptation():
    source = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    agent = [source[1], source[2], [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0]]
    result = compare_action_trajectories(agent, source)
    assert result["exact_source_action_fraction"] == pytest.approx(2 / 3)
    assert result["longest_exact_contiguous_match"] == 2
    assert result["shared_initial_exact_prefix"] == 0
    assert result["exact_contiguous_copy_fraction_min_4"] == 0.0
    assert result["dtw_mean_l2"] > 0


def test_action_comparison_distinguishes_long_copy_from_shared_primitives():
    source = [[float(index), 0, 0, 0, 0, 0, 1] for index in range(8)]
    agent = [
        [99.0, 0, 0, 0, 0, 0, 1],
        *source[2:7],
        [98.0, 0, 0, 0, 0, 0, 1],
    ]

    result = compare_action_trajectories(agent, source)

    assert result["longest_exact_contiguous_match"] == 5
    assert result["exact_contiguous_copy_fraction_min_4"] == pytest.approx(5 / 7)
    assert result["longest_exact_contiguous_fraction_of_agent"] == pytest.approx(
        5 / 7
    )


def test_run_audit_records_context_access_and_private_source_similarity(tmp_path):
    run = tmp_path / "run"
    source = tmp_path / "source_master"
    source_action = [0.1, 0, 0, 0, 0, 0, 1]
    _write_jsonl(
        source / "source_trajectory.jsonl",
        [
            {
                "source_action": {"normalized_vector_7d": source_action},
            }
        ],
    )
    _write_json(
        source / "p4_master_manifest.json",
        {
            "task": {"instruction": "open the drawer"},
            "capture": {
                "trajectory": {"path": "source_trajectory.jsonl"},
                "frames": [
                    {
                        "observation": "frames/frame_000000/observation.json"
                    }
                ],
            },
        },
    )
    initial_source = {
        "state": {"eef_pose_robot_base_xyzw_7d": [0, 0, 0, 0, 0, 0, 1]},
        "annotations": {
            "cameras": {
                "head": {
                    "task_entities": {
                        "entity_000": {"bbox_xyxy": [0, 0, 10, 10]}
                    }
                }
            }
        },
    }
    _write_json(
        source / "frames/frame_000000/observation.json", initial_source
    )
    _write_json(
        run / "experience_context_projection_receipt.json",
        {
            "target_task_instruction": "open the drawer",
            "experiences": [
                {
                    "experience_id": "experience_000",
                    "source_master": str(source),
                }
            ]
        },
    )
    _write_jsonl(
        run / "actions.jsonl",
        [
            {
                "recorded_at": "2026-08-30T00:00:10+00:00",
                "request": {"command": "start"},
                "response": {"ok": True},
            },
            {
                "recorded_at": "2026-08-30T00:00:20+00:00",
                "request": {"command": "osc_sequence", "actions": [source_action]},
                "response": {"ok": True},
            },
        ],
    )
    _write_jsonl(
        run / "codex_session.jsonl",
        [
            {
                "timestamp": "2026-08-30T00:00:12Z",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "CommandExecution",
                        "command": [
                            "sed",
                            "benchmark_inputs/experience_context/manifest.json",
                            "benchmark_inputs/current_observation/observation.json",
                        ],
                    },
                },
            },
            {
                "timestamp": "2026-08-30T00:00:13Z",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "ImageView",
                        "path": (
                            "file:///tmp/work/benchmark_inputs/experience_context/"
                            "experiences/experience_000/video/head_wrist_rgb.mp4"
                        ),
                    },
                },
            },
            {
                "timestamp": "2026-08-30T00:00:14Z",
                "payload": {
                    "type": "item_completed",
                    "item": {
                        "type": "AgentMessage",
                        "phase": "commentary",
                        "content": [
                            {"type": "Text", "text": "I inspected the experience."}
                        ],
                    },
                },
            },
        ],
    )
    _write_json(run / "result.json", {"status": "finished", "success": True})
    initial_query = {
        "state": {"eef_pose_robot_base_xyzw_7d": [0.03, 0.04, 0, 0, 0, 0, 1]},
        "annotations": {
            "cameras": {
                "head": {
                    "task_entities": {
                        "entity_000": {"bbox_xyxy": [2, 4, 12, 14]}
                    }
                }
            }
        },
    }
    _write_json(
        run / "private_observations/obs_000000/observation.json", initial_query
    )

    report = audit_experience_context_run(run)
    assert report["context_kind"] == "experience_context"
    assert report["context_access"]["event_count"] == 2
    assert "benchmark_inputs/experience_context/manifest.json" in report[
        "context_access"
    ]["unique_requested_paths"]
    assert report["current_observation_access_before_first_action"]["observed"]
    assert len(report["context_reference_messages"]) == 1
    comparison = report["source_action_comparisons"][0]
    assert comparison["exact_source_action_fraction"] == 1.0
    assert comparison["longest_exact_contiguous_match"] == 1
    alignment = report["initial_public_alignment"]["experiences"][0]
    assert alignment["source_task_matches_query"] is True
    assert alignment["eef_position_distance_m"] == pytest.approx(0.05)
    assert alignment["task_entity_bbox_center_deltas_px"][0][
        "query_minus_source_xy_px"
    ] == [2.0, 4.0]
