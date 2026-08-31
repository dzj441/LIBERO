import json
from pathlib import Path

import pytest

from libero.libero.agent_env.experiments import (
    load_experiment_matrix,
    summarize_experiment_runs,
    write_experiment_summary,
)
from scripts.run_agent_experiment_matrix import _freeze_matrix, build_launch_command


def _matrix_value():
    query = {
        "suite": "libero_goal",
        "task_id": 3,
        "init_state_id": 2,
        "seed": 11,
        "max_agent_steps": 100,
        "icl_condition": "none",
    }
    return {
        "schema_version": "libero.agent_experiment_matrix.v1",
        "name": "drawer_test",
        "task_family": "drawer",
        "hypothesis": "Support experience improves the query.",
        "capability_tags": ["causal_composition", "cross_episode_transfer"],
        "runs": [
            {
                "run_id": "drawer_direct_r0",
                "replicate_id": "r0",
                "condition": "direct",
                "mode": "single_episode",
                "profile": "level4",
                "episode": query,
            },
            {
                "run_id": "drawer_active_r0",
                "replicate_id": "r0",
                "condition": "active_support",
                "mode": "curriculum",
                "profile": "level4",
                "episodes": [
                    {
                        "suite": "libero_90",
                        "task_id": 7,
                        "init_state_id": 1,
                        "seed": 7,
                        "max_agent_steps": 50,
                        "icl_condition": "none",
                    },
                    query,
                ],
                "experience_guidance": "implicit",
            },
        ],
    }


def test_matrix_loads_explicit_runs_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "matrix.json"
    value = _matrix_value()
    path.write_text(json.dumps(value), encoding="utf-8")
    matrix = load_experiment_matrix(path)
    assert matrix["name"] == "drawer_test"
    assert matrix["runs"][0]["episode"]["max_agent_steps"] == 100
    assert len(matrix["runs"][1]["episodes"]) == 2

    value["runs"][1]["run_id"] = value["runs"][0]["run_id"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="run_id values must be unique"):
        load_experiment_matrix(path)


def test_summary_separates_query_and_total_curriculum_actions(tmp_path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_matrix_value()), encoding="utf-8")
    matrix = load_experiment_matrix(matrix_path)
    run = tmp_path / "runs" / "drawer_active_r0"
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "source_commit": "abc",
                "source_worktree_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "finished",
                "success": True,
                "all_episodes_success": True,
                "completed_episode_count": 2,
                "episode_count": 2,
                "accepted_agent_steps": 5,
                "episodes": [
                    {"accepted_agent_steps": 2, "success": True},
                    {"accepted_agent_steps": 3, "success": True},
                ],
                "launcher_finished_at": "2026-01-01T00:10:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "episode_index": 0,
            "request": {"command": "osc_sequence", "actions": [[0] * 7] * 4},
            "response": {"ok": True},
        },
        {
            "episode_index": 1,
            "request": {"command": "osc_sequence", "actions": [[0] * 7] * 6},
            "response": {"ok": True},
        },
    ]
    (run / "actions.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    session_events = [
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {
                    "type": "CommandExecution",
                    "command": ["sed", "benchmark_inputs/expert_demo/manifest.json"],
                },
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "item_completed",
                "item": {"type": "ImageView", "path": "head/rgb.png"},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 40,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    }
                },
            },
        },
    ]
    (run / "codex_session.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in session_events),
        encoding="utf-8",
    )
    (run / "experience_context_audit.json").write_text(
        json.dumps(
            {
                "current_observation_access_before_first_action": {
                    "observed": True
                },
                "initial_public_alignment": {"available": True},
                "source_action_comparisons": [
                    {
                        "exact_contiguous_copy_fraction_min_4": 0.25,
                        "longest_exact_contiguous_match": 12,
                    },
                    {
                        "exact_contiguous_copy_fraction_min_4": 0.1,
                        "longest_exact_contiguous_match": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_experiment_runs(matrix, tmp_path / "runs")
    row = next(item for item in summary["rows"] if item["condition"] == "active_support")
    assert row["query_agent_steps"] == 3
    assert row["total_agent_steps"] == 5
    assert row["episode_successes"] == [True, True]
    assert row["support_success_count"] == 1
    assert row["all_support_success"] is True
    assert row["query_native_osc_micro_actions"] == 6
    assert row["total_native_osc_micro_actions"] == 10
    assert row["wall_time_minutes"] == 10.0
    assert row["total_tokens"] == 120
    assert row["shell_command_count"] == 1
    assert row["image_view_count"] == 1
    assert row["expert_demo_access_events"] == 1
    assert row["experience_context_access_events"] == 0
    assert row["context_current_observation_before_first_action"] is True
    assert row["context_source_count"] == 2
    assert row["context_max_exact_copy_coverage_min_4"] == 0.25
    assert row["context_max_longest_exact_run"] == 12
    assert row["context_initial_alignment_available"] is True
    aggregate = next(
        item for item in summary["aggregates"] if item["condition"] == "active_support"
    )
    assert aggregate["query_success_wilson95_low"] < 1.0
    assert aggregate["query_success_wilson95_high"] == pytest.approx(1.0)
    assert summary["paired_comparisons_against_direct"] == []
    paths = write_experiment_summary(summary, tmp_path / "summary")
    assert all(path.is_file() for path in paths.values())


def test_matrix_launch_command_materializes_curriculum_plan(tmp_path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_matrix_value()), encoding="utf-8")
    matrix = load_experiment_matrix(matrix_path)
    run = matrix["runs"][1]
    command = build_launch_command(
        run,
        batch_root=tmp_path / "batch",
        launcher_root=Path(__file__).resolve().parents[2],
        artifact_root=tmp_path / "artifacts",
        render_gpu_device_id=2,
        resolution=256,
        initial_settle_control_steps=10,
        codex_bin="codex",
        codex_model="gpt-5.6-sol",
        codex_effort="high",
        https_proxy="http://127.0.0.1:7890",
    )
    plan = tmp_path / "batch" / "experiment_plans" / "drawer_active_r0.json"
    assert plan.is_file()
    assert "launch_agent_curriculum.py" in command[1]
    assert command[command.index("--render-gpu-device-id") + 1] == "2"
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    assert plan_value["episodes"][0]["fixed_demo_master"] is None


def test_resolved_matrix_is_immutable_within_a_batch(tmp_path):
    path = tmp_path / "experiment_matrix_resolved.json"
    matrix = {"schema_version": "test", "runs": []}
    _freeze_matrix(path, matrix)
    _freeze_matrix(path, matrix)
    with pytest.raises(ValueError, match="different resolved experiment matrix"):
        _freeze_matrix(path, {**matrix, "runs": [{"run_id": "changed"}]})


def test_matrix_supports_single_episode_experience_context(tmp_path):
    value = _matrix_value()
    value["runs"] = [
        {
            "run_id": "drawer_context_r0",
            "replicate_id": "r0",
            "condition": "matched_video",
            "mode": "single_episode",
            "profile": "level4",
            "episode": {
                "suite": "libero_goal",
                "task_id": 3,
                "init_state_id": 22,
                "seed": 11,
                "max_agent_steps": 100,
                "icl_condition": "experience_context",
                "experience_context_spec": (
                    "configs/agent_contexts/drawer_matched_video.json"
                ),
            },
        }
    ]
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(value), encoding="utf-8")
    matrix = load_experiment_matrix(matrix_path)
    command = build_launch_command(
        matrix["runs"][0],
        batch_root=tmp_path / "batch",
        launcher_root=Path(__file__).resolve().parents[2],
        artifact_root=Path(__file__).resolve().parents[2],
        render_gpu_device_id=0,
        resolution=256,
        initial_settle_control_steps=10,
        codex_bin="codex",
        codex_model="gpt-5.6-sol",
        codex_effort="high",
        https_proxy="http://127.0.0.1:7890",
    )
    assert command[command.index("--icl") + 1] == "experience_context"
    assert "--experience-context-spec" in command
