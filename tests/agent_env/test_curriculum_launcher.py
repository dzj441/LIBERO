import json
from pathlib import Path

import pytest

from scripts.launch_agent_curriculum import (
    _enrich_episodes,
    build_curriculum_prompt,
    load_curriculum_plan,
)


def test_curriculum_prompt_discloses_workflow_but_not_future_tasks():
    prompt = build_curriculum_prompt(
        episode_count=3,
        fixed_demo_possible=True,
    )
    assert "Complete 3 prepared LIBERO episodes" in prompt
    assert "task_instruction" in prompt
    assert "next_episode_available=true" in prompt
    assert "curriculum_complete=true" in prompt
    assert "benchmark_inputs/expert_demo/" in prompt
    assert "open the top drawer" not in prompt
    assert "put the bowl" not in prompt


def test_curriculum_plan_loads_without_exposing_task_instructions(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "libero.agent_curriculum_plan.v1",
                "name": "test plan",
                "profile": "level4",
                "episodes": [
                    {
                        "suite": "libero_90",
                        "task_id": 7,
                        "init_state_id": 0,
                        "seed": 1,
                        "icl_condition": "none",
                    },
                    {
                        "suite": "libero_goal",
                        "task_id": 3,
                        "init_state_id": 0,
                        "seed": 2,
                        "icl_condition": "none",
                    },
                ],
                "primary_metric_episode_index": 1,
            }
        ),
        encoding="utf-8",
    )
    plan = load_curriculum_plan(path, source_root=tmp_path)
    assert plan["name"] == "test plan"
    assert plan["primary_metric_episode_index"] == 1
    assert all("task_instruction" not in episode for episode in plan["episodes"])


def test_curriculum_episode_enrichment_resolves_tasks_and_rejects_bad_icl(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "scripts.launch_agent_curriculum._task_instruction",
        lambda suite, task_id: f"{suite} task {task_id}",
    )
    episodes = _enrich_episodes(
        [
            {
                "suite": "suite_a",
                "task_id": 1,
                "init_state_id": 2,
                "seed": 3,
                "icl_condition": "none",
            }
        ],
        source_root=tmp_path,
    )
    assert episodes[0]["task_instruction"] == "suite_a task 1"
    assert episodes[0]["fixed_demo_master"] is None

    with pytest.raises(ValueError, match="unsupported curriculum ICL"):
        _enrich_episodes(
            [
                {
                    "suite": "suite_a",
                    "task_id": 1,
                    "init_state_id": 2,
                    "seed": 3,
                    "icl_condition": "future_oracle",
                }
            ],
            source_root=tmp_path,
        )
