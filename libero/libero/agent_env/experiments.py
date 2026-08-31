"""Reproducible experiment-matrix contracts and persisted run summaries."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


EXPERIMENT_MATRIX_SCHEMA_VERSION = "libero.agent_experiment_matrix.v1"
EXPERIMENT_SUMMARY_SCHEMA_VERSION = "libero.agent_experiment_summary.v1"
RUN_MODES = frozenset({"single_episode", "curriculum"})
CONTEXT_CONDITIONS = frozenset(
    {
        "direct",
        "unrelated_support",
        "passive_demo",
        "active_support",
        "demo_assisted_active_support",
        "query_demo_upper_bound",
        "context_reset_control",
        "matched_text",
        "matched_procedural_text",
        "matched_video",
        "matched_observations",
        "matched_actions",
        "matched_full",
        "compositional_full",
        "irrelevant_full",
    }
)
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_experiment_matrix(path: str | Path) -> dict[str, Any]:
    """Load and validate one explicit, fully enumerated run matrix."""

    source = Path(path).expanduser().resolve()
    value = _read_json(source)
    if value.get("schema_version") != EXPERIMENT_MATRIX_SCHEMA_VERSION:
        raise ValueError("unsupported Agent experiment matrix schema")

    name = _identifier(value.get("name"), "matrix name")
    family = _nonempty_text(value.get("task_family"), "task_family")
    hypothesis = _nonempty_text(value.get("hypothesis"), "hypothesis")
    capability_tags = value.get("capability_tags")
    if not isinstance(capability_tags, list) or not capability_tags:
        raise ValueError("capability_tags must be a non-empty list")
    normalized_tags = [
        _identifier(item, "capability tag") for item in capability_tags
    ]
    if len(set(normalized_tags)) != len(normalized_tags):
        raise ValueError("capability_tags must be unique")

    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("experiment matrix requires at least one run")
    normalized_runs = [_normalize_run(item, index) for index, item in enumerate(runs)]
    run_ids = [item["run_id"] for item in normalized_runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_id values must be unique within one matrix")

    return {
        "schema_version": EXPERIMENT_MATRIX_SCHEMA_VERSION,
        "name": name,
        "task_family": family,
        "hypothesis": hypothesis,
        "capability_tags": normalized_tags,
        "source": source.as_posix(),
        "source_sha256": _file_sha256(source),
        "runs": normalized_runs,
    }


def summarize_experiment_runs(
    matrix: Mapping[str, Any], batch_root: str | Path
) -> dict[str, Any]:
    """Summarize every declared run without treating missing runs as failures."""

    root = Path(batch_root).expanduser().resolve()
    rows = [summarize_run(run, root / run["run_id"]) for run in matrix["runs"]]
    aggregates = []
    conditions = sorted({str(row["condition"]) for row in rows})
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        completed = [
            row
            for row in selected
            if row["status"] == "finished"
            and row["result_category"] != "infrastructure_failure"
        ]
        successes = [row for row in completed if row["query_success"] is True]
        interval = _wilson_interval(len(successes), len(completed))
        aggregates.append(
            {
                "condition": condition,
                "declared_runs": len(selected),
                "completed_runs": len(completed),
                "infrastructure_failures": sum(
                    row["result_category"] == "infrastructure_failure"
                    for row in selected
                ),
                "query_successes": len(successes),
                "query_success_rate": (
                    None if not completed else len(successes) / len(completed)
                ),
                "query_success_wilson95_low": (
                    None if interval is None else interval[0]
                ),
                "query_success_wilson95_high": (
                    None if interval is None else interval[1]
                ),
                "mean_query_agent_steps": _mean(
                    row["query_agent_steps"] for row in completed
                ),
                "mean_query_native_osc_micro_actions": _mean(
                    row["query_native_osc_micro_actions"] for row in completed
                ),
                "mean_wall_time_minutes": _mean(
                    row["wall_time_minutes"] for row in completed
                ),
                "mean_total_tokens": _mean(
                    row["total_tokens"] for row in completed
                ),
            }
        )
    return {
        "schema_version": EXPERIMENT_SUMMARY_SCHEMA_VERSION,
        "matrix_name": matrix["name"],
        "task_family": matrix["task_family"],
        "hypothesis": matrix["hypothesis"],
        "capability_tags": list(matrix["capability_tags"]),
        "matrix_source_sha256": matrix["source_sha256"],
        "batch_root": root.as_posix(),
        "rows": rows,
        "aggregates": aggregates,
        "paired_comparisons_against_direct": _paired_comparisons(rows),
    }


def summarize_run(run: Mapping[str, Any], directory: str | Path) -> dict[str, Any]:
    """Return one stable row from persisted evaluator-private artifacts."""

    root = Path(directory).expanduser().resolve()
    result = _read_json(root / "result.json", required=False)
    manifest = _read_json(root / "run_manifest.json", required=False)
    actions = _read_jsonl(root / "actions.jsonl")
    session = _read_jsonl(root / "codex_session.jsonl")
    context_audit = _read_json(
        root / "experience_context_audit.json", required=False
    )

    status = str(result.get("status") or ("not_started" if not root.exists() else "incomplete"))
    infrastructure_error = result.get("infrastructure_error")
    if infrastructure_error or status == "infrastructure_error":
        category = "infrastructure_failure"
    elif status == "finished":
        category = "success" if bool(result.get("success")) else "policy_failure"
    elif status == "aborted":
        category = "agent_or_server_aborted"
    else:
        category = status

    episode_rows = result.get("episodes") if isinstance(result.get("episodes"), list) else []
    query_steps = result.get("accepted_agent_steps")
    if episode_rows:
        query_steps = episode_rows[-1].get("accepted_agent_steps")
    episode_successes = [
        bool(episode.get("success"))
        for episode in episode_rows
        if isinstance(episode, dict) and episode.get("success") is not None
    ]
    support_successes = episode_successes[:-1] if episode_successes else []
    total_micro_actions = 0
    total_osc_sequence_calls = 0
    query_micro_actions = 0
    query_osc_sequence_calls = 0
    rejected_robot_calls = 0
    query_episode_index = (
        len(run["episodes"]) - 1 if run["mode"] == "curriculum" else 0
    )
    for event in actions:
        request = event.get("request")
        response = event.get("response")
        if not isinstance(request, dict) or request.get("command") != "osc_sequence":
            continue
        total_osc_sequence_calls += 1
        action_batch = request.get("actions")
        batch_size = len(action_batch) if isinstance(action_batch, list) else 0
        total_micro_actions += batch_size
        event_episode_index = _integer_or_none(event.get("episode_index"))
        if event_episode_index is None:
            event_episode_index = 0
        if event_episode_index == query_episode_index:
            query_osc_sequence_calls += 1
            query_micro_actions += batch_size
        if isinstance(response, dict) and response.get("ok") is False:
            rejected_robot_calls += 1

    token_usage = _last_token_usage(session)
    activity = _activity_counts(session)
    context_comparisons = context_audit.get("source_action_comparisons")
    if not isinstance(context_comparisons, list):
        context_comparisons = []
    valid_context_comparisons = [
        item for item in context_comparisons if isinstance(item, dict)
    ]
    initial_alignment = context_audit.get("initial_public_alignment")
    if not isinstance(initial_alignment, dict):
        initial_alignment = {}
    return {
        "run_id": run["run_id"],
        "replicate_id": run["replicate_id"],
        "condition": run["condition"],
        "mode": run["mode"],
        "directory": root.as_posix(),
        "status": status,
        "result_category": category,
        "query_success": (
            bool(result.get("success")) if status == "finished" else None
        ),
        "all_episodes_success": result.get("all_episodes_success"),
        "completed_episode_count": result.get(
            "completed_episode_count", 1 if status == "finished" else 0
        ),
        "episode_count": result.get("episode_count", 1),
        "episode_successes": episode_successes,
        "support_episode_count": max(0, len(episode_rows) - 1),
        "support_success_count": sum(support_successes),
        "all_support_success": (
            None if not support_successes else all(support_successes)
        ),
        "query_agent_steps": _integer_or_none(query_steps),
        "total_agent_steps": _integer_or_none(result.get("accepted_agent_steps")),
        "query_osc_sequence_calls": query_osc_sequence_calls,
        "query_native_osc_micro_actions": query_micro_actions,
        "total_osc_sequence_calls": total_osc_sequence_calls,
        "total_native_osc_micro_actions": total_micro_actions,
        "rejected_robot_calls": rejected_robot_calls,
        "wall_time_minutes": _wall_time_minutes(manifest, result),
        "input_tokens": _integer_or_none(token_usage.get("input_tokens")),
        "cached_input_tokens": _integer_or_none(
            token_usage.get("cached_input_tokens")
        ),
        "output_tokens": _integer_or_none(token_usage.get("output_tokens")),
        "reasoning_output_tokens": _integer_or_none(
            token_usage.get("reasoning_output_tokens")
        ),
        "total_tokens": _integer_or_none(token_usage.get("total_tokens")),
        "shell_command_count": activity["CommandExecution"],
        "image_view_count": activity["ImageView"],
        "agent_message_count": activity["AgentMessage"],
        "reasoning_summary_count": activity["Reasoning"],
        "expert_demo_access_events": activity["expert_demo_access"],
        "experience_context_access_events": activity[
            "experience_context_access"
        ],
        "context_current_observation_before_first_action": (
            context_audit.get("current_observation_access_before_first_action", {})
            .get("observed")
            if context_audit
            else None
        ),
        "context_source_count": (
            len(valid_context_comparisons) if context_audit else None
        ),
        "context_max_exact_copy_coverage_min_4": _max_numeric(
            item.get("exact_contiguous_copy_fraction_min_4")
            for item in valid_context_comparisons
        ),
        "context_max_longest_exact_run": _max_numeric(
            item.get("longest_exact_contiguous_match")
            for item in valid_context_comparisons
        ),
        "context_initial_alignment_available": (
            initial_alignment.get("available") if context_audit else None
        ),
        "source_commit": manifest.get("source_commit"),
        "source_worktree_dirty": manifest.get("source_worktree_dirty"),
        "infrastructure_error": infrastructure_error,
    }


def write_experiment_summary(
    summary: Mapping[str, Any], output_directory: str | Path
) -> dict[str, Path]:
    """Write lossless JSON plus spreadsheet- and report-friendly projections."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_summary.json"
    csv_path = output / "experiment_summary.csv"
    markdown_path = output / "experiment_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = list(summary["rows"])
    fieldnames = list(rows[0]) if rows else []
    buffer = io.StringIO()
    if fieldnames:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")
    markdown_path.write_text(_markdown_summary(summary), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _normalize_run(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"run {index} must be an object")
    run_id = _identifier(value.get("run_id"), f"run {index} run_id")
    replicate_id = _identifier(
        value.get("replicate_id"), f"run {index} replicate_id"
    )
    condition = _identifier(value.get("condition"), f"run {index} condition")
    if condition not in CONTEXT_CONDITIONS:
        raise ValueError(f"run {index} has unsupported context condition")
    mode = str(value.get("mode"))
    if mode not in RUN_MODES:
        raise ValueError(f"run {index} has unsupported mode")
    profile = _nonempty_text(value.get("profile", "level4"), "profile")
    normalized = {
        "run_id": run_id,
        "replicate_id": replicate_id,
        "condition": condition,
        "mode": mode,
        "profile": profile,
    }
    if mode == "single_episode":
        normalized["episode"] = _normalize_episode(value.get("episode"), index)
    else:
        episodes = value.get("episodes")
        if not isinstance(episodes, list) or len(episodes) < 2:
            raise ValueError(f"curriculum run {index} requires at least two episodes")
        normalized["episodes"] = [
            _normalize_episode(item, index) for item in episodes
        ]
        if any(
            episode["icl_condition"] == "experience_context"
            for episode in normalized["episodes"]
        ):
            raise ValueError(
                "experience_context is currently a single-episode condition"
            )
        normalized["experience_guidance"] = str(
            value.get("experience_guidance", "implicit")
        )
        if normalized["experience_guidance"] not in {"implicit", "explicit"}:
            raise ValueError(f"run {index} has invalid experience_guidance")
    return normalized


def _normalize_episode(value: Any, run_index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"run {run_index} episode must be an object")
    required = {"suite", "task_id", "init_state_id", "seed", "icl_condition"}
    if not required.issubset(value):
        raise ValueError(f"run {run_index} episode is missing required fields")
    icl = str(value["icl_condition"])
    if icl not in {"none", "fixed_demo", "experience_context"}:
        raise ValueError(f"run {run_index} episode has invalid icl_condition")
    master = value.get("fixed_demo_master")
    if (icl == "fixed_demo") != isinstance(master, str):
        raise ValueError(
            f"run {run_index} fixed_demo_master does not match icl_condition"
        )
    context_spec = value.get("experience_context_spec")
    if (icl == "experience_context") != isinstance(context_spec, str):
        raise ValueError(
            f"run {run_index} experience_context_spec does not match icl_condition"
        )
    max_steps = int(value.get("max_agent_steps", 50))
    if not 1 <= max_steps <= 100:
        raise ValueError(f"run {run_index} max_agent_steps must be in [1, 100]")
    return {
        "suite": _nonempty_text(value["suite"], "suite"),
        "task_id": int(value["task_id"]),
        "init_state_id": int(value["init_state_id"]),
        "seed": int(value["seed"]),
        "max_agent_steps": max_steps,
        "icl_condition": icl,
        "fixed_demo_master": master,
        "experience_context_spec": context_spec,
    }


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# {summary['matrix_name']} results",
        "",
        f"Task family: {summary['task_family']}",
        "",
        f"Hypothesis: {summary['hypothesis']}",
        "",
        "## Aggregate",
        "",
        "| Condition | Complete | Query success | SR | Wilson 95% CI | Mean query steps | Mean OSC micro-actions | Mean minutes | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["aggregates"]:
        rate = item["query_success_rate"]
        lines.append(
            "| {condition} | {completed_runs}/{declared_runs} | "
            "{query_successes} | {rate} | {interval} | {steps} | {micro} | {minutes} | {tokens} |".format(
                **item,
                rate="—" if rate is None else f"{rate:.3f}",
                interval=_format_interval(
                    item["query_success_wilson95_low"],
                    item["query_success_wilson95_high"],
                ),
                steps=_display(item["mean_query_agent_steps"]),
                micro=_display(item["mean_query_native_osc_micro_actions"]),
                minutes=_display(item["mean_wall_time_minutes"]),
                tokens=_display(item["mean_total_tokens"]),
            )
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Replicate | Condition | Status | Support success | Query success | Query steps | OSC micro-actions | Minutes | Tokens |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {run_id} | {replicate_id} | {condition} | {result_category} | "
            "{support} | {success} | {steps} | {micro} | {minutes} | {tokens} |".format(
                **row,
                support=_support_display(row),
                success=_display(row["query_success"]),
                steps=_display(row["query_agent_steps"]),
                micro=_display(row["query_native_osc_micro_actions"]),
                minutes=_display(row["wall_time_minutes"]),
                tokens=_display(row["total_tokens"]),
            )
        )
    comparisons = summary.get("paired_comparisons_against_direct", [])
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired comparisons against direct",
                "",
                "| Condition | Paired complete | SR delta | Improved | Degraded | Tied |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            lines.append(
                "| {condition} | {paired_complete} | {delta} | {improved} | "
                "{degraded} | {tied} |".format(
                    **comparison,
                    delta=_display(comparison["query_success_rate_delta"]),
                )
            )
    return "\n".join(lines) + "\n"


def _paired_comparisons(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    direct = {
        str(row["replicate_id"]): row
        for row in rows
        if row["condition"] == "direct"
        and row["status"] == "finished"
        and row["result_category"] != "infrastructure_failure"
    }
    if not direct:
        return []
    output = []
    conditions = sorted(
        {str(row["condition"]) for row in rows if row["condition"] != "direct"}
    )
    for condition in conditions:
        candidate = {
            str(row["replicate_id"]): row
            for row in rows
            if row["condition"] == condition
            and row["status"] == "finished"
            and row["result_category"] != "infrastructure_failure"
        }
        replicate_ids = sorted(direct.keys() & candidate.keys())
        if not replicate_ids:
            continue
        improved = degraded = tied = 0
        direct_successes = candidate_successes = 0
        for replicate_id in replicate_ids:
            direct_success = bool(direct[replicate_id]["query_success"])
            candidate_success = bool(candidate[replicate_id]["query_success"])
            direct_successes += direct_success
            candidate_successes += candidate_success
            if candidate_success and not direct_success:
                improved += 1
            elif direct_success and not candidate_success:
                degraded += 1
            else:
                tied += 1
        output.append(
            {
                "condition": condition,
                "paired_complete": len(replicate_ids),
                "query_success_rate_delta": (
                    candidate_successes - direct_successes
                )
                / len(replicate_ids),
                "improved": improved,
                "degraded": degraded,
                "tied": tied,
                "replicate_ids": replicate_ids,
            }
        )
    return output


def _last_token_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for record in records:
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
            usage = info["total_token_usage"]
    return usage


def _activity_counts(records: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "item_completed":
            continue
        item = payload.get("item")
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if isinstance(item_type, str):
            counts[item_type] += 1
        public_reference_fields = {
            key: item.get(key) for key in ("command", "path", "parsed_cmd")
        }
        if "benchmark_inputs/expert_demo" in json.dumps(
            public_reference_fields, sort_keys=True, ensure_ascii=False
        ):
            counts["expert_demo_access"] += 1
        if "benchmark_inputs/experience_context" in json.dumps(
            public_reference_fields, sort_keys=True, ensure_ascii=False
        ):
            counts["experience_context_access"] += 1
    return counts


def _wall_time_minutes(
    manifest: Mapping[str, Any], result: Mapping[str, Any]
) -> float | None:
    start = _parse_time(manifest.get("created_at"))
    end = _parse_time(result.get("launcher_finished_at") or result.get("finished_at"))
    if start is None or end is None:
        return None
    return max(0.0, (end - start).total_seconds() / 60.0)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _identifier(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if not SAFE_IDENTIFIER.fullmatch(text):
        raise ValueError(f"{label} contains unsafe characters")
    return text


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _integer_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return None if not numbers else sum(numbers) / len(numbers)


def _max_numeric(values: Iterable[Any]) -> float | int | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    return max(numbers) if numbers else None


def _wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _format_interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "—"
    return f"[{float(low):.3f}, {float(high):.3f}]"


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _support_display(row: Mapping[str, Any]) -> str:
    total = int(row.get("support_episode_count") or 0)
    if total == 0:
        return "—"
    return f"{int(row.get('support_success_count') or 0)}/{total}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
