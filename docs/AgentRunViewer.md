# LIBERO Agent Session Viewer

The read-only Viewer treats the persisted Codex session as the source of truth
for Agent activity. It aligns completed session items with the corresponding
`liberoctl` records in `actions.jsonl` and displays:

- public reasoning summaries and Agent messages;
- shell commands, exit status, stdout, and stderr;
- `ImageView` events and the exact historical image viewed at that point;
- robot requests, environment responses, and returned observations;
- state, proprioception, RGB, depth previews, initial annotations, and metric
  depth downloads;
- the evaluator-private continuous simulator video;
- benchmark task, runtime-injected user context, base instructions, and every
  developer message archived in the Codex session;
- model, reasoning effort, sandbox, approval, permission, collaboration, task
  lifecycle, completion timing, and aggregate token usage.

Hidden/raw reasoning content is never displayed. Artifacts are served through
an allowlist derived from the normalized Viewer response. Workspace files are
exposed only when the Codex session contains an explicit `ImageView` event for
that file.

The session coverage panel accounts for every persisted JSONL record as one of:

- rendered directly in the timeline;
- summarized once in the prompt/runtime/session panels;
- a structural protocol duplicate of a rendered completed item;
- deliberately hidden raw or encrypted model reasoning;
- unsupported, which produces an explicit Viewer warning instead of being
  silently discarded.

Public shell output, file-change content, Agent messages, and reasoning
summaries are not text-truncated by the Viewer. The JSONL remains the source of
truth. “Viewer complete” means that there are no unsupported public record
types and every explicit `ImageView` has a preserved image artifact. It does
not mean that hidden chain-of-thought is exposed.

## Ephemeral Agent workspace

The episode launcher now creates a randomized workspace on the system temporary
disk by default. Codex still writes its normal session to `$CODEX_HOME`; after
the process exits, the launcher copies that session into the evaluator-private
run, archives the prompt and workspace contract, and preserves files explicitly
opened through `ImageView`. The launcher leaves the inactive workspace on the
temporary disk and delegates eventual cleanup to the operating system.

Current-observation images are reconstructed from the per-step private
observation archive. Other explicitly viewed files, including expert-demo
contact sheets and Agent-created crops, are copied into `viewed_artifacts/`.
The episode remains non-resumable. Use `--keep-workspace` only when a stable,
named debug cwd is useful. This lifecycle is data hygiene, not a security sandbox: Codex
continues to run with the evaluator-selected capabilities.

## Launch

```bash
python scripts/run_agent_viewer.py \
  --host 0.0.0.0 \
  --port 8765 \
  --runs-root agent_runs
```

Open the printed code-server proxy URL or `http://127.0.0.1:8765/`.

The launcher already archives `codex_session.jsonl` and
`codex_session_metadata.json` inside each completed Agent run. The Viewer does
not read or mutate the live global Codex session store.
