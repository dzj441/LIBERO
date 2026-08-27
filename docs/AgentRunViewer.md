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
- task, base, and developer instructions archived in the Codex session.

Hidden/raw reasoning content is never displayed. Artifacts are served through
an allowlist derived from the normalized Viewer response. Workspace files are
exposed only when the Codex session contains an explicit `ImageView` event for
that file.

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
