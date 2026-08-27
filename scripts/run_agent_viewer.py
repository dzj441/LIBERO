#!/usr/bin/env python3
"""Launch the read-only LIBERO Agent/Codex session viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero.agent_env.run_viewer import (  # noqa: E402
    create_server,
    describe_server_urls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve persisted LIBERO Agent runs as a read-only web UI"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "agent_runs",
        help="Directory containing LIBERO Agent runs (default: repo/agent_runs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = create_server(args.host, args.port, args.runs_root)
    actual_port = int(server.server_address[1])
    print("LIBERO Agent Session Viewer is ready.", flush=True)
    for url in describe_server_urls(args.host, actual_port):
        print(url, flush=True)
    print(f"Runs root: {args.runs_root.expanduser().resolve()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping viewer.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
