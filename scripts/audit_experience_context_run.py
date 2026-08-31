#!/usr/bin/env python3
"""Write a public-evidence audit for one persisted LIBERO Agent run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero.libero.agent_env.context_audit import audit_experience_context_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_directory = args.run_directory.expanduser().resolve()
    report = audit_experience_context_run(run_directory)
    output = (args.output or run_directory / "experience_context_audit.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
