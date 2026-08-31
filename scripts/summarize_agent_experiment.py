#!/usr/bin/env python3
"""Rebuild JSON, CSV, and Markdown summaries for an Agent run matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from libero.libero.agent_env.experiments import (  # noqa: E402
    load_experiment_matrix,
    summarize_experiment_runs,
    write_experiment_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    args = parser.parse_args()
    matrix = load_experiment_matrix(args.matrix)
    summary = summarize_experiment_runs(matrix, args.batch_root)
    paths = write_experiment_summary(summary, args.batch_root)
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
