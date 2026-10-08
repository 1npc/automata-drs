#!/usr/bin/env python3
"""Summarize and pair a run_byt5_experiment.py JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.experiment import (  # noqa: E402
    summarize_experiment,
    summary_to_markdown,
)
from automata_drs.scoring import merge_semantic_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--scores", type=Path, help="semantic-score sidecar JSONL")
    args = parser.parse_args()

    with args.path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if args.scores is not None:
        with args.scores.open(encoding="utf-8") as source:
            scores = [json.loads(line) for line in source if line.strip()]
        records = merge_semantic_scores(records, scores)
    summary = summarize_experiment(records)
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(summary_to_markdown(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
