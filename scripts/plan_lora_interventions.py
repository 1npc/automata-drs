#!/usr/bin/env python3
"""Freeze exact U-failure IDs for the conditioned U/B/W/BF/WF analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.interventions import build_intervention_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="completed experiment JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    source_bytes = args.path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    plan = build_intervention_plan(
        records,
        input_path=str(args.path),
        input_sha256=source_sha256,
    )
    if hashlib.sha256(args.path.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError(
            "experiment JSONL changed while intervention IDs were being planned"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
