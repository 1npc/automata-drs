#!/usr/bin/env python3
"""Freeze or re-verify the two post-primary CUDA ranking cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.ranking_gate import (  # noqa: E402
    evaluate_adaptive_ranking_benchmarks,
    verify_frozen_adaptive_ranking_benchmarks,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--beam", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.greedy == args.beam:
        parser.error("greedy and beam reports must be distinct")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite ranking gate: {args.output}")
    try:
        report = evaluate_adaptive_ranking_benchmarks(
            (args.greedy, args.beam), repository_root=REPOSITORY_ROOT
        )
        if report["passed"] is not True:
            failed = [
                item["name"] for item in report["checks"] if not item["passed"]
            ]
            raise ValueError("ranking benchmark checks failed: " + ", ".join(failed))
        if args.verify_existing is not None:
            stored = json.loads(
                args.verify_existing.read_text(encoding="utf-8")
            )
            if not isinstance(stored, dict):
                raise ValueError("stored ranking gate is not a JSON object")
            verify_frozen_adaptive_ranking_benchmarks(report, stored)
            print(f"verified adaptive-ranking benchmark gate: {args.verify_existing}")
            return 0
        frozen = dict(report)
        frozen["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination_file:
            json.dump(frozen, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
        print(f"adaptive-ranking benchmark gate passed: {args.output}")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        raise AssertionError("unreachable") from error


if __name__ == "__main__":
    raise SystemExit(main())
