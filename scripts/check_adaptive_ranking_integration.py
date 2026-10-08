#!/usr/bin/env python3
"""Freeze or reverify the real-logit adaptive-ranking integration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.ranking_integration_gate import (  # noqa: E402
    evaluate_adaptive_ranking_integration,
    freeze_adaptive_ranking_integration,
    verify_frozen_adaptive_ranking_integration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, required=True)
    parser.add_argument("--beam", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite integration gate: {args.output}")
    try:
        report = evaluate_adaptive_ranking_integration(
            (args.greedy, args.beam),
            selection_path=args.selection,
            repository_root=REPOSITORY_ROOT,
        )
        if report["passed"] is not True:
            failed = [item["name"] for item in report["checks"] if not item["passed"]]
            raise ValueError("integration checks failed: " + ", ".join(failed))
        if args.verify_existing is not None:
            stored = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("stored integration gate is not a JSON object")
            verify_frozen_adaptive_ranking_integration(report, stored)
            print(f"verified adaptive-ranking integration gate: {args.verify_existing}")
            return 0
        assert args.output is not None
        frozen = freeze_adaptive_ranking_integration(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination_file:
            json.dump(frozen, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
        print(f"adaptive-ranking integration gate passed: {args.output}")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        raise AssertionError("unreachable") from error


if __name__ == "__main__":
    raise SystemExit(main())
