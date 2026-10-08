#!/usr/bin/env python3
"""Audit flat SBN strings in TSV/plain-text or JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.sbn import validate_sbn_lexemes  # noqa: E402


def iter_sbn(path: Path, json_field: str):
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if path.suffix == ".jsonl":
                yield line_number, str(json.loads(line)[json_field])
            elif "\t" in line:
                yield line_number, line.split("\t", 1)[1].strip()
            else:
                yield line_number, line.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--json-field", default="predict")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    total = 0
    invalid = 0
    reasons: Counter[str] = Counter()
    examples: list[tuple[int, str, str]] = []
    for line_number, sbn in iter_sbn(args.path, args.json_field):
        total += 1
        result = validate_sbn_lexemes(sbn)
        if result.valid:
            continue
        invalid += 1
        reason = result.reason or "unknown error"
        reasons[reason] += 1
        if len(examples) < args.examples:
            examples.append((line_number, reason, sbn))

    print(f"file: {args.path}")
    print(f"total: {total}")
    print(f"valid: {total - invalid}")
    print(f"invalid: {invalid}")
    if reasons:
        print("top reasons:")
        for reason, count in reasons.most_common(10):
            print(f"  {count:>6}  {reason}")
    if examples:
        print("examples:")
        for line_number, reason, sbn in examples:
            print(f"  line {line_number}: {reason}")
            print(f"    {sbn}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
