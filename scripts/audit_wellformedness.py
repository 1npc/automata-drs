#!/usr/bin/env python3
"""Audit SBN scope/binding locally and optionally against upstream Referee."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.sbn import (  # noqa: E402
    PMBRefereeOracle,
    check_sbn_state_wellformedness,
    sbn_state_to_clf,
    validate_sbn_lexemes,
)


def iter_sbn(path: Path, json_field: str):
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if path.suffix == ".jsonl":
                sbn = str(json.loads(line)[json_field])
            elif "\t" in line:
                sbn = line.split("\t", 1)[1].strip()
            else:
                sbn = line.strip()
            yield line_number, sbn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json-field", default="predict")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--referee",
        type=Path,
        help="path to upstream evaluation/clf_referee.py for cross-validation",
    )
    parser.add_argument(
        "--signature",
        type=Path,
        help="matching official CLF signature (requires PyYAML)",
    )
    args = parser.parse_args()

    if args.signature is not None and args.referee is None:
        parser.error("--signature requires --referee")
    upstream = (
        PMBRefereeOracle(args.referee, signature_path=args.signature)
        if args.referee is not None
        else None
    )

    total = 0
    structural_invalid = 0
    semantic_invalid = 0
    disagreements = 0
    reasons: Counter[str] = Counter()
    examples: list[tuple[Path, int, str, str]] = []

    for path in args.paths:
        for line_number, sbn in iter_sbn(path, args.json_field):
            total += 1
            parsed = validate_sbn_lexemes(sbn)
            if not parsed.valid:
                structural_invalid += 1
                reason = f"structural: {parsed.reason or 'unknown error'}"
                reasons[reason] += 1
                if len(examples) < args.examples:
                    examples.append((path, line_number, reason, sbn))
                continue

            local = check_sbn_state_wellformedness(parsed.state)
            if not local.valid:
                semantic_invalid += 1
                reason = f"{local.family}: {local.reason}"
                reasons[reason] += 1
                if len(examples) < args.examples:
                    examples.append((path, line_number, reason, sbn))

            if upstream is not None:
                official = upstream.check(sbn_state_to_clf(parsed.state))
                if official.valid != local.valid:
                    disagreements += 1
                    reason = (
                        f"oracle disagreement: local={local.reason!r}; "
                        f"upstream={official.reason!r}"
                    )
                    reasons[reason] += 1
                    if len(examples) < args.examples:
                        examples.append((path, line_number, reason, sbn))

    print(f"files: {', '.join(str(path) for path in args.paths)}")
    print(f"total: {total}")
    print(f"structural invalid: {structural_invalid}")
    print(f"semantic well-formed: {total - structural_invalid - semantic_invalid}")
    print(f"semantic invalid: {semantic_invalid}")
    if upstream is not None:
        print(f"upstream disagreements: {disagreements}")
    if reasons:
        print("top reasons:")
        for reason, count in reasons.most_common(10):
            print(f"  {count:>6}  {reason}")
    if examples:
        print("examples:")
        for path, line_number, reason, sbn in examples:
            print(f"  {path}:{line_number}: {reason}")
            print(f"    {sbn}")
    return 1 if structural_invalid or semantic_invalid or disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
