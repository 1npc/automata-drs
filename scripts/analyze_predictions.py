#!/usr/bin/env python3
"""Report strict validity and first blocked prefix events for JSONL outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import SBNPolicy, diagnose_sbn  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--prediction-field", default="predict")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--allow-cycles",
        action="store_true",
        help="mimic reference-parser recovery; strict SBN graphs are DAGs",
    )
    parser.add_argument(
        "--semantic-referee",
        action="store_true",
        help="enable the full W scope/binding policy",
    )
    parser.add_argument(
        "--configuration",
        choices=["S", "R", "G", "B", "W"],
        help="use an executable ablation policy (overrides legacy switches)",
    )
    args = parser.parse_args()

    if args.configuration and (args.allow_cycles or args.semantic_referee):
        parser.error(
            "--configuration cannot be combined with --allow-cycles or "
            "--semantic-referee"
        )
    policy = (
        SBNPolicy.for_configuration(args.configuration)
        if args.configuration
        else SBNPolicy(
            enforce_graph_acyclicity=not args.allow_cycles,
            enforce_referee_wellformedness=args.semantic_referee,
        )
    )

    rows = []
    prediction_families: Counter[str] = Counter()
    prediction_events: Counter[str] = Counter()
    label_invalid = 0
    with args.path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prediction = str(row[args.prediction_field])
            diagnostic = diagnose_sbn(prediction, policy)
            if args.label_field in row and not diagnose_sbn(
                str(row[args.label_field]), policy
            ).valid:
                label_invalid += 1
            if diagnostic.valid:
                continue
            assert diagnostic.family is not None
            assert diagnostic.event is not None
            prediction_families[diagnostic.family.value] += 1
            prediction_events[diagnostic.event] += 1
            rows.append(
                {
                    "line": line_number,
                    "family": diagnostic.family.value,
                    "event": diagnostic.event,
                    "char_offset": diagnostic.char_offset,
                    "byte_offset": diagnostic.byte_offset,
                    "blocked": diagnostic.offending_text,
                    "reason": diagnostic.reason,
                    "prefix_tail": prediction[
                        max(0, diagnostic.char_offset - 60) : diagnostic.char_offset
                    ],
                }
            )

    total = sum(1 for line in args.path.open(encoding="utf-8") if line.strip())
    report = {
        "file": str(args.path),
        "total": total,
        "strict_valid": total - len(rows),
        "strict_invalid": len(rows),
        "strict_valid_rate": (total - len(rows)) / total if total else 0.0,
        "invalid_labels": label_invalid,
        "allow_cycles": args.allow_cycles,
        "semantic_referee": args.semantic_referee,
        "configuration": args.configuration,
        "failure_families": dict(sorted(prediction_families.items())),
        "failure_events": dict(sorted(prediction_events.items())),
        "examples": rows[: args.examples],
    }
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"file: {report['file']}")
        print(f"total: {total}")
        print(f"strict valid: {report['strict_valid']}")
        print(f"strict invalid: {report['strict_invalid']}")
        print(f"strict valid rate: {report['strict_valid_rate']:.4%}")
        print(f"invalid labels: {label_invalid}")
        print(f"allow cycles: {args.allow_cycles}")
        print(f"semantic Referee: {args.semantic_referee}")
        print(f"configuration: {args.configuration or 'custom/default'}")
        print("failure families:")
        for family, count in prediction_families.most_common():
            print(f"  {count:>6}  {family}")
        print("failure events:")
        for event, count in prediction_events.most_common():
            print(f"  {count:>6}  {event}")
        if rows:
            print("examples:")
            for example in rows[: args.examples]:
                print(
                    f"  line {example['line']}: {example['family']} / "
                    f"{example['event']} blocks {example['blocked']!r} at "
                    f"byte {example['byte_offset']}"
                )
                print(f"    {example['reason']}")
                print(f"    ...{example['prefix_tail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
