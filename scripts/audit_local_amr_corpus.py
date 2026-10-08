#!/usr/bin/env python3
"""Audit paired local AMR data without confusing sentences with graph blocks."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_data import iter_amr_records  # noqa: E402
from automata_drs.penman import sha256_file, validate_penman  # noqa: E402


def _safe_reason(error: BaseException) -> str:
    first_line = str(error).splitlines()[0].strip()
    return (first_line or type(error).__name__)[:240]


def _coverage(total: int, accepted: int, reasons: Counter[str], examples: list) -> dict:
    return {
        "denominator": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_rate": accepted / total if total else None,
        "reasons": dict(sorted(reasons.items())),
        "examples": examples,
    }


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sentence/PENMAN pairing and graph acceptance in local AMR data."
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--expect-records", type=int, required=True)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--require-official-rate", type=float, default=1.0)
    parser.add_argument("--require-local-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.expect_records < 0:
        parser.error("--expect-records must be non-negative")
    if args.examples < 0:
        parser.error("--examples must be non-negative")
    for name in ("require_official_rate", "require_local_rate"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")

    observed_sha256 = sha256_file(args.corpus)
    if observed_sha256 != args.expect_sha256:
        parser.error(
            "corpus SHA256 mismatch: "
            f"expected {args.expect_sha256}, observed {observed_sha256}"
        )

    try:
        import penman
    except ImportError as error:
        parser.error(f"the independent penman oracle is unavailable: {error}")

    total = 0
    sentence_parenthesis_prefix = 0
    official_accepted = 0
    local_accepted = 0
    local_on_official_accepted = 0
    official_reasons: Counter[str] = Counter()
    local_reasons: Counter[str] = Counter()
    official_examples: list[dict] = []
    local_examples: list[dict] = []

    with args.corpus.open(encoding="utf-8") as source:
        for record in iter_amr_records(source):
            total += 1
            if record.sentence.lstrip().startswith("("):
                sentence_parenthesis_prefix += 1

            official_ok = False
            try:
                penman.decode(record.graph)
                official_ok = True
                official_accepted += 1
            except Exception as error:
                reason = _safe_reason(error)
                official_reasons[reason] += 1
                if len(official_examples) < args.examples:
                    official_examples.append(
                        {
                            "ordinal": record.ordinal,
                            "graph_start_line": record.graph_start_line,
                            "reason": reason,
                        }
                    )

            result = validate_penman(record.graph)
            if result.valid:
                local_accepted += 1
                if official_ok:
                    local_on_official_accepted += 1
            else:
                reason = result.reason or "unknown local rejection"
                local_reasons[reason] += 1
                if len(local_examples) < args.examples:
                    local_examples.append(
                        {
                            "ordinal": record.ordinal,
                            "graph_start_line": record.graph_start_line,
                            "reason": reason,
                        }
                    )

    official = _coverage(
        total, official_accepted, official_reasons, official_examples
    )
    local = _coverage(total, local_accepted, local_reasons, local_examples)
    local_on_official = {
        "denominator": official_accepted,
        "accepted": local_on_official_accepted,
        "rejected": official_accepted - local_on_official_accepted,
        "acceptance_rate": (
            local_on_official_accepted / official_accepted
            if official_accepted
            else None
        ),
    }
    checks = {
        "source_sha256_matches": observed_sha256 == args.expect_sha256,
        "record_count_matches": total == args.expect_records,
        "official_rate_requirement": (
            official["acceptance_rate"] is not None
            and official["acceptance_rate"] >= args.require_official_rate
        ),
        "local_rate_requirement": (
            local["acceptance_rate"] is not None
            and local["acceptance_rate"] >= args.require_local_rate
        ),
    }
    report = {
        "schema_version": 1,
        "audit": "local-paired-amr-corpus",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "hoshuhan/amr-3-parsed",
            "underlying_corpus": "AMR 3.0 (LDC2020T02)",
            "redistribution_status": "not-cleared",
        },
        "corpus": {
            "path": str(args.corpus),
            "file_name": args.corpus.name,
            "size_bytes": args.corpus.stat().st_size,
            "sha256": observed_sha256,
            "records": total,
            "sentence_parenthesis_prefix_records": sentence_parenthesis_prefix,
        },
        "oracle": {"name": "penman", "version": penman.__version__},
        "official_decode": official,
        "local_raw": local,
        "local_raw_on_official": local_on_official,
        "implementation": {
            "reader_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "amr_data.py"
            ),
            "state_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "automata_drs" / "penman" / "state.py"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "required_official_rate": args.require_official_rate,
            "required_local_rate": args.require_local_rate,
        },
    }
    _write_json_exclusive(args.output, report)

    print(f"report: {args.output}")
    print(f"records: {total}")
    print(f"official decode: {official_accepted}/{total}")
    print(f"local raw coverage: {local_accepted}/{total}")
    print(f"sentence parenthesis prefixes: {sentence_parenthesis_prefix}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
