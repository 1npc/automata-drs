#!/usr/bin/env python3
"""Audit a versioned PENMAN corpus against the incremental constraint state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.penman import audit_penman_corpus, sha256_file  # noqa: E402


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw and oracle-normalized PENMAN corpus strings with the "
            "incremental automata-drs validator."
        )
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-sha256")
    parser.add_argument("--source-repository-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--require-official-rate", type=float)
    parser.add_argument("--require-raw-coverage", type=float)
    args = parser.parse_args()

    if args.examples < 0:
        parser.error("--examples must be non-negative")
    for name in ("require_official_rate", "require_raw_coverage"):
        value = getattr(args, name)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")

    observed_sha256 = sha256_file(args.corpus)
    if args.expect_sha256 and observed_sha256 != args.expect_sha256:
        parser.error(
            "corpus SHA256 mismatch: "
            f"expected {args.expect_sha256}, observed {observed_sha256}"
        )

    try:
        import penman
    except ImportError as error:
        parser.error(f"the independent penman oracle is unavailable: {error}")

    report = audit_penman_corpus(
        args.corpus,
        decode=penman.decode,
        encode=penman.encode,
        oracle_name="penman",
        oracle_version=penman.__version__,
        source={
            "repository_url": args.source_repository_url,
            "revision": args.source_revision,
            "repository_file": args.source_file,
            "license": args.source_license,
        },
        example_limit=args.examples,
    )
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["implementation"] = {
        "state_sha256": sha256_file(
            REPOSITORY_ROOT / "src" / "automata_drs" / "penman" / "state.py"
        ),
        "audit_sha256": sha256_file(
            REPOSITORY_ROOT / "src" / "automata_drs" / "penman" / "audit.py"
        ),
    }

    gates = {
        "source_sha256_matches": (
            args.expect_sha256 is None or observed_sha256 == args.expect_sha256
        ),
        "official_rate_requirement": (
            args.require_official_rate is None
            or report["official_decode"]["acceptance_rate"]
            >= args.require_official_rate
        ),
        "raw_coverage_requirement": (
            args.require_raw_coverage is None
            or report["local_raw_on_official"]["acceptance_rate"]
            >= args.require_raw_coverage
        ),
    }
    report["gate"] = {
        "passed": all(gates.values()),
        "checks": gates,
        "required_official_rate": args.require_official_rate,
        "required_raw_coverage": args.require_raw_coverage,
    }
    _write_json_exclusive(args.output, report)

    print(f"report: {args.output}")
    print(f"records: {report['corpus']['records']}")
    print(
        "official decode: "
        f"{report['official_decode']['accepted']}/"
        f"{report['official_decode']['denominator']}"
    )
    print(
        "local raw coverage: "
        f"{report['local_raw_on_official']['accepted']}/"
        f"{report['local_raw_on_official']['denominator']}"
    )
    print(
        "local normalized coverage: "
        f"{report['local_normalized_on_official']['accepted']}/"
        f"{report['local_normalized_on_official']['denominator']}"
    )
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
