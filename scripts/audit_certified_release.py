#!/usr/bin/env python3
"""Audit whether completed SBN rows pass the fail-closed release boundary."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.release import assess_sbn_certified_release  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(value)
    return rows


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument(
        "--configurations", nargs="+", default=("BF", "WF"), choices=("B", "W", "BF", "WF")
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite release audit: {args.output}")
    configurations = tuple(args.configurations)
    if len(set(configurations)) != len(configurations):
        parser.error("configurations must be unique")

    manifest_path = Path(str(args.predictions) + ".manifest.json")
    try:
        manifest = _read_object(manifest_path)
        rows = _read_rows(args.predictions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    predictions_sha256 = sha256_file(args.predictions)
    if manifest.get("status") != "complete":
        parser.error("generation manifest is not complete")
    if manifest.get("output_sha256") != predictions_sha256:
        parser.error("generation output hash differs from complete manifest")
    if not set(configurations).issubset(set(manifest.get("configurations", []))):
        parser.error("requested release configurations are absent from manifest")
    completion_budget_factor = manifest.get("completion_budget_factor", 1.0)
    if isinstance(completion_budget_factor, bool) or not isinstance(
        completion_budget_factor, (int, float)
    ):
        parser.error("manifest completion budget factor is not numeric")

    selected = [
        row for row in rows if str(row.get("configuration", "")) in configurations
    ]
    expected_examples = manifest.get("expected_examples")
    if isinstance(expected_examples, bool) or not isinstance(expected_examples, int):
        parser.error("manifest expected_examples is not an integer")
    expected_rows = expected_examples * len(configurations)
    if len(selected) != expected_rows:
        parser.error(f"selected {len(selected)} rows; expected {expected_rows}")

    seen: set[tuple[str, str]] = set()
    guarantee_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for row in selected:
        identity = (str(row.get("id", "")), str(row.get("configuration", "")))
        if not identity[0] or identity in seen:
            parser.error(f"missing or duplicate release identity: {identity}")
        seen.add(identity)
        decision = assess_sbn_certified_release(
            row, completion_budget_factor=float(completion_budget_factor)
        )
        if decision.releasable:
            assert decision.guarantee is not None
            guarantee_counts[decision.guarantee] += 1
            continue
        reason_counts.update(decision.reasons)
        failures.append(
            {
                "id": identity[0],
                "configuration": identity[1],
                "reasons": list(decision.reasons),
            }
        )

    passed = not failures
    release_implementation = REPOSITORY_ROOT / "src" / "automata_drs" / "release.py"
    audit_implementation = Path(__file__).resolve()
    report = {
        "schema_version": 2,
        "audit": "sbn-certified-release",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "predictions": str(args.predictions),
            "predictions_sha256": predictions_sha256,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "configurations": list(configurations),
            "completion_budget_factor": completion_budget_factor,
        },
        "implementation": {
            "release": str(release_implementation.relative_to(REPOSITORY_ROOT)),
            "release_sha256": sha256_file(release_implementation),
            "audit": str(audit_implementation.relative_to(REPOSITORY_ROOT)),
            "audit_sha256": sha256_file(audit_implementation),
        },
        "population": {
            "examples": expected_examples,
            "rows": len(selected),
            "unique_identities": len(seen),
        },
        "release": {
            "releasable_rows": len(selected) - len(failures),
            "refused_rows": len(failures),
            "guarantee_counts": dict(sorted(guarantee_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "failures": failures,
        },
        "claim_boundary": (
            "policy-relative structural/well-formedness release only; "
            "not task-semantic correctness"
        ),
        "gate": {
            "passed": passed,
            "performance_threshold_used": False,
        },
    }
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        comparable = dict(report)
        stored_comparable = dict(stored)
        comparable.pop("created_at_utc", None)
        stored_comparable.pop("created_at_utc", None)
        if stored_comparable != comparable:
            parser.error("stored release audit differs from recomputed evidence")
        print(f"verified release audit: {args.verify_existing}")
        return 0
    assert args.output is not None
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite release audit: {args.output}")
    print(f"release audit: {len(selected) - len(failures)}/{len(selected)} releasable")
    print(f"report: {args.output}")
    print(f"gate passed: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
