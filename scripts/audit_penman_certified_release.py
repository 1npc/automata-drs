#!/usr/bin/env python3
"""Audit fail-closed release of online-constrained PENMAN rows."""

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
from automata_drs.release import (  # noqa: E402
    PENMAN_RELEASE_GUARANTEE,
    assess_penman_certified_release,
    has_complete_selective_release_accounting,
)


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
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite release audit: {args.output}")

    try:
        import penman
    except ImportError as error:
        parser.error(f"independent penman package is required: {error}")
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
    if manifest.get("task") != "amr" or "P" not in manifest.get("configurations", []):
        parser.error("manifest is not an AMR matrix containing P")
    expected_examples = manifest.get("expected_examples")
    if isinstance(expected_examples, bool) or not isinstance(expected_examples, int):
        parser.error("manifest expected_examples is not an integer")
    selected = [row for row in rows if row.get("configuration") == "P"]
    if len(selected) != expected_examples:
        parser.error(f"selected {len(selected)} P rows; expected {expected_examples}")

    seen: set[str] = set()
    guarantees: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    refusals: list[dict[str, Any]] = []
    for row in selected:
        identity = str(row.get("id", ""))
        if not identity or identity in seen:
            parser.error(f"missing or duplicate P identity: {identity!r}")
        seen.add(identity)
        raw = row.get("raw_prediction")
        independent_valid = False
        if isinstance(raw, str) and raw.strip():
            try:
                penman.decode(raw)
            except Exception:
                independent_valid = False
            else:
                independent_valid = True
        decision = assess_penman_certified_release(
            row, independent_penman_valid=independent_valid
        )
        if decision.releasable:
            assert decision.guarantee is not None
            guarantees[decision.guarantee] += 1
        else:
            reasons.update(decision.reasons)
            refusals.append({"id": identity, "reasons": list(decision.reasons)})

    release_implementation = REPOSITORY_ROOT / "src" / "automata_drs" / "release.py"
    audit_implementation = Path(__file__).resolve()
    release_summary = {
        "releasable_rows": len(selected) - len(refusals),
        "refused_rows": len(refusals),
        "guarantee_counts": dict(sorted(guarantees.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "refusals": refusals,
    }
    accounting_complete = has_complete_selective_release_accounting(
        release_summary,
        attempts=len(selected),
        guarantee=PENMAN_RELEASE_GUARANTEE,
    )
    report = {
        "schema_version": 2,
        "audit": "penman-certified-release",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "predictions": str(args.predictions),
            "predictions_sha256": predictions_sha256,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "configuration": "P",
        },
        "implementation": {
            "release": str(release_implementation.relative_to(REPOSITORY_ROOT)),
            "release_sha256": sha256_file(release_implementation),
            "audit": str(audit_implementation.relative_to(REPOSITORY_ROOT)),
            "audit_sha256": sha256_file(audit_implementation),
            "independent_penman_version": penman.__version__,
        },
        "population": {
            "examples": expected_examples,
            "rows": len(selected),
            "unique_identities": len(seen),
        },
        "release": release_summary,
        "claim_boundary": (
            "local plus independent PENMAN surface membership only; "
            "not AMR task-semantic correctness or guaranteed availability"
        ),
        "gate": {
            "passed": accounting_complete,
            "checks": {
                "complete_population_accounting": accounting_complete,
                "released_rows_certified": accounting_complete,
            },
            "all_attempts_releasable": len(refusals) == 0,
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
        if comparable != stored_comparable:
            parser.error("stored release audit differs from recomputed evidence")
        print(f"verified PENMAN release audit: {args.verify_existing}")
        return 0
    assert args.output is not None
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite release audit: {args.output}")
    print(
        f"PENMAN release audit: {len(selected) - len(refusals)}/{len(selected)} releasable"
    )
    print(f"report: {args.output}")
    print(f"integrity gate passed: {accounting_complete}")
    return 0 if accounting_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
