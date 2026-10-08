#!/usr/bin/env python3
"""Gate the AMR graph-boundary regression probe against frozen U outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _by_id(
    rows: list[dict[str, Any]], configuration: str
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("configuration") != configuration:
            continue
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{configuration} row has no string id")
        if identifier in selected:
            raise ValueError(f"duplicate {configuration} row: {identifier}")
        selected[identifier] = row
    return selected


def _score_f1(path: Path, configuration: str) -> float:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("gate", {}).get("passed"):
        raise ValueError(f"score gate did not pass: {path}")
    try:
        return float(
            report["groups"][configuration]["raw_primary_smatch"]["f1"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"missing {configuration} raw primary Smatch in {path}"
        ) from error


def build_report(
    *,
    baseline: Path,
    constrained: Path,
    finite_horizon: Path,
    baseline_score: Path,
    constrained_score: Path,
    finite_horizon_score: Path,
    expected_records: int,
) -> dict[str, Any]:
    u_rows = _by_id(_read_jsonl(baseline), "U")
    p_rows = _by_id(_read_jsonl(constrained), "P")
    pf_rows = _by_id(_read_jsonl(finite_horizon), "PF")
    identities_match = set(u_rows) == set(p_rows) == set(pf_rows)
    record_count_matches = (
        len(u_rows) == len(p_rows) == len(pf_rows) == expected_records
    )

    row_checks: list[dict[str, Any]] = []
    if identities_match:
        for identifier in sorted(u_rows):
            u_row, p_row, pf_row = u_rows[identifier], p_rows[identifier], pf_rows[identifier]
            p_valid = bool(p_row.get("validity", {}).get("raw_official_penman"))
            pf_valid = bool(pf_row.get("validity", {}).get("raw_official_penman"))
            p_stopped = bool(
                p_row.get("generation", {}).get(
                    "ended_with_structural_completion"
                )
            )
            pf_stopped = bool(
                pf_row.get("generation", {}).get(
                    "ended_with_structural_completion"
                )
            )
            p_hit_limit = bool(
                p_row.get("generation", {}).get("hit_max_new_tokens")
            )
            pf_hit_limit = bool(
                pf_row.get("generation", {}).get("hit_max_new_tokens")
            )
            row_checks.append(
                {
                    "id": identifier,
                    "p_raw_official_penman": p_valid,
                    "pf_raw_official_penman": pf_valid,
                    "p_structural_stop": p_stopped,
                    "pf_structural_stop": pf_stopped,
                    "p_hit_max_new_tokens": p_hit_limit,
                    "pf_hit_max_new_tokens": pf_hit_limit,
                    "p_prediction_exactly_matches_u": (
                        p_row.get("prediction") == u_row.get("prediction")
                    ),
                    "pf_prediction_exactly_matches_u": (
                        pf_row.get("prediction") == u_row.get("prediction")
                    ),
                }
            )

    u_f1 = _score_f1(baseline_score, "U")
    p_f1 = _score_f1(constrained_score, "P")
    pf_f1 = _score_f1(finite_horizon_score, "PF")
    score_tolerance = 1e-12
    row_gate = bool(row_checks) and all(
        check["p_raw_official_penman"]
        and check["pf_raw_official_penman"]
        and check["p_structural_stop"]
        and check["pf_structural_stop"]
        and not check["p_hit_max_new_tokens"]
        and not check["pf_hit_max_new_tokens"]
        and check["p_prediction_exactly_matches_u"]
        and check["pf_prediction_exactly_matches_u"]
        for check in row_checks
    )
    score_gate = (
        abs(p_f1 - u_f1) <= score_tolerance
        and abs(pf_f1 - u_f1) <= score_tolerance
    )
    passed = identities_match and record_count_matches and row_gate and score_gate
    inputs = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in {
            "baseline": baseline,
            "constrained": constrained,
            "finite_horizon": finite_horizon,
            "baseline_score": baseline_score,
            "constrained_score": constrained_score,
            "finite_horizon_score": finite_horizon_score,
        }.items()
    }
    return {
        "schema_version": 1,
        "audit": "amr-graph-boundary-regression",
        "hypothesis": (
            "accepting graph-closing BPE tokens with trailing record whitespace "
            "and stopping at the representation boundary preserves frozen U graphs"
        ),
        "expected_records": expected_records,
        "inputs": inputs,
        "scores": {"U": u_f1, "P": p_f1, "PF": pf_f1},
        "row_checks": row_checks,
        "gate": {
            "identities_match": identities_match,
            "record_count_matches": record_count_matches,
            "all_rows_valid_stopped_and_exact": row_gate,
            "smatch_exactly_preserved": score_gate,
            "passed": passed,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--finite-horizon", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--constrained-score", type=Path, required=True)
    parser.add_argument("--finite-horizon-score", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_records <= 0:
        parser.error("expected-records must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    try:
        report = build_report(
            baseline=args.baseline,
            constrained=args.constrained,
            finite_horizon=args.finite_horizon,
            baseline_score=args.baseline_score,
            constrained_score=args.constrained_score,
            finite_horizon_score=args.finite_horizon_score,
            expected_records=args.expected_records,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
