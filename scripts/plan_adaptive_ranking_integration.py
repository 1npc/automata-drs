#!/usr/bin/env python3
"""Freeze a pressure-plus-control population for real-logit ranking audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def line_number(row: dict[str, Any]) -> int:
    identifier = row.get("id")
    if not isinstance(identifier, str) or ":" not in identifier:
        raise ValueError("baseline row has no line-number identifier")
    try:
        value = int(identifier.rsplit(":", 1)[1])
    except ValueError as error:
        raise ValueError(f"invalid baseline identifier: {identifier}") from error
    if value <= 0:
        raise ValueError(f"invalid baseline line number: {value}")
    return value


def evenly_spaced_controls(
    candidates: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("control count must be positive")
    if len(candidates) < count:
        raise ValueError("not enough eligible controls")
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row["generation"]["output_tokens_including_eos"]),
            line_number(row),
        ),
    )
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise AssertionError("length-stratified control indices are not unique")
    return [ordered[index] for index in indices]


def build_plan(
    baseline_path: Path,
    failure_path: Path,
    data_path: Path,
    *,
    controls: int,
) -> dict[str, Any]:
    rows = read_jsonl(baseline_path)
    failures = read_object(failure_path)
    baseline_hash = sha256_file(baseline_path)
    if failures.get("input_sha256") != baseline_hash:
        raise ValueError("failure selection is not bound to the baseline JSONL")
    if any(row.get("configuration") != "U" for row in rows):
        raise ValueError("baseline must contain only U rows")
    row_by_line: dict[int, dict[str, Any]] = {}
    for row in rows:
        number = line_number(row)
        if number in row_by_line:
            raise ValueError(f"duplicate baseline line number: {number}")
        generation = row.get("generation")
        if (
            not isinstance(generation, dict)
            or isinstance(generation.get("output_tokens_including_eos"), bool)
            or not isinstance(generation.get("output_tokens_including_eos"), int)
            or generation["output_tokens_including_eos"] <= 0
        ):
            raise ValueError(f"invalid generation length at line {number}")
        row_by_line[number] = row

    selected_failures = failures.get("selected")
    if not isinstance(selected_failures, list) or not selected_failures:
        raise ValueError("failure selection has no selected rows")
    failure_lines = [
        int(item["line_number"])
        for item in selected_failures
        if isinstance(item, dict) and "line_number" in item
    ]
    if len(failure_lines) != len(selected_failures):
        raise ValueError("failure selection contains malformed rows")
    if len(set(failure_lines)) != len(failure_lines):
        raise ValueError("failure selection contains duplicate line numbers")
    missing = sorted(set(failure_lines) - set(row_by_line))
    if missing:
        raise ValueError(f"failure rows are absent from baseline: {missing}")
    for number in failure_lines:
        validity = row_by_line[number].get("validity")
        if not isinstance(validity, dict) or (
            validity.get("raw_sbn") is True
            and validity.get("raw_clf_referee") is not False
        ):
            raise ValueError(f"line {number} is not a recorded U failure")

    failure_set = set(failure_lines)
    control_candidates = [
        row
        for number, row in row_by_line.items()
        if number not in failure_set
        and isinstance(row.get("validity"), dict)
        and row["validity"].get("raw_sbn") is True
        and row["validity"].get("raw_clf_referee") is True
        and row.get("generation", {}).get("ended_with_eos") is True
        and row.get("generation", {}).get("hit_max_new_tokens") is False
    ]
    control_rows = evenly_spaced_controls(control_candidates, controls)
    control_lines = {line_number(row) for row in control_rows}

    selected: list[dict[str, Any]] = []
    for number in sorted(failure_set | control_lines):
        row = row_by_line[number]
        selected.append(
            {
                "id": row["id"],
                "line_number": number,
                "group": (
                    "u_failure_pressure"
                    if number in failure_set
                    else "length_stratified_valid_control"
                ),
                "u_output_tokens_including_eos": row["generation"][
                    "output_tokens_including_eos"
                ],
                "u_raw_sbn": row["validity"].get("raw_sbn"),
                "u_raw_clf_referee": row["validity"].get(
                    "raw_clf_referee"
                ),
            }
        )

    return {
        "schema_version": 1,
        "kind": "adaptive-ranking-real-logit-audit-selection",
        "baseline": str(baseline_path),
        "baseline_sha256": baseline_hash,
        "failure_selection": str(failure_path),
        "failure_selection_sha256": sha256_file(failure_path),
        "data": str(data_path),
        "data_sha256": sha256_file(data_path),
        "selection_rule": {
            "pressure": "all frozen U failures",
            "controls": (
                "U raw-SBN+Referee-valid completed rows at evenly spaced ranks "
                "after sorting by (U output tokens including EOS, line number)"
            ),
            "control_count": controls,
            "randomness_used": False,
        },
        "registered_audit": {
            "configurations": ["BF", "WF"],
            "decode_cells": [
                {"strategy": "greedy", "num_beams": 1, "retained": 1},
                {"strategy": "beam", "num_beams": 4, "retained": 8},
            ],
            "max_new_tokens": 256,
            "completion_budget_factor": 1.0,
            "adaptive_initial_k": 32,
            "same_logits_three_way_shadow": True,
            "returned_generation_path": "frozen_unstable_full_sort",
            "stable_adaptive_mismatches_required": 0,
            "frozen_stable_mismatches_required_for_drop_in_claim": 0,
            "performance_threshold_used": False,
        },
        "counts": {
            "baseline_rows": len(rows),
            "u_failure_pressure": len(failure_lines),
            "length_stratified_valid_controls": len(control_rows),
            "selected": len(selected),
            "generation_cells": len(selected) * 2 * 2,
        },
        "selected": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--failure-selection", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.controls <= 0:
        parser.error("controls must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite audit selection: {args.output}")
    try:
        plan = build_plan(
            args.baseline,
            args.failure_selection,
            args.data,
            controls=args.controls,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(plan, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        print(
            "froze adaptive-ranking real-logit audit selection: "
            f"{plan['counts']['selected']} rows / "
            f"{plan['counts']['generation_cells']} cells -> {args.output}"
        )
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
        raise AssertionError("unreachable") from error


if __name__ == "__main__":
    raise SystemExit(main())
