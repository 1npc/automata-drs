#!/usr/bin/env python3
"""Create or verify the immutable DRS suite-v2 to suite-v3 promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_EVALUATIONS = {
    "dev400",
    "dev400-fine",
    "dev400-beam4",
    "dev1132",
    "test1132",
    "tax117",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_report(
    v2_json: Path,
    v2_markdown: Path,
    v3_json: Path,
    v3_markdown: Path,
) -> dict[str, Any]:
    for path in (v2_json, v2_markdown, v3_json, v3_markdown):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe DRS suite artifact: {path}")
    v2 = _read(v2_json)
    v3 = _read(v3_json)
    v2_selection = v2.get("host_selection", {})
    v3_selection = v3.get("host_selection", {})
    selected_key = v3_selection.get("selected_key")
    selected_models = [
        row
        for row in v3.get("models", [])
        if isinstance(row, dict) and row.get("key") == selected_key
    ]
    baselines = (
        selected_models[0].get("method_baselines", [])
        if len(selected_models) == 1
        else []
    )
    evidence_hashes = v3.get("evidence", {}).get("input_sha256", {})
    observed_evaluations: set[str] = set()
    baseline_rows_valid = (
        isinstance(baselines, list)
        and len(baselines) == len(EXPECTED_EVALUATIONS)
        and isinstance(evidence_hashes, dict)
    )
    for row in baselines:
        if not isinstance(row, dict):
            baseline_rows_valid = False
            continue
        evaluation = row.get("evaluation")
        expected_strategy = (
            "beam4" if evaluation == "dev400-beam4" else "greedy"
        )
        path = row.get("path")
        row_valid = (
            isinstance(evaluation, str)
            and evaluation in EXPECTED_EVALUATIONS
            and evaluation not in observed_evaluations
            and row.get("configuration") == "U"
            and row.get("strategy") == expected_strategy
            and isinstance(row.get("group"), dict)
            and bool(row.get("group"))
            and isinstance(path, str)
            and _is_sha256(evidence_hashes.get(path))
        )
        baseline_rows_valid = baseline_rows_valid and row_valid
        if isinstance(evaluation, str):
            observed_evaluations.add(evaluation)
    baseline_rows_valid = (
        baseline_rows_valid and observed_evaluations == EXPECTED_EVALUATIONS
    )
    v2_config = v2.get("evidence", {}).get("config_sha256")
    v3_config = v3.get("evidence", {}).get("config_sha256")
    checks = {
        "source_is_legacy_full_suite": (
            v2.get("schema_version") == 6 and v2.get("scope") == "full"
        ),
        "target_is_current_full_suite": (
            v3.get("schema_version") == 7 and v3.get("scope") == "full"
        ),
        "selected_host_unchanged": (
            v2_selection.get("status") == "selected"
            and v3_selection.get("status") == "selected"
            and v2_selection.get("selected_key") == selected_key
            and isinstance(selected_key, str)
            and bool(selected_key)
        ),
        "frozen_config_unchanged": (
            _is_sha256(v2_config) and v2_config == v3_config
        ),
        "all_evaluation_specific_u_rows": baseline_rows_valid,
        "target_has_hash_bound_inputs": (
            bool(evidence_hashes)
            and all(_is_sha256(value) for value in evidence_hashes.values())
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("DRS suite-v3 promotion checks failed: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "audit": "drs-suite-v3-promotion",
        "source_v2": {
            "json": str(v2_json),
            "json_sha256": _sha256(v2_json),
            "markdown": str(v2_markdown),
            "markdown_sha256": _sha256(v2_markdown),
            "role": "immutable legacy collector output; never overwritten",
            "recomputed_under_schema_v7": False,
        },
        "target_v3": {
            "json": str(v3_json),
            "json_sha256": _sha256(v3_json),
            "markdown": str(v3_markdown),
            "markdown_sha256": _sha256(v3_markdown),
            "role": "independent schema-v7 recollection from frozen run evidence",
            "selected_key": selected_key,
            "evaluation_specific_u_rows": sorted(observed_evaluations),
        },
        "checks": checks,
        "claim_boundary": (
            "schema promotion only; no generation, scoring, statistical, "
            "release, or performance definition changed"
        ),
        "gate": {"passed": True, "performance_threshold_used": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2-json",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v2.json"),
    )
    parser.add_argument(
        "--v2-markdown",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v2.md"),
    )
    parser.add_argument(
        "--v3-json",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v3.json"),
    )
    parser.add_argument(
        "--v3-markdown",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v3.md"),
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(
            args.v2_json,
            args.v2_markdown,
            args.v3_json,
            args.v3_markdown,
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        try:
            stored = _read(args.verify_existing)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if stored != report:
            parser.error("stored DRS suite-v3 promotion gate differs from current evidence")
        print(f"verified DRS suite-v3 promotion: {args.verify_existing}")
        return 0
    assert args.output is not None
    if args.output.exists():
        parser.error(f"refusing to overwrite promotion gate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as destination_file:
            json.dump(report, destination_file, ensure_ascii=False, indent=2)
            destination_file.write("\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite promotion gate: {args.output}")
    print(f"DRS suite-v3 promotion gate: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
