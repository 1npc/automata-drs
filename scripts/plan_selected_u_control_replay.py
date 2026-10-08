#!/usr/bin/env python3
"""Freeze the batch-sensitivity repair plan exposed by the v1 pilot gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.host_selection import (  # noqa: E402
    verify_frozen_lora_host_selection,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/lora"))
    parser.add_argument(
        "--selection", type=Path, default=Path("runs/lora/selected-host.json")
    )
    parser.add_argument("--failed-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite replay plan: {args.output}")

    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes)
        selection = json.loads(selection_bytes)
        selected_key = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        pilot = config["method_experiment"]["pilot"]
        records = int(pilot["records"])
        run_root = args.runs_root / f"{selected_key}-r16-seed42"
        baseline_path = run_root / "dev-u-greedy.jsonl"
        matrix_path = run_root / f"dev{records}-u-bf-wf-greedy.jsonl"
        baseline_manifest_path = Path(str(baseline_path) + ".manifest.json")
        matrix_manifest_path = Path(str(matrix_path) + ".manifest.json")
        required = (
            args.failed_gate,
            baseline_path,
            baseline_manifest_path,
            matrix_path,
            matrix_manifest_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"missing replay-plan evidence: {missing}")
        failed_gate = read_object(args.failed_gate)
        checks = failed_gate.get("checks")
        if failed_gate.get("passed") is not False or not isinstance(checks, list):
            raise ValueError("input is not a failed v1 pilot gate")
        failed_checks = [check for check in checks if check.get("passed") is not True]
        if len(failed_checks) != 1 or failed_checks[0].get("name") != "u_surface_equivalence":
            raise ValueError(
                "v1 gate must fail only frozen-baseline surface equivalence"
            )
        observed_differences = failed_checks[0].get("observed")
        if not isinstance(observed_differences, list) or not observed_differences:
            raise ValueError("v1 gate records no batch-sensitive identifiers")

        baseline_manifest = read_object(baseline_manifest_path)
        matrix_manifest = read_object(matrix_manifest_path)
        if baseline_manifest.get("batch_size") != 8:
            raise ValueError("frozen host baseline is not the registered batch-8 run")
        if (
            matrix_manifest.get("batch_size") != 1
            or matrix_manifest.get("status") != "complete"
        ):
            raise ValueError("pilot matrix is not a complete batch-1 run")
        baseline = [
            row for row in read_rows(baseline_path)
            if row.get("configuration") == "U"
        ][:records]
        matrix_u = [
            row for row in read_rows(matrix_path)
            if row.get("configuration") == "U"
        ]
        if len(baseline) != records or len(matrix_u) != records:
            raise ValueError("replay-plan populations are incomplete")
        if [row.get("id") for row in baseline] != [row.get("id") for row in matrix_u]:
            raise ValueError("replay-plan populations differ")
        recomputed = [
            str(left["id"])
            for left, right in zip(baseline, matrix_u, strict=True)
            if left.get("raw_prediction") != right.get("raw_prediction")
        ]
        if recomputed != observed_differences:
            raise ValueError("v1 gate differences do not reproduce")
        replay_path = run_root / f"dev{records}-u-sequential-control-replay.jsonl"
        if replay_path.exists() or Path(str(replay_path) + ".manifest.json").exists():
            raise ValueError("control replay already exists before plan freeze")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    report = {
        "schema_version": 1,
        "kind": "selected-u-control-replay-plan",
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_key": selected_key,
        "trigger": (
            "The v1 integrity gate found batch-8 versus batch-1 greedy surface "
            "differences; every non-U-equivalence integrity check passed."
        ),
        "scope": (
            "Execution-control repair only. No task metric, method comparison, "
            "promotion threshold, test output, or tax output is used."
        ),
        "original_requirement": "u_surface_equivalent_to_frozen_baseline",
        "corrected_requirement": (
            "pilot U must exactly reproduce in an independent U-only batch-1 "
            "greedy run over the same ordered 400-example population"
        ),
        "observed_batch_sensitive_ids": observed_differences,
        "replay": {
            "path": str(replay_path),
            "records": records,
            "configuration": "U",
            "batch_size": 1,
            "num_beams": 1,
            "schedule": "rotate",
            "score_model_logprob": True,
            "max_new_tokens": int(pilot["max_new_tokens"]),
            "acceptance": "raw_prediction exact match to every pilot U cell",
            "performance_threshold_used": False,
        },
        "evidence": {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "failed_gate_path": str(args.failed_gate),
            "failed_gate_sha256": sha256(args.failed_gate),
            "baseline_path": str(baseline_path),
            "baseline_sha256": sha256(baseline_path),
            "baseline_manifest_sha256": sha256(baseline_manifest_path),
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256(matrix_path),
            "matrix_manifest_sha256": sha256(matrix_manifest_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"frozen U control replay plan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
