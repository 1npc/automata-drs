#!/usr/bin/env python3
"""Verify the independent batch-1 U replay against every pilot U cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite U replay audit: {args.output}")
    try:
        plan = read_object(args.plan)
        if (
            plan.get("schema_version") != 1
            or plan.get("kind") != "selected-u-control-replay-plan"
            or plan.get("status") != "frozen"
        ):
            raise ValueError("unsupported or unfrozen U replay plan")
        evidence = plan["evidence"]
        hash_paths = {
            "config_sha256": Path("configs/lora_baselines.json"),
            "selection_sha256": Path("runs/lora/selected-host.json"),
            "failed_gate_sha256": Path(evidence["failed_gate_path"]),
            "baseline_sha256": Path(evidence["baseline_path"]),
            "baseline_manifest_sha256": Path(
                str(evidence["baseline_path"]) + ".manifest.json"
            ),
            "matrix_sha256": Path(evidence["matrix_path"]),
            "matrix_manifest_sha256": Path(
                str(evidence["matrix_path"]) + ".manifest.json"
            ),
        }
        drift = {}
        for name, path in hash_paths.items():
            observed = sha256(path) if path.is_file() else None
            if observed != evidence[name]:
                drift[name] = {
                    "observed": observed,
                    "expected": evidence[name],
                }
        if drift:
            raise ValueError(f"U replay plan evidence drifted: {drift}")
        replay_contract = plan["replay"]
        replay_path = Path(replay_contract["path"])
        replay_manifest_path = Path(str(replay_path) + ".manifest.json")
        replay_manifest = read_object(replay_manifest_path)
        records = int(replay_contract["records"])
        expected_manifest = {
            "status": "complete",
            "batch_size": 1,
            "configurations": ["U"],
            "num_beams": 1,
            "schedule": "rotate",
            "score_model_logprob": True,
            "expected_examples": records,
            "expected_rows": records,
            "examples": records,
            "rows": records,
            "output_sha256": sha256(replay_path),
        }
        mismatches = {
            name: {"observed": replay_manifest.get(name), "expected": expected}
            for name, expected in expected_manifest.items()
            if replay_manifest.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"U replay manifest mismatch: {mismatches}")
        pilot_rows = [
            row for row in read_rows(Path(evidence["matrix_path"]))
            if row.get("configuration") == "U"
        ]
        replay_rows = read_rows(replay_path)
        if len(pilot_rows) != records or len(replay_rows) != records:
            raise ValueError("U replay or pilot population is incomplete")
        identity_failures = [
            index + 1
            for index, (pilot, replay) in enumerate(
                zip(pilot_rows, replay_rows, strict=True)
            )
            if pilot.get("id") != replay.get("id")
            or replay.get("configuration") != "U"
        ]
        surface_failures = [
            str(pilot.get("id"))
            for pilot, replay in zip(pilot_rows, replay_rows, strict=True)
            if pilot.get("raw_prediction") != replay.get("raw_prediction")
        ]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    report = {
        "schema_version": 1,
        "kind": "selected-u-control-replay-audit",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not identity_failures and not surface_failures,
        "records": records,
        "identity_failures": identity_failures,
        "surface_failures": surface_failures,
        "batch_sensitive_ids": plan["observed_batch_sensitive_ids"],
        "performance_threshold_used": False,
        "evidence": {
            "plan_path": str(args.plan),
            "plan_sha256": sha256(args.plan),
            "pilot_matrix_sha256": evidence["matrix_sha256"],
            "replay_path": str(replay_path),
            "replay_sha256": sha256(replay_path),
            "replay_manifest_sha256": sha256(replay_manifest_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
