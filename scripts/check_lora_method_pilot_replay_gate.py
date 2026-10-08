#!/usr/bin/env python3
"""Gate the pilot after the registered batch-1 U control replay.

The original gate is intentionally retained as failed evidence.  This v2 gate
can pass only when that gate failed *solely* because a batch-8 host-selection
baseline differed from batch-1 inference, and an independently generated
U-only batch-1 matrix exactly reproduces every pilot U surface.
"""

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
from automata_drs.lora_method_gate import (  # noqa: E402
    evaluate_lora_method_pilot,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--replay-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite v2 pilot gate: {args.output}")

    try:
        config_bytes = args.config.read_bytes()
        selection_bytes = args.selection.read_bytes()
        config = json.loads(config_bytes)
        selection = json.loads(selection_bytes)
        selected_key = verify_frozen_lora_host_selection(
            config, config_bytes, selection, args.runs_root
        )
        model_rows = [
            row for row in config["models"] if row["key"] == selected_key
        ]
        if len(model_rows) != 1:
            raise ValueError("selected host is absent or duplicated")
        current = evaluate_lora_method_pilot(
            config, model_rows[0], args.runs_root
        )
        normalized_current = json.loads(json.dumps(current, ensure_ascii=False))
        failed_checks = [
            check
            for check in current.get("checks", [])
            if check.get("passed") is not True
        ]
        if current.get("passed") is not False or len(failed_checks) != 1:
            raise ValueError("current v1 gate does not have exactly one failure")
        u_failure = failed_checks[0]
        if u_failure.get("name") != "u_surface_equivalence":
            raise ValueError("current v1 failure is not U surface equivalence")

        stored_failed = read_object(args.failed_gate)
        stored_core = {
            name: value
            for name, value in stored_failed.items()
            if name not in {"checked_at_utc", "config_sha256", "selection_sha256"}
        }
        if stored_core != normalized_current:
            raise ValueError("stored failed v1 gate differs from current evidence")
        config_digest = hashlib.sha256(config_bytes).hexdigest()
        selection_digest = hashlib.sha256(selection_bytes).hexdigest()
        if (
            stored_failed.get("config_sha256") != config_digest
            or stored_failed.get("selection_sha256") != selection_digest
        ):
            raise ValueError("stored failed v1 gate provenance mismatch")

        plan = read_object(args.plan)
        audit = read_object(args.replay_audit)
        if (
            plan.get("schema_version") != 1
            or plan.get("kind") != "selected-u-control-replay-plan"
            or plan.get("status") != "frozen"
            or plan.get("selected_key") != selected_key
        ):
            raise ValueError("unsupported U control replay plan")
        plan_evidence = plan.get("evidence", {})
        expected_plan_evidence = {
            "config_sha256": config_digest,
            "selection_sha256": selection_digest,
            "failed_gate_sha256": sha256(args.failed_gate),
            "baseline_sha256": current["evidence"]["baseline_sha256"],
            "matrix_sha256": current["evidence"]["matrix_sha256"],
            "matrix_manifest_sha256": current["evidence"][
                "matrix_manifest_sha256"
            ],
        }
        plan_mismatches = {
            name: {
                "observed": plan_evidence.get(name),
                "expected": expected,
            }
            for name, expected in expected_plan_evidence.items()
            if plan_evidence.get(name) != expected
        }
        if plan_mismatches:
            raise ValueError(f"U replay plan provenance mismatch: {plan_mismatches}")
        if plan.get("observed_batch_sensitive_ids") != u_failure.get("observed"):
            raise ValueError("U replay plan batch-sensitive identifiers changed")
        if plan.get("replay", {}).get("performance_threshold_used") is not False:
            raise ValueError("U replay plan uses a performance threshold")

        expected_records = int(config["method_experiment"]["pilot"]["records"])
        audit_evidence = audit.get("evidence", {})
        replay_path = Path(str(audit_evidence.get("replay_path", "")))
        replay_manifest_path = Path(str(replay_path) + ".manifest.json")
        audit_ok = (
            audit.get("schema_version") == 1
            and audit.get("kind") == "selected-u-control-replay-audit"
            and audit.get("status") == "complete"
            and audit.get("passed") is True
            and audit.get("records") == expected_records
            and audit.get("identity_failures") == []
            and audit.get("surface_failures") == []
            and audit.get("performance_threshold_used") is False
            and audit.get("batch_sensitive_ids") == u_failure.get("observed")
            and audit_evidence.get("plan_path") == str(args.plan)
            and audit_evidence.get("plan_sha256") == sha256(args.plan)
            and audit_evidence.get("pilot_matrix_sha256")
            == current["evidence"]["matrix_sha256"]
            and replay_path.is_file()
            and replay_manifest_path.is_file()
            and audit_evidence.get("replay_sha256") == sha256(replay_path)
            and audit_evidence.get("replay_manifest_sha256")
            == sha256(replay_manifest_path)
        )
        if not audit_ok:
            raise ValueError("U control replay audit is incomplete or drifted")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    report = {
        "schema_version": 2,
        "kind": "selected-lora-method-pilot-gate",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": selected_key,
        "passed": True,
        "protocol_correction": {
            "original_gate_passed": False,
            "only_original_failure": "u_surface_equivalence",
            "cause": "legacy batch-8 versus registered batch-1 execution-contract sensitivity",
            "corrected_control": "independent U-only batch-1 exact replay",
            "performance_threshold_used": False,
            "task_metrics_used_for_acceptance": False,
            "batch_sensitive_ids": u_failure["observed"],
        },
        "retained_v1_checks": [
            check for check in normalized_current["checks"]
            if check.get("name") != "u_surface_equivalence"
        ],
        "evidence": {
            **current["evidence"],
            "config_sha256": config_digest,
            "selection_sha256": selection_digest,
            "failed_v1_gate_path": str(args.failed_gate),
            "failed_v1_gate_sha256": sha256(args.failed_gate),
            "replay_plan_path": str(args.plan),
            "replay_plan_sha256": sha256(args.plan),
            "replay_audit_path": str(args.replay_audit),
            "replay_audit_sha256": sha256(args.replay_audit),
            "replay_sha256": audit_evidence["replay_sha256"],
            "replay_manifest_sha256": audit_evidence[
                "replay_manifest_sha256"
            ],
        },
    }
    if not all(check.get("passed") is True for check in report["retained_v1_checks"]):
        parser.error("a retained v1 integrity check did not pass")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"v2 selected-host method pilot gate passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
