"""Frozen execution-control contract for publication-scale method matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


KIND = "selected-publication-u-control-plan"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_outputs(
    config: dict[str, Any], model_key: str, runs_root: Path
) -> list[str]:
    run_root = runs_root / f"{model_key}-r16-seed42"
    values = []
    for row in config["method_experiment"]["publication_matrices"]:
        split = str(row["split"])
        records = int(row["records"])
        stem = (
            "tax-u-bf-wf-greedy"
            if split == "tax"
            else f"{split}{records}-u-bf-wf-greedy"
        )
        values.append(str(run_root / f"{stem}.jsonl"))
    return values


def verify_publication_control_plan(
    plan_path: Path,
    *,
    config: dict[str, Any],
    config_sha256: str,
    selection_sha256: str,
    model_key: str,
    runs_root: Path,
    pilot_gate_path: Path,
) -> dict[str, Any]:
    """Fail closed if the pre-output publication control plan drifted."""

    value = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("publication U-control plan must be a JSON object")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("kind") != KIND
        or value.get("status") != "frozen"
        or value.get("selected_key") != model_key
        or not isinstance(value.get("created_at_utc"), str)
    ):
        raise ValueError("unsupported or unfrozen publication U-control plan")
    expected_control = {
        "host_selection_u_role": "quality-only legacy batch-8 screen",
        "method_u_role": "within-matrix batch-1 greedy control",
        "method_u_logits_processor": None,
        "cross_contract_surface_identity_required": False,
        "pilot_batch1_exact_replay_required": True,
        "paired_population_required": True,
        "task_metric_threshold_used": False,
    }
    if value.get("control_contract") != expected_control:
        raise ValueError("publication U-control contract changed")
    expected_outputs = planned_outputs(config, model_key, runs_root)
    if value.get("planned_outputs") != expected_outputs:
        raise ValueError("publication U-control output population changed")
    evidence = value.get("evidence", {})
    expected_evidence = {
        "config_sha256": config_sha256,
        "selection_sha256": selection_sha256,
        "pilot_gate_path": str(pilot_gate_path),
        "pilot_gate_sha256": sha256_file(pilot_gate_path),
    }
    mismatches = {
        name: {"observed": evidence.get(name), "expected": expected}
        for name, expected in expected_evidence.items()
        if evidence.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f"publication U-control plan evidence changed: {mismatches}"
        )
    if value.get("outputs_absent_when_frozen") is not True:
        raise ValueError("publication U-control plan lacks pre-output assertion")
    return value

