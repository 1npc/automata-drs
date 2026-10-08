"""Machine-verifiable promotion gates for recent LoRA smoke runs."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SMOKE_CONFIGURATIONS = ("U", "B", "W", "BF", "WF")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def evaluate_lora_smoke_gate(
    model_row: dict[str, Any],
    run_root: Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Return an evidence-rich pass/fail report for one smoke adapter.

    This gate checks pipeline integrity, not task quality. In particular, U/B/W
    may be invalid after an eight-step smoke run; only BF/WF are required to
    finish in the accepting language without an all-masked decoder row.
    """

    audit_path = Path(str(run_root) + ".tokenizer-audit.json")
    train_manifest_path = run_root / "train_manifest.json"
    train_metrics_path = run_root / "train_metrics.json"
    prediction_path = run_root / "dev20-u-b-w-bf-wf-greedy.jsonl"
    prediction_manifest_path = Path(str(prediction_path) + ".manifest.json")
    adapter_path = run_root / "adapter" / "adapter_model.safetensors"
    required_paths = (
        audit_path,
        train_manifest_path,
        train_metrics_path,
        prediction_path,
        prediction_manifest_path,
        adapter_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {
            "model_key": model_row.get("key"),
            "run_root": str(run_root),
            "passed": False,
            "missing_artifacts": missing,
            "checks": [],
        }

    audit = _read_json(audit_path)
    train_manifest = _read_json(train_manifest_path)
    train_metrics = _read_json(train_metrics_path)
    prediction_manifest = _read_json(prediction_manifest_path)
    rows = _read_jsonl(prediction_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    revision = str(model_row["revision"])
    check(
        "tokenizer_revision",
        audit.get("requested_revision") == revision,
        audit.get("requested_revision"),
        revision,
    )
    roundtrip_failures = audit.get("target_roundtrip_failure_count")
    check("target_roundtrip_failures", roundtrip_failures == 0, roundtrip_failures, 0)
    total_max = audit.get("lengths", {}).get("total_max")
    check(
        "maximum_sequence_length",
        isinstance(total_max, int) and total_max <= max_length,
        total_max,
        f"<= {max_length}",
    )
    check(
        "training_status",
        train_manifest.get("status") == "complete",
        train_manifest.get("status"),
        "complete",
    )
    check(
        "training_revision",
        train_manifest.get("requested_revision") == revision,
        train_manifest.get("requested_revision"),
        revision,
    )
    check(
        "smoke_optimizer_steps",
        train_manifest.get("recipe", {}).get("max_steps") == 8,
        train_manifest.get("recipe", {}).get("max_steps"),
        8,
    )
    train_loss = train_metrics.get("train", {}).get("train_loss")
    eval_loss = train_metrics.get("eval", {}).get("eval_loss")
    check(
        "finite_train_loss",
        isinstance(train_loss, (int, float)) and math.isfinite(train_loss),
        train_loss,
        "finite",
    )
    check(
        "finite_eval_loss",
        isinstance(eval_loss, (int, float)) and math.isfinite(eval_loss),
        eval_loss,
        "finite",
    )
    adapter_bytes = adapter_path.stat().st_size
    check("nonempty_adapter", adapter_bytes > 0, adapter_bytes, "> 0 bytes")
    check(
        "prediction_revision",
        prediction_manifest.get("requested_revision") == revision,
        prediction_manifest.get("requested_revision"),
        revision,
    )

    smoke_records = int(model_row.get("smoke_records", 20))
    expected_total = smoke_records * len(SMOKE_CONFIGURATIONS)
    configuration_counts = Counter(str(row.get("configuration")) for row in rows)
    expected_counts = {
        configuration: smoke_records for configuration in SMOKE_CONFIGURATIONS
    }
    check(
        "prediction_matrix",
        len(rows) == expected_total and dict(configuration_counts) == expected_counts,
        {"records": len(rows), "configurations": dict(configuration_counts)},
        {"records": expected_total, "configurations": expected_counts},
    )
    identities = {
        (str(row.get("id")), str(row.get("configuration"))) for row in rows
    }
    check(
        "unique_prediction_cells",
        len(identities) == expected_total,
        len(identities),
        expected_total,
    )
    all_masked_rows = sum(
        int(row.get("constraint", {}).get("all_masked_rows", 0))
        for row in rows
        if row.get("constraint") is not None
    )
    check("all_masked_rows", all_masked_rows == 0, all_masked_rows, 0)
    bounded_rows = [
        row for row in rows if row.get("configuration") in {"BF", "WF"}
    ]
    bounded_invalid = [
        str(row.get("id")) + ":" + str(row.get("configuration"))
        for row in bounded_rows
        if row.get("validity", {}).get("raw_sbn") is not True
        or (
            row.get("configuration") == "WF"
            and row.get("validity", {}).get(
                "raw_clf_referee",
                row.get("validity", {}).get("clf_referee"),
            )
            is not True
        )
    ]
    check("bounded_outputs_valid", not bounded_invalid, bounded_invalid, [])
    bounded_truncated = [
        str(row.get("id")) + ":" + str(row.get("configuration"))
        for row in bounded_rows
        if row.get("generation", {}).get("hit_max_new_tokens") is True
    ]
    check("bounded_outputs_not_truncated", not bounded_truncated, bounded_truncated, [])
    canonical_scored = sum(
        row.get("generation", {}).get("canonical_tokenization") is not None
        for row in rows
    )
    canonical = sum(
        row.get("generation", {}).get("canonical_tokenization") is True
        for row in rows
    )

    return {
        "model_key": model_row.get("key"),
        "model": model_row.get("repo_id"),
        "revision": revision,
        "run_root": str(run_root),
        "passed": all(item["passed"] for item in checks),
        "missing_artifacts": [],
        "checks": checks,
        "diagnostics": {
            "adapter_bytes": adapter_bytes,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "smoke_records": smoke_records,
            "canonical_tokenization_scored": canonical_scored,
            "canonical_tokenization": canonical,
            "canonical_tokenization_rate": (
                canonical / canonical_scored if canonical_scored else None
            ),
        },
    }


__all__ = ["SMOKE_CONFIGURATIONS", "evaluate_lora_smoke_gate"]
