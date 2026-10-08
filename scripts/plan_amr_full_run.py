#!/usr/bin/env python3
"""Freeze a resource-safe full AMR recipe from passed preflight and smoke gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import sha256_file  # noqa: E402


MINIMUM_GPU_MARGIN_BYTES = 4 * 1024**3


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def build_plan(
    config_path: Path,
    preflight_path: Path,
    smoke_path: Path,
    smoke_run_root: Path,
) -> dict[str, Any]:
    config = _read_object(config_path)
    preflight = _read_object(preflight_path)
    smoke = _read_object(smoke_path)
    train_manifest_path = smoke_run_root / "train_manifest.json"
    train_metrics_path = smoke_run_root / "train_metrics.json"
    train_manifest = _read_object(train_manifest_path)
    train_metrics = _read_object(train_metrics_path)

    config_sha256 = sha256_file(config_path)
    preflight_sha256 = sha256_file(preflight_path)
    train_manifest_sha256 = sha256_file(train_manifest_path)
    train_metrics_sha256 = sha256_file(train_metrics_path)
    recommendation = preflight.get("recommendation", {})
    smoke_recipe = train_manifest.get("recipe", {})
    cuda = train_metrics.get("cuda_memory", {})
    batch_size = smoke_recipe.get("batch_size")
    accumulation = smoke_recipe.get("gradient_accumulation")
    epochs = config.get("training", {}).get("epochs")
    train_records = config.get("data", {}).get("train", {}).get("records")
    smoke_steps = smoke_recipe.get("max_steps")
    smoke_wall_seconds = train_metrics.get("wall_seconds")
    integers = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            batch_size,
            accumulation,
            epochs,
            train_records,
            smoke_steps,
        )
    )
    planned_steps = (
        math.ceil(train_records / (batch_size * accumulation)) * epochs
        if integers
        else None
    )
    seconds_per_smoke_step = (
        float(smoke_wall_seconds) / smoke_steps
        if isinstance(smoke_wall_seconds, (int, float))
        and not isinstance(smoke_wall_seconds, bool)
        and smoke_wall_seconds > 0
        and isinstance(smoke_steps, int)
        and smoke_steps > 0
        else None
    )
    estimated_wall_seconds = (
        seconds_per_smoke_step * planned_steps
        if seconds_per_smoke_step is not None and planned_steps is not None
        else None
    )
    peak_reserved = cuda.get("peak_reserved_bytes")
    device_total = cuda.get("device_total_bytes")
    gpu_margin = (
        device_total - peak_reserved
        if isinstance(device_total, int)
        and isinstance(peak_reserved, int)
        and not isinstance(device_total, bool)
        and not isinstance(peak_reserved, bool)
        else None
    )
    smoke_inputs = smoke.get("inputs", {})
    checks = {
        "prepared_before_outputs": config.get("status") == "prepared-not-launched",
        "preflight_passed": preflight.get("gate", {}).get("passed") is True,
        "smoke_passed": smoke.get("gate", {}).get("passed") is True,
        "current_config_bound": (
            preflight.get("inputs", {}).get("config_sha256") == config_sha256
        ),
        "smoke_binds_preflight": (
            smoke_inputs.get("preflight_gate_sha256") == preflight_sha256
        ),
        "smoke_binds_training": (
            smoke_inputs.get("train_manifest_sha256") == train_manifest_sha256
            and smoke_inputs.get("train_metrics_sha256") == train_metrics_sha256
        ),
        "smoke_training_complete": train_manifest.get("status") == "complete",
        "smoke_uses_preflight_length": (
            smoke_recipe.get("max_length") == recommendation.get("max_length")
        ),
        "same_effective_batch_as_registered": (
            integers
            and batch_size * accumulation
            == config["training"]["per_device_batch_size"]
            * config["training"]["gradient_accumulation_steps"]
        ),
        "safe_batch_is_observed_not_extrapolated": batch_size == 1,
        "training_cuda_memory_measured": (
            isinstance(peak_reserved, int)
            and isinstance(device_total, int)
            and 0 < peak_reserved <= device_total
        ),
        "minimum_gpu_memory_margin": (
            isinstance(gpu_margin, int)
            and gpu_margin >= MINIMUM_GPU_MARGIN_BYTES
        ),
        "serial_cpu_contract": (
            smoke_recipe.get("torch_threads")
            == config.get("resource_contract", {}).get("cpu_math_threads")
            and smoke_recipe.get("dataloader_num_workers") == 0
        ),
        "full_schedule_computable": (
            isinstance(planned_steps, int)
            and planned_steps > smoke_steps
            and estimated_wall_seconds is not None
        ),
    }
    return {
        "schema_version": 1,
        "audit": "amr-full-run-resource-plan",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(config_path),
            "config_sha256": config_sha256,
            "preflight_gate": str(preflight_path),
            "preflight_gate_sha256": preflight_sha256,
            "smoke_gate": str(smoke_path),
            "smoke_gate_sha256": sha256_file(smoke_path),
            "smoke_train_manifest": str(train_manifest_path),
            "smoke_train_manifest_sha256": train_manifest_sha256,
            "smoke_train_metrics": str(train_metrics_path),
            "smoke_train_metrics_sha256": train_metrics_sha256,
        },
        "recipe": {
            "max_length": recommendation.get("max_length"),
            "max_input_tokens": recommendation.get("max_input_tokens"),
            "max_new_tokens": recommendation.get("max_new_tokens"),
            "epochs": epochs,
            "per_device_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": (
                batch_size * accumulation if integers else None
            ),
            "planned_optimizer_steps": planned_steps,
            "batch_promotion_policy": (
                "reuse the exact worst-case-smoke batch; no unmeasured memory scaling"
            ),
        },
        "resource_evidence": {
            "smoke_peak_reserved_bytes": peak_reserved,
            "device_total_bytes": device_total,
            "observed_gpu_margin_bytes": gpu_margin,
            "required_gpu_margin_bytes": MINIMUM_GPU_MARGIN_BYTES,
            "smoke_wall_seconds": smoke_wall_seconds,
            "seconds_per_smoke_optimizer_step": seconds_per_smoke_step,
            "rough_full_training_wall_seconds": estimated_wall_seconds,
            "wall_time_estimate_is_scheduling_only": True,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "performance_threshold_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/amr_transfer_experiment.json")
    )
    parser.add_argument("--preflight-gate", type=Path, required=True)
    parser.add_argument("--smoke-gate", type=Path, required=True)
    parser.add_argument("--smoke-run-root", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        report = build_plan(
            args.config, args.preflight_gate, args.smoke_gate, args.smoke_run_root
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR full-run plan differs from current evidence")
        print(f"verified AMR full-run plan: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR full-run plan: {args.output}")
    print(f"planned optimizer steps: {report['recipe']['planned_optimizer_steps']}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
