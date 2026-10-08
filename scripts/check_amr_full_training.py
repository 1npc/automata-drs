#!/usr/bin/env python3
"""Immutable gate for the full selected-host AMR LoRA training run."""

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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/amr_transfer_experiment.json")
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/lora_baselines.json")
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        config = _read_object(args.config)
        registry = _read_object(args.registry)
        plan = _read_object(args.resource_plan)
        manifest_path = args.run_root / "train_manifest.json"
        metrics_path = args.run_root / "train_metrics.json"
        manifest = _read_object(manifest_path)
        metrics = _read_object(metrics_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    host = config["host"]
    models = [row for row in registry["models"] if row.get("key") == host["key"]]
    if len(models) != 1:
        parser.error("AMR host is missing or duplicated in model registry")
    model = models[0]
    recipe = manifest.get("recipe", {})
    planned = plan.get("recipe", {})
    training = config["training"]
    cuda = metrics.get("cuda_memory", {})
    train = manifest.get("train", {})
    evaluation = manifest.get("eval", {})
    adapter_model = args.run_root / "adapter" / "adapter_model.safetensors"
    adapter_config = args.run_root / "adapter" / "adapter_config.json"
    checks = {
        "resource_plan_passed": (
            plan.get("audit") == "amr-full-run-resource-plan"
            and plan.get("gate", {}).get("passed") is True
            and plan.get("gate", {}).get("performance_threshold_used") is False
            and plan.get("inputs", {}).get("config_sha256")
            == sha256_file(args.config)
        ),
        "training_complete": manifest.get("status") == "complete",
        "task_and_host": (
            manifest.get("task") == "amr"
            and manifest.get("model") == host["repo_id"]
            and manifest.get("model_loader") == host["model_loader"]
            and manifest.get("requested_revision") == host["revision"]
            and manifest.get("resolved_revision") == host["revision"]
        ),
        "full_data_bound": (
            manifest.get("train_source_sha256") == config["data"]["train"]["sha256"]
            and manifest.get("eval_source_sha256")
            == config["data"]["development"]["sha256"]
            and manifest.get("train_record_numbers") is None
            and train.get("records") == config["data"]["train"]["records"]
            and evaluation.get("records")
            == config["data"]["development"]["records"]
        ),
        "no_training_truncation": (
            isinstance(train.get("max_sequence_tokens"), int)
            and isinstance(evaluation.get("max_sequence_tokens"), int)
            and train["max_sequence_tokens"] <= planned.get("max_length", -1)
            and evaluation["max_sequence_tokens"] <= planned.get("max_length", -1)
        ),
        "promoted_batch_and_length": (
            recipe.get("max_length") == planned.get("max_length")
            and recipe.get("epochs") == planned.get("epochs")
            and recipe.get("max_steps") == -1
            and recipe.get("batch_size") == planned.get("per_device_batch_size")
            and recipe.get("gradient_accumulation")
            == planned.get("gradient_accumulation_steps")
            and recipe.get("effective_batch_size")
            == planned.get("effective_batch_size")
            and recipe.get("planned_optimizer_steps")
            == planned.get("planned_optimizer_steps")
        ),
        "fixed_lora_recipe": (
            recipe.get("learning_rate") == training["learning_rate"]
            and recipe.get("warmup_ratio") == training["warmup_ratio"]
            and recipe.get("lora_rank") == training["lora_rank"]
            and recipe.get("lora_alpha") == training["lora_alpha"]
            and recipe.get("lora_dropout") == training["lora_dropout"]
            and recipe.get("seed") == training["seed"]
            and recipe.get("precision") == "bf16"
            and recipe.get("target_modules")
            == model["training_target_modules_manifest"]
        ),
        "bounded_cpu_and_dataset": (
            recipe.get("torch_threads") == config["resource_contract"]["cpu_math_threads"]
            and recipe.get("dataloader_num_workers") == 0
            and recipe.get("dataset_storage")
            == "compact-uint32-input-ids; masks-and-labels-materialized-per-item"
        ),
        "loss_and_time_finite": (
            _finite(metrics.get("train", {}).get("train_loss"))
            and _finite(metrics.get("eval", {}).get("eval_loss"))
            and _finite(metrics.get("wall_seconds"))
        ),
        "training_memory_safe": (
            isinstance(cuda.get("peak_allocated_bytes"), int)
            and isinstance(cuda.get("peak_reserved_bytes"), int)
            and isinstance(cuda.get("device_total_bytes"), int)
            and 0 < cuda["peak_allocated_bytes"] <= cuda["peak_reserved_bytes"]
            <= cuda["device_total_bytes"]
            and cuda["device_total_bytes"] - cuda["peak_reserved_bytes"]
            >= plan["resource_evidence"]["required_gpu_margin_bytes"]
        ),
        "adapter_complete": adapter_model.is_file() and adapter_config.is_file(),
    }
    report = {
        "schema_version": 1,
        "audit": "amr-full-lora-training-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "registry": str(args.registry),
            "registry_sha256": sha256_file(args.registry),
            "resource_plan": str(args.resource_plan),
            "resource_plan_sha256": sha256_file(args.resource_plan),
            "train_manifest": str(manifest_path),
            "train_manifest_sha256": sha256_file(manifest_path),
            "train_metrics": str(metrics_path),
            "train_metrics_sha256": sha256_file(metrics_path),
            "adapter_model_sha256": (
                sha256_file(adapter_model) if adapter_model.is_file() else None
            ),
            "adapter_config_sha256": (
                sha256_file(adapter_config) if adapter_config.is_file() else None
            ),
        },
        "observed": {
            "train_records": train.get("records"),
            "eval_records": evaluation.get("records"),
            "planned_optimizer_steps": recipe.get("planned_optimizer_steps"),
            "wall_seconds": metrics.get("wall_seconds"),
            "cuda_memory": cuda,
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "performance_threshold_used": False,
        },
    }
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored full AMR training gate differs from current evidence")
        print(f"verified AMR training gate: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR training gate: {args.output}")
    print(f"checks: {sum(checks.values())}/{len(checks)}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
