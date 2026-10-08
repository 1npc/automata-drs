#!/usr/bin/env python3
"""Immutable integrity and safety gate for the worst-case AMR GPU smoke."""

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

from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
    sha256_file,
)
from automata_drs.release import (  # noqa: E402
    PENMAN_RELEASE_GUARANTEE,
    has_complete_selective_release_accounting,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(value)
    return rows


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
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
    parser.add_argument("--preflight-gate", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--paired", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite AMR smoke gate: {args.output}")

    config_path = Path("configs/amr_transfer_experiment.json")
    registry_path = Path("configs/lora_baselines.json")
    train_manifest_path = args.run_root / "train_manifest.json"
    train_metrics_path = args.run_root / "train_metrics.json"
    predictions_path = args.run_root / "dev-smoke-u-p-greedy.jsonl"
    generation_manifest_path = Path(str(predictions_path) + ".manifest.json")
    scores_path = args.scores or (
        args.run_root / "dev-smoke-u-p-greedy.smatch.json"
    )
    paired_path = args.paired or (
        args.run_root / "dev-smoke-u-p-greedy.paired.json"
    )
    release_audit_path = args.run_root / "dev-smoke-p-certified-release.json"
    try:
        config = _read_object(config_path)
        registry = _read_object(registry_path)
        preflight = _read_object(args.preflight_gate)
        train_manifest = _read_object(train_manifest_path)
        train_metrics = _read_object(train_metrics_path)
        generation_manifest = complete_generation_manifest(
            predictions_path, generation_manifest_path
        )
        scores = _read_object(scores_path)
        paired = _read_object(paired_path)
        release_audit = _read_object(release_audit_path)
        rows = _read_rows(predictions_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))

    host = config["host"]
    models = [row for row in registry["models"] if row["key"] == host["key"]]
    if len(models) != 1:
        parser.error("AMR host is missing or duplicated in model registry")
    model = models[0]
    recommendation = preflight["recommendation"]
    expected_ordinals = recommendation["smoke_train_record_ordinals"]
    expected_examples = int(model["smoke_records"])
    expected_rows = expected_examples * 2
    expected_target_manifest = model["training_target_modules_manifest"]
    training_cuda = train_metrics.get("cuda_memory", {})

    adapter_model = args.run_root / "adapter" / "adapter_model.safetensors"
    adapter_config = args.run_root / "adapter" / "adapter_config.json"
    checks: dict[str, bool] = {
        "preflight_gate": preflight.get("gate", {}).get("passed") is True,
        "preflight_config_hash": (
            preflight.get("inputs", {}).get("config_sha256")
            == sha256_file(config_path)
        ),
        "train_complete": train_manifest.get("status") == "complete",
        "train_task": train_manifest.get("task") == "amr",
        "train_model": train_manifest.get("model") == host["repo_id"],
        "train_loader": train_manifest.get("model_loader") == host["model_loader"],
        "train_requested_revision": (
            train_manifest.get("requested_revision") == host["revision"]
        ),
        "train_resolved_revision": (
            train_manifest.get("resolved_revision") == host["revision"]
        ),
        "train_source_hash": (
            train_manifest.get("train_source_sha256")
            == config["data"]["train"]["sha256"]
        ),
        "eval_source_hash": (
            train_manifest.get("eval_source_sha256")
            == config["data"]["development"]["sha256"]
        ),
        "worst_case_selection": (
            train_manifest.get("train_record_numbers") == expected_ordinals
        ),
        "train_record_count": (
            train_manifest.get("train", {}).get("records") == len(expected_ordinals)
        ),
        "eval_record_count": train_manifest.get("eval", {}).get("records") == 16,
        "no_training_truncation": (
            isinstance(train_manifest.get("train", {}).get("max_sequence_tokens"), int)
            and train_manifest["train"]["max_sequence_tokens"]
            <= recommendation["max_length"]
            and isinstance(train_manifest.get("eval", {}).get("max_sequence_tokens"), int)
            and train_manifest["eval"]["max_sequence_tokens"]
            <= recommendation["max_length"]
        ),
        "max_length": (
            train_manifest.get("recipe", {}).get("max_length")
            == recommendation["max_length"]
        ),
        "smoke_steps": (
            train_manifest.get("recipe", {}).get("max_steps")
            == recommendation["smoke_max_steps"]
        ),
        "smoke_batch": (
            train_manifest.get("recipe", {}).get("batch_size")
            == recommendation["smoke_per_device_batch_size"]
            and train_manifest.get("recipe", {}).get("gradient_accumulation")
            == recommendation["smoke_gradient_accumulation_steps"]
        ),
        "fixed_lora_recipe": (
            train_manifest.get("recipe", {}).get("learning_rate")
            == config["training"]["learning_rate"]
            and train_manifest.get("recipe", {}).get("warmup_ratio")
            == config["training"]["warmup_ratio"]
            and train_manifest.get("recipe", {}).get("lora_rank")
            == config["training"]["lora_rank"]
            and train_manifest.get("recipe", {}).get("lora_alpha")
            == config["training"]["lora_alpha"]
            and train_manifest.get("recipe", {}).get("lora_dropout")
            == config["training"]["lora_dropout"]
            and train_manifest.get("recipe", {}).get("seed")
            == config["training"]["seed"]
            and train_manifest.get("recipe", {}).get("precision") == "bf16"
        ),
        "cpu_and_dataset_contract": (
            train_manifest.get("recipe", {}).get("torch_threads") == 2
            and train_manifest.get("recipe", {}).get("dataloader_num_workers") == 0
            and train_manifest.get("recipe", {}).get("dataset_storage")
            == "compact-uint32-input-ids; masks-and-labels-materialized-per-item"
        ),
        "lora_target_manifest": (
            train_manifest.get("recipe", {}).get("target_modules")
            == expected_target_manifest
        ),
        "training_metrics_finite": (
            _finite(train_metrics.get("train", {}).get("train_loss"))
            and _finite(train_metrics.get("eval", {}).get("eval_loss"))
            and _finite(train_metrics.get("wall_seconds"))
        ),
        "training_memory_measured": (
            isinstance(training_cuda.get("peak_allocated_bytes"), int)
            and 0 < training_cuda["peak_allocated_bytes"]
            <= training_cuda.get("peak_reserved_bytes", -1)
            <= training_cuda.get("device_total_bytes", -1)
        ),
        "adapter_files": adapter_model.is_file() and adapter_config.is_file(),
        "generation_complete": generation_manifest.get("status") == "complete",
        "generation_task": generation_manifest.get("task") == "amr",
        "generation_model": generation_manifest.get("model") == host["repo_id"],
        "generation_loader": (
            generation_manifest.get("model_loader") == host["model_loader"]
        ),
        "generation_revision": (
            generation_manifest.get("requested_revision") == host["revision"]
        ),
        "generation_config_hash": (
            generation_manifest.get("experiment_config_sha256")
            == sha256_file(config_path)
        ),
        "generation_source_hash": (
            generation_manifest.get("source_sha256")
            == config["data"]["development"]["sha256"]
        ),
        "generation_population": (
            generation_manifest.get("limit") == expected_examples
            and generation_manifest.get("expected_examples") == expected_examples
            and generation_manifest.get("expected_rows") == expected_rows
            and generation_manifest.get("configurations") == ["U", "P"]
            and generation_manifest.get("num_beams") == 1
            and generation_manifest.get("batch_size") == 1
            and generation_manifest.get("schedule") == "rotate"
        ),
        "generation_adapter": (
            generation_manifest.get("adapter") == str(args.run_root / "adapter")
        ),
        "generation_lengths": (
            generation_manifest.get("max_input_tokens")
            == recommendation["max_input_tokens"]
            and generation_manifest.get("max_new_tokens")
            == recommendation["max_new_tokens"]
        ),
        "generation_scored": generation_manifest.get("score_model_logprob") is True,
        "row_count": len(rows) == expected_rows,
        "score_gate": scores.get("gate", {}).get("passed") is True,
        "score_prediction_hash": (
            scores.get("predictions", {}).get("sha256")
            == sha256_file(predictions_path)
        ),
        "score_groups": (
            set(scores.get("groups", {})) == {"U", "P"}
            and all(
                scores["groups"][name].get("records") == expected_examples
                for name in ("U", "P")
            )
        ),
        "paired_gate": paired.get("gate", {}).get("passed") is True,
        "paired_population": (
            paired.get("population", {}).get("records") == expected_examples
            and paired.get("population", {}).get("paired") is True
        ),
        "certified_release_accounting": (
            release_audit.get("schema_version") == 2
            and release_audit.get("audit") == "penman-certified-release"
            and release_audit.get("gate", {}).get("passed") is True
            and release_audit.get("gate", {}).get("performance_threshold_used")
            is False
            and release_audit.get("input", {}).get("predictions_sha256")
            == sha256_file(predictions_path)
            and release_audit.get("population", {}).get("examples")
            == expected_examples
            and release_audit.get("population", {}).get("rows")
            == expected_examples
            and has_complete_selective_release_accounting(
                release_audit.get("release", {}),
                attempts=expected_examples,
                guarantee=PENMAN_RELEASE_GUARANTEE,
            )
        ),
    }

    identities: dict[str, set[str]] = {"U": set(), "P": set()}
    p_accepting_count = 0
    p_all_masked_rows = 0
    truncations = {"U": 0, "P": 0}
    finite_scores = True
    unconstrained_has_no_processor = True
    peak_within_device = True
    for row in rows:
        configuration = str(row.get("configuration", ""))
        identifier = str(row.get("id", ""))
        if configuration not in identities or identifier in identities[configuration]:
            checks["row_identities_well_formed"] = False
            continue
        identities[configuration].add(identifier)
        generation = row.get("generation", {})
        truncations[configuration] += int(
            bool(generation.get("hit_max_new_tokens"))
        )
        peak = generation.get("peak_gpu_bytes")
        peak_within_device = peak_within_device and isinstance(peak, int) and (
            peak <= generation_manifest.get("cuda_device_total_bytes", -1)
        )
        score = row.get("score", {})
        finite_scores = finite_scores and _finite(score.get("raw_model_logprob"))
        if configuration == "U":
            unconstrained_has_no_processor = (
                unconstrained_has_no_processor and row.get("constraint") is None
            )
        else:
            validity = row.get("validity", {})
            p_accepting_count += int(
                validity.get("raw_local_penman") is True
                and validity.get("raw_official_penman") is True
                and generation.get("ended_with_eos") is True
            )
            constraint = row.get("constraint")
            p_all_masked_rows += (
                int(constraint.get("all_masked_rows", 0))
                if isinstance(constraint, dict)
                else 0
            )
    checks.setdefault("row_identities_well_formed", True)
    checks.update(
        {
            "complete_identity_pairing": (
                len(identities["U"]) == expected_examples
                and identities["U"] == identities["P"]
            ),
            "finite_raw_model_scores": finite_scores,
            "u_has_no_logits_processor": unconstrained_has_no_processor,
            "peak_gpu_within_device": peak_within_device,
            "paired_reports_search_outcomes": (
                paired.get("search", {}).get("p_all_masked_rows")
                == p_all_masked_rows
                and paired.get("search", {}).get("truncations") == truncations
            ),
        }
    )
    release = release_audit.get("release", {})
    releasable_rows = release.get("releasable_rows")
    refused_rows = release.get("refused_rows")
    report = {
        "schema_version": 1,
        "audit": "amr-worst-case-lora-smoke-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "preflight_gate": str(args.preflight_gate),
            "preflight_gate_sha256": sha256_file(args.preflight_gate),
            "train_manifest": str(train_manifest_path),
            "train_manifest_sha256": sha256_file(train_manifest_path),
            "train_metrics": str(train_metrics_path),
            "train_metrics_sha256": sha256_file(train_metrics_path),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "scores": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "paired": str(paired_path),
            "paired_sha256": sha256_file(paired_path),
            "certified_release_audit": str(release_audit_path),
            "certified_release_audit_sha256": sha256_file(release_audit_path),
            "adapter_model_sha256": (
                sha256_file(adapter_model) if adapter_model.is_file() else None
            ),
        },
        "observed": {
            "train_records": train_manifest.get("train", {}).get("records"),
            "eval_records": train_manifest.get("eval", {}).get("records"),
            "prediction_rows": len(rows),
            "training_peak_allocated_bytes": training_cuda.get(
                "peak_allocated_bytes"
            ),
            "training_peak_reserved_bytes": training_cuda.get(
                "peak_reserved_bytes"
            ),
            "training_device_total_bytes": training_cuda.get(
                "device_total_bytes"
            ),
            "p_raw_penman_valid": sum(
                row.get("configuration") == "P"
                and row.get("validity", {}).get("raw_official_penman") is True
                for row in rows
            ),
            "peak_gpu_bytes": max(
                (
                    int(row.get("generation", {}).get("peak_gpu_bytes", 0))
                    for row in rows
                ),
                default=0,
            ),
        },
        "reported_outcomes": {
            "promotion_policy": (
                "selective-release integrity only; acceptance, all-masked, and "
                "hard-horizon counts are empirical outcomes, not thresholds"
            ),
            "p_attempts": expected_examples,
            "p_accepting_with_eos": p_accepting_count,
            "p_all_masked_rows": p_all_masked_rows,
            "truncations": truncations,
            "p_releasable_rows": releasable_rows,
            "p_refused_rows": refused_rows,
            "p_release_coverage": (
                releasable_rows / expected_examples
                if isinstance(releasable_rows, int)
                else None
            ),
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
            parser.error("stored AMR smoke gate differs from recomputed evidence")
        print(f"verified AMR smoke gate: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR smoke gate: {args.output}")
        print(f"report: {args.output}")
    print(f"checks: {sum(checks.values())}/{len(checks)}")
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
