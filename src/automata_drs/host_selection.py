"""Fail-closed, development-only selection of the primary LoRA host model."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .causal_lm import SBNRecord, iter_sbn_records


SPLIT_PATHS = {
    "dev": Path("data/en/dev/standard.sbn"),
    "test": Path("data/en/test/standard.sbn"),
    "tax": Path("data/en/test/tax_challenge.sbn"),
}


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_rate(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]: {number!r}")
    return number


def _validate_prediction_jsonl(
    path: Path,
    *,
    expected_rows: list[SBNRecord],
    corpus_label: Path,
) -> tuple[list[dict[str, Any]], str]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank prediction line {line_number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"prediction line is not an object: {path}:{line_number}"
                )
            identifier = str(row.get("id", ""))
            if not identifier or identifier in seen:
                raise ValueError(
                    f"missing or duplicate prediction id {identifier!r}: "
                    f"{path}:{line_number}"
                )
            seen.add(identifier)
            if row.get("configuration") != "U":
                raise ValueError(f"non-U row in baseline: {path}:{line_number}")
            decode = row.get("decode", {})
            if decode.get("strategy") != "greedy" or decode.get("num_beams") != 1:
                raise ValueError(f"non-greedy row in baseline: {path}:{line_number}")
            rows.append(row)
    if len(rows) != len(expected_rows):
        raise ValueError(
            f"expected {len(expected_rows)} baseline records, found "
            f"{len(rows)}: {path}"
        )
    for index, (row, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        line_suffix = expected.identifier.rsplit(":", 1)[-1]
        expected_id = f"{corpus_label}:{line_suffix}"
        mismatches = []
        if row.get("id") != expected_id:
            mismatches.append("id")
        if row.get("input") != expected.source:
            mismatches.append("input")
        if row.get("reference") != expected.target:
            mismatches.append("reference")
        if mismatches:
            raise ValueError(
                f"baseline row {index + 1} differs from {corpus_label}: "
                + ", ".join(mismatches)
            )
    return rows, _sha256(path)


def _validate_score_jsonl(
    path: Path, prediction_rows: list[dict[str, Any]]
) -> tuple[list[float], str]:
    scores: list[float] = []
    with path.open(encoding="utf-8") as source:
        for line_number, (line, prediction) in enumerate(
            zip(source, prediction_rows, strict=True), start=1
        ):
            if not line.strip():
                raise ValueError(f"blank score line {line_number}: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"score line is not an object: {path}:{line_number}")
            if (
                row.get("id") != prediction.get("id")
                or row.get("configuration") != "U"
                or row.get("decode", {}).get("strategy") != "greedy"
                or row.get("decode", {}).get("num_beams") != 1
            ):
                raise ValueError(
                    f"score/prediction identity mismatch: {path}:{line_number}"
                )
            scores.append(
                _finite_rate(
                    row.get("semantic_scores", {}).get("sbn_smatch"),
                    f"sbn_smatch at {path}:{line_number}",
                )
            )
    if len(scores) != len(prediction_rows):
        raise ValueError(
            f"expected {len(prediction_rows)} score rows, found {len(scores)}: "
            f"{path}"
        )
    return scores, _sha256(path)


def _u_group(summary: dict[str, Any], expected_records: int) -> dict[str, Any]:
    matches = [
        row
        for row in summary.get("groups", [])
        if row.get("configuration") == "U" and row.get("strategy") == "greedy"
    ]
    if len(matches) != 1:
        raise ValueError("summary must contain exactly one U/greedy group")
    group = matches[0]
    if int(group.get("total", -1)) != expected_records:
        raise ValueError(
            f"summary total must be {expected_records}, found {group.get('total')!r}"
        )
    return group


def _validate_training_contract(
    manifest: dict[str, Any],
    model_row: dict[str, Any],
    shared_recipe: dict[str, Any],
    *,
    train_records: int,
    eval_records: int,
) -> None:
    if manifest.get("resolved_revision") != model_row["revision"]:
        raise ValueError("training resolved revision does not match pin")
    manifest_contract = model_row.get("training_manifest_contract")
    if manifest_contract == "model-loader-v1":
        if manifest.get("model_loader") != model_row["model_loader"]:
            raise ValueError("training model loader does not match config")
    elif manifest_contract == "legacy-no-model-loader-v0":
        if manifest.get("model_loader") is not None:
            raise ValueError("legacy training manifest unexpectedly has model loader")
    else:
        raise ValueError(
            f"unsupported training manifest contract: {manifest_contract!r}"
        )
    if (
        manifest.get("train_path") != "data/en/train/gold.sbn"
        or manifest.get("eval_path") != "data/en/dev/standard.sbn"
        or manifest.get("train", {}).get("records") != train_records
        or manifest.get("eval", {}).get("records") != eval_records
    ):
        raise ValueError("training corpus path/count contract does not match")

    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("training manifest has no recipe")
    target_modules = recipe.get("target_modules")
    expected_target_modules = model_row.get(
        "training_target_modules_manifest"
    )
    if not isinstance(expected_target_modules, (str, dict)):
        raise ValueError(
            "model has no registered training target-module manifest"
        )
    batch_size = int(shared_recipe["per_device_batch_size"])
    accumulation = int(shared_recipe["gradient_accumulation_steps"])
    epochs = float(shared_recipe["epochs"])
    optimizer_steps = (
        math.ceil(math.ceil(train_records / batch_size) / accumulation)
        * int(epochs)
    )
    warmup_steps = math.ceil(
        optimizer_steps * float(shared_recipe["warmup_ratio"])
    )
    expected = {
        "max_length": int(shared_recipe["max_length"]),
        "epochs": epochs,
        "max_steps": -1,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "learning_rate": float(shared_recipe["learning_rate"]),
        "planned_optimizer_steps": optimizer_steps,
        "warmup_steps": warmup_steps,
        "lora_rank": int(shared_recipe["lora_rank"]),
        "lora_alpha": int(shared_recipe["lora_alpha"]),
        "lora_dropout": float(shared_recipe["lora_dropout"]),
        "precision": shared_recipe["precision"],
        "seed": int(shared_recipe["seed"]),
    }
    mismatches = {
        name: {"observed": recipe.get(name), "expected": value}
        for name, value in expected.items()
        if recipe.get(name) != value
    }
    if target_modules != expected_target_modules:
        mismatches["target_modules"] = {
            "observed": target_modules,
            "expected": expected_target_modules,
        }
    if mismatches:
        raise ValueError(
            "training recipe differs from shared contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    parameters = manifest.get("parameters", {})
    total = int(parameters.get("total", 0))
    trainable = int(parameters.get("trainable", 0))
    if not 0 < trainable < total < 10_000_000_000:
        raise ValueError("training parameter counts violate sub-10B LoRA contract")


def _validate_adapter_contract(
    adapter: dict[str, Any],
    model_row: dict[str, Any],
    shared_recipe: dict[str, Any],
) -> None:
    expected = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "bias": "none",
        "use_dora": False,
        "use_rslora": False,
        "r": int(shared_recipe["lora_rank"]),
        "lora_alpha": int(shared_recipe["lora_alpha"]),
        "lora_dropout": float(shared_recipe["lora_dropout"]),
        "base_model_name_or_path": model_row["repo_id"],
    }
    mismatches = {
        name: {"observed": adapter.get(name), "expected": value}
        for name, value in expected.items()
        if adapter.get(name) != value
    }
    targets = adapter.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or len(set(targets)) != len(targets)
        or any(not isinstance(name, str) or not name for name in targets)
    ):
        mismatches["target_modules"] = {
            "observed": targets,
            "expected": "non-empty unique module-name list",
        }
    if mismatches:
        raise ValueError(
            "saved adapter differs from LoRA contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _validate_tokenizer_audit_contract(
    audit: dict[str, Any],
    model_row: dict[str, Any],
    shared_recipe: dict[str, Any],
    *,
    train_records: int,
) -> None:
    if (
        audit.get("model") != model_row["repo_id"]
        or audit.get("requested_revision") != model_row["revision"]
        or audit.get("records") != train_records
        or audit.get("target_roundtrip_failure_count") != 0
    ):
        raise ValueError("tokenizer audit model/revision/coverage differs")
    total_max = audit.get("lengths", {}).get("total_max")
    if (
        not isinstance(total_max, int)
        or total_max > int(shared_recipe["max_length"])
    ):
        raise ValueError("tokenizer audit exceeds frozen max length")
    contract = model_row.get("tokenizer_audit_contract")
    if contract == "run-root-resolved-revision-v1":
        if audit.get("resolved_revision") != model_row["revision"]:
            raise ValueError("tokenizer audit resolved revision does not match pin")
    elif contract == "legacy-train-name-no-resolved-revision-v0":
        if audit.get("resolved_revision") is not None:
            raise ValueError(
                "legacy tokenizer audit unexpectedly has resolved revision"
            )
    else:
        raise ValueError(f"unsupported tokenizer audit contract: {contract!r}")


def _candidate(
    model_row: dict[str, Any],
    runs_root: Path,
    *,
    split: str,
    expected_records: int,
    repository_root: Path,
    shared_recipe: dict[str, Any],
) -> dict[str, Any]:
    key = str(model_row["key"])
    run_root = runs_root / f"{key}-r16-seed42"
    tokenizer_contract = model_row.get("tokenizer_audit_contract")
    if tokenizer_contract == "legacy-train-name-no-resolved-revision-v0":
        tokenizer_audit_path = runs_root / f"{key}-train-tokenizer-audit.json"
    elif tokenizer_contract == "run-root-resolved-revision-v1":
        tokenizer_audit_path = Path(str(run_root) + ".tokenizer-audit.json")
    else:
        tokenizer_audit_path = Path(str(run_root) + ".unsupported-audit-contract")
    paths = {
        "adapter_config": run_root / "adapter" / "adapter_config.json",
        "adapter_weights": run_root / "adapter" / "adapter_model.safetensors",
        "train_manifest": run_root / "train_manifest.json",
        "train_metrics": run_root / "train_metrics.json",
        "tokenizer_audit": tokenizer_audit_path,
        "predictions": run_root / f"{split}-u-greedy.jsonl",
        "generation_manifest": run_root / f"{split}-u-greedy.jsonl.manifest.json",
        "scores": run_root / f"{split}-u-greedy.smatch.jsonl",
        "score_manifest": run_root / f"{split}-u-greedy.smatch.jsonl.manifest.json",
        "summary": run_root / f"{split}-u-greedy.summary.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    result: dict[str, Any] = {
        "key": key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "parameters_billions": float(model_row["parameters_billions"]),
        "priority": int(model_row["priority"]),
        "run_root": str(run_root),
        "eligible": False,
        "reasons": [],
        "metrics": None,
        "evidence": {},
    }
    if missing:
        result["reasons"].append("missing artifacts: " + ", ".join(missing))
        return result

    try:
        if split not in SPLIT_PATHS:
            raise ValueError(f"unsupported selection split: {split}")
        corpus_label = SPLIT_PATHS[split]
        corpus_path = repository_root / corpus_label
        expected_rows = list(iter_sbn_records(corpus_path))
        if len(expected_rows) != expected_records:
            raise ValueError(
                f"configured {split} count is {expected_records}, but corpus has "
                f"{len(expected_rows)} rows: {corpus_path}"
            )
        train_path = repository_root / "data/en/train/gold.sbn"
        train_records = sum(1 for _ in iter_sbn_records(train_path))
        train_manifest = _read_object(paths["train_manifest"])
        train_metrics = _read_object(paths["train_metrics"])
        adapter_config = _read_object(paths["adapter_config"])
        tokenizer_audit = _read_object(paths["tokenizer_audit"])
        generation_manifest = _read_object(paths["generation_manifest"])
        score_manifest = _read_object(paths["score_manifest"])
        summary = _read_object(paths["summary"])
        if train_manifest.get("status") != "complete":
            raise ValueError("training manifest status is not complete")
        for manifest_name, manifest in (
            ("training", train_manifest),
            ("generation", generation_manifest),
        ):
            if manifest.get("model") != model_row["repo_id"]:
                raise ValueError(f"{manifest_name} model does not match config")
            if manifest.get("requested_revision") != model_row["revision"]:
                raise ValueError(
                    f"{manifest_name} requested revision does not match config"
                )
        _validate_training_contract(
            train_manifest,
            model_row,
            shared_recipe,
            train_records=train_records,
            eval_records=expected_records,
        )
        _validate_adapter_contract(adapter_config, model_row, shared_recipe)
        _validate_tokenizer_audit_contract(
            tokenizer_audit,
            model_row,
            shared_recipe,
            train_records=train_records,
        )
        expected_generation_fields = {
            "path": str(corpus_label),
            "start": 0,
            "limit": -1,
            "line_numbers": None,
            "configurations": ["U"],
            "num_beams": 1,
            "schedule": "rotate",
        }
        generation_contract = model_row.get("baseline_generation_contract")
        split_contracts = model_row.get("baseline_generation_split_contracts")
        if split_contracts is not None:
            if (
                not isinstance(split_contracts, dict)
                or split_contracts.get(split) != generation_contract
            ):
                raise ValueError(
                    "development generation contract differs from registered "
                    "split-specific contract"
                )
        if generation_contract == "batch8-unscored-v1":
            expected_generation_fields.update(
                {"batch_size": 8, "score_model_logprob": False}
            )
        elif generation_contract == "legacy-sequential-unscored-v0":
            expected_generation_fields.update(
                {"batch_size": None, "score_model_logprob": None}
            )
        else:
            raise ValueError(
                "unsupported baseline generation contract: "
                f"{generation_contract!r}"
            )
        generation_mismatches = [
            name
            for name, expected in expected_generation_fields.items()
            if generation_manifest.get(name) != expected
        ]
        if generation_mismatches:
            raise ValueError(
                "generation manifest differs from frozen dev baseline: "
                + ", ".join(generation_mismatches)
            )
        eval_loss = float(train_metrics.get("eval", {}).get("eval_loss"))
        if not math.isfinite(eval_loss):
            raise ValueError("development loss is not finite")
        prediction_rows, prediction_sha = _validate_prediction_jsonl(
            paths["predictions"],
            expected_rows=expected_rows,
            corpus_label=corpus_label,
        )
        score_values, score_sha = _validate_score_jsonl(
            paths["scores"], prediction_rows
        )
        if score_manifest.get("input_sha256") != prediction_sha:
            raise ValueError("score manifest input hash does not match predictions")
        if score_manifest.get("output_sha256") != score_sha:
            raise ValueError("score manifest output hash does not match score rows")
        if int(score_manifest.get("records", -1)) != expected_records:
            raise ValueError("score manifest record count does not match dev corpus")
        if score_manifest.get("groups") != [
            {
                "configuration": "U",
                "strategy": "greedy",
                "records": expected_records,
            }
        ]:
            raise ValueError("score manifest groups are not exactly U/greedy dev")
        group = _u_group(summary, expected_records)
        computed_smatch = sum(score_values) / expected_records
        invalid_raw_referee_rows = []
        invalid_legacy_fallback_rows = []
        for row in prediction_rows:
            validity = row.get("validity", {})
            raw_sbn = validity.get("raw_sbn", validity.get("sbn")) is True
            if "raw_clf_referee" in validity:
                if raw_sbn and not isinstance(
                    validity.get("raw_clf_referee"), bool
                ):
                    invalid_raw_referee_rows.append(str(row.get("id")))
            else:
                raw_prediction = row.get(
                    "raw_prediction", row.get("prediction")
                )
                if (
                    not isinstance(raw_prediction, str)
                    or not isinstance(row.get("prediction"), str)
                    or raw_prediction != row.get("prediction")
                ):
                    invalid_legacy_fallback_rows.append(str(row.get("id")))
        if invalid_raw_referee_rows:
            raise ValueError(
                "raw-SBN-valid rows have no raw Referee decision: "
                + ", ".join(invalid_raw_referee_rows[:10])
            )
        if invalid_legacy_fallback_rows:
            raise ValueError(
                "legacy Referee fallback requires identical raw/postprocessed "
                "surfaces: "
                + ", ".join(invalid_legacy_fallback_rows[:10])
            )
        computed_postprocessed_referee = sum(
            row.get("validity", {}).get("clf_referee") is True
            for row in prediction_rows
        ) / expected_records
        computed_raw_referee = sum(
            row.get("validity", {}).get(
                "raw_sbn", row.get("validity", {}).get("sbn")
            )
            is True
            and row.get("validity", {}).get(
                "raw_clf_referee",
                row.get("validity", {}).get("clf_referee"),
            )
            is True
            for row in prediction_rows
        ) / expected_records
        computed_raw_sbn = sum(
            row.get("validity", {}).get("raw_sbn") is True
            for row in prediction_rows
        ) / expected_records
        observed_postprocessed_referee = _finite_rate(
            group.get("referee_valid_rate_total"),
            "referee_valid_rate_total",
        )
        observed_raw_referee = group.get("raw_referee_valid_rate_total")
        if observed_raw_referee is not None:
            observed_raw_referee = _finite_rate(
                observed_raw_referee, "raw_referee_valid_rate_total"
            )
        metrics = {
            "sbn_smatch_mean": _finite_rate(
                group.get("sbn_smatch_mean"), "sbn_smatch_mean"
            ),
            # The frozen tie-breaker name predates the explicit raw field. Its
            # executable definition is raw structural validity AND the raw
            # Referee-aligned decision, matching the primary method metric.
            "referee_valid_rate_total": computed_raw_referee,
            "postprocessed_referee_valid_rate_total": (
                observed_postprocessed_referee
            ),
            "raw_sbn_valid_rate": _finite_rate(
                group.get("raw_sbn_valid_rate"), "raw_sbn_valid_rate"
            ),
            "truncated": int(group.get("truncated", 0)),
            "eval_loss": eval_loss,
        }
        recomputed = {
            "sbn_smatch_mean": computed_smatch,
            "postprocessed_referee_valid_rate_total": (
                computed_postprocessed_referee
            ),
            "raw_sbn_valid_rate": computed_raw_sbn,
        }
        disagreements = {
            name: {
                "summary": (
                    observed_postprocessed_referee
                    if name == "postprocessed_referee_valid_rate_total"
                    else metrics[name]
                ),
                "recomputed": value,
            }
            for name, value in recomputed.items()
            if not math.isclose(
                (
                    observed_postprocessed_referee
                    if name == "postprocessed_referee_valid_rate_total"
                    else metrics[name]
                ),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        if observed_raw_referee is not None and not math.isclose(
            observed_raw_referee,
            computed_raw_referee,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            disagreements["raw_referee_valid_rate_total"] = {
                "summary": observed_raw_referee,
                "recomputed": computed_raw_referee,
            }
        if disagreements:
            raise ValueError(
                "development summary differs from prediction/score rows: "
                + json.dumps(disagreements, sort_keys=True)
            )
        if metrics["truncated"] < 0 or metrics["truncated"] > expected_records:
            raise ValueError("truncated count is out of range")
        result["metrics"] = metrics
        result["evidence"] = {
            "prediction_records": len(prediction_rows),
            "prediction_sha256": prediction_sha,
            "corpus_path": str(corpus_label),
            "corpus_sha256": _sha256(corpus_path),
            "train_corpus_path": "data/en/train/gold.sbn",
            "train_corpus_sha256": _sha256(train_path),
            "generation_manifest_sha256": _sha256(paths["generation_manifest"]),
            "baseline_generation_contract": generation_contract,
            "training_manifest_contract": model_row[
                "training_manifest_contract"
            ],
            "training_target_modules_manifest": train_manifest["recipe"][
                "target_modules"
            ],
            "summary_sha256": _sha256(paths["summary"]),
            "score_sha256": score_sha,
            "score_manifest_sha256": _sha256(paths["score_manifest"]),
            "train_manifest_sha256": _sha256(paths["train_manifest"]),
            "tokenizer_audit_sha256": _sha256(paths["tokenizer_audit"]),
            "adapter_config_sha256": _sha256(paths["adapter_config"]),
            "adapter_weights_sha256": _sha256(paths["adapter_weights"]),
            "score_manifest": score_manifest,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        result["reasons"].append(str(error))
        return result
    result["eligible"] = True
    return result


def select_lora_host(
    config: dict[str, Any],
    runs_root: Path,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    """Select one model using only the pre-registered development criterion."""

    rule = config.get("host_selection")
    if not isinstance(rule, dict):
        raise ValueError("config has no host_selection object")
    if rule.get("split") != "dev" or rule.get("test_metrics_used") is not False:
        raise ValueError("host selection must be development-only")
    if rule.get("referee_validity_surface") != "raw_generation":
        raise ValueError("host-selection Referee tie-breaker must use raw generation")
    if (
        rule.get("primary_metric") != "sbn_smatch_mean"
        or rule.get("direction") != "max"
    ):
        raise ValueError("unsupported primary host-selection rule")
    expected_ties = [
        "referee_valid_rate_total",
        "raw_sbn_valid_rate",
        "parameters_billions:min",
        "priority:min",
    ]
    if rule.get("tie_breakers") != expected_ties:
        raise ValueError("host-selection tie breakers differ from the frozen rule")
    expected_records = int(rule.get("required_records", 0))
    if expected_records <= 0:
        raise ValueError("required_records must be positive")
    candidates = [
        _candidate(
            row,
            runs_root,
            split="dev",
            expected_records=expected_records,
            repository_root=repository_root,
            shared_recipe=config["shared_recipe"],
        )
        for row in sorted(config["models"], key=lambda row: row["priority"])
    ]
    ineligible = [candidate for candidate in candidates if not candidate["eligible"]]
    if rule.get("require_all_models_complete") is not True:
        raise ValueError("all configured models must be required before selection")
    if ineligible:
        details = "; ".join(
            f"{candidate['key']}: {', '.join(candidate['reasons'])}"
            for candidate in ineligible
        )
        raise ValueError("host selection is not ready: " + details)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate["metrics"]["sbn_smatch_mean"],
            -candidate["metrics"]["referee_valid_rate_total"],
            -candidate["metrics"]["raw_sbn_valid_rate"],
            candidate["parameters_billions"],
            candidate["priority"],
        ),
    )
    return {
        "schema_version": 1,
        "status": "selected",
        "selection_rule": dict(rule),
        "selected_key": ranked[0]["key"],
        "selected_model": ranked[0]["model"],
        "ranking": [candidate["key"] for candidate in ranked],
        "candidates": candidates,
    }


def verify_frozen_lora_host_selection(
    config: dict[str, Any],
    config_bytes: bytes,
    selection: dict[str, Any],
    runs_root: Path,
    repository_root: Path = Path("."),
) -> str:
    """Verify that a stored development-only selection still matches evidence."""

    if selection.get("schema_version") != 1 or selection.get("status") != "selected":
        raise ValueError("unsupported or incomplete frozen host selection")
    expected_config_sha = hashlib.sha256(config_bytes).hexdigest()
    if selection.get("config_sha256") != expected_config_sha:
        raise ValueError("frozen host selection config hash does not match")
    if not isinstance(selection.get("selected_at_utc"), str):
        raise ValueError("frozen host selection has no selection timestamp")
    current = select_lora_host(
        config, runs_root, repository_root=repository_root
    )
    evidence_fields = (
        "selection_rule",
        "selected_key",
        "selected_model",
        "ranking",
        "candidates",
    )
    differences = [
        field for field in evidence_fields if selection.get(field) != current.get(field)
    ]
    if differences:
        raise ValueError(
            "frozen host selection differs from current evidence: "
            + ", ".join(differences)
        )
    return str(current["selected_key"])


__all__ = ["select_lora_host", "verify_frozen_lora_host_selection"]
