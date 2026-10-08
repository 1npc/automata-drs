"""Fail-closed inspection of staged recent-model LoRA baselines."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .causal_lm import iter_sbn_records
from .lora_gate import evaluate_lora_smoke_gate


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


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def _prediction_count(
    path: Path,
    *,
    expected_records: int,
    expected_configuration: str = "U",
    expected_rows: list[Any] | None = None,
    corpus_label: Path | None = None,
) -> int:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank prediction line: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object prediction: {path}:{line_number}")
            identifier = str(row.get("id", ""))
            if not identifier or identifier in seen:
                raise ValueError(
                    f"missing or duplicate prediction id {identifier!r}: "
                    f"{path}:{line_number}"
                )
            seen.add(identifier)
            if row.get("configuration") != expected_configuration:
                raise ValueError(
                    f"unexpected configuration in {path}:{line_number}"
                )
            decode = row.get("decode", {})
            if decode.get("strategy") != "greedy" or decode.get("num_beams") != 1:
                raise ValueError(f"non-greedy prediction: {path}:{line_number}")
            rows.append(row)
    if len(rows) != expected_records:
        raise ValueError(
            f"expected {expected_records} predictions, found {len(rows)}: {path}"
        )
    if expected_rows is not None:
        if corpus_label is None or len(expected_rows) != expected_records:
            raise ValueError("invalid expected-corpus contract")
        for index, (row, expected) in enumerate(
            zip(rows, expected_rows, strict=True), start=1
        ):
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
                    f"prediction row {index} differs from {corpus_label}: "
                    + ", ".join(mismatches)
                )
    return len(rows)


def _validate_tokenizer_audit(
    path: Path, model_row: dict[str, Any], *, max_length: int
) -> None:
    audit = _read_object(path)
    if audit.get("requested_revision") != model_row["revision"]:
        raise ValueError(f"tokenizer audit revision mismatch: {path}")
    if audit.get("target_roundtrip_failure_count") != 0:
        raise ValueError(f"tokenizer audit has round-trip failures: {path}")
    maximum = audit.get("lengths", {}).get("total_max")
    if not isinstance(maximum, int) or maximum > max_length:
        raise ValueError(
            f"tokenizer audit maximum must be <= {max_length}, found {maximum!r}: "
            f"{path}"
        )


def _validate_training(
    run_root: Path,
    audit_path: Path,
    model_row: dict[str, Any],
    *,
    max_length: int,
) -> None:
    required = {
        "tokenizer audit": audit_path,
        "training manifest": run_root / "train_manifest.json",
        "training metrics": run_root / "train_metrics.json",
        "adapter config": run_root / "adapter" / "adapter_config.json",
        "adapter weights": run_root / "adapter" / "adapter_model.safetensors",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError("incomplete training artifacts: " + "; ".join(missing))
    _validate_tokenizer_audit(audit_path, model_row, max_length=max_length)
    manifest = _read_object(required["training manifest"])
    if manifest.get("status") != "complete":
        raise ValueError(f"training manifest is not complete: {required['training manifest']}")
    if manifest.get("model") != model_row["repo_id"]:
        raise ValueError(f"training model mismatch: {required['training manifest']}")
    if manifest.get("requested_revision") != model_row["revision"]:
        raise ValueError(f"training revision mismatch: {required['training manifest']}")
    metrics = _read_object(required["training metrics"])
    _finite(metrics.get("train", {}).get("train_loss"), "train_loss")
    _finite(metrics.get("eval", {}).get("eval_loss"), "eval_loss")
    if required["adapter weights"].stat().st_size <= 0:
        raise ValueError(f"empty adapter weights: {required['adapter weights']}")


def _artifact_pair(path: Path) -> tuple[Path, Path]:
    return path, Path(str(path) + ".manifest.json")


def _validate_generation(
    path: Path,
    model_row: dict[str, Any],
    *,
    expected_records: int,
    expected_rows: list[Any] | None = None,
    corpus_label: Path | None = None,
) -> None:
    prediction_path, manifest_path = _artifact_pair(path)
    present = [prediction_path.is_file(), manifest_path.is_file()]
    if any(present) and not all(present):
        raise ValueError(
            "prediction and manifest must either both exist or both be absent: "
            f"{prediction_path}"
        )
    if not all(present):
        raise FileNotFoundError(str(prediction_path))
    _prediction_count(
        prediction_path,
        expected_records=expected_records,
        expected_rows=expected_rows,
        corpus_label=corpus_label,
    )
    manifest = _read_object(manifest_path)
    if manifest.get("model") != model_row["repo_id"]:
        raise ValueError(f"generation model mismatch: {manifest_path}")
    if manifest.get("requested_revision") != model_row["revision"]:
        raise ValueError(f"generation revision mismatch: {manifest_path}")
    if manifest.get("configurations") != ["U"]:
        raise ValueError(f"generation is not U-only: {manifest_path}")
    if manifest.get("num_beams") != 1:
        raise ValueError(f"generation is not greedy: {manifest_path}")
    if corpus_label is not None:
        expected_contract = {
            "path": str(corpus_label),
            "start": 0,
            "limit": -1,
            "line_numbers": None,
            "schedule": "rotate",
        }
        split = next(
            (
                name
                for name, relative_path in SPLIT_PATHS.items()
                if relative_path == corpus_label
            ),
            None,
        )
        split_contracts = model_row.get(
            "baseline_generation_split_contracts", {}
        )
        generation_contract = split_contracts.get(
            split,
            model_row.get(
                "baseline_generation_contract", "batch8-unscored-v1"
            ),
        )
        if generation_contract == "batch8-unscored-v1":
            expected_contract.update(
                {"batch_size": 8, "score_model_logprob": False}
            )
        elif generation_contract == "legacy-sequential-unscored-v0":
            expected_contract.update(
                {"batch_size": None, "score_model_logprob": None}
            )
        elif generation_contract == "legacy-sequential-scored-v1":
            expected_contract.update(
                {"batch_size": 1, "score_model_logprob": True}
            )
        else:
            raise ValueError(
                "unsupported baseline generation contract: "
                f"{generation_contract!r}"
            )
        mismatches = [
            name
            for name, expected in expected_contract.items()
            if manifest.get(name) != expected
        ]
        if mismatches:
            raise ValueError(
                f"generation manifest differs from full {corpus_label} "
                f"contract: {', '.join(mismatches)}"
            )


def _score_paths(prediction: Path) -> dict[str, Path]:
    prefix = prediction.with_suffix("")
    scores = Path(str(prefix) + ".smatch.jsonl")
    return {
        "scores": scores,
        "score_manifest": Path(str(scores) + ".manifest.json"),
        "summary_markdown": Path(str(prefix) + ".summary.md"),
        "summary_json": Path(str(prefix) + ".summary.json"),
        "failures": Path(str(prefix) + ".failures.json"),
    }


def _validate_scores(prediction: Path, *, expected_records: int) -> None:
    paths = _score_paths(prediction)
    present = {name: path.is_file() for name, path in paths.items()}
    if any(present.values()) and not all(present.values()):
        details = ", ".join(
            f"{name}={'present' if exists else 'missing'}"
            for name, exists in present.items()
        )
        raise ValueError(f"partial score artifacts for {prediction}: {details}")
    if not all(present.values()):
        raise FileNotFoundError(str(prediction))

    prediction_rows = []
    with prediction.open(encoding="utf-8") as source:
        prediction_rows = [json.loads(line) for line in source if line.strip()]
    score_rows = []
    with paths["scores"].open(encoding="utf-8") as source:
        score_rows = [json.loads(line) for line in source if line.strip()]
    if len(score_rows) != expected_records:
        raise ValueError(
            f"expected {expected_records} scores, found {len(score_rows)}: "
            f"{paths['scores']}"
        )
    if len(prediction_rows) != expected_records:
        raise ValueError(
            f"expected {expected_records} scored predictions, found "
            f"{len(prediction_rows)}: {prediction}"
        )
    for line_number, (prediction_row, score_row) in enumerate(
        zip(prediction_rows, score_rows, strict=True), start=1
    ):
        score = score_row.get("semantic_scores", {}).get("sbn_smatch")
        if (
            score_row.get("id") != prediction_row.get("id")
            or score_row.get("configuration") != prediction_row.get("configuration")
            or score_row.get("decode") != prediction_row.get("decode")
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(
                f"score/prediction identity or value mismatch: "
                f"{paths['scores']}:{line_number}"
            )
    manifest = _read_object(paths["score_manifest"])
    if manifest.get("input_sha256") != _sha256(prediction):
        raise ValueError(f"score input hash mismatch: {paths['score_manifest']}")
    if manifest.get("output_sha256") != _sha256(paths["scores"]):
        raise ValueError(f"score output hash mismatch: {paths['score_manifest']}")
    if int(manifest.get("records", -1)) != expected_records:
        raise ValueError(f"score record count mismatch: {paths['score_manifest']}")
    expected_groups = [
        {"configuration": "U", "strategy": "greedy", "records": expected_records}
    ]
    if manifest.get("groups") != expected_groups:
        raise ValueError(f"score groups mismatch: {paths['score_manifest']}")
    summary = _read_object(paths["summary_json"])
    groups = [
        row
        for row in summary.get("groups", [])
        if row.get("configuration") == "U" and row.get("strategy") == "greedy"
    ]
    if len(groups) != 1 or int(groups[0].get("total", -1)) != expected_records:
        raise ValueError(f"summary group mismatch: {paths['summary_json']}")
    group = groups[0]
    raw_sbn_valid = sum(
        row.get("validity", {}).get(
            "raw_sbn", row.get("validity", {}).get("sbn")
        )
        is True
        for row in prediction_rows
    )
    referee_scored = sum(
        row.get("validity", {}).get("clf_referee") is not None
        for row in prediction_rows
    )
    referee_valid = sum(
        row.get("validity", {}).get("clf_referee") is True
        for row in prediction_rows
    )
    raw_referee_scored = sum(
        row.get("validity", {}).get(
            "raw_sbn", row.get("validity", {}).get("sbn")
        )
        is True
        and row.get("validity", {}).get(
            "raw_clf_referee",
            row.get("validity", {}).get("clf_referee"),
        )
        is not None
        for row in prediction_rows
    )
    raw_referee_valid = sum(
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
    )
    expected_summary = {
        "raw_sbn_valid": raw_sbn_valid,
        "referee_scored": referee_scored,
        "referee_valid": referee_valid,
        "exact_serialization": sum(
            row.get("prediction") == row.get("reference")
            for row in prediction_rows
        ),
        "truncated": sum(
            row.get("generation", {}).get("hit_max_new_tokens") is True
            for row in prediction_rows
        ),
        "canonical_tokenization_scored": sum(
            row.get("generation", {}).get("canonical_tokenization") is not None
            for row in prediction_rows
        ),
        "canonical_tokenization": sum(
            row.get("generation", {}).get("canonical_tokenization") is True
            for row in prediction_rows
        ),
    }
    if "raw_referee_scored" in group or "raw_referee_valid" in group:
        expected_summary.update(
            {
                "raw_referee_scored": raw_referee_scored,
                "raw_referee_valid": raw_referee_valid,
            }
        )
    mismatches = {
        name: {"observed": group.get(name), "recomputed": expected}
        for name, expected in expected_summary.items()
        if group.get(name) != expected
    }
    computed_smatch = sum(
        float(row["semantic_scores"]["sbn_smatch"])
        for row in score_rows
    ) / expected_records
    observed_smatch = _finite(
        group.get("sbn_smatch_mean"), "summary sbn_smatch_mean"
    )
    if not math.isclose(
        observed_smatch, computed_smatch, rel_tol=0.0, abs_tol=1e-12
    ):
        mismatches["sbn_smatch_mean"] = {
            "observed": observed_smatch,
            "recomputed": computed_smatch,
        }
    if mismatches:
        raise ValueError(
            f"summary differs from prediction/score rows: {paths['summary_json']}: "
            + json.dumps(mismatches, sort_keys=True)
        )


def verify_complete_lora_baseline_outputs(
    config: dict[str, Any], runs_root: Path, repository_root: Path
) -> dict[str, Any]:
    """Verify every full-split baseline row, score, summary, and source binding."""

    evidence: dict[str, dict[str, dict[str, str]]] = {}
    for model_row in sorted(config["models"], key=lambda row: row["priority"]):
        key = str(model_row["key"])
        run_root = runs_root / f"{key}-r16-seed42"
        model_evidence: dict[str, dict[str, str]] = {}
        for split, corpus_label in SPLIT_PATHS.items():
            expected_rows = list(
                iter_sbn_records(repository_root / corpus_label)
            )
            prediction = run_root / f"{split}-u-greedy.jsonl"
            _validate_generation(
                prediction,
                model_row,
                expected_records=len(expected_rows),
                expected_rows=expected_rows,
                corpus_label=corpus_label,
            )
            _validate_scores(prediction, expected_records=len(expected_rows))
            paths = _score_paths(prediction)
            model_evidence[split] = {
                "corpus_sha256": _sha256(repository_root / corpus_label),
                "prediction_sha256": _sha256(prediction),
                "generation_manifest_sha256": _sha256(
                    Path(str(prediction) + ".manifest.json")
                ),
                "score_sha256": _sha256(paths["scores"]),
                "score_manifest_sha256": _sha256(paths["score_manifest"]),
                "summary_sha256": _sha256(paths["summary_json"]),
            }
        evidence[key] = model_evidence
    return {
        "schema_version": 1,
        "passed": True,
        "models": evidence,
    }


def _smoke_report(
    model_row: dict[str, Any], runs_root: Path, *, max_length: int
) -> dict[str, Any]:
    run_root = runs_root / f"{model_row['key']}-smoke"
    audit_path = Path(str(run_root) + ".tokenizer-audit.json")
    gate_path = run_root / "smoke_gate.json"
    if not run_root.exists() and not audit_path.exists() and not gate_path.exists():
        return {"state": "absent", "reasons": [], "run_root": str(run_root)}
    try:
        computed = evaluate_lora_smoke_gate(
            model_row, run_root, max_length=max_length
        )
        if not computed["passed"]:
            failed = [
                item["name"] for item in computed.get("checks", []) if not item["passed"]
            ]
            missing = computed.get("missing_artifacts", [])
            details = missing or failed or ["unknown smoke-gate failure"]
            raise ValueError("smoke gate failed: " + ", ".join(details))
        if not gate_path.exists():
            return {
                "state": "ready_for_gate",
                "reasons": [],
                "run_root": str(run_root),
            }
        stored = _read_object(gate_path)
        if stored != computed:
            raise ValueError(f"stored smoke gate differs from recomputation: {gate_path}")
        return {"state": "complete", "reasons": [], "run_root": str(run_root)}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "state": "invalid",
            "reasons": [str(error)],
            "run_root": str(run_root),
        }


def _full_report(
    model_row: dict[str, Any],
    runs_root: Path,
    repository_root: Path,
    *,
    max_length: int,
) -> dict[str, Any]:
    run_root = runs_root / f"{model_row['key']}-r16-seed42"
    audit_path = Path(str(run_root) + ".tokenizer-audit.json")
    if not run_root.exists() and not audit_path.exists():
        return {
            "state": "absent",
            "reasons": [],
            "run_root": str(run_root),
            "missing_generations": ["dev20", "dev", "test", "tax"],
            "missing_scores": ["dev", "test", "tax"],
        }
    try:
        _validate_training(
            run_root,
            audit_path,
            model_row,
            max_length=max_length,
        )
        corpus_rows = {
            split: list(iter_sbn_records(repository_root / relative_path))
            for split, relative_path in SPLIT_PATHS.items()
        }
        corpus_counts = {
            split: len(rows) for split, rows in corpus_rows.items()
        }
        generations = {
            "dev20": (run_root / "dev20-u-sequential-greedy.jsonl", 20),
            "dev": (run_root / "dev-u-greedy.jsonl", corpus_counts["dev"]),
            "test": (run_root / "test-u-greedy.jsonl", corpus_counts["test"]),
            "tax": (run_root / "tax-u-greedy.jsonl", corpus_counts["tax"]),
        }
        missing_generations: list[str] = []
        for name, (path, expected) in generations.items():
            try:
                if name == "dev20":
                    _validate_generation(
                        path, model_row, expected_records=expected
                    )
                else:
                    _validate_generation(
                        path,
                        model_row,
                        expected_records=expected,
                        expected_rows=corpus_rows[name],
                        corpus_label=SPLIT_PATHS[name],
                    )
            except FileNotFoundError:
                missing_generations.append(name)
        if len(missing_generations) == len(generations):
            state = "training_complete"
        elif missing_generations:
            state = "generation_partial"
        else:
            state = "unscored"

        missing_scores: list[str] = []
        if not missing_generations:
            for split in ("dev", "test", "tax"):
                path, expected = generations[split]
                try:
                    _validate_scores(path, expected_records=expected)
                except FileNotFoundError:
                    missing_scores.append(split)
            if not missing_scores:
                state = "complete"
            elif len(missing_scores) < 3:
                state = "scoring_partial"
        return {
            "state": state,
            "reasons": [],
            "run_root": str(run_root),
            "corpus_counts": corpus_counts,
            "missing_generations": missing_generations,
            "missing_scores": missing_scores,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "state": "invalid",
            "reasons": [str(error)],
            "run_root": str(run_root),
            "missing_generations": [],
            "missing_scores": [],
        }


def inspect_lora_baseline(
    config: dict[str, Any],
    model_key: str,
    runs_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Inspect one baseline and identify its next safe idempotent action."""

    matches = [row for row in config["models"] if row["key"] == model_key]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate model key: {model_key}")
    model_row = matches[0]
    max_length = int(config["shared_recipe"]["max_length"])
    smoke = _smoke_report(model_row, runs_root, max_length=max_length)
    full = _full_report(
        model_row,
        runs_root,
        repository_root,
        max_length=max_length,
    )
    promotion_violation = (
        full["state"] != "absent" and smoke["state"] != "complete"
    )
    if promotion_violation:
        smoke["reasons"].append(
            "full run exists without a complete stored smoke promotion gate"
        )
        next_action = "inspect_invalid_artifacts"
        resumable = False
    elif smoke["state"] == "invalid" or full["state"] == "invalid":
        next_action = "inspect_invalid_artifacts"
        resumable = False
    elif smoke["state"] == "absent":
        next_action = "run_smoke"
        resumable = True
    elif smoke["state"] == "ready_for_gate":
        next_action = "write_smoke_gate"
        resumable = True
    elif full["state"] == "absent":
        next_action = "run_full"
        resumable = True
    elif full["state"] in {"training_complete", "generation_partial"}:
        next_action = "resume_full_generation"
        resumable = True
    elif full["state"] in {"unscored", "scoring_partial"}:
        next_action = "finalize_missing_scores"
        resumable = True
    elif full["state"] == "complete":
        next_action = "done"
        resumable = True
    else:
        raise AssertionError(f"unhandled LoRA state: {smoke['state']}, {full['state']}")
    return {
        "schema_version": 1,
        "model_key": model_key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "smoke": smoke,
        "full": full,
        "resumable": resumable,
        "next_action": next_action,
    }


__all__ = [
    "SPLIT_PATHS",
    "inspect_lora_baseline",
    "verify_complete_lora_baseline_outputs",
]
