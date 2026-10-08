"""Integrity gate for the frozen selected-host constrained-decoding pilot."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .paired_analysis import paired_report_mismatches
from .sbn import SBNPolicy, validate_sbn


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _evaluate_lora_matrix(
    model_row: dict[str, Any],
    runs_root: Path,
    *,
    split: str,
    expected_records: int,
    matrix_path: Path,
    max_new_tokens: int,
    expected_manifest_limit: int,
    performance_threshold_used: bool,
    paired_parameters: dict[str, Any],
    require_frozen_u_surface: bool,
) -> dict[str, Any]:
    """Evaluate one scored U/BF/WF matrix and its registered U control."""

    configurations = ("U", "BF", "WF")

    key = str(model_row["key"])
    run_root = runs_root / f"{key}-r16-seed42"
    baseline_path = run_root / f"{split}-u-greedy.jsonl"
    matrix_manifest_path = Path(str(matrix_path) + ".manifest.json")
    prefix = matrix_path.with_suffix("")
    scores_path = Path(str(prefix) + ".smatch.jsonl")
    score_manifest_path = Path(str(scores_path) + ".manifest.json")
    summary_path = Path(str(prefix) + ".summary.json")
    paired_path = Path(str(prefix) + ".paired-statistics.json")
    required_paths = (
        baseline_path,
        matrix_path,
        matrix_manifest_path,
        scores_path,
        score_manifest_path,
        summary_path,
        paired_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "model_key": key,
            "run_root": str(run_root),
            "passed": False,
            "missing_artifacts": missing,
            "checks": [],
        }

    baseline_rows = _read_jsonl(baseline_path)
    matrix_rows = _read_jsonl(matrix_path)
    manifest = _read_object(matrix_manifest_path)
    score_manifest = _read_object(score_manifest_path)
    score_rows = _read_jsonl(scores_path)
    summary = _read_object(summary_path)
    paired = _read_object(paired_path)
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

    baseline_u = [
        row
        for row in baseline_rows
        if row.get("configuration") == "U"
        and row.get("decode", {}).get("strategy") == "greedy"
        and row.get("decode", {}).get("num_beams") == 1
    ][:expected_records]
    baseline_ids = [str(row.get("id", "")) for row in baseline_u]
    check(
        "baseline_population",
        len(baseline_u) == expected_records
        and len(set(baseline_ids)) == expected_records
        and all(baseline_ids),
        {"records": len(baseline_u), "unique_ids": len(set(baseline_ids))},
        {"records": expected_records, "unique_ids": expected_records},
    )

    expected_total = expected_records * len(configurations)
    identities = [
        (str(row.get("id", "")), str(row.get("configuration", "")))
        for row in matrix_rows
    ]
    counts = Counter(configuration for _, configuration in identities)
    expected_counts = {configuration: expected_records for configuration in configurations}
    check(
        "prediction_matrix",
        len(matrix_rows) == expected_total
        and len(set(identities)) == expected_total
        and dict(counts) == expected_counts,
        {
            "records": len(matrix_rows),
            "unique_cells": len(set(identities)),
            "configurations": dict(counts),
        },
        {
            "records": expected_total,
            "unique_cells": expected_total,
            "configurations": expected_counts,
        },
    )
    matrix_ids = {
        configuration: {
            identifier for identifier, observed in identities if observed == configuration
        }
        for configuration in configurations
    }
    check(
        "paired_population",
        all(matrix_ids[configuration] == set(baseline_ids) for configuration in configurations),
        {configuration: len(ids) for configuration, ids in matrix_ids.items()},
        {configuration: expected_records for configuration in configurations},
    )

    u_by_id = {
        str(row.get("id")): row
        for row in matrix_rows
        if row.get("configuration") == "U"
    }
    baseline_by_id = {str(row.get("id")): row for row in baseline_u}
    u_differences = [
        identifier
        for identifier in baseline_ids
        if identifier not in u_by_id
        or baseline_by_id[identifier].get("raw_prediction")
        != u_by_id[identifier].get("raw_prediction")
    ]
    if require_frozen_u_surface:
        check("u_surface_equivalence", not u_differences, u_differences, [])
    else:
        u_processor_failures = [
            str(row.get("id"))
            for row in matrix_rows
            if row.get("configuration") == "U"
            and row.get("constraint") is not None
        ]
        check(
            "u_within_matrix_control",
            not u_processor_failures,
            {
                "contract": "batch-1 greedy without logits processor",
                "processor_failures": u_processor_failures,
                "legacy_cross_contract_surface_differences": u_differences,
            },
            {
                "contract": "batch-1 greedy without logits processor",
                "processor_failures": [],
                "legacy_cross_contract_surface_differences": "diagnostic-only",
            },
        )

    decode_failures = [
        f"{row.get('id')}:{row.get('configuration')}"
        for row in matrix_rows
        if row.get("decode", {}).get("strategy") != "greedy"
        or row.get("decode", {}).get("num_beams") != 1
    ]
    check("greedy_decode", not decode_failures, decode_failures, [])
    all_masked_rows = sum(
        int(row.get("constraint", {}).get("all_masked_rows", 0))
        for row in matrix_rows
        if row.get("constraint") is not None
    )
    check("all_masked_rows", all_masked_rows == 0, all_masked_rows, 0)

    # BF/WF are not merely labels for valid completed surfaces: the claimed
    # finite-horizon intervention requires the constructive certificate to be
    # active on every nonterminal logits row.  Completed-output validation
    # alone cannot distinguish BF/WF from an accidental B/W run, so bind the
    # gate to the processor's independently recorded activity counters.
    bounded_activity_failures: list[dict[str, Any]] = []
    for row in matrix_rows:
        configuration = row.get("configuration")
        if configuration not in {"BF", "WF"}:
            continue
        constraint = row.get("constraint")
        identifier = f"{row.get('id')}:{configuration}"
        if not isinstance(constraint, dict):
            bounded_activity_failures.append(
                {"cell": identifier, "reason": "missing constraint statistics"}
            )
            continue
        integer_fields = (
            "calls",
            "rows",
            "bounded_rows",
            "candidates_tested",
            "candidates_accepted",
            "allowed_total",
            "all_masked_rows",
            "completion_certificates_tested",
            "completion_certificates_accepted",
        )
        values: dict[str, int] = {}
        invalid_fields: list[str] = []
        for name in integer_fields:
            value = constraint.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                invalid_fields.append(name)
            else:
                values[name] = value
        if invalid_fields:
            bounded_activity_failures.append(
                {
                    "cell": identifier,
                    "reason": "invalid nonnegative integer counters",
                    "fields": invalid_fields,
                }
            )
            continue
        rows = values["rows"]
        activity_ok = (
            rows > 0
            and values["calls"] == rows
            and values["bounded_rows"] == rows
            and values["all_masked_rows"] == 0
            and values["allowed_total"] == rows
            and values["candidates_tested"]
            >= values["candidates_accepted"]
            >= values["completion_certificates_tested"]
            >= values["completion_certificates_accepted"]
            == values["allowed_total"]
        )
        if not activity_ok:
            bounded_activity_failures.append(
                {
                    "cell": identifier,
                    "reason": "finite-horizon counters are inconsistent",
                    "observed": values,
                }
            )
    check(
        "bounded_completion_activity",
        not bounded_activity_failures,
        bounded_activity_failures,
        [],
    )
    truncated = [
        f"{row.get('id')}:{row.get('configuration')}"
        for row in matrix_rows
        if row.get("generation", {}).get("hit_max_new_tokens") is True
        or row.get("generation", {}).get("ended_with_eos") is not True
    ]
    check("bounded_completion", not truncated, truncated, [])

    policy_failures: list[str] = []
    wf_semantic_failures: list[str] = []
    for row in matrix_rows:
        configuration = row.get("configuration")
        if configuration not in {"BF", "WF"}:
            continue
        base_configuration = "B" if configuration == "BF" else "W"
        result = validate_sbn(
            str(row.get("raw_prediction", "")),
            SBNPolicy.for_configuration(base_configuration),
        )
        if not result.valid or row.get("validity", {}).get("raw_sbn") is not True:
            policy_failures.append(f"{row.get('id')}:{configuration}")
        if (
            configuration == "WF"
            and row.get("validity", {}).get(
                "raw_clf_referee",
                row.get("validity", {}).get("clf_referee"),
            )
            is not True
        ):
            wf_semantic_failures.append(str(row.get("id")))
    check("bounded_policy_acceptance", not policy_failures, policy_failures, [])
    check(
        "wf_semantic_wellformedness",
        not wf_semantic_failures,
        wf_semantic_failures,
        [],
    )

    score_failures = [
        f"{row.get('id')}:{row.get('configuration')}"
        for row in matrix_rows
        if not _is_finite(row.get("score", {}).get("raw_model_logprob"))
        or not _is_finite(row.get("score", {}).get("raw_model_mean_logprob"))
        or int(row.get("score", {}).get("scored_tokens", 0)) <= 0
    ]
    check("finite_unmasked_logprob", not score_failures, score_failures, [])
    timing_failures = [
        f"{row.get('id')}:{row.get('configuration')}"
        for row in matrix_rows
        if any(
            not _is_finite(row.get("timing", {}).get(name))
            or float(row.get("timing", {}).get(name)) < 0
            for name in ("generation_ms", "model_score_ms", "constraint_ms")
        )
    ]
    check("finite_nonnegative_timings", not timing_failures, timing_failures, [])

    manifest_expected = {
        "status": "complete",
        "model": model_row["repo_id"],
        "requested_revision": model_row["revision"],
        "limit": expected_manifest_limit,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "batch_size": 1,
        "configurations": list(configurations),
        "completion_budget_factor": 1.0,
        "score_model_logprob": True,
        "constraint_lifting": "exact-bytelevel-bpe-score-ranked",
        "schedule": "rotate",
        "expected_examples": expected_records,
        "expected_rows": expected_total,
        "examples": expected_records,
        "rows": expected_total,
        "output_sha256": _sha256(matrix_path),
    }
    manifest_mismatches = {
        name: {"observed": manifest.get(name), "expected": expected}
        for name, expected in manifest_expected.items()
        if manifest.get(name) != expected
    }
    check("generation_manifest", not manifest_mismatches, manifest_mismatches, {})
    gpu_index = manifest.get("physical_gpu_index")
    gpu_name = manifest.get("cuda_device_name")
    gpu_total_bytes = manifest.get("cuda_device_total_bytes")
    required_gpu_bytes = int(model_row.get("safetensors_bytes", 0)) + 6 * 1024**3
    hardware_ok = (
        gpu_index in {"0", "1"}
        and isinstance(gpu_name, str)
        and bool(gpu_name.strip())
        and isinstance(gpu_total_bytes, int)
        and not isinstance(gpu_total_bytes, bool)
        and gpu_total_bytes >= required_gpu_bytes > 6 * 1024**3
    )
    check(
        "generation_hardware",
        hardware_ok,
        {
            "physical_gpu_index": gpu_index,
            "cuda_device_name": gpu_name,
            "cuda_device_total_bytes": gpu_total_bytes,
        },
        {
            "physical_gpu_index": "0 or 1",
            "cuda_device_name": "non-empty",
            "cuda_device_total_bytes_at_least": required_gpu_bytes,
        },
    )

    score_identities = {
        (str(row.get("id", "")), str(row.get("configuration", "")))
        for row in score_rows
    }
    check(
        "semantic_score_matrix",
        len(score_rows) == expected_total and score_identities == set(identities),
        {"records": len(score_rows), "unique_cells": len(score_identities)},
        {"records": expected_total, "unique_cells": expected_total},
    )
    expected_score_groups = {
        (configuration, "greedy", expected_records)
        for configuration in configurations
    }
    observed_score_groups = {
        (row.get("configuration"), row.get("strategy"), row.get("records"))
        for row in score_manifest.get("groups", [])
    }
    score_manifest_ok = (
        score_manifest.get("input_sha256") == _sha256(matrix_path)
        and score_manifest.get("output_sha256") == _sha256(scores_path)
        and score_manifest.get("records") == expected_total
        and observed_score_groups == expected_score_groups
    )
    check(
        "semantic_score_manifest",
        score_manifest_ok,
        {
            "records": score_manifest.get("records"),
            "groups": sorted(observed_score_groups),
        },
        {"records": expected_total, "groups": sorted(expected_score_groups)},
    )

    groups = {
        str(row.get("configuration")): row
        for row in summary.get("groups", [])
        if row.get("strategy") == "greedy"
    }
    pairs = {
        str(row.get("configuration")): row
        for row in summary.get("paired_vs_u", [])
        if row.get("strategy") == "greedy"
    }
    summary_ok = set(groups) == set(configurations) and set(pairs) == {"BF", "WF"}
    if summary_ok:
        summary_ok = all(
            int(groups[configuration].get("total", -1)) == expected_records
            and _is_finite(groups[configuration].get("sbn_smatch_mean"))
            for configuration in configurations
        ) and all(
            int(pairs[configuration].get("paired", -1)) == expected_records
            and int(pairs[configuration].get("missing_unconstrained", -1)) == 0
            for configuration in ("BF", "WF")
        )
    check(
        "paired_summary",
        summary_ok,
        {"groups": sorted(groups), "pairs": sorted(pairs)},
        {"groups": list(configurations), "pairs": ["BF", "WF"]},
    )
    paired_mismatches = paired_report_mismatches(
        paired,
        expected_total_records=expected_total,
        expected_configurations=("BF", "WF"),
        expected_records_per_comparison=expected_records,
        expected_population_scope="fixed-without-u-outcome-selection",
        prediction_sha256=_sha256(matrix_path),
        scores_sha256=_sha256(scores_path),
        registered_parameters=paired_parameters,
    )
    check(
        "paired_statistics",
        not paired_mismatches,
        paired_mismatches,
        [],
    )
    check(
        "no_performance_promotion_threshold",
        performance_threshold_used is False,
        performance_threshold_used,
        False,
    )

    return {
        "schema_version": 1,
        "model_key": key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "run_root": str(run_root),
        "passed": all(item["passed"] for item in checks),
        "missing_artifacts": [],
        "checks": checks,
        "evidence": {
            "baseline_sha256": _sha256(baseline_path),
            "matrix_sha256": _sha256(matrix_path),
            "matrix_manifest_sha256": _sha256(matrix_manifest_path),
            "scores_sha256": _sha256(scores_path),
            "score_manifest_sha256": _sha256(score_manifest_path),
            "summary_sha256": _sha256(summary_path),
            "paired_statistics_sha256": _sha256(paired_path),
            "records_per_configuration": expected_records,
            "performance_threshold_used": performance_threshold_used,
        },
    }


def verify_frozen_lora_method_pilot(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
) -> None:
    """Verify a stored pilot gate against recomputed artifacts and hashes."""

    if stored.get("schema_version") == 2:
        _verify_replay_corrected_method_pilot(
            current,
            stored,
            config_sha256=config_sha256,
            selection_sha256=selection_sha256,
        )
        return
    if current.get("passed") is not True:
        raise ValueError("current method pilot evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored method pilot gate is unsupported or did not pass")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored method pilot gate has no timestamp")
    # Normalize tuples and other JSON-compatible containers exactly as the
    # persisted gate was serialized before comparing immutable evidence.
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected["config_sha256"] = config_sha256
    expected["selection_sha256"] = selection_sha256
    observed = {
        key: value for key, value in stored.items() if key != "checked_at_utc"
    }
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "stored method pilot gate differs from current evidence: "
            + ", ".join(differing)
        )


def _verify_replay_corrected_method_pilot(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
) -> None:
    """Verify the explicit v2 repair for a batch-sensitive U baseline.

    The legacy gate remains failed evidence.  The correction is accepted only
    for one U-equivalence failure and an immutable, independently checked
    batch-1 replay; no other integrity failure can be waived.
    """

    if (
        stored.get("kind") != "selected-lora-method-pilot-gate"
        or stored.get("passed") is not True
        or not isinstance(stored.get("checked_at_utc"), str)
    ):
        raise ValueError("stored replay-corrected pilot gate is unsupported")
    normalized_current = json.loads(json.dumps(current, ensure_ascii=False))
    normalized_stored = json.loads(json.dumps(stored, ensure_ascii=False))
    failed = [
        check
        for check in current.get("checks", [])
        if check.get("passed") is not True
    ]
    if (
        current.get("passed") is not False
        or len(failed) != 1
        or failed[0].get("name") != "u_surface_equivalence"
    ):
        raise ValueError(
            "replay correction requires exactly one U-equivalence failure"
        )
    retained = [
        check
        for check in normalized_current["checks"]
        if check.get("name") != "u_surface_equivalence"
    ]
    if normalized_stored.get("retained_v1_checks") != retained or not all(
        check.get("passed") is True for check in retained
    ):
        raise ValueError("replay-corrected pilot did not retain every v1 check")
    correction = stored.get("protocol_correction", {})
    expected_correction = {
        "original_gate_passed": False,
        "only_original_failure": "u_surface_equivalence",
        "cause": "legacy batch-8 versus registered batch-1 execution-contract sensitivity",
        "corrected_control": "independent U-only batch-1 exact replay",
        "performance_threshold_used": False,
        "task_metrics_used_for_acceptance": False,
        "batch_sensitive_ids": failed[0].get("observed"),
    }
    if correction != expected_correction:
        raise ValueError("replay-corrected pilot protocol record changed")

    evidence = stored.get("evidence", {})
    for name, expected in current.get("evidence", {}).items():
        if evidence.get(name) != expected:
            raise ValueError(f"replay-corrected pilot evidence changed: {name}")
    if (
        evidence.get("config_sha256") != config_sha256
        or evidence.get("selection_sha256") != selection_sha256
    ):
        raise ValueError("replay-corrected pilot config or selection changed")
    external = (
        ("failed_v1_gate_path", "failed_v1_gate_sha256"),
        ("replay_plan_path", "replay_plan_sha256"),
        ("replay_audit_path", "replay_audit_sha256"),
    )
    for path_name, digest_name in external:
        path_value = evidence.get(path_name)
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"replay-corrected pilot has no {path_name}")
        path = Path(path_value)
        if not path.is_file() or evidence.get(digest_name) != _sha256(path):
            raise ValueError(f"replay-corrected pilot source drifted: {path_name}")

    plan = _read_object(Path(evidence["replay_plan_path"]))
    audit = _read_object(Path(evidence["replay_audit_path"]))
    failed_gate = _read_object(Path(evidence["failed_v1_gate_path"]))
    failed_core = {
        name: value
        for name, value in failed_gate.items()
        if name not in {"checked_at_utc", "config_sha256", "selection_sha256"}
    }
    if failed_core != normalized_current:
        raise ValueError("replay-corrected pilot v1 failure no longer reproduces")
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "selected-u-control-replay-plan"
        or plan.get("status") != "frozen"
        or plan.get("replay", {}).get("performance_threshold_used") is not False
        or plan.get("observed_batch_sensitive_ids") != failed[0].get("observed")
    ):
        raise ValueError("replay-corrected pilot plan is unsupported")
    audit_evidence = audit.get("evidence", {})
    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != "selected-u-control-replay-audit"
        or audit.get("status") != "complete"
        or audit.get("passed") is not True
        or audit.get("identity_failures") != []
        or audit.get("surface_failures") != []
        or audit.get("performance_threshold_used") is not False
        or audit.get("batch_sensitive_ids") != failed[0].get("observed")
        or audit_evidence.get("plan_path") != evidence["replay_plan_path"]
        or audit_evidence.get("plan_sha256") != evidence["replay_plan_sha256"]
        or audit_evidence.get("pilot_matrix_sha256")
        != evidence.get("matrix_sha256")
        or audit_evidence.get("replay_sha256") != evidence.get("replay_sha256")
        or audit_evidence.get("replay_manifest_sha256")
        != evidence.get("replay_manifest_sha256")
    ):
        raise ValueError("replay-corrected pilot audit is incomplete or changed")
    replay_path_value = audit_evidence.get("replay_path")
    if not isinstance(replay_path_value, str) or not replay_path_value:
        raise ValueError("replay-corrected pilot audit has no replay path")
    replay_path = Path(replay_path_value)
    replay_manifest_path = Path(str(replay_path) + ".manifest.json")
    if (
        not replay_path.is_file()
        or not replay_manifest_path.is_file()
        or _sha256(replay_path) != evidence.get("replay_sha256")
        or _sha256(replay_manifest_path)
        != evidence.get("replay_manifest_sha256")
    ):
        raise ValueError("replay-corrected pilot replay evidence drifted")


def evaluate_lora_method_pilot(
    config: dict[str, Any],
    model_row: dict[str, Any],
    runs_root: Path,
) -> dict[str, Any]:
    """Return a pilot pass/fail report without using a performance threshold."""

    method = config.get("method_experiment", {})
    pilot = method.get("pilot", {})
    expected_requirements = {
        "u_surface_equivalent_to_frozen_baseline": True,
        "bounded_outputs_accepting_under_policy": True,
        "wf_semantic_wellformedness": True,
        "all_masked_rows": 0,
        "truncated_rows": 0,
        "finite_unmasked_logprob_for_every_output": True,
        "performance_threshold_used": False,
    }
    if pilot.get("requirements") != expected_requirements:
        raise ValueError("method pilot requirements differ from the frozen gate")
    if pilot.get("split") != "dev" or pilot.get("selection") != "first-corpus-order":
        raise ValueError("unsupported method pilot population")
    configurations = tuple(pilot.get("configurations", []))
    if configurations != ("U", "BF", "WF"):
        raise ValueError("method pilot configurations must be U/BF/WF")
    if pilot.get("strategy") != "greedy":
        raise ValueError("method pilot strategy must be greedy")
    expected_records = int(pilot.get("records", 0))
    if expected_records <= 0:
        raise ValueError("method pilot record count must be positive")
    run_root = runs_root / f"{model_row['key']}-r16-seed42"
    return _evaluate_lora_matrix(
        model_row,
        runs_root,
        split="dev",
        expected_records=expected_records,
        matrix_path=run_root / f"dev{expected_records}-u-bf-wf-greedy.jsonl",
        max_new_tokens=int(pilot["max_new_tokens"]),
        expected_manifest_limit=expected_records,
        performance_threshold_used=bool(
            pilot["requirements"]["performance_threshold_used"]
        ),
        paired_parameters=method["paired_statistics"],
        require_frozen_u_surface=True,
    )


def evaluate_lora_publication_matrix(
    config: dict[str, Any],
    model_row: dict[str, Any],
    runs_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate a complete pre-registered dev, test, or tax method matrix."""

    matches = [
        row
        for row in config.get("method_experiment", {}).get(
            "publication_matrices", []
        )
        if row.get("split") == split
    ]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicate publication matrix: {split}")
    matrix = matches[0]
    if matrix.get("configurations") != ["U", "BF", "WF"]:
        raise ValueError(f"publication matrix is not U/BF/WF: {split}")
    expected_records = int(matrix.get("records", 0))
    if expected_records <= 0:
        raise ValueError(f"publication matrix record count is invalid: {split}")
    run_root = runs_root / f"{model_row['key']}-r16-seed42"
    stem = (
        "tax-u-bf-wf-greedy"
        if split == "tax"
        else f"{split}{expected_records}-u-bf-wf-greedy"
    )
    return _evaluate_lora_matrix(
        model_row,
        runs_root,
        split=split,
        expected_records=expected_records,
        matrix_path=run_root / f"{stem}.jsonl",
        max_new_tokens=int(
            config["method_experiment"]["pilot"]["max_new_tokens"]
        ),
        expected_manifest_limit=expected_records,
        performance_threshold_used=False,
        paired_parameters=config["method_experiment"]["paired_statistics"],
        require_frozen_u_surface=False,
    )


def verify_frozen_lora_publication_matrix(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    split: str,
    config_sha256: str,
    selection_sha256: str,
    pilot_gate_sha256: str,
    u_control_plan_sha256: str,
) -> None:
    """Verify a stored publication gate against current immutable evidence."""

    if current.get("passed") is not True:
        raise ValueError("current publication-matrix evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored publication gate is unsupported or did not pass")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored publication gate has no timestamp")
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected.update(
        {
            "split": split,
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
            "pilot_gate_sha256": pilot_gate_sha256,
            "u_control_plan_sha256": u_control_plan_sha256,
        }
    )
    observed = {
        key: value for key, value in stored.items() if key != "checked_at_utc"
    }
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "stored publication gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "evaluate_lora_method_pilot",
    "evaluate_lora_publication_matrix",
    "verify_frozen_lora_method_pilot",
    "verify_frozen_lora_publication_matrix",
]
