"""Integrity gate for failure-conditioned selected-host interventions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .interventions import build_intervention_plan
from .paired_analysis import paired_report_mismatches
from .sbn import SBNPolicy, validate_sbn


CONFIGURATIONS = ("U", "B", "W", "BF", "WF")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_lora_intervention(
    config: dict[str, Any],
    model_row: dict[str, Any],
    runs_root: Path,
    split: str,
    *,
    artifact_suffix: str = "",
) -> dict[str, Any]:
    """Verify one conditioned U/B/W/BF/WF intervention artifact."""

    if artifact_suffix not in {"", "-v2"}:
        raise ValueError(f"unsupported intervention artifact suffix: {artifact_suffix}")

    contract = config.get("method_experiment", {}).get(
        "conditioned_intervention", {}
    )
    expected_contract = {
        "configurations": list(CONFIGURATIONS),
        "strategy": "greedy",
        "membership_conditioned_on_u_failure": True,
        "performance_threshold_used": False,
    }
    mismatches = [
        name for name, value in expected_contract.items() if contract.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "conditioned intervention differs from frozen contract: "
            + ", ".join(mismatches)
        )
    if split not in contract.get("splits", []):
        raise ValueError(f"split is not pre-registered for intervention: {split}")

    key = str(model_row["key"])
    run_root = runs_root / f"{key}-r16-seed42"
    baseline = run_root / f"{split}-u-greedy.jsonl"
    plan_path = run_root / f"{split}-u-invalid-selection.json"
    initial_required = (baseline, plan_path)
    missing = [str(path) for path in initial_required if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "model_key": key,
            "split": split,
            "passed": False,
            "missing_artifacts": missing,
            "checks": [],
        }

    plan = _json(plan_path)
    baseline_rows = _jsonl(baseline)
    baseline_by_id = {
        str(row.get("id")): row
        for row in baseline_rows
        if row.get("configuration") == "U"
    }
    recomputed_plan = build_intervention_plan(
        baseline_rows,
        input_path=str(baseline),
        input_sha256=_sha(baseline),
    )
    selected = plan.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("intervention plan selected field is not a list")
    selected_ids = [str(item.get("id", "")) for item in selected]
    selected_lines = [int(item.get("line_number", 0)) for item in selected]
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

    plan_ok = (
        plan == recomputed_plan
        and plan.get("schema_version") == 1
        and plan.get("input") == str(baseline)
        and plan.get("input_sha256") == _sha(baseline)
        and plan.get("selection_criterion") == contract.get("selection_criterion")
        and plan.get("selected_records") == len(selected)
        and len(set(selected_ids)) == len(selected_ids)
        and len(set(selected_lines)) == len(selected_lines)
        and all(selected_ids)
        and all(line > 0 for line in selected_lines)
        and all(
            identifier.rsplit(":", 1)[-1] == str(line)
            for identifier, line in zip(selected_ids, selected_lines, strict=True)
        )
        and set(selected_ids) <= set(baseline_by_id)
    )
    check("frozen_selection_plan", plan_ok, len(selected), plan.get("selected_records"))
    if not selected:
        return {
            "schema_version": 1,
            "model_key": key,
            "model": model_row["repo_id"],
            "revision": model_row["revision"],
            "split": split,
            "passed": all(item["passed"] for item in checks),
            "missing_artifacts": [],
            "checks": checks,
            "diagnostics": {
                "artifact_suffix": artifact_suffix,
                "selected_records": 0,
                "conditioned_population": True,
                "generation_skipped": True,
            },
            "evidence": {
                "baseline_sha256": _sha(baseline),
                "plan_sha256": _sha(plan_path),
            },
        }

    output = run_root / (
        f"{split}-u-invalid-u-b-w-bf-wf-greedy{artifact_suffix}.jsonl"
    )
    generation_manifest = Path(str(output) + ".manifest.json")
    prefix = output.with_suffix("")
    scores = Path(str(prefix) + ".smatch.jsonl")
    score_manifest = Path(str(scores) + ".manifest.json")
    summary = Path(str(prefix) + ".summary.json")
    paired_statistics = Path(str(prefix) + ".paired-statistics.json")
    required = (
        output,
        generation_manifest,
        scores,
        score_manifest,
        summary,
        paired_statistics,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "schema_version": 1,
            "model_key": key,
            "split": split,
            "passed": False,
            "missing_artifacts": missing,
            "checks": checks,
        }

    rows = _jsonl(output)
    identities = [
        (str(row.get("id", "")), str(row.get("configuration", "")))
        for row in rows
    ]
    expected_total = len(selected) * len(CONFIGURATIONS)
    counts = Counter(configuration for _, configuration in identities)
    matrix_ok = (
        len(rows) == expected_total
        and len(set(identities)) == expected_total
        and counts == Counter(
            {configuration: len(selected) for configuration in CONFIGURATIONS}
        )
        and {
            identifier for identifier, _ in identities
        } == set(selected_ids)
    )
    check("conditioned_matrix", matrix_ok, len(rows), expected_total)

    u_by_id = {
        str(row.get("id")): row
        for row in rows
        if row.get("configuration") == "U"
    }
    u_differences = [
        identifier
        for identifier in selected_ids
        if identifier not in u_by_id
        or u_by_id[identifier].get("raw_prediction")
        != baseline_by_id[identifier].get("raw_prediction")
    ]
    check("selected_u_surface_equivalence", not u_differences, u_differences, [])

    decode_failures: list[str] = []
    constraint_failures: list[str] = []
    bounded_failures: list[str] = []
    unbounded_completion_failures: list[str] = []
    score_failures: list[str] = []
    timing_failures: list[str] = []
    all_masked = 0
    truncated = Counter()
    for row in rows:
        identifier = str(row.get("id"))
        configuration = str(row.get("configuration"))
        decode = row.get("decode", {})
        if decode.get("strategy") != "greedy" or decode.get("num_beams") != 1:
            decode_failures.append(f"{identifier}:{configuration}")
        constraint = row.get("constraint")
        if configuration == "U":
            if constraint is not None:
                constraint_failures.append(f"{identifier}:U")
        elif not isinstance(constraint, dict):
            constraint_failures.append(f"{identifier}:{configuration}")
        if isinstance(constraint, dict):
            all_masked += int(constraint.get("all_masked_rows", 0))
        hit_limit = row.get("generation", {}).get("hit_max_new_tokens") is True
        if hit_limit:
            truncated[configuration] += 1
        if configuration in {"BF", "WF"}:
            base = "B" if configuration == "BF" else "W"
            accepted = validate_sbn(
                str(row.get("raw_prediction", "")),
                SBNPolicy.for_configuration(base),
            ).valid
            if (
                hit_limit
                or row.get("generation", {}).get("ended_with_eos") is not True
                or not accepted
                or row.get("validity", {}).get("raw_sbn") is not True
                or (
                    configuration == "WF"
                    and row.get("validity", {}).get(
                        "raw_clf_referee",
                        row.get("validity", {}).get("clf_referee"),
                    )
                    is not True
                )
            ):
                bounded_failures.append(f"{identifier}:{configuration}")
        elif configuration in {"B", "W"} and not hit_limit:
            accepted = validate_sbn(
                str(row.get("raw_prediction", "")),
                SBNPolicy.for_configuration(configuration),
            ).valid
            if (
                row.get("generation", {}).get("ended_with_eos") is not True
                or not accepted
                or (
                    configuration == "W"
                    and row.get("validity", {}).get(
                        "raw_clf_referee",
                        row.get("validity", {}).get("clf_referee"),
                    )
                    is not True
                )
            ):
                unbounded_completion_failures.append(
                    f"{identifier}:{configuration}"
                )
        if (
            not _finite(row.get("score", {}).get("raw_model_logprob"))
            or not _finite(row.get("score", {}).get("raw_model_mean_logprob"))
            or int(row.get("score", {}).get("scored_tokens", 0)) <= 0
        ):
            score_failures.append(f"{identifier}:{configuration}")
        if any(
            not _finite(row.get("timing", {}).get(name))
            or float(row.get("timing", {}).get(name)) < 0
            for name in ("generation_ms", "model_score_ms", "constraint_ms")
        ):
            timing_failures.append(f"{identifier}:{configuration}")
    check("greedy_decode", not decode_failures, decode_failures, [])
    check("constraint_statistics", not constraint_failures, constraint_failures, [])
    check("all_masked_rows", all_masked == 0, all_masked, 0)
    check("bounded_outputs", not bounded_failures, bounded_failures, [])
    check(
        "unbounded_acceptance_when_finished",
        not unbounded_completion_failures,
        unbounded_completion_failures,
        [],
    )
    check("finite_unmasked_logprob", not score_failures, score_failures, [])
    check("finite_nonnegative_timings", not timing_failures, timing_failures, [])

    manifest = _json(generation_manifest)
    expected_manifest = {
        "model": model_row["repo_id"],
        "requested_revision": model_row["revision"],
        "line_numbers": selected_lines,
        "max_new_tokens": int(contract["max_new_tokens"]),
        "num_beams": 1,
        "batch_size": 1,
        "configurations": list(CONFIGURATIONS),
        "score_model_logprob": True,
        "schedule": (
            "cell-preserving-derived-v2"
            if artifact_suffix == "-v2"
            else "rotate"
        ),
    }
    manifest_mismatches = {
        name: {"observed": manifest.get(name), "expected": expected}
        for name, expected in expected_manifest.items()
        if manifest.get(name) != expected
    }
    if artifact_suffix == "-v2":
        correction_config = Path(
            "configs/qwen_drs_intervention_correction_v2.json"
        )
        supplement = run_root / (
            f"{split}-u-invalid-b-w-bf-wf-greedy-v2.jsonl"
        )
        supplement_manifest = Path(str(supplement) + ".manifest.json")
        baseline_manifest = Path(str(baseline) + ".manifest.json")
        source_files = (
            correction_config,
            supplement,
            supplement_manifest,
            baseline_manifest,
        )
        missing_sources = [str(path) for path in source_files if not path.is_file()]
        sources = manifest.get("sources", {})
        expected_sources = None
        if not missing_sources:
            expected_sources = {
                "U": {
                    "path": str(baseline),
                    "sha256": _sha(baseline),
                    "manifest_path": str(baseline_manifest),
                    "manifest_sha256": _sha(baseline_manifest),
                },
                "B/W/BF/WF": {
                    "path": str(supplement),
                    "sha256": _sha(supplement),
                    "manifest_path": str(supplement_manifest),
                    "manifest_sha256": _sha(supplement_manifest),
                },
                "selection": {
                    "path": str(plan_path),
                    "sha256": _sha(plan_path),
                },
                "correction_config": {
                    "path": str(correction_config),
                    "sha256": _sha(correction_config),
                },
            }
        derived_ok = (
            manifest.get("kind") == "derived-lora-conditioned-intervention-v2"
            and manifest.get("rows") == expected_total
            and manifest.get("expected_rows") == expected_total
            and manifest.get("output_sha256") == _sha(output)
            and not missing_sources
            and sources == expected_sources
        )
        if not derived_ok:
            manifest_mismatches["derived_v2_sources"] = {
                "observed": {
                    "kind": manifest.get("kind"),
                    "rows": manifest.get("rows"),
                    "expected_rows": manifest.get("expected_rows"),
                    "output_sha256": manifest.get("output_sha256"),
                    "missing_sources": missing_sources,
                    "sources": sources,
                },
                "expected": {
                    "kind": "derived-lora-conditioned-intervention-v2",
                    "rows": expected_total,
                    "expected_rows": expected_total,
                    "output_sha256": _sha(output),
                    "missing_sources": [],
                    "sources": expected_sources,
                },
            }
    check("generation_manifest", not manifest_mismatches, manifest_mismatches, {})

    score_rows = _jsonl(scores)
    score_ids = {
        (str(row.get("id")), str(row.get("configuration"))) for row in score_rows
    }
    score_meta = _json(score_manifest)
    expected_score_groups = {
        (configuration, "greedy", len(selected))
        for configuration in CONFIGURATIONS
    }
    observed_score_groups = {
        (item.get("configuration"), item.get("strategy"), item.get("records"))
        for item in score_meta.get("groups", [])
    }
    score_ok = (
        len(score_rows) == expected_total
        and score_ids == set(identities)
        and score_meta.get("input_sha256") == _sha(output)
        and score_meta.get("output_sha256") == _sha(scores)
        and score_meta.get("records") == expected_total
        and observed_score_groups == expected_score_groups
    )
    check("official_score_matrix", score_ok, len(score_rows), expected_total)
    summary_value = _json(summary)
    groups = {
        str(item.get("configuration")): item
        for item in summary_value.get("groups", [])
        if item.get("strategy") == "greedy"
    }
    pairs = {
        str(item.get("configuration")): item
        for item in summary_value.get("paired_vs_u", [])
        if item.get("strategy") == "greedy"
    }
    summary_ok = set(groups) == set(CONFIGURATIONS) and set(pairs) == {
        "B", "W", "BF", "WF"
    }
    if summary_ok:
        summary_ok = all(
            int(groups[name].get("total", -1)) == len(selected)
            and _finite(groups[name].get("sbn_smatch_mean"))
            for name in CONFIGURATIONS
        ) and all(
            int(pairs[name].get("paired", -1)) == len(selected)
            and int(pairs[name].get("missing_unconstrained", -1)) == 0
            for name in ("B", "W", "BF", "WF")
        )
    check("conditioned_summary", summary_ok, sorted(groups), list(CONFIGURATIONS))
    paired_value = _json(paired_statistics)
    paired_mismatches = paired_report_mismatches(
        paired_value,
        expected_total_records=expected_total,
        expected_configurations=("B", "W", "BF", "WF"),
        expected_records_per_comparison=len(selected),
        expected_population_scope="failure-conditioned-on-u-invalidity",
        prediction_sha256=_sha(output),
        scores_sha256=_sha(scores),
        registered_parameters=config["method_experiment"]["paired_statistics"],
    )
    check("paired_statistics", not paired_mismatches, paired_mismatches, [])

    return {
        "schema_version": 1,
        "model_key": key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "split": split,
        "passed": all(item["passed"] for item in checks),
        "missing_artifacts": [],
        "checks": checks,
        "diagnostics": {
            "artifact_suffix": artifact_suffix,
            "selected_records": len(selected),
            "structural_invalid": int(plan.get("structural_invalid", 0)),
            "referee_only_invalid": int(plan.get("referee_only_invalid", 0)),
            "failure_families": dict(
                Counter(str(item.get("failure_family")) for item in selected)
            ),
            "truncated_by_configuration": dict(truncated),
            "conditioned_population": True,
            "population_validity_estimate_allowed": False,
        },
        "evidence": {
            "baseline_sha256": _sha(baseline),
            "plan_sha256": _sha(plan_path),
            "output_sha256": _sha(output),
            "generation_manifest_sha256": _sha(generation_manifest),
            "score_sha256": _sha(scores),
            "score_manifest_sha256": _sha(score_manifest),
            "summary_sha256": _sha(summary),
            "paired_statistics_sha256": _sha(paired_statistics),
            **(
                {
                    "correction_config_sha256": _sha(
                        Path("configs/qwen_drs_intervention_correction_v2.json")
                    )
                }
                if artifact_suffix == "-v2"
                else {}
            ),
        },
    }


def verify_frozen_lora_intervention(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
) -> None:
    """Verify a stored intervention gate against current source evidence."""

    if current.get("passed") is not True:
        raise ValueError("current intervention evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored intervention gate is unsupported or did not pass")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored intervention gate has no timestamp")
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected.update(
        {
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
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
            "stored intervention gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "CONFIGURATIONS",
    "evaluate_lora_intervention",
    "verify_frozen_lora_intervention",
]
