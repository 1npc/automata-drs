"""Integrity checks for the selected-host U/B/W/BF/WF layer ablation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .lora_layer_ablation import (
    FULL_LADDER,
    SUPPLEMENT_CONFIGURATIONS,
    combine_layer_ablation_rows,
)
from .paired_analysis import paired_report_mismatches
from .sbn import SBNPolicy, validate_sbn


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_lora_layer_ablation(
    config: dict[str, Any],
    model_row: dict[str, Any],
    runs_root: Path,
) -> dict[str, Any]:
    """Verify generated B/W cells and their cell-preserving full-ladder join."""

    ablation = config.get("method_experiment", {}).get("layer_ablation", {})
    frozen_contract = {
        "split": "dev",
        "selection": "same-first-corpus-order-as-pilot",
        "generated_configurations": ["B", "W"],
        "derived_full_ladder": ["U", "B", "W", "BF", "WF"],
        "strategy": "greedy",
        "performance_threshold_used": False,
    }
    mismatched_contract = [
        key for key, value in frozen_contract.items() if ablation.get(key) != value
    ]
    if mismatched_contract:
        raise ValueError(
            "layer-ablation config differs from the frozen contract: "
            + ", ".join(mismatched_contract)
        )
    records = int(ablation.get("records", 0))
    if records <= 0:
        raise ValueError("layer-ablation record count must be positive")
    pilot_records = int(config["method_experiment"]["pilot"]["records"])
    if records != pilot_records:
        raise ValueError("layer ablation and pilot must use the same population")

    key = str(model_row["key"])
    run_root = runs_root / f"{key}-r16-seed42"
    pilot = run_root / f"dev{records}-u-bf-wf-greedy.jsonl"
    supplement = run_root / f"dev{records}-b-w-greedy.jsonl"
    combined = run_root / f"dev{records}-u-b-w-bf-wf-greedy.jsonl"
    supplement_manifest = Path(str(supplement) + ".manifest.json")
    combined_manifest = Path(str(combined) + ".manifest.json")
    prefix = combined.with_suffix("")
    scores = Path(str(prefix) + ".smatch.jsonl")
    score_manifest = Path(str(scores) + ".manifest.json")
    summary = Path(str(prefix) + ".summary.json")
    paired_statistics = Path(str(prefix) + ".paired-statistics.json")
    required = (
        pilot,
        Path(str(pilot) + ".manifest.json"),
        supplement,
        supplement_manifest,
        combined,
        combined_manifest,
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
            "passed": False,
            "missing_artifacts": missing,
            "checks": [],
        }

    pilot_rows = _read_jsonl(pilot)
    supplement_rows = _read_jsonl(supplement)
    combined_rows = _read_jsonl(combined)
    expected_rows, join_evidence = combine_layer_ablation_rows(
        pilot_rows, supplement_rows, expected_records=records
    )
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

    check(
        "cell_preserving_join",
        combined_rows == expected_rows,
        len(combined_rows),
        len(expected_rows),
    )
    generated_counts = Counter(
        str(row.get("configuration")) for row in supplement_rows
    )
    check(
        "supplement_matrix",
        generated_counts == Counter({"B": records, "W": records}),
        dict(generated_counts),
        {"B": records, "W": records},
    )
    decode_failures: list[str] = []
    policy_failures: list[str] = []
    score_failures: list[str] = []
    timing_failures: list[str] = []
    missing_constraint_stats: list[str] = []
    all_masked = 0
    truncated = Counter()
    for row in supplement_rows:
        identifier = str(row.get("id"))
        configuration = str(row.get("configuration"))
        decode = row.get("decode", {})
        if decode.get("strategy") != "greedy" or decode.get("num_beams") != 1:
            decode_failures.append(f"{identifier}:{configuration}")
        constraint_value = row.get("constraint")
        if not isinstance(constraint_value, dict):
            missing_constraint_stats.append(f"{identifier}:{configuration}")
        constraint = constraint_value or {}
        all_masked += int(constraint.get("all_masked_rows", 0))
        hit_limit = row.get("generation", {}).get("hit_max_new_tokens") is True
        if hit_limit:
            truncated[configuration] += 1
        else:
            result = validate_sbn(
                str(row.get("raw_prediction", "")),
                SBNPolicy.for_configuration(configuration),
            )
            if (
                row.get("generation", {}).get("ended_with_eos") is not True
                or not result.valid
                or row.get("validity", {}).get("raw_sbn") is not True
                or (
                    configuration == "W"
                    and row.get("validity", {}).get(
                        "raw_clf_referee",
                        row.get("validity", {}).get("clf_referee"),
                    )
                    is not True
                )
            ):
                policy_failures.append(f"{identifier}:{configuration}")
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
    check(
        "constraint_statistics_present",
        not missing_constraint_stats,
        missing_constraint_stats,
        [],
    )
    check("all_masked_rows", all_masked == 0, all_masked, 0)
    check(
        "accepting_untruncated_outputs",
        not policy_failures,
        policy_failures,
        [],
    )
    check("finite_unmasked_logprob", not score_failures, score_failures, [])
    check("finite_nonnegative_timings", not timing_failures, timing_failures, [])

    generation_manifest = _read_json(supplement_manifest)
    expected_generation = {
        "model": model_row["repo_id"],
        "requested_revision": model_row["revision"],
        "limit": records,
        "max_new_tokens": int(ablation["max_new_tokens"]),
        "num_beams": 1,
        "batch_size": 1,
        "configurations": list(SUPPLEMENT_CONFIGURATIONS),
        "score_model_logprob": True,
        "schedule": "rotate",
    }
    generation_mismatches = {
        name: {"observed": generation_manifest.get(name), "expected": expected}
        for name, expected in expected_generation.items()
        if generation_manifest.get(name) != expected
    }
    check("supplement_manifest", not generation_mismatches, generation_mismatches, {})

    derivation = _read_json(combined_manifest)
    source_evidence = derivation.get("sources", [])
    source_by_path = {
        str(item.get("path")): item for item in source_evidence if isinstance(item, dict)
    }
    derivation_ok = (
        derivation.get("schema_version") == 1
        and derivation.get("status") == "complete"
        and derivation.get("kind") == "derived-lora-layer-ablation"
        and derivation.get("output_sha256") == _sha(combined)
        and derivation.get("derived_full_ladder") == list(FULL_LADDER)
        and derivation.get("derivation") == "cell-preserving-join-by-id"
        and source_by_path.get(str(pilot), {}).get("sha256") == _sha(pilot)
        and source_by_path.get(str(supplement), {}).get("sha256") == _sha(supplement)
        and source_by_path.get(str(pilot), {}).get("manifest_sha256")
        == _sha(Path(str(pilot) + ".manifest.json"))
        and source_by_path.get(str(supplement), {}).get("manifest_sha256")
        == _sha(supplement_manifest)
    )
    check("derived_manifest", derivation_ok, derivation.get("kind"), "hash-valid join")

    score_rows = _read_jsonl(scores)
    score_ids = {
        (str(row.get("id")), str(row.get("configuration"))) for row in score_rows
    }
    combined_ids = {
        (str(row.get("id")), str(row.get("configuration"))) for row in combined_rows
    }
    score_meta = _read_json(score_manifest)
    expected_total = records * len(FULL_LADDER)
    expected_groups = {
        (configuration, "greedy", records) for configuration in FULL_LADDER
    }
    observed_groups = {
        (item.get("configuration"), item.get("strategy"), item.get("records"))
        for item in score_meta.get("groups", [])
    }
    score_ok = (
        len(score_rows) == expected_total
        and score_ids == combined_ids
        and score_meta.get("input_sha256") == _sha(combined)
        and score_meta.get("output_sha256") == _sha(scores)
        and score_meta.get("records") == expected_total
        and observed_groups == expected_groups
    )
    check("official_score_matrix", score_ok, len(score_rows), expected_total)

    summary_value = _read_json(summary)
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
    summary_ok = set(groups) == set(FULL_LADDER) and set(pairs) == {
        "B", "W", "BF", "WF"
    }
    if summary_ok:
        summary_ok = all(
            int(groups[name].get("total", -1)) == records
            and _finite(groups[name].get("sbn_smatch_mean"))
            for name in FULL_LADDER
        ) and all(
            int(pairs[name].get("paired", -1)) == records
            and int(pairs[name].get("missing_unconstrained", -1)) == 0
            for name in ("B", "W", "BF", "WF")
        )
    check("full_ladder_summary", summary_ok, sorted(groups), list(FULL_LADDER))
    paired_value = _read_json(paired_statistics)
    paired_mismatches = paired_report_mismatches(
        paired_value,
        expected_total_records=expected_total,
        expected_configurations=("B", "W", "BF", "WF"),
        expected_records_per_comparison=records,
        expected_population_scope="fixed-without-u-outcome-selection",
        prediction_sha256=_sha(combined),
        scores_sha256=_sha(scores),
        registered_parameters=config["method_experiment"]["paired_statistics"],
    )
    check("paired_statistics", not paired_mismatches, paired_mismatches, [])
    check(
        "no_performance_threshold",
        ablation["performance_threshold_used"] is False,
        ablation["performance_threshold_used"],
        False,
    )

    return {
        "schema_version": 1,
        "model_key": key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "passed": all(item["passed"] for item in checks),
        "missing_artifacts": [],
        "checks": checks,
        "diagnostics": {
            "truncated_B": truncated["B"],
            "truncated_W": truncated["W"],
            "truncation_is_measured_not_gated": True,
        },
        "evidence": {
            "pilot_sha256": _sha(pilot),
            "supplement_sha256": _sha(supplement),
            "combined_sha256": _sha(combined),
            "derived_manifest_sha256": _sha(combined_manifest),
            "score_sha256": _sha(scores),
            "score_manifest_sha256": _sha(score_manifest),
            "summary_sha256": _sha(summary),
            "paired_statistics_sha256": _sha(paired_statistics),
            "join": join_evidence,
        },
    }


def verify_frozen_lora_layer_ablation(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
    pilot_gate_sha256: str,
) -> None:
    """Verify a stored layer gate against recomputed artifacts and hashes."""

    if current.get("passed") is not True:
        raise ValueError("current layer-ablation evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored layer-ablation gate is unsupported or did not pass")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored layer-ablation gate has no timestamp")
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected.update(
        {
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
            "pilot_gate_sha256": pilot_gate_sha256,
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
            "stored layer-ablation gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "evaluate_lora_layer_ablation",
    "verify_frozen_lora_layer_ablation",
]
