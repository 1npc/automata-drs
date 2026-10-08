"""Integrity gate for the frozen selected-host beam-selection study."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .beam_selection import combine_beam_selection_rows
from .paired_analysis import experiment_identity, paired_report_mismatches


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_lora_beam_selection(
    config: dict[str, Any],
    model_row: dict[str, Any],
    runs_root: Path,
    *,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    method = config.get("method_experiment")
    if not isinstance(method, dict):
        raise ValueError("config has no method_experiment object")
    study = method.get("beam_selection_study")
    if not isinstance(study, dict):
        raise ValueError("config has no beam_selection_study object")
    pilot = method.get("pilot")
    if not isinstance(pilot, dict):
        raise ValueError("config has no method pilot object")
    frozen_records = int(pilot.get("records", 0))
    frozen_horizon = int(pilot.get("max_new_tokens", 0))
    if frozen_records <= 0 or frozen_horizon <= 0:
        raise ValueError("method pilot records/horizon are invalid")
    expected_contract = {
        "split": "dev",
        "selection": "same-first-corpus-order-as-pilot",
        "records": frozen_records,
        "num_beams": 4,
        "max_new_tokens": frozen_horizon,
        "generated_conditions": ["U4-candidates", "W", "WF"],
        "derived_comparison": ["U", "PostW", "W", "WF"],
        "u_num_return_sequences": 4,
        "online_num_return_sequences": 1,
        "strategy": "beam",
        "performance_threshold_used": False,
    }
    differences = {
        key: {"observed": study.get(key), "expected": value}
        for key, value in expected_contract.items()
        if study.get(key) != value
    }
    if differences:
        raise ValueError(f"frozen beam study changed: {differences}")

    key = str(model_row["key"])
    run_root = runs_root / f"{key}-r16-seed42"
    records = int(study["records"])
    beams = int(study["num_beams"])
    stem = f"dev{records}-u-postw-w-wf-beam{beams}"
    candidates = run_root / f"dev{records}-u{beams}-beam-candidates.jsonl"
    online = run_root / f"dev{records}-w-wf-beam{beams}.jsonl"
    matrix = run_root / f"{stem}.jsonl"
    candidate_manifest_path = Path(str(candidates) + ".manifest.json")
    online_manifest_path = Path(str(online) + ".manifest.json")
    matrix_manifest_path = Path(str(matrix) + ".manifest.json")
    prefix = matrix.with_suffix("")
    scores = Path(str(prefix) + ".smatch.jsonl")
    score_manifest_path = Path(str(scores) + ".manifest.json")
    summary_path = Path(str(prefix) + ".summary.json")
    paired_path = Path(str(prefix) + ".paired-statistics.json")
    baseline_path = run_root / "dev-u-greedy.jsonl"
    required = (
        baseline_path,
        candidates,
        candidate_manifest_path,
        online,
        online_manifest_path,
        matrix,
        matrix_manifest_path,
        scores,
        score_manifest_path,
        summary_path,
        paired_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
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
    candidate_rows = _read_jsonl(candidates)
    online_rows = _read_jsonl(online)
    matrix_rows = _read_jsonl(matrix)
    score_rows = _read_jsonl(scores)
    candidate_manifest = _read_object(candidate_manifest_path)
    online_manifest = _read_object(online_manifest_path)
    matrix_manifest = _read_object(matrix_manifest_path)
    score_manifest = _read_object(score_manifest_path)
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
    ][:records]
    identifiers = [str(row.get("id", "")) for row in baseline_u]
    check(
        "baseline_population",
        len(identifiers) == records
        and len(set(identifiers)) == records
        and all(identifiers),
        {"records": len(identifiers), "unique": len(set(identifiers))},
        {"records": records, "unique": records},
    )
    try:
        recomputed = combine_beam_selection_rows(
            candidate_rows, online_rows, identifiers, num_beams=beams
        )
        recompute_error = None
    except (KeyError, TypeError, ValueError) as error:
        recomputed = []
        recompute_error = str(error)
    check(
        "exact_derived_matrix",
        recompute_error is None and matrix_rows == recomputed,
        recompute_error or _sha256(matrix),
        "byte-equivalent JSON objects from frozen candidate/online inputs",
    )

    configurations = ("U", "PostW", "W", "WF")
    identities = [
        (str(row.get("id", "")), str(row.get("configuration", "")))
        for row in matrix_rows
    ]
    counts = Counter(configuration for _, configuration in identities)
    expected_counts = {configuration: records for configuration in configurations}
    check(
        "prediction_matrix",
        len(matrix_rows) == records * len(configurations)
        and len(set(identities)) == len(matrix_rows)
        and dict(counts) == expected_counts,
        {"rows": len(matrix_rows), "counts": dict(counts)},
        {"rows": records * len(configurations), "counts": expected_counts},
    )
    check(
        "paired_population",
        all(
            {
                identifier
                for identifier, observed in identities
                if observed == configuration
            }
            == set(identifiers)
            for configuration in configurations
        ),
        "matrix identifiers by configuration",
        "same frozen first-400 identifiers",
    )
    decode_failures = [
        (row.get("id"), row.get("configuration"))
        for row in matrix_rows
        if row.get("decode", {}).get("strategy") != "beam"
        or row.get("decode", {}).get("num_beams") != beams
    ]
    check("beam_decode_contract", not decode_failures, decode_failures[:10], [])
    nonfinite_logprob = [
        (row.get("id"), row.get("configuration"))
        for row in matrix_rows
        if not _finite(row.get("score", {}).get("raw_model_logprob"))
        or not _finite(row.get("score", {}).get("raw_model_mean_logprob"))
    ]
    check("finite_raw_logprob", not nonfinite_logprob, nonfinite_logprob[:10], [])
    online_failures = [
        (row.get("id"), row.get("configuration"))
        for row in matrix_rows
        if row.get("configuration") in {"W", "WF"}
        and (
            row.get("validity", {}).get("raw_sbn") is not True
            or row.get("validity", {}).get(
                "raw_clf_referee",
                row.get("validity", {}).get("clf_referee"),
            )
            is not True
        )
    ]
    check("online_outputs_w_valid", not online_failures, online_failures[:10], [])
    wf_truncated = [
        row.get("id")
        for row in matrix_rows
        if row.get("configuration") == "WF"
        and row.get("generation", {}).get("hit_max_new_tokens") is True
    ]
    check("wf_not_truncated", not wf_truncated, wf_truncated[:10], [])
    all_masked = sum(
        int(row.get("constraint", {}).get("all_masked_rows", 0))
        for row in matrix_rows
        if row.get("configuration") in {"W", "WF"}
        and isinstance(row.get("constraint"), dict)
    )
    check("online_all_masked_rows", all_masked == 0, all_masked, 0)

    fallbacks = sum(
        row.get("beam_selection", {}).get("fallback_to_u_top1") is True
        for row in matrix_rows
        if row.get("configuration") == "PostW"
    )
    selected_valid = sum(
        row.get("validity", {}).get("raw_sbn") is True
        and row.get("validity", {}).get(
            "raw_clf_referee",
            row.get("validity", {}).get("clf_referee"),
        )
        is True
        for row in matrix_rows
        if row.get("configuration") == "PostW"
    )
    check(
        "postw_accounting",
        0 <= fallbacks <= records and 0 <= selected_valid <= records,
        {"fallbacks": fallbacks, "w_valid_selected": selected_valid},
        "counts within frozen population",
    )

    candidate_manifest_ok = (
        candidate_manifest.get("status") == "complete"
        and candidate_manifest.get("kind") == "unconstrained-beam-candidates"
        and candidate_manifest.get("model") == model_row["repo_id"]
        and candidate_manifest.get("model_loader") == model_row["model_loader"]
        and candidate_manifest.get("requested_revision") == model_row["revision"]
        and candidate_manifest.get("output_sha256") == _sha256(candidates)
        and candidate_manifest.get("records") == records
        and candidate_manifest.get("rows") == records * beams
        and candidate_manifest.get("num_beams") == beams
    )
    check("candidate_manifest", candidate_manifest_ok, candidate_manifest, "frozen")
    online_manifest_ok = (
        online_manifest.get("configurations") == ["W", "WF"]
        and online_manifest.get("model") == model_row["repo_id"]
        and online_manifest.get("model_loader") == model_row["model_loader"]
        and online_manifest.get("requested_revision") == model_row["revision"]
        and online_manifest.get("num_beams") == beams
        and online_manifest.get("limit") == records
        and online_manifest.get("max_new_tokens") == study["max_new_tokens"]
    )
    check("online_manifest", online_manifest_ok, online_manifest, "frozen")
    matrix_manifest_ok = (
        matrix_manifest.get("status") == "complete"
        and matrix_manifest.get("kind") == "derived-beam-selection-matrix"
        and matrix_manifest.get("candidate_sha256") == _sha256(candidates)
        and matrix_manifest.get("online_sha256") == _sha256(online)
        and matrix_manifest.get("output_sha256") == _sha256(matrix)
        and matrix_manifest.get("study_contract") == study
        and matrix_manifest.get("records") == records
        and matrix_manifest.get("rows") == records * len(configurations)
        and matrix_manifest.get("configurations") == list(configurations)
        and matrix_manifest.get("num_beams") == beams
        and (
            config_sha256 is None
            or matrix_manifest.get("config_sha256") == config_sha256
        )
    )
    check("derived_manifest", matrix_manifest_ok, matrix_manifest, "frozen")

    prediction_cells = {experiment_identity(row) for row in matrix_rows}
    score_cells = {experiment_identity(row) for row in score_rows}
    score_ok = (
        len(score_cells) == len(score_rows) == len(matrix_rows)
        and score_cells == prediction_cells
        and score_manifest.get("input_sha256") == _sha256(matrix)
        and score_manifest.get("output_sha256") == _sha256(scores)
        and score_manifest.get("records") == len(matrix_rows)
    )
    check("official_scores", score_ok, len(score_rows), len(matrix_rows))
    summary_groups = {
        (str(row.get("configuration")), str(row.get("strategy"))): row
        for row in summary.get("groups", [])
        if isinstance(row, dict)
    }
    summary_ok = (
        summary.get("records") == len(matrix_rows)
        and set(summary_groups)
        == {(configuration, f"beam{beams}") for configuration in configurations}
        and all(row.get("total") == records for row in summary_groups.values())
    )
    check("summary", summary_ok, sorted(summary_groups), "complete beam4 groups")
    paired_mismatches = paired_report_mismatches(
        paired,
        expected_total_records=len(matrix_rows),
        expected_configurations=("PostW", "W", "WF"),
        expected_records_per_comparison=records,
        expected_population_scope="fixed-without-u-outcome-selection",
        prediction_sha256=_sha256(matrix),
        scores_sha256=_sha256(scores),
        registered_parameters=method["paired_statistics"],
        expected_strategy=f"beam{beams}",
    )
    check("paired_statistics", not paired_mismatches, paired_mismatches, [])

    source_hashes = {str(path): _sha256(path) for path in required}
    return {
        "schema_version": 1,
        "model_key": key,
        "run_root": str(run_root),
        "passed": all(item["passed"] for item in checks),
        "performance_threshold_used": False,
        "missing_artifacts": [],
        "checks": checks,
        "diagnostics": {
            "records": records,
            "num_beams": beams,
            "postw_fallbacks": fallbacks,
            "postw_w_valid_selected": selected_valid,
        },
        "source_sha256": source_hashes,
    }


def verify_frozen_lora_beam_selection(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
    publication_gate_sha256: str,
) -> None:
    if current.get("passed") is not True:
        raise ValueError("current beam-selection evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored beam-selection gate is unsupported or did not pass")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored beam-selection gate has no timestamp")
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected.update(
        {
            "config_sha256": config_sha256,
            "selection_sha256": selection_sha256,
            "publication_gate_sha256": publication_gate_sha256,
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
            "stored beam-selection gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "evaluate_lora_beam_selection",
    "verify_frozen_lora_beam_selection",
]
