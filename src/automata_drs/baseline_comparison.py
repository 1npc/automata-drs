"""Paired quality inference across recent-model LoRA baselines."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .lora_run_state import (
    SPLIT_PATHS,
    verify_complete_lora_baseline_outputs,
)
from .diagnostics import ConstraintFamily, diagnose_sbn
from .paired_analysis import exact_mcnemar_p, paired_bootstrap_intervals
from .sbn import SBNPolicy


CONSTRAINT_LADDER = ("S", "R", "G", "B", "W")


@dataclass(frozen=True, slots=True)
class _Metric:
    name: str
    kind: str
    extract: Callable[[Mapping[str, Any]], float]


def _validity(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("validity")
    if not isinstance(value, Mapping):
        raise ValueError("baseline row has no validity object")
    return value


def _raw_sbn(row: Mapping[str, Any]) -> float:
    validity = _validity(row)
    return float(validity.get("raw_sbn", validity.get("sbn")) is True)


def _wellformed(row: Mapping[str, Any]) -> float:
    validity = _validity(row)
    return float(
        validity.get("raw_sbn", validity.get("sbn")) is True
        and validity.get(
            "raw_clf_referee", validity.get("clf_referee")
        )
        is True
    )


def _exact(row: Mapping[str, Any]) -> float:
    if "prediction" not in row or "reference" not in row:
        raise ValueError("baseline exact match needs prediction/reference")
    return float(row["prediction"] == row["reference"])


def _canonical(row: Mapping[str, Any]) -> float:
    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("baseline row has no generation object")
    value = generation.get("canonical_tokenization")
    if not isinstance(value, bool):
        raise ValueError("baseline row has no canonical-tokenization decision")
    return float(value)


def _smatch(row: Mapping[str, Any]) -> float:
    semantic = row.get("semantic_scores")
    if not isinstance(semantic, Mapping):
        raise ValueError("baseline row has no semantic score object")
    value = semantic.get("sbn_smatch")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("baseline row has invalid SBN Smatch")
    return float(value)


METRICS = (
    _Metric("raw_sbn_valid", "binary", _raw_sbn),
    _Metric("wellformed_valid", "binary", _wellformed),
    _Metric("exact_serialization", "binary", _exact),
    _Metric("canonical_tokenization", "binary", _canonical),
    _Metric("sbn_smatch", "continuous", _smatch),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(row)
    return rows


def _load_scored_rows(
    runs_root: Path, key: str, split: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    run_root = runs_root / f"{key}-r16-seed42"
    prediction_path = run_root / f"{split}-u-greedy.jsonl"
    score_path = run_root / f"{split}-u-greedy.smatch.jsonl"
    predictions = _read_jsonl(prediction_path)
    scores = _read_jsonl(score_path)
    if len(predictions) != len(scores):
        raise ValueError(f"unpaired baseline score rows: {key}/{split}")
    enriched = []
    for line_number, (prediction, score) in enumerate(
        zip(predictions, scores, strict=True), start=1
    ):
        if (
            score.get("id") != prediction.get("id")
            or score.get("configuration") != "U"
            or score.get("decode") != prediction.get("decode")
        ):
            raise ValueError(
                f"baseline score identity mismatch: {key}/{split}:{line_number}"
            )
        row = dict(prediction)
        row["semantic_scores"] = score.get("semantic_scores")
        # Trigger strict validation before the row enters any vector.
        for metric in METRICS:
            metric.extract(row)
        enriched.append(row)
    return enriched, {
        "predictions_sha256": _sha256(prediction_path),
        "scores_sha256": _sha256(score_path),
    }


def _raw_prediction(row: Mapping[str, Any]) -> tuple[str, bool]:
    if "raw_prediction" in row:
        value = row["raw_prediction"]
        explicit = True
    else:
        value = row.get("prediction")
        explicit = False
    if not isinstance(value, str):
        raise ValueError("baseline row has no string raw prediction surface")
    return value, explicit


def _constraint_profile(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay raw U surfaces through the monotonic S/R/G/B/W ladder."""

    if not rows:
        raise ValueError(f"empty baseline constraint population: {model}/{split}")
    identifiers = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"invalid baseline ID in constraint replay: {model}/{split}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate baseline ID in constraint replay: {model}/{split}")

    accepted = Counter({configuration: 0 for configuration in CONSTRAINT_LADDER})
    first_blocking = Counter(
        {configuration: 0 for configuration in (*CONSTRAINT_LADDER, "accepted")}
    )
    w_families = Counter({family.value: 0 for family in ConstraintFamily})
    w_events = Counter({"character": 0, "eos": 0})
    truncated = 0
    truncated_b_invalid = 0
    truncated_w_invalid = 0
    explicit_raw = 0
    raw_postprocessed_different = 0
    profiles: list[dict[str, Any]] = []

    for row in rows:
        raw_prediction, explicit = _raw_prediction(row)
        explicit_raw += int(explicit)
        postprocessed_prediction = row.get("prediction")
        if not isinstance(postprocessed_prediction, str):
            raise ValueError("baseline row has no string postprocessed prediction")
        raw_postprocessed_different += int(
            raw_prediction != postprocessed_prediction
        )
        diagnostics = {
            configuration: diagnose_sbn(
                raw_prediction, SBNPolicy.for_configuration(configuration)
            )
            for configuration in CONSTRAINT_LADDER
        }
        decisions = [diagnostics[configuration].valid for configuration in CONSTRAINT_LADDER]
        if any(
            not left and right
            for left, right in zip(decisions, decisions[1:])
        ):
            raise ValueError(
                f"non-monotonic constraint replay for {model}/{split}/{row.get('id')}"
            )
        for configuration, decision in zip(
            CONSTRAINT_LADDER, decisions, strict=True
        ):
            accepted[configuration] += int(decision)
        blocking = next(
            (
                configuration
                for configuration, decision in zip(
                    CONSTRAINT_LADDER, decisions, strict=True
                )
                if not decision
            ),
            "accepted",
        )
        first_blocking[blocking] += 1

        validity = _validity(row)
        stored_b = validity.get("raw_sbn", validity.get("sbn")) is True
        has_raw_referee = "raw_clf_referee" in validity
        stored_referee = (
            validity.get("raw_clf_referee")
            if has_raw_referee
            else validity.get("clf_referee")
        )
        if not has_raw_referee and raw_prediction != postprocessed_prediction:
            raise ValueError(
                f"legacy Referee fallback requires identical raw/postprocessed "
                f"surfaces: {model}/{split}/{row.get('id')}"
            )
        # Structurally invalid rows are not necessarily submitted to the
        # external Referee, so their stored decision may legitimately be null.
        # A B-valid row must carry the decision needed to validate W replay.
        if stored_b and not isinstance(stored_referee, bool):
            raise ValueError(
                f"baseline row has no raw Referee decision: "
                f"{model}/{split}/{row.get('id')}"
            )
        stored_w = stored_b and stored_referee is True
        if diagnostics["B"].valid != stored_b:
            raise ValueError(
                f"stored/raw B validity mismatch for {model}/{split}/{row.get('id')}: "
                f"stored={stored_b}, replay={diagnostics['B'].valid}"
            )
        if diagnostics["W"].valid != stored_w:
            raise ValueError(
                f"stored/raw W validity mismatch for {model}/{split}/{row.get('id')}: "
                f"stored={stored_w}, replay={diagnostics['W'].valid}"
            )

        w_diagnostic = diagnostics["W"]
        w_outcome = "accepted"
        if not w_diagnostic.valid:
            if w_diagnostic.family is None or w_diagnostic.event not in w_events:
                raise ValueError(
                    f"incomplete W diagnostic for {model}/{split}/{row.get('id')}"
                )
            w_outcome = w_diagnostic.family.value
            w_families[w_outcome] += 1
            w_events[w_diagnostic.event] += 1

        generation = row.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError("baseline row has no generation object")
        hit_limit = generation.get("hit_max_new_tokens")
        if not isinstance(hit_limit, bool):
            raise ValueError(
                f"baseline row has no truncation decision: "
                f"{model}/{split}/{row.get('id')}"
            )
        truncated += int(hit_limit)
        truncated_b_invalid += int(hit_limit and not diagnostics["B"].valid)
        truncated_w_invalid += int(hit_limit and not diagnostics["W"].valid)
        profiles.append(
            {
                "id": row["id"],
                "first_blocking_layer": blocking,
                "b_valid": diagnostics["B"].valid,
                "w_valid": diagnostics["W"].valid,
                "w_outcome": w_outcome,
            }
        )

    total = len(rows)
    report = {
        "model": model,
        "split": split,
        "records": total,
        "selection_use": False,
        "surface": {
            "explicit_raw_prediction": explicit_raw,
            "prediction_fallback": total - explicit_raw,
            "raw_postprocessed_different": raw_postprocessed_different,
        },
        "acceptance": {
            configuration: {
                "accepted": accepted[configuration],
                "rate": accepted[configuration] / total,
            }
            for configuration in CONSTRAINT_LADDER
        },
        "first_blocking_layer": dict(first_blocking),
        "w_failure_family": dict(w_families),
        "w_failure_event": dict(w_events),
        "truncation": {
            "hit_max_new_tokens": truncated,
            "and_b_invalid": truncated_b_invalid,
            "and_w_invalid": truncated_w_invalid,
        },
    }
    return report, profiles


def summarize_constraint_headroom(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    split: str,
) -> dict[str, Any]:
    """Return the reviewer-facing constraint-replay summary for raw U rows."""

    report, _ = _constraint_profile(rows, model=model, split=split)
    return report


def _transition_rows(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    transitions = Counter(
        (str(left_row[field]), str(right_row[field]))
        for left_row, right_row in zip(left, right, strict=True)
    )
    return [
        {"left": left_value, "right": right_value, "count": count}
        for (left_value, right_value), count in sorted(transitions.items())
    ]


def _paired_constraint_headroom(
    left_profiles: Sequence[Mapping[str, Any]],
    right_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if [row["id"] for row in left_profiles] != [row["id"] for row in right_profiles]:
        raise ValueError("constraint-headroom IDs differ between paired models")
    overlap = Counter(
        (
            "both_valid"
            if left["w_valid"] and right["w_valid"]
            else "left_only_valid"
            if left["w_valid"]
            else "right_only_valid"
            if right["w_valid"]
            else "both_invalid"
        )
        for left, right in zip(left_profiles, right_profiles, strict=True)
    )
    return {
        "w_validity_overlap": {
            key: overlap[key]
            for key in (
                "both_valid",
                "left_only_valid",
                "right_only_valid",
                "both_invalid",
            )
        },
        "first_blocking_layer_transitions": _transition_rows(
            left_profiles, right_profiles, "first_blocking_layer"
        ),
        "w_outcome_transitions": _transition_rows(
            left_profiles, right_profiles, "w_outcome"
        ),
    }


def _holm_within_split(comparisons: list[dict[str, Any]]) -> None:
    for metric in (row for row in METRICS if row.kind == "binary"):
        family = sorted(
            (
                (float(comparison["metrics"][metric.name]["mcnemar_exact_p"]), index)
                for index, comparison in enumerate(comparisons)
            ),
            key=lambda item: (item[0], item[1]),
        )
        running = 0.0
        total = len(family)
        for rank, (p_value, index) in enumerate(family):
            running = max(running, min(1.0, (total - rank) * p_value))
            comparisons[index]["metrics"][metric.name][
                "mcnemar_exact_p_holm"
            ] = running


def analyze_lora_baseline_comparisons(
    config: dict[str, Any],
    runs_root: Path,
    repository_root: Path,
    *,
    bootstrap_resamples: int = 5_000,
    confidence: float = 0.95,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Compare all model pairs on identical split rows without selecting a host."""

    verification = verify_complete_lora_baseline_outputs(
        config, runs_root, repository_root
    )
    models = sorted(config["models"], key=lambda row: row["priority"])
    if len(models) < 2:
        raise ValueError("baseline comparison requires at least two models")
    comparisons: list[dict[str, Any]] = []
    constraint_headroom: list[dict[str, Any]] = []
    source: dict[str, dict[str, Any]] = {}
    for split in ("dev", "test", "tax"):
        by_model: dict[str, list[dict[str, Any]]] = {}
        profiles_by_model: dict[str, list[dict[str, Any]]] = {}
        for model in models:
            key = str(model["key"])
            rows, hashes = _load_scored_rows(runs_root, key, split)
            by_model[key] = rows
            headroom, profiles = _constraint_profile(
                rows, model=key, split=split
            )
            constraint_headroom.append(headroom)
            profiles_by_model[key] = profiles
            source.setdefault(key, {})[split] = hashes
        first_key = str(models[0]["key"])
        identifiers = [str(row["id"]) for row in by_model[first_key]]
        references = [row.get("reference") for row in by_model[first_key]]
        for model in models[1:]:
            key = str(model["key"])
            if [str(row["id"]) for row in by_model[key]] != identifiers:
                raise ValueError(f"baseline ID order differs: {split}/{key}")
            if [row.get("reference") for row in by_model[key]] != references:
                raise ValueError(f"baseline references differ: {split}/{key}")
        population_hash = hashlib.sha256(
            ("\n".join(identifiers) + "\n").encode("utf-8")
        ).hexdigest()
        derived_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}\0baseline-models\0{split}\0{population_hash}".encode()
            ).digest()[:8],
            "big",
        )
        pair_vectors: dict[
            tuple[str, str], dict[str, tuple[list[float], list[float]]]
        ] = {}
        delta_vectors: dict[str, list[float]] = {}
        for left_model, right_model in itertools.combinations(models, 2):
            left = str(left_model["key"])
            right = str(right_model["key"])
            metrics: dict[str, tuple[list[float], list[float]]] = {}
            for metric in METRICS:
                left_values = [metric.extract(row) for row in by_model[left]]
                right_values = [metric.extract(row) for row in by_model[right]]
                metrics[metric.name] = (left_values, right_values)
                delta_vectors[f"{left}->{right}:{metric.name}"] = [
                    right_value - left_value
                    for left_value, right_value in zip(
                        left_values, right_values, strict=True
                    )
                ]
            pair_vectors[(left, right)] = metrics
        intervals = paired_bootstrap_intervals(
            delta_vectors,
            resamples=bootstrap_resamples,
            confidence=confidence,
            seed=derived_seed,
        )
        split_comparisons = []
        for (left, right), metrics in pair_vectors.items():
            metric_reports = {}
            for metric in METRICS:
                left_values, right_values = metrics[metric.name]
                deltas = delta_vectors[f"{left}->{right}:{metric.name}"]
                lower, upper = intervals[f"{left}->{right}:{metric.name}"]
                report = {
                    "kind": metric.kind,
                    "paired": len(deltas),
                    "left_mean": statistics.fmean(left_values),
                    "right_mean": statistics.fmean(right_values),
                    "delta_right_minus_left": statistics.fmean(deltas),
                    "delta_median": statistics.median(deltas),
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                }
                if metric.kind == "binary":
                    fixed = sum(
                        left_value == 0.0 and right_value == 1.0
                        for left_value, right_value in zip(
                            left_values, right_values, strict=True
                        )
                    )
                    regressed = sum(
                        left_value == 1.0 and right_value == 0.0
                        for left_value, right_value in zip(
                            left_values, right_values, strict=True
                        )
                    )
                    report.update(
                        {
                            "right_fixed": fixed,
                            "right_regressed": regressed,
                            "discordant": fixed + regressed,
                            "mcnemar_exact_p": exact_mcnemar_p(
                                fixed, regressed
                            ),
                        }
                    )
                metric_reports[metric.name] = report
            split_comparisons.append(
                {
                    "split": split,
                    "left": left,
                    "right": right,
                    "paired": len(identifiers),
                    "population_sha256": population_hash,
                    "bootstrap_seed": derived_seed,
                    "metrics": metric_reports,
                    "constraint_headroom": _paired_constraint_headroom(
                        profiles_by_model[left], profiles_by_model[right]
                    ),
                }
            )
        _holm_within_split(split_comparisons)
        comparisons.extend(split_comparisons)
    return {
        "schema_version": 2,
        "analysis": "paired-recent-lora-baseline-quality-and-constraint-headroom",
        "selection_use": False,
        "parameters": {
            "paired_unit": "example",
            "bootstrap": "paired-percentile",
            "bootstrap_resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
            "mcnemar": "two-sided-exact-conditional-binomial",
            "multiplicity": "Holm within split and binary metric across model pairs",
            "delta_direction": "right_minus_left",
            "constraint_replay": {
                "surface": "raw_prediction with legacy prediction fallback",
                "ladder": list(CONSTRAINT_LADDER),
                "stored_consistency": ["raw B", "raw W"],
                "host_selection_use": False,
            },
        },
        "source": source,
        "baseline_verification": verification,
        "constraint_headroom": constraint_headroom,
        "comparisons": comparisons,
    }


__all__ = [
    "CONSTRAINT_LADDER",
    "METRICS",
    "analyze_lora_baseline_comparisons",
    "summarize_constraint_headroom",
]
