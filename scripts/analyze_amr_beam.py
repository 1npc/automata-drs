#!/usr/bin/env python3
"""Paired AMR inference for U, offline PostP, and online-P beam decoding."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_beam_selection import BEAM_RANKING_METHOD  # noqa: E402
from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
    sha256_file,
)
from automata_drs.paired_analysis import exact_mcnemar_p  # noqa: E402


CONDITIONS = ("U", "PostP", "P")


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


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _f1(matches: int, predicted: int, gold: int) -> float:
    if predicted == 0 or gold == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / gold
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _bootstrap_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(pairs)
    deltas: list[float] = []
    for _ in range(resamples):
        baseline_matches = baseline_predicted = baseline_gold = 0
        treatment_matches = treatment_predicted = treatment_gold = 0
        for _ in range(size):
            baseline, treatment = pairs[generator.randrange(size)]
            baseline_matches += int(baseline["matches"])
            baseline_predicted += int(baseline["predicted_triples"])
            baseline_gold += int(baseline["gold_triples"])
            treatment_matches += int(treatment["matches"])
            treatment_predicted += int(treatment["predicted_triples"])
            treatment_gold += int(treatment["gold_triples"])
        deltas.append(
            _f1(treatment_matches, treatment_predicted, treatment_gold)
            - _f1(baseline_matches, baseline_predicted, baseline_gold)
        )
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (count - index) * values[name])
        adjusted[name] = min(1.0, running)
    return adjusted


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
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--predecessor-gate", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("resamples must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite AMR beam report: {args.output}")

    try:
        manifest = complete_generation_manifest(args.predictions)
        scores = _read_object(args.scores)
        predecessor = _read_object(args.predecessor_gate)
        rows = _read_rows(args.predictions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if manifest.get("task") != "amr" or manifest.get("configurations") != list(CONDITIONS):
        parser.error("predictions are not the fixed U/PostP/P AMR beam matrix")
    if manifest.get("derivation") != (
        "u-top1-and-first-raw-penman-valid-u-beam-plus-online-p"
    ):
        parser.error("AMR beam derivation is not registered")
    if not scores.get("gate", {}).get("passed"):
        parser.error("AMR beam Smatch report gate did not pass")
    if scores.get("predictions", {}).get("sha256") != sha256_file(args.predictions):
        parser.error("AMR beam Smatch report does not bind predictions")

    by_condition: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in CONDITIONS
    }
    for row in rows:
        condition = str(row.get("configuration", ""))
        identifier = str(row.get("id", ""))
        if condition not in by_condition or not identifier or identifier in by_condition[condition]:
            parser.error(f"invalid or duplicate beam identity: {condition} {identifier}")
        by_condition[condition][identifier] = row
    identities = set(by_condition["U"])
    if not identities or any(set(group) != identities for group in by_condition.values()):
        parser.error("AMR beam condition identity sets are not exactly paired")

    scored: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in CONDITIONS
    }
    for row in scores.get("paired_rows", []):
        condition = str(row.get("configuration", ""))
        identifier = str(row.get("id", ""))
        if condition in scored:
            if identifier in scored[condition]:
                parser.error(f"duplicate AMR beam score: {condition} {identifier}")
            scored[condition][identifier] = row
    if any(set(group) != identities for group in scored.values()):
        parser.error("AMR beam score identities differ from predictions")

    comparison_specs = (
        ("postp_minus_u", "U", "PostP"),
        ("p_minus_u", "U", "P"),
        ("p_minus_postp", "PostP", "P"),
    )
    comparisons: dict[str, dict[str, Any]] = {}
    mcnemar_values: dict[str, float] = {}
    for comparison_index, (name, baseline_name, treatment_name) in enumerate(
        comparison_specs
    ):
        repairs = regressions = wins = losses = ties = changed = 0
        sufficient_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        logprob_deltas: list[float] = []
        for identifier in sorted(identities):
            baseline = by_condition[baseline_name][identifier]
            treatment = by_condition[treatment_name][identifier]
            if baseline.get("reference") != treatment.get("reference"):
                parser.error(f"reference mismatch for {identifier}")
            baseline_score = scored[baseline_name][identifier]
            treatment_score = scored[treatment_name][identifier]
            baseline_valid = bool(baseline_score["raw_official_penman"])
            treatment_valid = bool(treatment_score["raw_official_penman"])
            repairs += int(not baseline_valid and treatment_valid)
            regressions += int(baseline_valid and not treatment_valid)
            baseline_smatch = baseline_score["raw_smatch"]
            treatment_smatch = treatment_score["raw_smatch"]
            sufficient_pairs.append((baseline_smatch, treatment_smatch))
            baseline_f1 = float(baseline_smatch["f1"])
            treatment_f1 = float(treatment_smatch["f1"])
            wins += int(treatment_f1 > baseline_f1)
            losses += int(treatment_f1 < baseline_f1)
            ties += int(treatment_f1 == baseline_f1)
            changed += int(
                baseline.get("raw_prediction") != treatment.get("raw_prediction")
            )
            baseline_logprob = _finite(
                baseline.get("score", {}).get("raw_model_mean_logprob")
            )
            treatment_logprob = _finite(
                treatment.get("score", {}).get("raw_model_mean_logprob")
            )
            if baseline_logprob is not None and treatment_logprob is not None:
                logprob_deltas.append(treatment_logprob - baseline_logprob)
        baseline_corpus = scores["groups"][baseline_name]["raw_primary_smatch"]
        treatment_corpus = scores["groups"][treatment_name]["raw_primary_smatch"]
        delta = float(treatment_corpus["f1"]) - float(baseline_corpus["f1"])
        lower, upper = _bootstrap_delta(
            sufficient_pairs,
            seed=args.seed + comparison_index,
            resamples=args.resamples,
        )
        mcnemar = exact_mcnemar_p(repairs, regressions)
        mcnemar_values[name] = mcnemar
        comparisons[name] = {
            "baseline": baseline_name,
            "treatment": treatment_name,
            "corpus_smatch_delta": delta,
            "delta_ci_lower": lower,
            "delta_ci_upper": upper,
            "repairs": repairs,
            "regressions": regressions,
            "exact_mcnemar_p": mcnemar,
            "per_example_wins": wins,
            "per_example_losses": losses,
            "per_example_ties": ties,
            "changed_raw_predictions": changed,
            "mean_raw_model_mean_logprob_delta": _mean(logprob_deltas),
            "logprob_paired_records": len(logprob_deltas),
        }
    adjusted = _holm(mcnemar_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_mcnemar_p"] = value

    selected_ranks: Counter[int] = Counter()
    valid_candidate_counts: list[float] = []
    fallbacks = 0
    selection_contract = True
    postp_nonfallback_valid = True
    finite_emitted_scores = True
    truncations = {condition: 0 for condition in CONDITIONS}
    online_p_all_masked = 0
    online_p_accepting_rows = 0
    online_p_search_evidence_complete = True
    timing: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for row in by_condition[condition].values():
            score_value = _finite(row.get("score", {}).get("raw_model_logprob"))
            finite_emitted_scores = finite_emitted_scores and score_value is not None
            generation = row.get("generation", {})
            truncations[condition] += int(
                bool(generation.get("hit_max_new_tokens"))
            )
            generation_ms = _finite(row.get("timing", {}).get("generation_ms"))
            if generation_ms is not None:
                timing[condition].append(generation_ms)
            if condition == "P":
                constraint = row.get("constraint")
                masked = None if not isinstance(constraint, dict) else constraint.get(
                    "all_masked_rows"
                )
                evidence_ok = (
                    isinstance(masked, int)
                    and not isinstance(masked, bool)
                    and masked >= 0
                )
                online_p_search_evidence_complete = (
                    online_p_search_evidence_complete and evidence_ok
                )
                if evidence_ok:
                    online_p_all_masked += masked
                validity = row.get("validity", {})
                online_p_accepting_rows += int(
                    validity.get("raw_local_penman") is True
                    and validity.get("raw_official_penman") is True
                    and generation.get("ended_with_eos") is True
                    and generation.get("hit_max_new_tokens") is False
                )
            if condition != "PostP":
                continue
            selection = row.get("beam_selection")
            if not isinstance(selection, dict):
                selection_contract = False
                continue
            rank = selection.get("selected_candidate_rank")
            fallback = selection.get("fallback_to_u_top1")
            candidate_count = selection.get("candidate_count")
            valid_count = selection.get("raw_penman_valid_candidate_count")
            valid_ranks = selection.get("raw_penman_valid_candidate_ranks")
            selection_contract = selection_contract and (
                selection.get("method")
                == "first-raw-penman-valid-in-final-beam-order"
                and selection.get("ranking_method") == BEAM_RANKING_METHOD
                and isinstance(rank, int)
                and not isinstance(rank, bool)
                and isinstance(fallback, bool)
                and isinstance(candidate_count, int)
                and not isinstance(candidate_count, bool)
                and candidate_count == manifest.get("num_beams")
                and isinstance(valid_count, int)
                and not isinstance(valid_count, bool)
                and isinstance(valid_ranks, list)
                and valid_count == len(valid_ranks)
                and (fallback is False or (rank == 0 and valid_count == 0))
                and (fallback is True or (valid_count > 0 and rank == valid_ranks[0]))
            )
            if isinstance(rank, int) and not isinstance(rank, bool):
                selected_ranks[rank] += 1
            fallbacks += int(fallback is True)
            count = valid_count
            if isinstance(count, int) and not isinstance(count, bool):
                valid_candidate_counts.append(float(count))
            if fallback is False:
                validity = row.get("validity", {})
                postp_nonfallback_valid = postp_nonfallback_valid and (
                    validity.get("raw_local_penman") is True
                    and validity.get("raw_official_penman") is True
                )

    checks = {
        "full_development_gate_passed": predecessor.get("gate", {}).get("passed") is True,
        "complete_identity_pairing": True,
        "score_hash_matches": True,
        "selection_contract": selection_contract,
        "postp_nonfallback_outputs_accepting": postp_nonfallback_valid,
        "online_p_search_evidence_complete": online_p_search_evidence_complete,
        "finite_emitted_raw_model_scores": finite_emitted_scores,
    }
    report = {
        "schema_version": 2,
        "audit": "amr-beam-u-postp-p-analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": {
            "records": len(identities),
            "paired": True,
            "conditions": list(CONDITIONS),
        },
        "inputs": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "scores": str(args.scores),
            "scores_sha256": sha256_file(args.scores),
            "predecessor_gate": str(args.predecessor_gate),
            "predecessor_gate_sha256": sha256_file(args.predecessor_gate),
        },
        "groups": {
            condition: {
                "raw_penman_valid_rate": scores["groups"][condition][
                    "raw_official_penman_rate"
                ],
                "raw_corpus_smatch_f1": scores["groups"][condition][
                    "raw_primary_smatch"
                ]["f1"],
                "generation_ms_mean": _mean(timing[condition]),
            }
            for condition in CONDITIONS
        },
        "comparisons": comparisons,
        "selection": {
            "fallbacks": fallbacks,
            "fallback_rate": fallbacks / len(identities),
            "selected_rank_counts": {
                str(rank): count for rank, count in sorted(selected_ranks.items())
            },
            "raw_penman_valid_candidates_mean": _mean(valid_candidate_counts),
            "ranking_method": BEAM_RANKING_METHOD,
            "numeric_step_score_history_retained": False,
        },
        "search": {
            "online_p_all_masked_rows": online_p_all_masked,
            "truncations": truncations,
            "truncated_any_condition": any(truncations.values()),
            "online_p_accepting_rows": online_p_accepting_rows,
            "online_p_nonaccepting_rows": (
                len(identities) - online_p_accepting_rows
            ),
        },
        "uncertainty": {
            "bootstrap_resamples": args.resamples,
            "seed": args.seed,
            "mcnemar_family": list(comparisons),
            "holm_correction": True,
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
            parser.error("stored AMR beam report differs from recomputed evidence")
        print(f"verified AMR beam report: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR beam report: {args.output}")
        print(f"report: {args.output}")
    for name, comparison in comparisons.items():
        print(
            f"{name}: Smatch delta={comparison['corpus_smatch_delta']:.6f} "
            f"CI=[{comparison['delta_ci_lower']:.6f}, "
            f"{comparison['delta_ci_upper']:.6f}]"
        )
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
