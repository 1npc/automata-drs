#!/usr/bin/env python3
"""Paired U/P inference from complete AMR generation and Smatch evidence."""

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

from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.paired_analysis import exact_mcnemar_p  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
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
                raise ValueError(f"row {line_number} is not an object")
            rows.append(value)
    return rows


def _f1(matches: int, predicted: int, gold: int) -> float:
    if predicted == 0 or gold == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / gold
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _bootstrap_corpus_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None]:
    if not pairs or resamples <= 0:
        return None, None
    generator = random.Random(seed)
    deltas: list[float] = []
    size = len(pairs)
    for _ in range(resamples):
        u_match = u_predicted = u_gold = 0
        p_match = p_predicted = p_gold = 0
        for _ in range(size):
            u_score, p_score = pairs[generator.randrange(size)]
            u_match += int(u_score["matches"])
            u_predicted += int(u_score["predicted_triples"])
            u_gold += int(u_score["gold_triples"])
            p_match += int(p_score["matches"])
            p_predicted += int(p_score["predicted_triples"])
            p_gold += int(p_score["gold_triples"])
        deltas.append(
            _f1(p_match, p_predicted, p_gold)
            - _f1(u_match, u_predicted, u_gold)
        )
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _bootstrap_mean_delta(
    values: list[float], *, seed: int, resamples: int
) -> tuple[float | None, float | None]:
    if not values or resamples <= 0:
        return None, None
    generator = random.Random(seed)
    size = len(values)
    samples = [
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(resamples)
    ]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


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
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=5000)
    args = parser.parse_args()
    if args.resamples <= 0:
        parser.error("resamples must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite paired report: {args.output}")

    try:
        rows = _read_jsonl(args.predictions)
        scores = _read_json(args.scores)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    if not scores.get("gate", {}).get("passed"):
        parser.error("AMR Smatch report gate did not pass")
    if scores.get("predictions", {}).get("sha256") != sha256_file(args.predictions):
        parser.error("AMR Smatch report does not bind the prediction file")

    by_condition: dict[str, dict[str, dict[str, Any]]] = {"U": {}, "P": {}}
    for row in rows:
        configuration = str(row.get("configuration", ""))
        if configuration not in by_condition:
            continue
        identifier = str(row.get("id", ""))
        if not identifier or identifier in by_condition[configuration]:
            parser.error(f"missing or duplicate {configuration} identity: {identifier}")
        by_condition[configuration][identifier] = row
    if set(by_condition["U"]) != set(by_condition["P"]):
        parser.error("U and P identity sets are not exactly paired")
    identities = sorted(by_condition["U"])
    if not identities:
        parser.error("no paired U/P rows")

    score_by_condition: dict[str, dict[str, dict[str, Any]]] = {"U": {}, "P": {}}
    for row in scores.get("paired_rows", []):
        configuration = str(row.get("configuration", ""))
        identifier = str(row.get("id", ""))
        if configuration in score_by_condition:
            if identifier in score_by_condition[configuration]:
                parser.error(f"duplicate scored identity: {configuration} {identifier}")
            score_by_condition[configuration][identifier] = row
    for configuration in ("U", "P"):
        if set(score_by_condition[configuration]) != set(identities):
            parser.error(f"{configuration} score identities differ from predictions")

    repairs = regressions = both_valid = both_invalid = 0
    wins = losses = ties = 0
    changed = 0
    likelihood_deltas: list[float] = []
    generation_u: list[float] = []
    generation_p: list[float] = []
    constraint_p: list[float] = []
    all_masked_p = 0
    truncations = {"U": 0, "P": 0}
    error_reasons: dict[str, Counter[str]] = {"U": Counter(), "P": Counter()}
    smatch_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for identifier in identities:
        u = by_condition["U"][identifier]
        p = by_condition["P"][identifier]
        if u.get("reference") != p.get("reference"):
            parser.error(f"reference mismatch for {identifier}")
        u_score = score_by_condition["U"][identifier]["raw_smatch"]
        p_score = score_by_condition["P"][identifier]["raw_smatch"]
        smatch_pairs.append((u_score, p_score))
        u_valid = bool(score_by_condition["U"][identifier]["raw_official_penman"])
        p_valid = bool(score_by_condition["P"][identifier]["raw_official_penman"])
        repairs += int(not u_valid and p_valid)
        regressions += int(u_valid and not p_valid)
        both_valid += int(u_valid and p_valid)
        both_invalid += int(not u_valid and not p_valid)
        if not u_valid:
            error_reasons["U"][
                str(u.get("validity", {}).get("raw_official_penman_reason") or "unknown")
            ] += 1
        if not p_valid:
            error_reasons["P"][
                str(p.get("validity", {}).get("raw_official_penman_reason") or "unknown")
            ] += 1

        u_f1 = float(u_score["f1"])
        p_f1 = float(p_score["f1"])
        wins += int(p_f1 > u_f1)
        losses += int(p_f1 < u_f1)
        ties += int(p_f1 == u_f1)
        changed += int(u.get("raw_prediction") != p.get("raw_prediction"))

        u_logprob = _finite_number(u.get("score", {}).get("raw_model_mean_logprob"))
        p_logprob = _finite_number(p.get("score", {}).get("raw_model_mean_logprob"))
        if u_logprob is not None and p_logprob is not None:
            likelihood_deltas.append(p_logprob - u_logprob)
        for condition, row, target in (
            ("U", u, generation_u), ("P", p, generation_p)
        ):
            generation_ms = _finite_number(row.get("timing", {}).get("generation_ms"))
            if generation_ms is not None:
                target.append(generation_ms)
            truncations[condition] += int(
                bool(row.get("generation", {}).get("hit_max_new_tokens"))
            )
        constraint_ms = _finite_number(p.get("timing", {}).get("constraint_ms"))
        if constraint_ms is not None:
            constraint_p.append(constraint_ms)
        constraint = p.get("constraint")
        if isinstance(constraint, dict):
            all_masked_p += int(constraint.get("all_masked_rows", 0))

    u_group = scores["groups"]["U"]["raw_primary_smatch"]
    p_group = scores["groups"]["P"]["raw_primary_smatch"]
    observed_delta = float(p_group["f1"]) - float(u_group["f1"])
    ci_lower, ci_upper = _bootstrap_corpus_delta(
        smatch_pairs, seed=args.seed, resamples=args.resamples
    )
    likelihood_mean = _mean(likelihood_deltas)
    likelihood_lower, likelihood_upper = _bootstrap_mean_delta(
        likelihood_deltas, seed=args.seed + 1, resamples=args.resamples
    )
    mcnemar = exact_mcnemar_p(repairs, regressions)
    report = {
        "schema_version": 1,
        "audit": "amr-paired-u-p-analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "population": {
            "records": len(identities),
            "paired": True,
            "conditioned_on_u_failure": False,
        },
        "inputs": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "scores": str(args.scores),
            "scores_sha256": sha256_file(args.scores),
        },
        "validity": {
            "u_valid": both_valid + regressions,
            "p_valid": both_valid + repairs,
            "repairs": repairs,
            "regressions": regressions,
            "both_valid": both_valid,
            "both_invalid": both_invalid,
            "exact_mcnemar_p": mcnemar,
            "holm_adjusted_p": mcnemar,
            "error_reasons": {
                condition: dict(sorted(counter.items()))
                for condition, counter in error_reasons.items()
            },
        },
        "smatch": {
            "u_corpus_f1": float(u_group["f1"]),
            "p_corpus_f1": float(p_group["f1"]),
            "delta_p_minus_u": observed_delta,
            "delta_ci_lower": ci_lower,
            "delta_ci_upper": ci_upper,
            "bootstrap_resamples": args.resamples,
            "seed": args.seed,
            "per_example_wins": wins,
            "per_example_losses": losses,
            "per_example_ties": ties,
        },
        "probability": {
            "paired_records": len(likelihood_deltas),
            "mean_raw_model_logprob_delta_p_minus_u": likelihood_mean,
            "delta_ci_lower": likelihood_lower,
            "delta_ci_upper": likelihood_upper,
        },
        "search": {
            "changed_raw_predictions": changed,
            "u_generation_ms_mean": _mean(generation_u),
            "p_generation_ms_mean": _mean(generation_p),
            "p_constraint_ms_mean": _mean(constraint_p),
            "truncations": truncations,
            "p_all_masked_rows": all_masked_p,
        },
        "gate": {
            "passed": True,
            "complete_identity_pairing": True,
            "score_hash_matches": True,
            "performance_threshold_used": False,
        },
    }
    if args.verify_existing is not None:
        try:
            stored = _read_json(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR paired report differs from recomputed evidence")
        print(f"verified AMR paired report: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite paired report: {args.output}")
        print(f"report: {args.output}")
    print(
        f"P-U Smatch={observed_delta:.6f} "
        f"CI=[{ci_lower:.6f}, {ci_upper:.6f}]"
    )
    print(f"validity repairs={repairs} regressions={regressions} p={mcnemar:.6g}")
    print("gate passed: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
