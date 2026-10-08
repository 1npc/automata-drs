#!/usr/bin/env python3
"""Fail-closed gate and paired analysis for a registered AMR boundary matrix."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from statistics import mean
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.paired_analysis import exact_mcnemar_p  # noqa: E402


CONFIGURATIONS = ("U", "P", "PF", "WCD")


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
                raise ValueError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _integer(value: Any) -> int | None:
    numeric = _number(value)
    if numeric is None or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _f1(matches: int, predicted: int, gold: int) -> float:
    if predicted == 0 or gold == 0:
        return 0.0
    precision = matches / predicted
    recall = matches / gold
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (
        precision + recall
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _linear_quantile(values: list[int], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile over no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_corpus_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None]:
    if not pairs or resamples <= 0:
        return None, None
    generator = random.Random(seed)
    size = len(pairs)
    deltas: list[float] = []
    for _ in range(resamples):
        base_matches = base_predicted = base_gold = 0
        target_matches = target_predicted = target_gold = 0
        for _ in range(size):
            base, target = pairs[generator.randrange(size)]
            base_matches += int(base["matches"])
            base_predicted += int(base["predicted_triples"])
            base_gold += int(base["gold_triples"])
            target_matches += int(target["matches"])
            target_predicted += int(target["predicted_triples"])
            target_gold += int(target["gold_triples"])
        deltas.append(
            _f1(target_matches, target_predicted, target_gold)
            - _f1(base_matches, base_predicted, base_gold)
        )
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _write_exclusive(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _manifest_expectations(
    contract: dict[str, Any], configurations: tuple[str, ...]
) -> dict[str, Any]:
    generation = contract["generation"]
    data = contract["data"]
    model = contract["model"]
    rows = int(data["limit"]) * len(configurations)
    return {
        "status": "complete",
        "task": "amr",
        "model": model["repo_id"],
        "model_loader": model["model_loader"],
        "requested_revision": model["revision"],
        "resolved_revision": model["revision"],
        "adapter": model["adapter"],
        "path": data["path"],
        "start": data["start"],
        "limit": data["limit"],
        "max_input_tokens": generation["max_input_tokens"],
        "max_new_tokens": generation["max_new_tokens"],
        "num_beams": generation["num_beams"],
        "batch_size": generation["batch_size"],
        "configurations": list(configurations),
        "score_model_logprob": generation["score_model_logprob"],
        "prompt_template": generation["prompt_template"],
        "output_envelope": generation["output_envelope"],
        "source_sha256": data["sha256"],
        "expected_examples": data["limit"],
        "expected_rows": rows,
        "examples": data["limit"],
        "rows": rows,
    }


def _load_bundle(
    contract: dict[str, Any],
    *,
    predictions_key: str,
    scores_key: str,
    configurations: tuple[str, ...],
    failures: list[str],
) -> dict[str, Any]:
    generation = contract["generation"]
    predictions_path = _resolve(generation[predictions_key])
    scores_path = _resolve(generation[scores_key])
    rows = _read_jsonl(predictions_path)
    prediction_sha = sha256_file(predictions_path)
    manifest_path = Path(str(predictions_path) + ".manifest.json")
    manifest = _read_object(manifest_path)
    expected_manifest = _manifest_expectations(contract, configurations)
    expected_manifest["output_sha256"] = prediction_sha
    if len(configurations) == 1:
        expected_manifest["schedule"] = "fixed"
        expected_manifest["stop_on_structural_completion"] = True
    else:
        expected_manifest["schedule"] = "rotate"
        expected_manifest["stop_on_structural_completion"] = True
    mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    label = "+".join(configurations)
    if mismatches:
        failures.append(f"{label}:generation-manifest-contract")
    if manifest.get("physical_gpu_index") not in {"0", "1"}:
        failures.append(f"{label}:physical-gpu")
    expected_rows = int(contract["data"]["limit"]) * len(configurations)
    if len(rows) != expected_rows:
        failures.append(f"{label}:prediction-row-count")
    counts = Counter(str(row.get("configuration")) for row in rows)
    if counts != Counter({name: int(contract["data"]["limit"]) for name in configurations}):
        failures.append(f"{label}:prediction-configuration-counts")

    scores = _read_object(scores_path)
    scorer = scores.get("scorer")
    score_gate = scores.get("gate")
    score_predictions = scores.get("predictions")
    score_groups = scores.get("groups")
    scoring = contract["scoring"]
    if not isinstance(scorer, dict) or any(
        (
            scorer.get("name") != scoring["implementation"],
            scorer.get("version") != scoring["version"],
            scorer.get("solver") != scoring["solver"],
            scorer.get("optimality_required") is not scoring["optimality_required"],
        )
    ):
        failures.append(f"{label}:score-contract")
    if not isinstance(score_gate, dict) or (
        score_gate.get("passed") is not True
        or score_gate.get("all_alignments_optimal") is not True
        or score_gate.get("complete_identity_pairing") is not True
    ):
        failures.append(f"{label}:score-gate")
    if not isinstance(score_predictions, dict) or (
        score_predictions.get("sha256") != prediction_sha
        or score_predictions.get("rows") != len(rows)
    ):
        failures.append(f"{label}:score-prediction-binding")
    if not isinstance(score_groups, dict) or set(score_groups) != set(configurations):
        failures.append(f"{label}:score-configurations")

    return {
        "predictions_path": predictions_path,
        "predictions_sha256": prediction_sha,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_mismatches": mismatches,
        "rows": rows,
        "scores_path": scores_path,
        "scores_sha256": sha256_file(scores_path),
        "scores": scores,
    }


def _condition_rows(
    bundles: list[dict[str, Any]], failures: list[str]
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    list[str],
]:
    by_condition = {name: {} for name in CONFIGURATIONS}
    score_by_condition = {name: {} for name in CONFIGURATIONS}
    for bundle in bundles:
        for row in bundle["rows"]:
            configuration = str(row.get("configuration", ""))
            identifier = str(row.get("id", ""))
            if configuration not in by_condition or not identifier:
                failures.append("unknown-or-empty-prediction-identity")
                continue
            if identifier in by_condition[configuration]:
                failures.append(f"{configuration}:duplicate-prediction-id")
            by_condition[configuration][identifier] = row
        paired_rows = bundle["scores"].get("paired_rows")
        if not isinstance(paired_rows, list):
            failures.append("score-paired-rows-missing")
            continue
        for row in paired_rows:
            if not isinstance(row, dict):
                failures.append("score-paired-row-not-object")
                continue
            configuration = str(row.get("configuration", ""))
            identifier = str(row.get("id", ""))
            if configuration not in score_by_condition or not identifier:
                failures.append("unknown-or-empty-score-identity")
                continue
            if identifier in score_by_condition[configuration]:
                failures.append(f"{configuration}:duplicate-score-id")
            score_by_condition[configuration][identifier] = row

    base_ids = list(by_condition["U"])
    expected_set = set(base_ids)
    for configuration in CONFIGURATIONS:
        if list(by_condition[configuration]) != base_ids:
            failures.append(f"{configuration}:unpaired-or-reordered-identifiers")
        if set(score_by_condition[configuration]) != expected_set:
            failures.append(f"{configuration}:score-identities")
    return by_condition, score_by_condition, base_ids


def _summarize_condition(
    rows: dict[str, dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    *,
    configuration: str,
    failures: list[str],
) -> dict[str, Any]:
    selected = list(rows.values())
    raw_valid = 0
    scoreable = 0
    ended_eos = 0
    ended_structural = 0
    hit_max = 0
    canonical = 0
    all_masked = 0
    processor_rows = 0
    generation_times: list[float] = []
    constraint_times: list[float] = []
    output_tokens: list[float] = []
    constraint_totals = Counter()
    for identifier, row in rows.items():
        scored = score_rows.get(identifier, {})
        row_valid = _nested(row, "validity", "raw_official_penman") is True
        scored_valid = scored.get("raw_official_penman") is True
        if row_valid != scored_valid:
            failures.append(f"{configuration}:row-score-validity-mismatch")
        raw_valid += int(row_valid)
        scoreable += int(scored.get("raw_smatchpp_scoreable") is True)
        ended_eos += int(_nested(row, "generation", "ended_with_eos") is True)
        ended_structural += int(
            _nested(row, "generation", "ended_with_structural_completion") is True
        )
        hit_max += int(_nested(row, "generation", "hit_max_new_tokens") is True)
        canonical += int(_nested(row, "generation", "canonical_tokenization") is True)
        for keys, target in (
            (("timing", "generation_ms"), generation_times),
            (("timing", "constraint_ms"), constraint_times),
            (("generation", "output_tokens_including_eos"), output_tokens),
        ):
            value = _number(_nested(row, *keys))
            if value is None or value < 0:
                failures.append(f"{configuration}:invalid-{'-'.join(keys)}")
            else:
                target.append(value)
        constraint = row.get("constraint")
        if isinstance(constraint, dict):
            for key in (
                "rows",
                "all_masked_rows",
                "allowed_total",
                "completion_certificates_tested",
                "completion_certificates_accepted",
                "prefix_rows_certified",
                "prefix_rows_unavailable",
                "carried_tokens",
                "cached_tokens",
                "replanned_tokens",
                "rejected_without_certificate",
            ):
                if key not in constraint:
                    continue
                value = _integer(constraint[key])
                if value is None:
                    failures.append(f"{configuration}:invalid-{key}")
                else:
                    constraint_totals[key] += value
            all_masked += _integer(constraint.get("all_masked_rows")) or 0
            processor_rows += _integer(constraint.get("rows")) or 0

    return {
        "rows": len(selected),
        "raw_official_penman_valid": raw_valid,
        "raw_smatchpp_scoreable": scoreable,
        "ended_with_eos": ended_eos,
        "ended_with_structural_completion": ended_structural,
        "hit_max_new_tokens": hit_max,
        "canonical_tokenization": canonical,
        "all_masked_rows": all_masked,
        "processor_rows": processor_rows,
        "generation_ms_mean": _mean(generation_times),
        "constraint_ms_mean": _mean(constraint_times),
        "output_tokens_mean": _mean(output_tokens),
        "constraint": dict(sorted(constraint_totals.items())),
    }


def build_report(config_path: Path) -> dict[str, Any]:
    contract = _read_object(config_path)
    if contract.get("schema_version") != 1 or contract.get("kind") != (
        "wcd-amr-boundary-development-probe"
    ):
        raise ValueError("unexpected AMR boundary contract")
    generation = contract.get("generation")
    data = contract.get("data")
    confirmation = contract.get("confirmation_window")
    horizon = contract.get("horizon_selection")
    if not all(isinstance(value, dict) for value in (generation, data, confirmation, horizon)):
        raise ValueError("AMR boundary contract is incomplete")
    assert isinstance(generation, dict)
    assert isinstance(data, dict)
    assert isinstance(confirmation, dict)
    assert isinstance(horizon, dict)
    if generation.get("configurations") != list(CONFIGURATIONS):
        raise ValueError("AMR boundary configurations changed")
    if (data.get("start"), data.get("limit"), generation.get("max_new_tokens")) != (
        0,
        32,
        288,
    ):
        raise ValueError("AMR development window or horizon changed")
    if data.get("selection_performance_conditioned") is not False:
        raise ValueError("AMR development population is performance-conditioned")
    if horizon.get("boundary_outcomes_observed_before_registration") is not False:
        raise ValueError("AMR boundary outcomes were already observed")
    if (
        confirmation.get("start"),
        confirmation.get("limit"),
        confirmation.get("max_new_tokens"),
        confirmation.get("configurations"),
        confirmation.get("outcomes_observed_before_freeze"),
    ) != (32, 96, 288, list(CONFIGURATIONS), False):
        raise ValueError("AMR confirmation window is not frozen")

    failures: list[str] = []
    predecessor_bindings: dict[str, Any] = {}
    for path_key, hash_key in (
        ("predecessor_predictions", "predecessor_predictions_sha256"),
        ("predecessor_manifest", "predecessor_manifest_sha256"),
        ("predecessor_scores", "predecessor_scores_sha256"),
    ):
        path = _resolve(str(horizon[path_key]))
        observed = sha256_file(path)
        if observed != horizon[hash_key]:
            failures.append(f"horizon:{path_key}-hash")
        predecessor_bindings[path_key] = {
            "path": str(path),
            "sha256": observed,
        }
    predecessor_rows = _read_jsonl(
        _resolve(str(horizon["predecessor_predictions"]))
    )
    observed_lengths = sorted(
        int(_nested(row, "generation", "output_tokens_including_eos"))
        for row in predecessor_rows
    )
    if observed_lengths != horizon.get("observed_output_lengths_including_eos_sorted"):
        failures.append("horizon:registered-lengths")
    observed_p90 = _linear_quantile(observed_lengths, 0.90)
    selected_horizon = math.ceil(observed_p90 / 16) * 16
    if not math.isclose(observed_p90, float(horizon.get("observed_p90")), abs_tol=1e-12):
        failures.append("horizon:registered-p90")
    if selected_horizon != horizon.get("selected_max_new_tokens"):
        failures.append("horizon:selection-rule")

    predecessor_gates: dict[str, Any] = {}
    for name, registered in contract["promotion_predecessors"].items():
        path = _resolve(registered["path"])
        observed = sha256_file(path)
        report = _read_object(path)
        if observed != registered["sha256"]:
            failures.append(f"predecessor:{name}-hash")
        if report.get("gate", {}).get("passed") is not True:
            failures.append(f"predecessor:{name}-gate")
        predecessor_gates[name] = {"path": str(path), "sha256": observed}

    up = _load_bundle(
        contract,
        predictions_key="up_predictions",
        scores_key="up_scores",
        configurations=("U", "P"),
        failures=failures,
    )
    pf = _load_bundle(
        contract,
        predictions_key="pf_predictions",
        scores_key="pf_scores",
        configurations=("PF",),
        failures=failures,
    )
    wcd = _load_bundle(
        contract,
        predictions_key="wcd_predictions",
        scores_key="wcd_scores",
        configurations=("WCD",),
        failures=failures,
    )
    bundles = [up, pf, wcd]
    by_condition, score_by_condition, identities = _condition_rows(bundles, failures)
    if len(identities) != 32:
        failures.append("paired-population-size")
    for identifier in identities:
        u = by_condition["U"].get(identifier, {})
        for configuration in CONFIGURATIONS[1:]:
            row = by_condition[configuration].get(identifier, {})
            if row.get("input") != u.get("input") or row.get("reference") != u.get("reference"):
                failures.append(f"{configuration}:input-or-reference-mismatch")

    summaries = {
        configuration: _summarize_condition(
            by_condition[configuration],
            score_by_condition[configuration],
            configuration=configuration,
            failures=failures,
        )
        for configuration in CONFIGURATIONS
    }
    for configuration in ("PF", "WCD"):
        summary = summaries[configuration]
        if summary["rows"] != 32:
            failures.append(f"{configuration}:row-count")
        if summary["raw_official_penman_valid"] != 32:
            failures.append(f"{configuration}:raw-penman-release")
        if summary["ended_with_structural_completion"] != 32:
            failures.append(f"{configuration}:structural-completion")
        if summary["hit_max_new_tokens"] != 0:
            failures.append(f"{configuration}:horizon-exhaustion")
        if summary["all_masked_rows"] != 0:
            failures.append(f"{configuration}:all-masked")

    wcd_constraint = summaries["WCD"]["constraint"]
    if wcd_constraint.get("prefix_rows_unavailable") != 0:
        failures.append("WCD:prefix-unavailable")
    if wcd_constraint.get("prefix_rows_certified") != summaries["WCD"]["processor_rows"]:
        failures.append("WCD:prefix-accounting")
    admitted = sum(
        int(wcd_constraint.get(key, 0))
        for key in ("carried_tokens", "cached_tokens", "replanned_tokens")
    )
    if admitted != wcd_constraint.get("allowed_total"):
        failures.append("WCD:decision-accounting")
    if summaries["WCD"]["processor_rows"] and admitted == 0:
        failures.append("WCD:no-certificate-activity")

    pf_wcd_raw_exact = 0
    pf_wcd_prediction_exact = 0
    for identifier in identities:
        pf_row = by_condition["PF"].get(identifier, {})
        wcd_row = by_condition["WCD"].get(identifier, {})
        pf_wcd_raw_exact += int(pf_row.get("raw_prediction") == wcd_row.get("raw_prediction"))
        pf_wcd_prediction_exact += int(pf_row.get("prediction") == wcd_row.get("prediction"))
    if pf_wcd_raw_exact != 32 or pf_wcd_prediction_exact != 32:
        failures.append("WCD:PF-output-equivalence")

    score_groups: dict[str, dict[str, Any]] = {}
    for bundle in bundles:
        groups = bundle["scores"].get("groups", {})
        if isinstance(groups, dict):
            score_groups.update(groups)
    paired: dict[str, dict[str, Any]] = {}
    scoring = contract["scoring"]
    for offset, target in enumerate(("P", "PF", "WCD")):
        repairs = regressions = both_valid = both_invalid = 0
        wins = losses = ties = changes = accepting_rescues = 0
        score_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        generation_deltas: list[float] = []
        constraint_deltas: list[float] = []
        for identifier in identities:
            base_row = by_condition["U"][identifier]
            target_row = by_condition[target][identifier]
            base_score = score_by_condition["U"][identifier]["raw_smatch"]
            target_score = score_by_condition[target][identifier]["raw_smatch"]
            score_pairs.append((base_score, target_score))
            base_valid = score_by_condition["U"][identifier].get("raw_official_penman") is True
            target_valid = score_by_condition[target][identifier].get("raw_official_penman") is True
            repairs += int(not base_valid and target_valid)
            regressions += int(base_valid and not target_valid)
            both_valid += int(base_valid and target_valid)
            both_invalid += int(not base_valid and not target_valid)
            base_f1 = float(base_score["f1"])
            target_f1 = float(target_score["f1"])
            wins += int(target_f1 > base_f1)
            losses += int(target_f1 < base_f1)
            ties += int(target_f1 == base_f1)
            changes += int(base_row.get("raw_prediction") != target_row.get("raw_prediction"))
            base_accepting = base_valid and (
                _nested(base_row, "generation", "ended_with_eos") is True
            ) and (_nested(base_row, "generation", "hit_max_new_tokens") is not True)
            target_accepting = target_valid and (
                _nested(target_row, "generation", "ended_with_eos") is True
                or _nested(target_row, "generation", "ended_with_structural_completion") is True
            ) and (_nested(target_row, "generation", "hit_max_new_tokens") is not True)
            accepting_rescues += int(not base_accepting and target_accepting)
            base_generation = _number(_nested(base_row, "timing", "generation_ms"))
            target_generation = _number(_nested(target_row, "timing", "generation_ms"))
            if base_generation is not None and target_generation is not None:
                generation_deltas.append(target_generation - base_generation)
            base_constraint = _number(_nested(base_row, "timing", "constraint_ms"))
            target_constraint = _number(_nested(target_row, "timing", "constraint_ms"))
            if base_constraint is not None and target_constraint is not None:
                constraint_deltas.append(target_constraint - base_constraint)
        base_f1 = float(score_groups["U"]["raw_primary_smatch"]["f1"])
        target_f1 = float(score_groups[target]["raw_primary_smatch"]["f1"])
        ci_lower, ci_upper = _bootstrap_corpus_delta(
            score_pairs,
            seed=int(scoring["seed"]) + offset,
            resamples=int(scoring["bootstrap_resamples"]),
        )
        paired[f"{target}-U"] = {
            "records": len(identities),
            "changed_raw_predictions": changes,
            "validity": {
                "repairs": repairs,
                "regressions": regressions,
                "both_valid": both_valid,
                "both_invalid": both_invalid,
                "exact_mcnemar_p": exact_mcnemar_p(repairs, regressions),
            },
            "accepting_rescues": accepting_rescues,
            "smatchpp": {
                "u_corpus_f1": base_f1,
                "target_corpus_f1": target_f1,
                "delta_target_minus_u": target_f1 - base_f1,
                "delta_ci_lower": ci_lower,
                "delta_ci_upper": ci_upper,
                "bootstrap_resamples": scoring["bootstrap_resamples"],
                "seed": int(scoring["seed"]) + offset,
                "per_example_wins": wins,
                "per_example_losses": losses,
                "per_example_ties": ties,
            },
            "timing": {
                "generation_ms_delta_mean": _mean(generation_deltas),
                "constraint_ms_delta_mean": _mean(constraint_deltas),
            },
        }

    bundle_inputs = {}
    for name, bundle in (("UP", up), ("PF", pf), ("WCD", wcd)):
        bundle_inputs[name] = {
            "predictions": str(bundle["predictions_path"]),
            "predictions_sha256": bundle["predictions_sha256"],
            "manifest": str(bundle["manifest_path"]),
            "manifest_sha256": bundle["manifest_sha256"],
            "scores": str(bundle["scores_path"]),
            "scores_sha256": bundle["scores_sha256"],
            "manifest_mismatches": bundle["manifest_mismatches"],
        }
    failures = sorted(set(failures))
    return {
        "schema_version": 1,
        "kind": "wcd-amr-boundary-development-probe-gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "registration": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "timing": "registered-before-four-condition-boundary-generation",
        },
        "horizon_audit": {
            "predecessors": predecessor_bindings,
            "observed_lengths": observed_lengths,
            "observed_p90": observed_p90,
            "selected_max_new_tokens": selected_horizon,
        },
        "predecessor_gates": predecessor_gates,
        "inputs": bundle_inputs,
        "population": {
            "records": len(identities),
            "start": data["start"],
            "limit": data["limit"],
            "paired": True,
            "performance_conditioned": False,
        },
        "summaries": summaries,
        "paired": paired,
        "equivalence": {
            "comparison": "WCD-PF",
            "raw_prediction_exact": pf_wcd_raw_exact,
            "prediction_exact": pf_wcd_prediction_exact,
            "records": len(identities),
        },
        "confirmation_window": confirmation,
        "gate": {
            "passed": not failures,
            "failures": failures,
            "complete_identity_pairing": not any(
                "identity" in failure or "paired-population" in failure
                for failure in failures
            ),
            "performance_threshold_used": False,
        },
    }


def _without_timestamp(report: dict[str, Any]) -> dict[str, Any]:
    value = dict(report)
    value.pop("created_at_utc", None)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite gate: {args.output}")
    try:
        report = build_report(args.config)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR boundary gate differs from recomputed evidence")
        print(f"verified AMR boundary gate: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite gate: {args.output}")
        print(f"gate: {args.output}")
    print(f"gate passed: {report['gate']['passed']}")
    if report["gate"]["failures"]:
        print("failures: " + ", ".join(report["gate"]["failures"]), file=sys.stderr)
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
