"""Fail-closed paired inference for constrained-decoding experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
import random
import statistics
from typing import Any


@dataclass(frozen=True, slots=True)
class _Metric:
    name: str
    kind: str
    extract: Callable[[dict[str, Any]], float]


def _strategy_key(row: Mapping[str, Any]) -> str:
    decode = row.get("decode")
    if not isinstance(decode, Mapping):
        raise ValueError("experiment row has no decode object")
    strategy = decode.get("strategy")
    if strategy == "greedy":
        return "greedy"
    if strategy == "beam":
        beams = decode.get("num_beams")
        if not isinstance(beams, int) or beams <= 0:
            raise ValueError("beam row has invalid num_beams")
        return f"beam{beams}"
    raise ValueError(f"unsupported decode strategy: {strategy!r}")


def experiment_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("id", "")),
        str(row.get("configuration", "")),
        _strategy_key(row),
    )


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing or non-finite paired metric {label}")
    return float(value)


def _validity(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("validity")
    if not isinstance(value, Mapping):
        raise ValueError("experiment row has no validity object")
    return value


def _raw_sbn(row: dict[str, Any]) -> float:
    validity = _validity(row)
    return float(validity.get("raw_sbn", validity.get("sbn")) is True)


def _sbn(row: dict[str, Any]) -> float:
    return float(_validity(row).get("sbn") is True)


def _wellformed(row: dict[str, Any]) -> float:
    validity = _validity(row)
    return float(
        validity.get("raw_sbn", validity.get("sbn")) is True
        and validity.get(
            "raw_clf_referee", validity.get("clf_referee")
        )
        is True
    )


def _exact(row: dict[str, Any]) -> float:
    if "prediction" not in row or "reference" not in row:
        raise ValueError("exact serialization requires prediction and reference")
    return float(row["prediction"] == row["reference"])


def _nested_number(row: Mapping[str, Any], group: str, name: str) -> float:
    container = row.get(group)
    if not isinstance(container, Mapping):
        raise ValueError(f"experiment row has no {group} object")
    return _finite(container.get(name), f"{group}.{name}")


METRICS = (
    _Metric("raw_sbn_valid", "binary", _raw_sbn),
    _Metric("sbn_valid", "binary", _sbn),
    _Metric("wellformed_valid", "binary", _wellformed),
    _Metric("exact_serialization", "binary", _exact),
    _Metric(
        "sbn_smatch",
        "continuous",
        lambda row: _nested_number(row, "semantic_scores", "sbn_smatch"),
    ),
    _Metric(
        "raw_model_mean_logprob",
        "continuous",
        lambda row: _nested_number(
            row, "score", "raw_model_mean_logprob"
        ),
    ),
    _Metric(
        "generation_ms",
        "continuous",
        lambda row: _nested_number(row, "timing", "generation_ms"),
    ),
    _Metric(
        "constraint_ms",
        "continuous",
        lambda row: _nested_number(row, "timing", "constraint_ms"),
    ),
    _Metric(
        "decoder_steps",
        "continuous",
        lambda row: _nested_number(row, "generation", "decoder_steps"),
    ),
)


def exact_mcnemar_p(fixed: int, regressed: int) -> float:
    """Two-sided exact McNemar p-value using the conditional binomial test."""

    if fixed < 0 or regressed < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    tail = min(fixed, regressed)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    denominator = 2**discordant
    return min(1.0, numerator / denominator)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _derived_seed(seed: int, strategy: str, population_hash: str) -> int:
    digest = sha256(
        f"{seed}\0{strategy}\0{population_hash}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_intervals(
    deltas: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    lengths = {len(values) for values in deltas.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("paired bootstrap metrics need one shared nonempty population")
    size = next(iter(lengths))
    names = tuple(deltas)
    samples = {name: [] for name in names}
    population = range(size)
    rng = random.Random(seed)
    for _ in range(resamples):
        indices = rng.choices(population, k=size)
        for name in names:
            values = deltas[name]
            samples[name].append(
                sum(values[index] for index in indices) / size
            )
    alpha = (1.0 - confidence) / 2.0
    return {
        name: (
            _percentile(samples[name], alpha),
            _percentile(samples[name], 1.0 - alpha),
        )
        for name in names
    }


def paired_bootstrap_intervals(
    deltas: Mapping[str, Sequence[float]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Public deterministic paired-percentile bootstrap over shared rows."""

    return _bootstrap_intervals(
        deltas,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


def _holm_adjust(
    comparisons: list[dict[str, Any]], metrics: Sequence[_Metric]
) -> None:
    binary_names = [metric.name for metric in metrics if metric.kind == "binary"]
    for name in binary_names:
        family = sorted(
            (
                (float(comparison["metrics"][name]["mcnemar_exact_p"]), index)
                for index, comparison in enumerate(comparisons)
            ),
            key=lambda item: (item[0], item[1]),
        )
        running = 0.0
        total = len(family)
        for rank, (p_value, comparison_index) in enumerate(family):
            adjusted = min(1.0, (total - rank) * p_value)
            running = max(running, adjusted)
            comparisons[comparison_index]["metrics"][name][
                "mcnemar_exact_p_holm"
            ] = running


def analyze_paired_experiment(
    records: Iterable[dict[str, Any]],
    *,
    configurations: Sequence[str] | None = None,
    baseline_configuration: str = "U",
    metric_names: Sequence[str] | None = None,
    bootstrap_resamples: int = 5_000,
    confidence: float = 0.95,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Analyze complete baseline-versus-treatment cells using paired examples.

    The historical default remains ``U`` with every registered metric. A
    caller may instead name another observed baseline (for example ``W`` in a
    bounded-horizon stress matrix) and request only metrics that the frozen
    run actually recorded. Both choices are persisted in the report.
    """

    if not isinstance(baseline_configuration, str) or not baseline_configuration:
        raise ValueError("baseline configuration must be a nonempty string")
    metrics_by_name = {metric.name: metric for metric in METRICS}
    if metric_names is None:
        selected_metrics = METRICS
    else:
        requested_metrics = list(metric_names)
        if not requested_metrics or len(set(requested_metrics)) != len(
            requested_metrics
        ):
            raise ValueError("metric names must be nonempty and unique")
        unknown_metrics = sorted(set(requested_metrics) - set(metrics_by_name))
        if unknown_metrics:
            raise ValueError(f"unknown paired metrics: {unknown_metrics}")
        selected_metrics = tuple(
            metric for metric in METRICS if metric.name in requested_metrics
        )

    rows = list(records)
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        identity = experiment_identity(row)
        if not all(identity):
            raise ValueError(f"incomplete experiment identity: {identity}")
        if identity in indexed:
            raise ValueError(f"duplicate experiment cell: {identity}")
        indexed[identity] = row
        groups[(identity[1], identity[2])].add(identity[0])

    observed_configurations = sorted(
        {
            configuration
            for configuration, _ in groups
            if configuration != baseline_configuration
        }
    )
    selected = (
        list(configurations)
        if configurations is not None
        else observed_configurations
    )
    if not selected:
        raise ValueError("paired analysis requires a non-U configuration")
    if len(set(selected)) != len(selected) or baseline_configuration in selected:
        raise ValueError(
            "paired configurations must be unique and exclude the baseline"
        )
    missing_configurations = sorted(set(selected) - set(observed_configurations))
    if missing_configurations:
        raise ValueError(
            f"requested configurations are absent: {missing_configurations}"
        )

    comparisons: list[dict[str, Any]] = []
    for configuration in selected:
        strategies = sorted(
            strategy
            for observed, strategy in groups
            if observed == configuration
        )
        for strategy in strategies:
            constrained_ids = groups[(configuration, strategy)]
            baseline_ids = groups.get((baseline_configuration, strategy), set())
            if constrained_ids != baseline_ids:
                missing_baseline = sorted(constrained_ids - baseline_ids)
                missing_constrained = sorted(baseline_ids - constrained_ids)
                raise ValueError(
                    f"unpaired population for {configuration}/{strategy}: "
                    f"missing {baseline_configuration}="
                    f"{missing_baseline[:5]}, "
                    f"missing constrained={missing_constrained[:5]}"
                )
            identifiers = sorted(constrained_ids)
            if not identifiers:
                raise ValueError(
                    f"empty paired population for {configuration}/{strategy}"
                )

            pairs = [
                (
                    indexed[(identifier, baseline_configuration, strategy)],
                    indexed[(identifier, configuration, strategy)],
                )
                for identifier in identifiers
            ]
            delta_vectors: dict[str, list[float]] = {}
            values_by_metric: dict[str, tuple[list[float], list[float]]] = {}
            for metric in selected_metrics:
                unconstrained_values = [metric.extract(pair[0]) for pair in pairs]
                constrained_values = [metric.extract(pair[1]) for pair in pairs]
                values_by_metric[metric.name] = (
                    unconstrained_values,
                    constrained_values,
                )
                delta_vectors[metric.name] = [
                    constrained - unconstrained
                    for unconstrained, constrained in zip(
                        unconstrained_values, constrained_values
                    )
                ]

            population_hash = sha256(
                ("\n".join(identifiers) + "\n").encode("utf-8")
            ).hexdigest()
            # Comparisons on the same paired population deliberately share
            # resampled indices. Identical delta vectors therefore receive
            # identical Monte Carlo intervals instead of seed noise that
            # merely reflects the configuration label.
            comparison_seed = _derived_seed(
                seed, strategy, population_hash
            )
            intervals = _bootstrap_intervals(
                delta_vectors,
                resamples=bootstrap_resamples,
                confidence=confidence,
                seed=comparison_seed,
            )
            metric_reports: dict[str, Any] = {}
            for metric in selected_metrics:
                unconstrained_values, constrained_values = values_by_metric[
                    metric.name
                ]
                deltas = delta_vectors[metric.name]
                lower, upper = intervals[metric.name]
                metric_report = {
                    "kind": metric.kind,
                    "paired": len(pairs),
                    "unconstrained_mean": statistics.fmean(
                        unconstrained_values
                    ),
                    "constrained_mean": statistics.fmean(
                        constrained_values
                    ),
                    "delta_mean": statistics.fmean(deltas),
                    "delta_median": statistics.median(deltas),
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                }
                if metric.kind == "binary":
                    fixed = sum(
                        unconstrained == 0.0 and constrained == 1.0
                        for unconstrained, constrained in zip(
                            unconstrained_values, constrained_values
                        )
                    )
                    regressed = sum(
                        unconstrained == 1.0 and constrained == 0.0
                        for unconstrained, constrained in zip(
                            unconstrained_values, constrained_values
                        )
                    )
                    metric_report.update(
                        {
                            "fixed": fixed,
                            "regressed": regressed,
                            "discordant": fixed + regressed,
                            "mcnemar_exact_p": exact_mcnemar_p(
                                fixed, regressed
                            ),
                        }
                    )
                metric_reports[metric.name] = metric_report

            comparisons.append(
                {
                    "configuration": configuration,
                    "strategy": strategy,
                    "paired": len(pairs),
                    "population_ids_sha256": population_hash,
                    "bootstrap_seed": comparison_seed,
                    "metrics": metric_reports,
                }
            )

    _holm_adjust(comparisons, selected_metrics)
    return {
        "schema_version": 1,
        "analysis": "paired-constrained-decoding",
        "records": len(rows),
        "unique_examples": len({identity[0] for identity in indexed}),
        "parameters": {
            "paired_unit": "example",
            "baseline_configuration": baseline_configuration,
            "bootstrap": "paired-percentile",
            "bootstrap_resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
            "mcnemar": "two-sided-exact-conditional-binomial",
            "multiplicity": "Holm within each binary metric across comparisons",
        },
        "comparisons": comparisons,
        "interpretation": {
            "delta_direction": (
                f"treatment minus {baseline_configuration} baseline"
            ),
            "timing_ci_captures_example_variation_not_repeated_system_noise": True,
            "conditioned_populations_are_population_estimates": False,
        },
    }


def paired_report_mismatches(
    report: Mapping[str, Any],
    *,
    expected_total_records: int,
    expected_configurations: Sequence[str],
    expected_records_per_comparison: int,
    expected_population_scope: str,
    prediction_sha256: str,
    scores_sha256: str,
    registered_parameters: Mapping[str, Any],
    expected_strategy: str = "greedy",
) -> list[str]:
    """Return fail-closed integrity mismatches for a persisted analysis."""

    mismatches: list[str] = []
    if report.get("schema_version") != 1:
        mismatches.append("schema_version")
    if report.get("analysis") != "paired-constrained-decoding":
        mismatches.append("analysis")
    if report.get("records") != expected_total_records:
        mismatches.append("records")
    if report.get("population_scope") != expected_population_scope:
        mismatches.append("population_scope")
    gate = report.get("gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        mismatches.append("gate")
    source = report.get("source")
    if not isinstance(source, Mapping):
        mismatches.append("source")
        source = {}
    if source.get("predictions_sha256") != prediction_sha256:
        mismatches.append("prediction_sha256")
    if source.get("scores_sha256") != scores_sha256:
        mismatches.append("scores_sha256")

    expected_parameters = {
        "paired_unit": registered_parameters.get("paired_unit"),
        "baseline_configuration": registered_parameters.get(
            "baseline_configuration"
        ),
        "bootstrap": registered_parameters.get("bootstrap"),
        "bootstrap_resamples": registered_parameters.get(
            "bootstrap_resamples"
        ),
        "confidence": registered_parameters.get("confidence"),
        "seed": registered_parameters.get("seed"),
        "mcnemar": registered_parameters.get("mcnemar"),
        "multiplicity": registered_parameters.get("multiplicity"),
    }
    if report.get("parameters") != expected_parameters:
        mismatches.append("parameters")

    registered_metrics = set(registered_parameters.get("primary_metrics", [])) | set(
        registered_parameters.get("secondary_metrics", [])
    )
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        mismatches.append("comparisons")
        return mismatches
    keyed = {
        (str(item.get("configuration")), str(item.get("strategy"))): item
        for item in comparisons
        if isinstance(item, Mapping)
    }
    expected_keys = {
        (str(configuration), expected_strategy)
        for configuration in expected_configurations
    }
    if len(keyed) != len(comparisons) or set(keyed) != expected_keys:
        mismatches.append("comparison_cells")
        return mismatches
    for key, comparison in keyed.items():
        if comparison.get("paired") != expected_records_per_comparison:
            mismatches.append(f"paired:{key[0]}:{key[1]}")
        metrics = comparison.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != registered_metrics:
            mismatches.append(f"metrics:{key[0]}:{key[1]}")
            continue
        for name, metric in metrics.items():
            if not isinstance(metric, Mapping):
                mismatches.append(f"metric_object:{key[0]}:{name}")
                continue
            numeric_fields = (
                "unconstrained_mean",
                "constrained_mean",
                "delta_mean",
                "delta_median",
                "delta_ci_lower",
                "delta_ci_upper",
            )
            if any(
                not isinstance(metric.get(field), (int, float))
                or not math.isfinite(float(metric[field]))
                for field in numeric_fields
            ):
                mismatches.append(f"finite_metric:{key[0]}:{name}")
            elif float(metric["delta_ci_lower"]) > float(
                metric["delta_ci_upper"]
            ):
                mismatches.append(f"metric_interval:{key[0]}:{name}")
            if metric.get("kind") == "binary" and any(
                not isinstance(metric.get(field), (int, float))
                or not 0.0 <= float(metric[field]) <= 1.0
                for field in ("mcnemar_exact_p", "mcnemar_exact_p_holm")
            ):
                mismatches.append(f"binary_p_value:{key[0]}:{name}")
    return mismatches


__all__ = [
    "analyze_paired_experiment",
    "exact_mcnemar_p",
    "experiment_identity",
    "paired_bootstrap_intervals",
    "paired_report_mismatches",
]
