"""Exact AMR graph scoring with the official Smatch++ implementation.

The production configuration uses Smatch++'s AMR standardizer and ILP solver.
Unlike the legacy Smatch hill climber, every reported alignment carries an
optimality certificate (objective value equals the solver upper bound).
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def create_amr_smatchpp(*, max_solver_seconds: int = 240) -> Any:
    """Create the frozen best-practice Smatch++ AMR evaluator."""

    if max_solver_seconds <= 0:
        raise ValueError("max_solver_seconds must be positive")
    from smatchpp import Smatchpp, eval_statistics, solvers
    from smatchpp.formalism.amr.tools import AMRStandardizer

    return Smatchpp(
        alignmentsolver=solvers.ILP(max_seconds=max_solver_seconds),
        graph_standardizer=AMRStandardizer(),
        printer=eval_statistics.ResultPrinter(
            score_type="micro",
            do_bootstrap=False,
            output_format="json",
        ),
    )


def _standardized_graph(measure: Any, surface: str) -> list[tuple[str, str, str]]:
    graph = measure.graph_reader.string2graph(surface)
    graph = measure.graph_standardizer.standardize(graph)
    if not graph or not any(relation == ":instance" for _, relation, _ in graph):
        raise ValueError("surface has no AMR instance triple")
    return graph


def smatchpp_scoreable(measure: Any, surface: str) -> bool:
    """Return whether Smatch++ can construct a non-empty AMR graph."""

    try:
        _standardized_graph(measure, surface)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    return True


def _zero_prediction_score(measure: Any, reference: str) -> dict[str, Any]:
    gold = len(_standardized_graph(measure, reference))
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "matches": 0,
        "matches_forward": 0.0,
        "matches_reverse": 0.0,
        "predicted_triples": 0,
        "gold_triples": gold,
        "scoreable": False,
        "alignment_required": False,
        "alignment_optimal": True,
        "optimization_value": None,
        "optimization_upper_bound": None,
        "optimality_gap": 0.0,
    }


def _score_from_statistics(
    *,
    matches_forward: float,
    matches_reverse: float,
    predicted: float,
    gold: float,
) -> tuple[float, float, float]:
    precision = matches_forward / predicted if predicted else 0.0
    recall = matches_reverse / gold if gold else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def score_amr_pair(
    measure: Any,
    prediction: str | None,
    reference: str,
) -> dict[str, Any]:
    """Score one prediction/reference pair and expose the ILP certificate."""

    _standardized_graph(measure, reference)
    if prediction is None or not smatchpp_scoreable(measure, prediction):
        return _zero_prediction_score(measure, reference)

    match, status, _ = measure.process_pair(prediction, reference)
    statistics = match.get("main")
    if statistics is None or len(statistics) != 4:
        raise ValueError("Smatch++ returned malformed main match statistics")
    forward, reverse, predicted, gold = (float(value) for value in statistics)
    objective_value, objective_upper_bound = (float(value) for value in status)
    optimality_gap = max(0.0, objective_upper_bound - objective_value)
    alignment_optimal = math.isclose(
        objective_value,
        objective_upper_bound,
        rel_tol=1e-9,
        abs_tol=1e-8,
    )
    precision, recall, f1 = _score_from_statistics(
        matches_forward=forward,
        matches_reverse=reverse,
        predicted=predicted,
        gold=gold,
    )
    if not math.isclose(forward, reverse, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("Smatch++ ID matching produced asymmetric match counts")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": int(round(forward)),
        "matches_forward": forward,
        "matches_reverse": reverse,
        "predicted_triples": int(round(predicted)),
        "gold_triples": int(round(gold)),
        "scoreable": True,
        "alignment_required": True,
        "alignment_optimal": alignment_optimal,
        "optimization_value": objective_value,
        "optimization_upper_bound": objective_upper_bound,
        "optimality_gap": optimality_gap,
    }


def aggregate_amr_scores(pair_scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pairwise Smatch++ sufficient statistics into micro scores."""

    forward = sum(float(score["matches_forward"]) for score in pair_scores)
    reverse = sum(float(score["matches_reverse"]) for score in pair_scores)
    predicted = sum(int(score["predicted_triples"]) for score in pair_scores)
    gold = sum(int(score["gold_triples"]) for score in pair_scores)
    precision, recall, f1 = _score_from_statistics(
        matches_forward=forward,
        matches_reverse=reverse,
        predicted=predicted,
        gold=gold,
    )
    required = [score for score in pair_scores if score["alignment_required"]]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": int(round(forward)),
        "matches_forward": forward,
        "matches_reverse": reverse,
        "predicted_triples": predicted,
        "gold_triples": gold,
        "records": len(pair_scores),
        "scoreable_records": sum(bool(score["scoreable"]) for score in pair_scores),
        "invalid_records": sum(not bool(score["scoreable"]) for score in pair_scores),
        "alignment_records": len(required),
        "optimal_alignment_records": sum(
            bool(score["alignment_optimal"]) for score in required
        ),
        "all_alignments_optimal": all(
            bool(score["alignment_optimal"]) for score in required
        ),
        "max_optimality_gap": max(
            (float(score["optimality_gap"]) for score in required),
            default=0.0,
        ),
    }


def score_amr_corpus(
    measure: Any,
    predictions: Iterable[str | None],
    references: Iterable[str],
) -> dict[str, Any]:
    """Return exact micro Smatch++ with invalid outputs retained as zeros."""

    prediction_values = list(predictions)
    reference_values = list(references)
    if len(prediction_values) != len(reference_values):
        raise ValueError("prediction/reference counts differ")
    cache: dict[tuple[str | None, str], dict[str, Any]] = {}
    scores: list[dict[str, Any]] = []
    for prediction, reference in zip(
        prediction_values, reference_values, strict=True
    ):
        key = (prediction, reference)
        if key not in cache:
            cache[key] = score_amr_pair(measure, prediction, reference)
        scores.append(cache[key])
    return aggregate_amr_scores(scores)
