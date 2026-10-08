"""Exact adaptive score ranking for expensive token-certificate predicates.

This module is deliberately not wired into the frozen publication generator
yet.  It provides a separately testable optimization primitive: grow a GPU
top-k pool, close every tie at the boundary, and stop only after the required
number of certified tokens is known to precede every omitted token under the
total order ``(-score, token_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


TokenPredicate = Callable[[int], bool]


@dataclass(frozen=True, slots=True)
class RankedFilterStats:
    vocabulary_size: int
    required: int
    topk_rounds: int
    peak_candidate_pool: int
    cumulative_candidate_pool: int
    python_scalar_values_materialized: int
    boundary_ties_added: int
    certificate_tests: int
    full_vocabulary_reached: bool


def _validate_scores(scores: Any, required: int) -> int:
    if getattr(scores, "ndim", None) != 1:
        raise ValueError("scores must be a one-dimensional tensor")
    vocabulary_size = int(scores.shape[0])
    if vocabulary_size <= 0:
        raise ValueError("scores must contain at least one token")
    if required <= 0:
        raise ValueError("required must be positive")
    if not bool(scores.isfinite().all().item()):
        raise ValueError("scores must all be finite")
    return vocabulary_size


def _ordered_ids(scores: Any, token_ids: list[int]) -> list[int]:
    values = [
        float(value)
        for value in scores[token_ids].detach().to("cpu").tolist()
    ]
    return [
        token_id
        for token_id, _ in sorted(
            zip(token_ids, values, strict=True),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def exhaustive_certified_tokens(
    scores: Any,
    certify: TokenPredicate,
    *,
    required: int,
) -> tuple[tuple[int, ...], RankedFilterStats]:
    """Reference result after certifying every vocabulary item."""

    vocabulary_size = _validate_scores(scores, required)
    certified = [
        token_id
        for token_id in range(vocabulary_size)
        if certify(token_id)
    ]
    ordered = _ordered_ids(scores, certified) if certified else []
    return (
        tuple(ordered[:required]),
        RankedFilterStats(
            vocabulary_size=vocabulary_size,
            required=required,
            topk_rounds=0,
            peak_candidate_pool=vocabulary_size,
            cumulative_candidate_pool=vocabulary_size,
            # Token IDs come from Python's range; only certified scores are
            # copied from the tensor for deterministic ordering.
            python_scalar_values_materialized=len(certified),
            boundary_ties_added=0,
            certificate_tests=vocabulary_size,
            full_vocabulary_reached=True,
        ),
    )


def adaptive_certified_tokens(
    scores: Any,
    certify: TokenPredicate,
    *,
    required: int,
    initial_k: int = 32,
) -> tuple[tuple[int, ...], RankedFilterStats]:
    """Return the exact top certified tokens without sorting the full tensor.

    At every round, all tokens tied with the k-th score are included.  Once
    ``required`` certified tokens occur in that closed pool, every omitted
    token has a strictly lower score, so the returned prefix equals exhaustive
    certification under the documented total order.  Certificate results are
    memoized across expansions.
    """

    vocabulary_size = _validate_scores(scores, required)
    if initial_k <= 0:
        raise ValueError("initial_k must be positive")
    k = min(vocabulary_size, max(required, initial_k))
    decisions: dict[int, bool] = {}
    rounds = 0
    peak_pool = 0
    cumulative_pool = 0
    boundary_ties_added = 0
    full_vocabulary_reached = False

    while True:
        rounds += 1
        top_values = scores.topk(k, largest=True, sorted=False).values
        cutoff = top_values.min()
        candidate_ids = [
            int(token_id)
            for token_id in (scores >= cutoff).nonzero().flatten().tolist()
        ]
        peak_pool = max(peak_pool, len(candidate_ids))
        cumulative_pool += len(candidate_ids)
        boundary_ties_added = max(
            boundary_ties_added, len(candidate_ids) - k
        )
        ordered = _ordered_ids(scores, candidate_ids)
        accepted: list[int] = []
        for token_id in ordered:
            decision = decisions.get(token_id)
            if decision is None:
                decision = bool(certify(token_id))
                decisions[token_id] = decision
            if decision:
                accepted.append(token_id)
                if len(accepted) >= required:
                    return (
                        tuple(accepted),
                        RankedFilterStats(
                            vocabulary_size=vocabulary_size,
                            required=required,
                            topk_rounds=rounds,
                            peak_candidate_pool=peak_pool,
                            cumulative_candidate_pool=cumulative_pool,
                            # Every round copies candidate IDs and their
                            # corresponding scores; expanded pools are copied
                            # again and are deliberately counted again.
                            python_scalar_values_materialized=(
                                2 * cumulative_pool
                            ),
                            boundary_ties_added=boundary_ties_added,
                            certificate_tests=len(decisions),
                            full_vocabulary_reached=(k == vocabulary_size),
                        ),
                    )
        if k == vocabulary_size:
            full_vocabulary_reached = True
            return (
                tuple(accepted),
                RankedFilterStats(
                    vocabulary_size=vocabulary_size,
                    required=required,
                    topk_rounds=rounds,
                    peak_candidate_pool=peak_pool,
                    cumulative_candidate_pool=cumulative_pool,
                    python_scalar_values_materialized=2 * cumulative_pool,
                    boundary_ties_added=boundary_ties_added,
                    certificate_tests=len(decisions),
                    full_vocabulary_reached=full_vocabulary_reached,
                ),
            )
        k = min(vocabulary_size, k * 2)


__all__ = [
    "RankedFilterStats",
    "adaptive_certified_tokens",
    "exhaustive_certified_tokens",
]
