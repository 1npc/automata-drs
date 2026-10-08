"""Post-primary real-logit audit for adaptive subword ranking.

This module is intentionally separate from :mod:`automata_drs.subword` and
the frozen publication generator.  It provides two experimental processors:

* a stable full-sort reference with the declared ``(-score, token_id)`` order;
* an exact tie-closed adaptive implementation of that same order.

``ThreeWaySubwordRankingAuditProcessor`` runs the frozen, stable, and adaptive
processors on the *same* logits row while returning the frozen mask.  Thus the
audit cannot change the publication trajectory.  Stable/adaptive disagreement
fails immediately; frozen/stable tie differences are counted because the
frozen processor uses PyTorch's default unstable ``argsort``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .byt5 import ByT5ConstraintState
from .ranking import RankedFilterStats, adaptive_certified_tokens
from .subword import SubwordSBNLogitsProcessor


@dataclass(frozen=True, slots=True)
class AdaptiveSubwordRankingStats:
    ranked_rows: int
    topk_rounds: int
    peak_candidate_pool: int
    cumulative_candidate_pool: int
    python_scalar_values_materialized: int
    boundary_ties_added: int
    certificate_tests: int
    full_vocabulary_rows: int


@dataclass(frozen=True, slots=True)
class RankingDifference:
    call_index: int
    row_index: int
    frozen_token_ids: tuple[int, ...]
    stable_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThreeWayRankingAuditStats:
    calls: int
    rows: int
    frozen_stable_mismatches: int
    stable_adaptive_mismatches: int
    frozen_stable_examples: tuple[RankingDifference, ...]


def _mask_from_allowlists(
    input_ids: Any,
    scores: Any,
    allowlists: list[list[int] | None],
) -> Any:
    vocabulary_size = int(scores.shape[-1])
    positions: list[int] = []
    for row_index, allowed in enumerate(allowlists):
        token_ids = range(vocabulary_size) if allowed is None else allowed
        offset = row_index * vocabulary_size
        positions.extend(offset + token_id for token_id in token_ids)
    masked = scores.new_full(scores.shape, float("-inf"))
    if positions:
        indices = input_ids.new_tensor(positions)
        masked_flat = masked.reshape(-1)
        scores_flat = scores.reshape(-1)
        masked_flat[indices] = scores_flat[indices]
    return masked


class StableSubwordSBNLogitsProcessor(SubwordSBNLogitsProcessor):
    """Full-sort SBN processor with an explicit ascending-token-ID tie break."""

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        started = time.perf_counter()
        prefixes = input_ids.tolist()
        # Natural input order is token-ID order; stable sorting therefore
        # realizes (-score, token_id), matching the adaptive primitive.
        ranked_rows = scores.argsort(
            dim=-1, descending=True, stable=True
        ).tolist()
        allowlists: list[list[int] | None] = []
        for prefix, ranked in zip(prefixes, ranked_rows):
            state = self.prefix_states.state_for(prefix)
            if state is not None and state.is_accepting:
                allowlists.append(None)
                continue
            allowed: list[int] = []
            if state is not None:
                generated = len(self.prefix_states.normalize_prefix(prefix))
                allowed = self._ranked_allowed(
                    state, ranked, generated=generated
                )
            allowlists.append(allowed)
            self._stats_rows += 1
            self._stats_allowed_total += len(allowed)
            self._stats_allowed_min = (
                len(allowed)
                if self._stats_allowed_min is None
                else min(self._stats_allowed_min, len(allowed))
            )
            self._stats_allowed_max = (
                len(allowed)
                if self._stats_allowed_max is None
                else max(self._stats_allowed_max, len(allowed))
            )
            if not allowed:
                self._stats_all_masked_rows += 1
        masked = _mask_from_allowlists(input_ids, scores, allowlists)
        self._stats_calls += 1
        self._stats_seconds += time.perf_counter() - started
        return masked


class AdaptiveSubwordSBNLogitsProcessor(SubwordSBNLogitsProcessor):
    """Exact adaptive ranking under the stable reference's total order."""

    def __init__(self, *args: Any, initial_k: int = 32, **kwargs: Any) -> None:
        if initial_k <= 0:
            raise ValueError("initial_k must be positive")
        self.initial_k = initial_k
        super().__init__(*args, **kwargs)

    def reset_stats(self) -> None:
        super().reset_stats()
        self._ranking_ranked_rows = 0
        self._ranking_topk_rounds = 0
        self._ranking_peak_candidate_pool = 0
        self._ranking_cumulative_candidate_pool = 0
        self._ranking_python_scalar_values_materialized = 0
        self._ranking_boundary_ties_added = 0
        self._ranking_certificate_tests = 0
        self._ranking_full_vocabulary_rows = 0

    def _record_ranking_stats(self, stats: RankedFilterStats) -> None:
        self._ranking_ranked_rows += 1
        self._ranking_topk_rounds += stats.topk_rounds
        self._ranking_peak_candidate_pool = max(
            self._ranking_peak_candidate_pool, stats.peak_candidate_pool
        )
        self._ranking_cumulative_candidate_pool += (
            stats.cumulative_candidate_pool
        )
        self._ranking_python_scalar_values_materialized += (
            stats.python_scalar_values_materialized
        )
        self._ranking_boundary_ties_added += stats.boundary_ties_added
        self._ranking_certificate_tests += stats.certificate_tests
        self._ranking_full_vocabulary_rows += int(
            stats.full_vocabulary_reached
        )

    def _adaptive_allowed(
        self,
        state: ByT5ConstraintState,
        score_row: Any,
        *,
        generated: int,
    ) -> list[int]:
        if self.bounded_completion:
            self._stats_bounded_rows += 1
        remaining_after_token = (
            None
            if self.max_new_tokens is None
            else self.max_new_tokens - generated - 1
        )

        def certify(token_id: int) -> bool:
            self._stats_candidates_tested += 1
            successor = self.controller.transition(state, token_id)
            if successor is None:
                return False
            self._stats_candidates_accepted += 1
            if not self.bounded_completion:
                return True
            self._stats_completion_certificates_tested += 1
            if remaining_after_token is None:
                raise AssertionError("missing completion budget")
            distance = self._successor_completion_distance(
                successor, remaining_after_token
            )
            if distance is None:
                return False
            self._stats_completion_certificates_accepted += 1
            self._stats_completion_witness_tokens_total += distance
            slack = remaining_after_token - distance
            self._stats_completion_slack_total += slack
            self._stats_completion_slack_min = (
                slack
                if self._stats_completion_slack_min is None
                else min(self._stats_completion_slack_min, slack)
            )
            return True

        selected, ranking_stats = adaptive_certified_tokens(
            score_row,
            certify,
            required=self.retained_tokens_per_row,
            initial_k=self.initial_k,
        )
        self._record_ranking_stats(ranking_stats)
        return list(selected)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        started = time.perf_counter()
        prefixes = input_ids.tolist()
        allowlists: list[list[int] | None] = []
        for row_index, prefix in enumerate(prefixes):
            state = self.prefix_states.state_for(prefix)
            if state is not None and state.is_accepting:
                allowlists.append(None)
                continue
            allowed: list[int] = []
            if state is not None:
                generated = len(self.prefix_states.normalize_prefix(prefix))
                allowed = self._adaptive_allowed(
                    state, scores[row_index], generated=generated
                )
            allowlists.append(allowed)
            self._stats_rows += 1
            self._stats_allowed_total += len(allowed)
            self._stats_allowed_min = (
                len(allowed)
                if self._stats_allowed_min is None
                else min(self._stats_allowed_min, len(allowed))
            )
            self._stats_allowed_max = (
                len(allowed)
                if self._stats_allowed_max is None
                else max(self._stats_allowed_max, len(allowed))
            )
            if not allowed:
                self._stats_all_masked_rows += 1
        masked = _mask_from_allowlists(input_ids, scores, allowlists)
        self._stats_calls += 1
        self._stats_seconds += time.perf_counter() - started
        return masked

    def ranking_stats(self) -> AdaptiveSubwordRankingStats:
        return AdaptiveSubwordRankingStats(
            ranked_rows=self._ranking_ranked_rows,
            topk_rounds=self._ranking_topk_rounds,
            peak_candidate_pool=self._ranking_peak_candidate_pool,
            cumulative_candidate_pool=self._ranking_cumulative_candidate_pool,
            python_scalar_values_materialized=(
                self._ranking_python_scalar_values_materialized
            ),
            boundary_ties_added=self._ranking_boundary_ties_added,
            certificate_tests=self._ranking_certificate_tests,
            full_vocabulary_rows=self._ranking_full_vocabulary_rows,
        )


class ThreeWaySubwordRankingAuditProcessor:
    """Shadow three rankers on one logits stream and return the frozen mask."""

    def __init__(
        self,
        *,
        frozen: SubwordSBNLogitsProcessor,
        stable: StableSubwordSBNLogitsProcessor,
        adaptive: AdaptiveSubwordSBNLogitsProcessor,
        max_examples: int = 20,
    ) -> None:
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative")
        if len({id(frozen), id(stable), id(adaptive)}) != 3:
            raise ValueError("three-way audit requires independent processors")
        self.frozen = frozen
        self.stable = stable
        self.adaptive = adaptive
        self.max_examples = max_examples
        self._calls = 0
        self._rows = 0
        self._frozen_stable_mismatches = 0
        self._stable_adaptive_mismatches = 0
        self._examples: list[RankingDifference] = []

    @classmethod
    def from_shared_configuration(
        cls,
        *,
        initial_k: int = 32,
        max_examples: int = 20,
        **processor_kwargs: Any,
    ) -> "ThreeWaySubwordRankingAuditProcessor":
        return cls(
            frozen=SubwordSBNLogitsProcessor(**processor_kwargs),
            stable=StableSubwordSBNLogitsProcessor(**processor_kwargs),
            adaptive=AdaptiveSubwordSBNLogitsProcessor(
                **processor_kwargs, initial_k=initial_k
            ),
            max_examples=max_examples,
        )

    @staticmethod
    def _selected(masked_row: Any) -> tuple[int, ...]:
        return tuple(
            int(token_id)
            for token_id in masked_row.isfinite().nonzero().flatten().tolist()
        )

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        frozen_mask = self.frozen(input_ids, scores)
        stable_mask = self.stable(input_ids, scores)
        adaptive_mask = self.adaptive(input_ids, scores)
        call_index = self._calls
        self._calls += 1
        for row_index in range(int(scores.shape[0])):
            self._rows += 1
            frozen_ids = self._selected(frozen_mask[row_index])
            stable_ids = self._selected(stable_mask[row_index])
            adaptive_ids = self._selected(adaptive_mask[row_index])
            if stable_ids != adaptive_ids:
                self._stable_adaptive_mismatches += 1
                raise AssertionError(
                    "stable/adaptive ranking mismatch at "
                    f"call {call_index}, row {row_index}: "
                    f"{stable_ids!r} != {adaptive_ids!r}"
                )
            if frozen_ids != stable_ids:
                self._frozen_stable_mismatches += 1
                if len(self._examples) < self.max_examples:
                    self._examples.append(
                        RankingDifference(
                            call_index=call_index,
                            row_index=row_index,
                            frozen_token_ids=frozen_ids,
                            stable_token_ids=stable_ids,
                        )
                    )
        return frozen_mask

    def stats(self) -> ThreeWayRankingAuditStats:
        return ThreeWayRankingAuditStats(
            calls=self._calls,
            rows=self._rows,
            frozen_stable_mismatches=self._frozen_stable_mismatches,
            stable_adaptive_mismatches=self._stable_adaptive_mismatches,
            frozen_stable_examples=tuple(self._examples),
        )


__all__ = [
    "AdaptiveSubwordRankingStats",
    "AdaptiveSubwordSBNLogitsProcessor",
    "RankingDifference",
    "StableSubwordSBNLogitsProcessor",
    "ThreeWayRankingAuditStats",
    "ThreeWaySubwordRankingAuditProcessor",
]
