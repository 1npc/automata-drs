"""Tokenizer-neutral token masking over surface-string fragments.

This module deliberately does not guess how a Hugging Face token maps to
surface text. Byte-level BPE and SentencePiece tokenizers need an exact adapter
that supplies the fragments. Once supplied, the constraint logic is shared.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, Hashable, Iterable, Mapping, Protocol, TypeVar, cast


class IncrementalConstraintState(Protocol):
    """Tokenizer-neutral, immutable/hashable structured-language state."""

    def __hash__(self) -> int: ...

    @property
    def is_extendable(self) -> bool: ...

    @property
    def is_accepting(self) -> bool: ...

    def advance_text(self, fragment: str) -> "IncrementalConstraintState": ...

    def finish(self) -> "IncrementalConstraintState": ...


StateT = TypeVar("StateT", bound=IncrementalConstraintState)


@dataclass(frozen=True, slots=True)
class SurfaceVocabulary:
    """A model vocabulary represented by exact decoded text fragments."""

    fragments: Mapping[int, str]
    eos_token_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fragments", dict(self.fragments))
        object.__setattr__(self, "eos_token_ids", frozenset(self.eos_token_ids))
        if any(token_id < 0 for token_id in self.fragments):
            raise ValueError("token IDs must be non-negative")
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("EOS token IDs must be non-negative")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.fragments) | set(self.eos_token_ids)))


@dataclass(frozen=True, slots=True)
class TransitionCacheInfo:
    hits: int
    misses: int
    size: int
    evictions: int = 0


CacheStateT = TypeVar("CacheStateT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class _CompletionDistanceEntry:
    distance: int | None = None
    max_failed_budget: int = -1


class CompletionDistanceCache(Generic[CacheStateT]):
    """Bounded monotone cache for constructive completion certificates.

    A deterministic completion planner has one eventual witness length ``d``
    for a state, when it finds a witness at all. Once ``d`` is observed, all
    larger budgets return ``d`` and all smaller budgets return ``None``.
    Conversely, failure with budget ``b`` implies failure for every smaller
    budget along the same deterministic plan, but says nothing about a larger
    budget. Storing this dominance information by state avoids duplicate
    ``(state, budget)`` entries while preserving the planner's exact decision.

    Entries are LRU-bounded because immutable structured states can be large
    and publication-scale decoding spans thousands of independent examples.
    """

    def __init__(self, max_entries: int = 100_000):
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            CacheStateT, _CompletionDistanceEntry
        ] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(
        self, state: CacheStateT, budget: int
    ) -> tuple[bool, int | None]:
        """Return ``(reusable, result)`` for ``state`` at ``budget``."""

        if budget < 0:
            raise ValueError("completion budget must be non-negative")
        entry = self._entries.get(state)
        if entry is None:
            self._misses += 1
            return False, None
        self._entries.move_to_end(state)
        if entry.distance is not None:
            self._hits += 1
            return True, (
                entry.distance if entry.distance <= budget else None
            )
        if budget <= entry.max_failed_budget:
            self._hits += 1
            return True, None
        self._misses += 1
        return False, None

    def store(
        self,
        state: CacheStateT,
        budget: int,
        distance: int | None,
    ) -> None:
        if budget < 0:
            raise ValueError("completion budget must be non-negative")
        if distance is not None and not 0 <= distance <= budget:
            raise ValueError("witness distance must fit its completion budget")
        if not self.max_entries:
            return

        previous = self._entries.pop(
            state, _CompletionDistanceEntry()
        )
        if distance is None:
            entry = _CompletionDistanceEntry(
                distance=previous.distance,
                max_failed_budget=max(previous.max_failed_budget, budget),
            )
        else:
            if (
                previous.distance is not None
                and previous.distance != distance
            ):
                raise AssertionError(
                    "deterministic completion distance changed for one state"
                )
            entry = _CompletionDistanceEntry(
                distance=distance,
                max_failed_budget=previous.max_failed_budget,
            )
        if len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        self._entries[state] = entry

    def cache_info(self) -> TransitionCacheInfo:
        return TransitionCacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._entries),
            evictions=self._evictions,
        )

    def clear(self) -> None:
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0


class TokenConstraintController(Generic[StateT]):
    """Cache model-token transitions for any incremental constraint state."""

    def __init__(self, vocabulary: SurfaceVocabulary):
        self.vocabulary = vocabulary
        self._cache: dict[
            tuple[StateT, int], StateT | None
        ] = {}
        self._hits = 0
        self._misses = 0

    def transition(
        self, state: StateT, token_id: int
    ) -> StateT | None:
        """Return the successor, or ``None`` when the token must be masked."""
        key = (state, token_id)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1

        if token_id in self.vocabulary.eos_token_ids:
            finished = cast(StateT, state.finish())
            successor = finished if finished.is_accepting else None
        else:
            fragment = self.vocabulary.fragments.get(token_id)
            if not fragment:
                successor = None
            else:
                advanced = cast(StateT, state.advance_text(fragment))
                successor = advanced if advanced.is_extendable else None

        self._cache[key] = successor
        return successor

    def allowed_token_ids(
        self,
        state: StateT,
        candidate_ids: Iterable[int] | None = None,
    ) -> frozenset[int]:
        candidates = self.vocabulary.token_ids if candidate_ids is None else candidate_ids
        return frozenset(
            token_id
            for token_id in candidates
            if self.transition(state, token_id) is not None
        )

    def boolean_mask(
        self, state: StateT, vocab_size: int
    ) -> tuple[bool, ...]:
        """Return a dependency-free mask suitable for conversion to a tensor."""
        if vocab_size < 0:
            raise ValueError("vocab_size must be non-negative")
        allowed = self.allowed_token_ids(
            state, (token_id for token_id in range(vocab_size))
        )
        return tuple(token_id in allowed for token_id in range(vocab_size))

    def cache_info(self) -> TransitionCacheInfo:
        return TransitionCacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
