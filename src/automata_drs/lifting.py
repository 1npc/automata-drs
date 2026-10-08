"""Representation-neutral exact lifting from byte-level BPE to constraints.

The SBN decoder has a representation-specific finite-budget completion
planner, but ordinary prefix masking should not be tied to SBN.  This module
bridges exact raw token bytes to any immutable
``IncrementalConstraintState``.  It keeps an incomplete UTF-8 scalar outside
the representation state, so a Unicode character split across model tokens is
never decoded through replacement characters.

Representations that accept non-ASCII text must supply a small guard for an
incomplete UTF-8 scalar.  The guard answers whether *some* completion of that
already validated byte prefix can preserve the current representation prefix.
Failing closed when no guard is supplied avoids admitting a token that ends in
an unverified partial scalar.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable, Generic, Iterable, Sequence, TypeVar, cast

from .certificates import (
    CertificateSearchResult,
    WitnessCarryingController,
    WitnessCarryingProcessorStats,
    WitnessDecisionKind,
    certify_constructive_plan,
)
from .decoding import (
    CompletionDistanceCache,
    IncrementalConstraintState,
    TransitionCacheInfo,
)
from .subword import ByteLevelVocabulary


StateT = TypeVar("StateT", bound=IncrementalConstraintState)
IncompleteUTF8Guard = Callable[[StateT, bytes], bool]
SurfaceCompletionPlanner = Callable[[StateT], str | None]
SurfaceEncoder = Callable[[str], Iterable[int]]


@dataclass(frozen=True, slots=True)
class UTF8ByteConstraintState(Generic[StateT]):
    """A representation state plus a proper prefix of one UTF-8 scalar."""

    representation: StateT
    pending_utf8: bytes = b""

    @property
    def is_extendable(self) -> bool:
        return self.representation.is_extendable

    @property
    def is_accepting(self) -> bool:
        return not self.pending_utf8 and self.representation.is_accepting


@dataclass(frozen=True, slots=True)
class ByteLevelProcessorStats:
    calls: int
    rows: int
    seconds: float
    candidates_tested: int
    candidates_accepted: int
    allowed_total: int
    allowed_min: int | None
    allowed_max: int | None
    all_masked_rows: int


@dataclass(frozen=True, slots=True)
class ByteLevelFiniteHorizonProcessorStats:
    """Online PF activity, keeping prefix and certificate filters distinct."""

    calls: int
    rows: int
    seconds: float
    candidates_tested: int
    candidates_accepted: int
    allowed_total: int
    allowed_min: int | None
    allowed_max: int | None
    all_masked_rows: int
    bounded_rows: int
    completion_certificates_tested: int
    completion_certificates_accepted: int
    completion_witness_tokens_total: int
    completion_slack_total: int
    completion_slack_min: int | None


ByteLevelWitnessCarryingProcessorStats = WitnessCarryingProcessorStats


class ByteLevelTokenConstraintController(Generic[StateT]):
    """Cache exact byte-level token transitions for any surface language."""

    def __init__(
        self,
        vocabulary: ByteLevelVocabulary,
        *,
        incomplete_utf8_is_extendable: IncompleteUTF8Guard[StateT] | None = None,
        max_cache_entries: int = 100_000,
    ) -> None:
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries must be non-negative")
        self.vocabulary = vocabulary
        self.incomplete_utf8_is_extendable = incomplete_utf8_is_extendable
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[
            tuple[UTF8ByteConstraintState[StateT], int],
            UTF8ByteConstraintState[StateT] | None,
        ] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @staticmethod
    def initial(state: StateT) -> UTF8ByteConstraintState[StateT]:
        if not state.is_extendable:
            raise ValueError("initial representation state is not extendable")
        return UTF8ByteConstraintState(state)

    def _advance_bytes(
        self,
        state: UTF8ByteConstraintState[StateT],
        fragment: bytes,
    ) -> UTF8ByteConstraintState[StateT] | None:
        if not state.is_extendable or not fragment:
            return None
        candidate = state.pending_utf8 + fragment
        pending = b""
        try:
            text = candidate.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            if (
                error.reason != "unexpected end of data"
                or error.end != len(candidate)
            ):
                return None
            complete = candidate[: error.start]
            pending = candidate[error.start :]
            if not pending or len(pending) > 3:
                return None
            try:
                text = complete.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return None

        advanced = cast(StateT, state.representation.advance_text(text))
        if not advanced.is_extendable:
            return None
        if pending:
            guard = self.incomplete_utf8_is_extendable
            if guard is None or not guard(advanced, pending):
                return None
        return UTF8ByteConstraintState(advanced, pending)

    def _compute_transition(
        self,
        state: UTF8ByteConstraintState[StateT],
        token_id: int,
    ) -> UTF8ByteConstraintState[StateT] | None:
        if token_id in self.vocabulary.eos_token_ids:
            if state.pending_utf8:
                return None
            finished = state.representation.finish()
            if finished is None:
                return None
            finished = cast(StateT, finished)
            return (
                UTF8ByteConstraintState(finished)
                if finished.is_accepting
                else None
            )
        fragment = self.vocabulary.fragments.get(token_id)
        return None if fragment is None else self._advance_bytes(state, fragment)

    def transition(
        self,
        state: UTF8ByteConstraintState[StateT],
        token_id: int,
    ) -> UTF8ByteConstraintState[StateT] | None:
        key = (state, int(token_id))
        if key in self._cache:
            self._hits += 1
            successor = self._cache[key]
            self._cache.move_to_end(key)
            return successor
        self._misses += 1
        successor = self._compute_transition(state, int(token_id))
        if self.max_cache_entries:
            if len(self._cache) >= self.max_cache_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = successor
        return successor

    def allowed_token_ids(
        self,
        state: UTF8ByteConstraintState[StateT],
        candidate_ids: Iterable[int] | None = None,
    ) -> frozenset[int]:
        candidates = (
            self.vocabulary.token_ids
            if candidate_ids is None
            else candidate_ids
        )
        return frozenset(
            int(token_id)
            for token_id in candidates
            if self.transition(state, int(token_id)) is not None
        )

    def cache_info(self) -> TransitionCacheInfo:
        return TransitionCacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            evictions=self._evictions,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0


@dataclass(frozen=True, slots=True)
class ByteLevelCompletionWitness(Generic[StateT]):
    """One exact token trace from a lifted prefix to accepting EOS.

    ``appended_surface_bytes`` deliberately stores bytes rather than text. If
    the candidate token ended inside a UTF-8 scalar, the first witness bytes
    are continuation bytes and are not independently decodable; together with
    the input state's ``pending_utf8`` they still replay exactly.
    """

    token_ids: tuple[int, ...]
    appended_surface_bytes: bytes
    final_state: UTF8ByteConstraintState[StateT]

    @property
    def distance(self) -> int:
        """Number of remaining model-token steps, including EOS."""

        return len(self.token_ids)


class ByteLevelFiniteHorizonPlanner(Generic[StateT]):
    """Construct and replay one accepting byte-token suffix within a budget.

    The representation supplies a deterministic surface-suffix constructor;
    this shared layer handles pending UTF-8, exact byte-level tokenization,
    raw-byte fallback, the model-token budget, and accepting EOS replay. A
    returned witness is positively sound because every token has already been
    advanced through ``ByteLevelTokenConstraintController``. ``None`` is only
    failure of this constructor and never a proof of bounded unreachability.
    """

    def __init__(
        self,
        controller: ByteLevelTokenConstraintController[StateT],
        *,
        construct_surface_suffix: SurfaceCompletionPlanner[StateT],
        encode_surface: SurfaceEncoder | None = None,
    ) -> None:
        if not controller.vocabulary.eos_token_ids:
            raise ValueError("finite-horizon planning requires an EOS token")
        self.controller = controller
        self.construct_surface_suffix = construct_surface_suffix
        self.encode_surface = encode_surface

    def _encode_exact(self, text: str) -> tuple[int, ...] | None:
        if not text:
            return ()
        expected = text.encode("utf-8")
        if self.encode_surface is not None:
            proposed = tuple(int(value) for value in self.encode_surface(text))
            recovered = b"".join(
                self.controller.vocabulary.fragments.get(token_id, b"")
                for token_id in proposed
            )
            if proposed and recovered == expected:
                return proposed
        try:
            return tuple(
                self.controller.vocabulary.byte_token_ids[byte]
                for byte in expected
            )
        except KeyError:
            return None

    def completion_within(
        self,
        state: UTF8ByteConstraintState[StateT],
        remaining: int,
    ) -> ByteLevelCompletionWitness[StateT] | None:
        if remaining < 0:
            raise ValueError("remaining token budget must be non-negative")
        if state.is_accepting:
            return ByteLevelCompletionWitness((), b"", state)
        if not state.is_extendable or remaining == 0:
            return None

        cursor = state
        token_ids: list[int] = []
        appended = bytearray()

        # A model candidate may end inside one UTF-8 scalar. The surface
        # constructor cannot express continuation bytes as standalone text, so
        # complete that scalar first through exact raw-byte transitions.
        while cursor.pending_utf8:
            if len(token_ids) >= remaining:
                return None
            choice: tuple[int, UTF8ByteConstraintState[StateT]] | None = None
            for byte in range(0x80, 0xC0):
                token_id = self.controller.vocabulary.byte_token_ids.get(byte)
                if token_id is None:
                    continue
                successor = self.controller.transition(cursor, token_id)
                if successor is not None:
                    choice = (token_id, successor)
                    break
            if choice is None:
                return None
            token_id, cursor = choice
            token_ids.append(token_id)
            appended.extend(self.controller.vocabulary.fragments[token_id])

        suffix = self.construct_surface_suffix(cursor.representation)
        if suffix is None:
            return None
        surface_ids = self._encode_exact(suffix)
        if surface_ids is None:
            return None
        # One further token is always reserved for EOS. A representation that
        # is already finished was handled at method entry.
        if len(token_ids) + len(surface_ids) + 1 > remaining:
            return None
        for token_id in surface_ids:
            successor = self.controller.transition(cursor, token_id)
            if successor is None:
                return None
            cursor = successor
            token_ids.append(token_id)
            appended.extend(self.controller.vocabulary.fragments[token_id])

        for eos_token_id in sorted(self.controller.vocabulary.eos_token_ids):
            finished = self.controller.transition(cursor, eos_token_id)
            if finished is None or not finished.is_accepting:
                continue
            token_ids.append(eos_token_id)
            return ByteLevelCompletionWitness(
                tuple(token_ids), bytes(appended), finished
            )
        return None

    def certificate_within(
        self,
        state: UTF8ByteConstraintState[StateT],
        remaining: int,
    ) -> CertificateSearchResult[UTF8ByteConstraintState[StateT]]:
        """Return a verified generic certificate or an honest ``UNKNOWN``.

        ``completion_within`` remains the behavior-compatible surface-planner
        API used by the frozen PF decoder. This method is the representation-
        neutral bridge for witness-carrying decoding: it independently replays
        the proposed native-token suffix and deliberately does not turn a
        constructive planner miss into a bounded-unreachability claim.
        """

        return certify_constructive_plan(
            state,
            max_steps=remaining,
            propose=lambda current, budget: (
                witness.token_ids
                if (witness := self.completion_within(current, budget))
                is not None
                else None
            ),
            transition=self.controller.transition,
            is_accepting=lambda current: current.is_accepting,
        )


class ByteLevelPrefixStateCache(Generic[StateT]):
    """Reconstruct representation states after arbitrary beam reordering."""

    def __init__(
        self,
        controller: ByteLevelTokenConstraintController[StateT],
        *,
        prompt_token_ids: Sequence[int],
        initial_state: StateT,
    ) -> None:
        self.controller = controller
        self.prompt_token_ids = tuple(int(value) for value in prompt_token_ids)
        self._states: dict[
            tuple[int, ...], UTF8ByteConstraintState[StateT] | None
        ] = {(): controller.initial(initial_state)}

    def normalize_prefix(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        prefix = tuple(int(token_id) for token_id in token_ids)
        prompt_size = len(self.prompt_token_ids)
        if prefix[:prompt_size] != self.prompt_token_ids:
            raise ValueError("generation prefix does not match the fixed prompt")
        return prefix[prompt_size:]

    def state_for(
        self, token_ids: Sequence[int]
    ) -> UTF8ByteConstraintState[StateT] | None:
        prefix = self.normalize_prefix(token_ids)
        cached = self._states.get(prefix, ...)
        if cached is not ...:
            return cached

        anchor = len(prefix) - 1
        while prefix[:anchor] not in self._states:
            anchor -= 1
        state = self._states[prefix[:anchor]]
        for length in range(anchor + 1, len(prefix) + 1):
            current = prefix[:length]
            state = (
                None
                if state is None
                else self.controller.transition(state, current[-1])
            )
            self._states[current] = state
        return state

    @property
    def size(self) -> int:
        return len(self._states)


class ByteLevelConstraintLogitsProcessor(Generic[StateT]):
    """Score-ranked hard mask shared by PENMAN and future representations."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: Sequence[int],
        initial_state: StateT,
        vocabulary: ByteLevelVocabulary | None = None,
        incomplete_utf8_is_extendable: IncompleteUTF8Guard[StateT] | None = None,
        retained_tokens_per_row: int = 1,
        max_cache_entries: int = 100_000,
    ) -> None:
        if retained_tokens_per_row <= 0:
            raise ValueError("retained_tokens_per_row must be positive")
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary or ByteLevelVocabulary.from_tokenizer(tokenizer)
        self.controller = ByteLevelTokenConstraintController(
            self.vocabulary,
            incomplete_utf8_is_extendable=incomplete_utf8_is_extendable,
            max_cache_entries=max_cache_entries,
        )
        self.prefix_states = ByteLevelPrefixStateCache(
            self.controller,
            prompt_token_ids=prompt_token_ids,
            initial_state=initial_state,
        )
        self.retained_tokens_per_row = retained_tokens_per_row
        self.reset_stats()

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return False
        return self.controller.transition(state, int(token_id)) is not None

    def prefix_can_finish(self, token_ids: Sequence[int]) -> bool:
        """Whether EOS could accept the current generated surface.

        This asks ``finish()`` on an immutable copy instead of requiring EOS
        to have been consumed already.  A caller can consequently stop at the
        representation boundary before a model-specific transport envelope
        (for example ``</amr>``) begins.
        """

        state = self.prefix_states.state_for(token_ids)
        if state is None or state.pending_utf8:
            return False
        finished = state.representation.finish()
        return bool(finished is not None and finished.is_accepting)

    def _ranked_allowed(
        self,
        state: UTF8ByteConstraintState[StateT],
        ranked_token_ids: Iterable[int],
    ) -> list[int]:
        accepted: list[int] = []
        for token_id_value in ranked_token_ids:
            token_id = int(token_id_value)
            self._stats_candidates_tested += 1
            successor = self.controller.transition(state, token_id)
            if successor is None:
                continue
            self._stats_candidates_accepted += 1
            accepted.append(token_id)
            if len(accepted) >= self.retained_tokens_per_row:
                break
        return accepted

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        started = time.perf_counter()
        prefixes = input_ids.tolist()
        ranked_rows = scores.argsort(dim=-1, descending=True).tolist()
        allowlists: list[list[int] | None] = []
        for prefix, ranked in zip(prefixes, ranked_rows):
            state = self.prefix_states.state_for(prefix)
            if state is not None and state.is_accepting:
                allowlists.append(None)
                continue
            allowed = [] if state is None else self._ranked_allowed(state, ranked)
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
        self._stats_calls += 1
        self._stats_seconds += time.perf_counter() - started
        return masked

    def stats(self) -> ByteLevelProcessorStats:
        return ByteLevelProcessorStats(
            calls=self._stats_calls,
            rows=self._stats_rows,
            seconds=self._stats_seconds,
            candidates_tested=self._stats_candidates_tested,
            candidates_accepted=self._stats_candidates_accepted,
            allowed_total=self._stats_allowed_total,
            allowed_min=self._stats_allowed_min,
            allowed_max=self._stats_allowed_max,
            all_masked_rows=self._stats_all_masked_rows,
        )

    def reset_stats(self) -> None:
        self._stats_calls = 0
        self._stats_rows = 0
        self._stats_seconds = 0.0
        self._stats_candidates_tested = 0
        self._stats_candidates_accepted = 0
        self._stats_allowed_total = 0
        self._stats_allowed_min: int | None = None
        self._stats_allowed_max: int | None = None
        self._stats_all_masked_rows = 0


class ByteLevelStructuralCompletionStoppingCriteria(Generic[StateT]):
    """Stop exactly when the lifted representation can accept EOS.

    Transformers calls stopping criteria after appending the selected token.
    The companion logits processor validates that closing token first; this
    criterion then ends generation before transport-envelope tokens can be
    mistaken for representation syntax.
    """

    def __init__(
        self, processor: ByteLevelConstraintLogitsProcessor[StateT]
    ) -> None:
        self.processor = processor

    def completed_rows(
        self, prefixes: Sequence[Sequence[int]]
    ) -> tuple[bool, ...]:
        return tuple(self.processor.prefix_can_finish(prefix) for prefix in prefixes)

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        del scores, kwargs
        completed = self.completed_rows(input_ids.tolist())
        return input_ids.new_tensor(completed).bool()


class ByteLevelFiniteHorizonConstraintLogitsProcessor(
    ByteLevelConstraintLogitsProcessor[StateT]
):
    """Score-ranked PF mask with a replayed accepting suffix per candidate.

    This class is intentionally separate from ordinary prefix masking. Every
    admitted token must first pass the representation transition and then have
    one positively replayed model-token path to accepting EOS inside the actual
    remaining generation horizon. A planner failure rejects the candidate but
    is never interpreted as a proof of unreachability.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: Sequence[int],
        initial_state: StateT,
        construct_surface_suffix: SurfaceCompletionPlanner[StateT],
        max_new_tokens: int,
        vocabulary: ByteLevelVocabulary | None = None,
        incomplete_utf8_is_extendable: IncompleteUTF8Guard[StateT] | None = None,
        encode_surface: SurfaceEncoder | None = None,
        retained_tokens_per_row: int = 1,
        max_cache_entries: int = 100_000,
        max_completion_cache_entries: int = 20_000,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("finite-horizon masking requires max_new_tokens")
        if max_completion_cache_entries < 0:
            raise ValueError("max_completion_cache_entries must be non-negative")
        super().__init__(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            initial_state=initial_state,
            vocabulary=vocabulary,
            incomplete_utf8_is_extendable=incomplete_utf8_is_extendable,
            retained_tokens_per_row=retained_tokens_per_row,
            max_cache_entries=max_cache_entries,
        )
        self.max_new_tokens = int(max_new_tokens)
        self.planner = ByteLevelFiniteHorizonPlanner(
            self.controller,
            construct_surface_suffix=construct_surface_suffix,
            encode_surface=encode_surface or self._encode_surface,
        )
        self._completion_cache = CompletionDistanceCache[
            UTF8ByteConstraintState[StateT]
        ](max_completion_cache_entries)

    def _encode_surface(self, text: str) -> tuple[int, ...]:
        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("tokenizer returned a non-singleton batch")
            encoded = encoded[0]
        return tuple(int(token_id) for token_id in encoded)

    def _completion_distance_within(
        self,
        state: UTF8ByteConstraintState[StateT],
        remaining: int,
    ) -> int | None:
        if remaining < 0:
            return None
        reusable, cached = self._completion_cache.lookup(state, remaining)
        if reusable:
            return cached
        witness = self.planner.completion_within(state, remaining)
        distance = None if witness is None else witness.distance
        self._completion_cache.store(state, remaining, distance)
        return distance

    def completion_cache_info(self) -> TransitionCacheInfo:
        return self._completion_cache.cache_info()

    def clear_completion_cache(self) -> None:
        self._completion_cache.clear()

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return False
        successor = self.controller.transition(state, int(token_id))
        if successor is None:
            return False
        generated = len(self.prefix_states.normalize_prefix(token_ids))
        remaining_after_token = self.max_new_tokens - generated - 1
        return (
            self._completion_distance_within(successor, remaining_after_token)
            is not None
        )

    def _ranked_finite_horizon_allowed(
        self,
        state: UTF8ByteConstraintState[StateT],
        ranked_token_ids: Iterable[int],
        *,
        generated: int,
        prefix_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        del prefix_token_ids
        accepted: list[int] = []
        remaining_after_token = self.max_new_tokens - generated - 1
        self._stats_bounded_rows += 1
        for token_id_value in ranked_token_ids:
            token_id = int(token_id_value)
            self._stats_candidates_tested += 1
            successor = self.controller.transition(state, token_id)
            if successor is None:
                continue
            self._stats_candidates_accepted += 1
            self._stats_completion_certificates_tested += 1
            distance = self._completion_distance_within(
                successor, remaining_after_token
            )
            if distance is None:
                continue
            self._stats_completion_certificates_accepted += 1
            self._stats_completion_witness_tokens_total += distance
            slack = remaining_after_token - distance
            self._stats_completion_slack_total += slack
            self._stats_completion_slack_min = (
                slack
                if self._stats_completion_slack_min is None
                else min(self._stats_completion_slack_min, slack)
            )
            accepted.append(token_id)
            if len(accepted) >= self.retained_tokens_per_row:
                break
        return accepted

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        started = time.perf_counter()
        prefixes = input_ids.tolist()
        ranked_rows = scores.argsort(dim=-1, descending=True).tolist()
        allowlists: list[list[int] | None] = []
        for prefix, ranked in zip(prefixes, ranked_rows):
            state = self.prefix_states.state_for(prefix)
            if state is not None and state.is_accepting:
                allowlists.append(None)
                continue
            allowed: list[int] = []
            if state is not None:
                generated = len(self.prefix_states.normalize_prefix(prefix))
                allowed = self._ranked_finite_horizon_allowed(
                    state,
                    ranked,
                    generated=generated,
                    prefix_token_ids=self.prefix_states.normalize_prefix(
                        prefix
                    ),
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
        self._stats_calls += 1
        self._stats_seconds += time.perf_counter() - started
        return masked

    def stats(self) -> ByteLevelFiniteHorizonProcessorStats:
        return ByteLevelFiniteHorizonProcessorStats(
            calls=self._stats_calls,
            rows=self._stats_rows,
            seconds=self._stats_seconds,
            candidates_tested=self._stats_candidates_tested,
            candidates_accepted=self._stats_candidates_accepted,
            allowed_total=self._stats_allowed_total,
            allowed_min=self._stats_allowed_min,
            allowed_max=self._stats_allowed_max,
            all_masked_rows=self._stats_all_masked_rows,
            bounded_rows=self._stats_bounded_rows,
            completion_certificates_tested=(
                self._stats_completion_certificates_tested
            ),
            completion_certificates_accepted=(
                self._stats_completion_certificates_accepted
            ),
            completion_witness_tokens_total=(
                self._stats_completion_witness_tokens_total
            ),
            completion_slack_total=self._stats_completion_slack_total,
            completion_slack_min=self._stats_completion_slack_min,
        )

    def reset_stats(self) -> None:
        super().reset_stats()
        self._stats_bounded_rows = 0
        self._stats_completion_certificates_tested = 0
        self._stats_completion_certificates_accepted = 0
        self._stats_completion_witness_tokens_total = 0
        self._stats_completion_slack_total = 0
        self._stats_completion_slack_min: int | None = None


class ByteLevelWitnessCarryingLogitsProcessor(
    ByteLevelFiniteHorizonConstraintLogitsProcessor[StateT]
):
    """Finite-horizon mask that preserves prefix-owned residual witnesses.

    Unlike PF, WCD does not need to reconstruct the same completion at every
    step. The verified head is always an admissible fallback; a higher-scoring
    deviation is admitted only after its successor receives a replacement
    certificate. The shared :class:`WitnessCarryingController` contains the
    representation-neutral policy, while this class only adapts Hugging Face
    score rows and the byte-level PENMAN state.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: Sequence[int],
        initial_state: StateT,
        construct_surface_suffix: SurfaceCompletionPlanner[StateT],
        max_new_tokens: int,
        vocabulary: ByteLevelVocabulary | None = None,
        incomplete_utf8_is_extendable: IncompleteUTF8Guard[StateT] | None = None,
        encode_surface: SurfaceEncoder | None = None,
        retained_tokens_per_row: int = 1,
        max_cache_entries: int = 100_000,
        max_completion_cache_entries: int = 20_000,
        max_prefix_certificates: int = 20_000,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            initial_state=initial_state,
            construct_surface_suffix=construct_surface_suffix,
            max_new_tokens=max_new_tokens,
            vocabulary=vocabulary,
            incomplete_utf8_is_extendable=incomplete_utf8_is_extendable,
            encode_surface=encode_surface,
            retained_tokens_per_row=retained_tokens_per_row,
            max_cache_entries=max_cache_entries,
            max_completion_cache_entries=max_completion_cache_entries,
        )
        self.witnesses = WitnessCarryingController(
            transition=self.controller.transition,
            plan=self.planner.certificate_within,
            max_prefix_certificates=max_prefix_certificates,
        )

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return False
        prefix = self.prefix_states.normalize_prefix(token_ids)
        remaining_after_token = self.max_new_tokens - len(prefix) - 1
        return self.witnesses.certify_token(
            prefix,
            state,
            int(token_id),
            remaining_after_token=remaining_after_token,
        ).allowed

    def _ranked_finite_horizon_allowed(
        self,
        state: UTF8ByteConstraintState[StateT],
        ranked_token_ids: Iterable[int],
        *,
        generated: int,
        prefix_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        if prefix_token_ids is None:
            raise ValueError("WCD requires the complete generated prefix")
        prefix = tuple(int(token_id) for token_id in prefix_token_ids)
        self._stats_bounded_rows += 1
        current = self.witnesses.ensure_prefix(
            prefix, state, self.max_new_tokens - generated
        )
        if current.certificate is None:
            self._stats_wcd_prefix_rows_unavailable += 1
            return []
        self._stats_wcd_prefix_rows_certified += 1

        accepted: list[int] = []
        remaining_after_token = self.max_new_tokens - generated - 1
        for token_id_value in ranked_token_ids:
            token_id = int(token_id_value)
            self._stats_candidates_tested += 1
            decision = self.witnesses.certify_token(
                prefix,
                state,
                token_id,
                remaining_after_token=remaining_after_token,
            )
            if decision.kind is WitnessDecisionKind.INVALID_TRANSITION:
                continue
            self._stats_candidates_accepted += 1
            if decision.kind is WitnessDecisionKind.NO_CERTIFICATE:
                self._stats_wcd_rejected_without_certificate += 1
                continue
            if decision.kind is WitnessDecisionKind.CARRIED:
                self._stats_wcd_carried_tokens += 1
            elif decision.kind is WitnessDecisionKind.CACHED:
                self._stats_wcd_cached_tokens += 1
            elif decision.kind is WitnessDecisionKind.REPLANNED:
                self._stats_wcd_replanned_tokens += 1
            else:  # pragma: no cover - enum exhaustiveness guard
                raise AssertionError(f"unexpected WCD decision: {decision.kind}")
            accepted.append(token_id)
            if len(accepted) >= self.retained_tokens_per_row:
                break
        return accepted

    def stats(self) -> WitnessCarryingProcessorStats:
        return WitnessCarryingProcessorStats(
            calls=self._stats_calls,
            rows=self._stats_rows,
            seconds=self._stats_seconds,
            candidates_tested=self._stats_candidates_tested,
            candidates_structurally_valid=self._stats_candidates_accepted,
            allowed_total=self._stats_allowed_total,
            allowed_min=self._stats_allowed_min,
            allowed_max=self._stats_allowed_max,
            all_masked_rows=self._stats_all_masked_rows,
            prefix_rows_certified=self._stats_wcd_prefix_rows_certified,
            prefix_rows_unavailable=self._stats_wcd_prefix_rows_unavailable,
            carried_tokens=self._stats_wcd_carried_tokens,
            cached_tokens=self._stats_wcd_cached_tokens,
            replanned_tokens=self._stats_wcd_replanned_tokens,
            rejected_without_certificate=(
                self._stats_wcd_rejected_without_certificate
            ),
        )

    def reset_stats(self) -> None:
        super().reset_stats()
        self._stats_wcd_prefix_rows_certified = 0
        self._stats_wcd_prefix_rows_unavailable = 0
        self._stats_wcd_carried_tokens = 0
        self._stats_wcd_cached_tokens = 0
        self._stats_wcd_replanned_tokens = 0
        self._stats_wcd_rejected_without_certificate = 0


__all__ = [
    "ByteLevelCompletionWitness",
    "ByteLevelConstraintLogitsProcessor",
    "ByteLevelFiniteHorizonConstraintLogitsProcessor",
    "ByteLevelFiniteHorizonProcessorStats",
    "ByteLevelFiniteHorizonPlanner",
    "ByteLevelPrefixStateCache",
    "ByteLevelProcessorStats",
    "ByteLevelStructuralCompletionStoppingCriteria",
    "ByteLevelTokenConstraintController",
    "ByteLevelWitnessCarryingLogitsProcessor",
    "ByteLevelWitnessCarryingProcessorStats",
    "IncompleteUTF8Guard",
    "SurfaceCompletionPlanner",
    "SurfaceEncoder",
    "UTF8ByteConstraintState",
]
