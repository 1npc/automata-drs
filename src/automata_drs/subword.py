"""Exact byte-level-BPE lifting for decoder-only constrained generation.

Byte-level BPE token strings encode arbitrary byte sequences.  Decoding each
token independently is not exact when a Unicode scalar is split across two
tokens: the tokenizer emits replacement characters for both pieces.  This
module instead recovers the original bytes from the tokenizer vocabulary and
feeds them through the same incremental UTF-8/SBN state used by ByT5.

The logits processor uses score-ranked certification.  For greedy decoding,
the first valid token is exactly the constrained argmax.  For beam width B,
retaining the best 2B valid tokens from every beam row preserves Hugging Face's
next-step global 2B candidate selection while avoiding a full 100k-vocabulary
automaton traversal.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .byt5 import ByT5ConstraintState, ByT5SBNLogitsProcessor
from .certificates import (
    CertificateSearchResult,
    WitnessCarryingController,
    WitnessCarryingProcessorStats,
    WitnessDecisionKind,
    certify_constructive_plan,
)
from .decoding import CompletionDistanceCache, TransitionCacheInfo
from .sbn import SBNPolicy


def gpt2_byte_decoder() -> dict[str, int]:
    """Return the inverse alphabet used by Hugging Face ByteLevel decoders."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {
        chr(codepoint): byte
        for byte, codepoint in zip(byte_values, codepoints)
    }


@dataclass(frozen=True, slots=True)
class ByteLevelVocabulary:
    """Exact raw-byte fragment for every ordinary model token."""

    fragments: Mapping[int, bytes]
    eos_token_ids: frozenset[int]
    byte_token_ids: Mapping[int, int]

    def __post_init__(self) -> None:
        fragments = {
            int(token_id): bytes(fragment)
            for token_id, fragment in self.fragments.items()
            if fragment
        }
        eos = frozenset(int(token_id) for token_id in self.eos_token_ids)
        byte_tokens = {
            int(byte): int(token_id)
            for byte, token_id in self.byte_token_ids.items()
        }
        if any(token_id < 0 for token_id in fragments) or any(
            token_id < 0 for token_id in eos
        ):
            raise ValueError("token IDs must be non-negative")
        if any(not 0 <= byte <= 255 for byte in byte_tokens):
            raise ValueError("single-byte lookup keys must be in 0..255")
        for byte, token_id in byte_tokens.items():
            if fragments.get(token_id) != bytes((byte,)):
                raise ValueError("single-byte lookup does not match fragments")
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "eos_token_ids", eos)
        object.__setattr__(self, "byte_token_ids", byte_tokens)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "ByteLevelVocabulary":
        """Build a vocabulary from a fast tokenizer with a ByteLevel decoder."""

        backend = getattr(tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if decoder is None or "ByteLevel" not in type(decoder).__name__:
            # tokenizers exposes extension types whose class name is the most
            # stable public discriminator across its Python releases.
            raise ValueError("tokenizer does not expose a ByteLevel decoder")
        inverse = gpt2_byte_decoder()
        special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
        fragments: dict[int, bytes] = {}
        byte_tokens: dict[int, int] = {}
        for token, token_id_value in tokenizer.get_vocab().items():
            token_id = int(token_id_value)
            if token_id in special_ids:
                continue
            try:
                fragment = bytes(inverse[character] for character in token)
            except KeyError as error:
                raise ValueError(
                    f"token {token!r} cannot be inverted through ByteLevel"
                ) from error
            if not fragment:
                continue
            fragments[token_id] = fragment
            if len(fragment) == 1:
                byte_tokens.setdefault(fragment[0], token_id)
        missing_bytes = sorted(set(range(256)) - set(byte_tokens))
        if missing_bytes:
            raise ValueError(
                "ByteLevel vocabulary lacks raw-byte fallbacks: "
                + ", ".join(str(byte) for byte in missing_bytes[:16])
            )
        eos_ids = frozenset(
            int(token_id)
            for token_id in (
                tokenizer.eos_token_id
                if isinstance(tokenizer.eos_token_id, (list, tuple))
                else [tokenizer.eos_token_id]
            )
            if token_id is not None
        )
        if not eos_ids:
            raise ValueError("tokenizer must define at least one EOS token")
        return cls(
            fragments=fragments,
            eos_token_ids=eos_ids,
            byte_token_ids=byte_tokens,
        )

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.fragments) | set(self.eos_token_ids)))


class SubwordTokenConstraintController:
    """Cached transitions from byte-level-BPE tokens to SBN states."""

    def __init__(
        self,
        vocabulary: ByteLevelVocabulary,
        *,
        max_cache_entries: int = 100_000,
    ) -> None:
        if max_cache_entries < 0:
            raise ValueError("max_cache_entries must be non-negative")
        self.vocabulary = vocabulary
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[
            tuple[ByT5ConstraintState, int], ByT5ConstraintState | None
        ] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _compute_transition(
        self, state: ByT5ConstraintState, token_id: int
    ) -> ByT5ConstraintState | None:
        if token_id in self.vocabulary.eos_token_ids:
            return state.finish()
        fragment = self.vocabulary.fragments.get(token_id)
        if fragment is None:
            return None
        cursor: ByT5ConstraintState | None = state
        for byte in fragment:
            cursor = None if cursor is None else cursor.advance_byte(byte)
            if cursor is None:
                return None
        return cursor

    def transition(
        self, state: ByT5ConstraintState, token_id: int
    ) -> ByT5ConstraintState | None:
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
        state: ByT5ConstraintState,
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


class CausalPrefixStateCache:
    """Recover SBN states after stripping the fixed causal-LM prompt."""

    def __init__(
        self,
        controller: SubwordTokenConstraintController,
        *,
        prompt_token_ids: Sequence[int],
        policy: SBNPolicy | None = None,
    ) -> None:
        self.controller = controller
        self.prompt_token_ids = tuple(int(token_id) for token_id in prompt_token_ids)
        self._initial = ByT5ConstraintState.initial(policy)
        self._states: dict[
            tuple[int, ...], ByT5ConstraintState | None
        ] = {(): self._initial}

    def normalize_prefix(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        prefix = tuple(int(token_id) for token_id in token_ids)
        prompt_size = len(self.prompt_token_ids)
        if prefix[:prompt_size] != self.prompt_token_ids:
            raise ValueError("generation prefix does not match the fixed prompt")
        return prefix[prompt_size:]

    def state_for(
        self, token_ids: Sequence[int]
    ) -> ByT5ConstraintState | None:
        prefix = self.normalize_prefix(token_ids)
        cached = self._states.get(prefix, ...)
        if cached is not ...:
            return cached
        parent = self.state_for(self.prompt_token_ids + prefix[:-1])
        state = (
            None
            if parent is None
            else self.controller.transition(parent, prefix[-1])
        )
        self._states[prefix] = state
        return state

    @property
    def size(self) -> int:
        return len(self._states)


@dataclass(frozen=True, slots=True)
class SubwordProcessorStats:
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


class SubwordSBNLogitsProcessor:
    """Score-ranked hard SBN mask for byte-level-BPE causal language models."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: Sequence[int],
        vocabulary: ByteLevelVocabulary | None = None,
        policy: SBNPolicy | None = None,
        retained_tokens_per_row: int = 1,
        max_new_tokens: int | None = None,
        bounded_completion: bool = False,
        completion_budget_factor: float = 1.0,
        max_completion_cache_entries: int = 100_000,
    ) -> None:
        if retained_tokens_per_row <= 0:
            raise ValueError("retained_tokens_per_row must be positive")
        if bounded_completion and (max_new_tokens is None or max_new_tokens <= 0):
            raise ValueError("bounded completion requires max_new_tokens")
        if not 0 < completion_budget_factor <= 1:
            raise ValueError("completion_budget_factor must be in (0, 1]")
        if max_completion_cache_entries < 0:
            raise ValueError(
                "max_completion_cache_entries must be non-negative"
            )
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary or ByteLevelVocabulary.from_tokenizer(tokenizer)
        self.controller = SubwordTokenConstraintController(self.vocabulary)
        self.prefix_states = CausalPrefixStateCache(
            self.controller,
            prompt_token_ids=prompt_token_ids,
            policy=policy,
        )
        self.retained_tokens_per_row = retained_tokens_per_row
        self.max_new_tokens = max_new_tokens
        self.bounded_completion = bounded_completion
        self.completion_budget_factor = completion_budget_factor
        self._bounded_finish_cache = CompletionDistanceCache[
            ByT5ConstraintState
        ](max_completion_cache_entries)
        self.reset_stats()

    def _encode_completion(self, text: str) -> tuple[int, ...]:
        ids = tuple(
            int(token_id)
            for token_id in self.tokenizer(
                text, add_special_tokens=False
            )["input_ids"]
        )
        encoded = b"".join(
            self.vocabulary.fragments.get(token_id, b"") for token_id in ids
        )
        if ids and encoded == text.encode("utf-8"):
            return ids
        # Any byte-level BPE vocabulary must contain all raw byte fallbacks.
        return tuple(
            self.vocabulary.byte_token_ids[byte]
            for byte in text.encode("utf-8")
        )

    def _completion_token_plan(
        self, state: ByT5ConstraintState
    ) -> tuple[int, ...]:
        if state.pending_utf8:
            return ()
        suffix = ByT5SBNLogitsProcessor._suggest_completion_suffix(state.sbn)
        return self._encode_completion(suffix) if suffix else ()

    def _completion_distance_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> int | None:
        if state.is_accepting:
            return 0
        if remaining <= 0:
            return None
        reusable, cached = self._bounded_finish_cache.lookup(state, remaining)
        if reusable:
            return cached
        witness = self._completion_token_witness_within(state, remaining)
        distance = None if witness is None else len(witness)
        self._bounded_finish_cache.store(state, remaining, distance)
        return distance

    def _completion_token_witness_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> tuple[int, ...] | None:
        """Construct one exact native-token suffix ending in accepting EOS."""

        compiled = self._surface_compiled_completion_token_witness_within(
            state, remaining
        )
        if compiled is not None:
            return compiled
        return self._incremental_completion_token_witness_within(state, remaining)

    def _surface_compiled_completion_token_witness_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> tuple[int, ...] | None:
        """Compile the complete repair surface before native tokenization.

        The representation planner emits semantic surface fragments (finish a
        target, add a placeholder concept, and so on).  Tokenizing each
        fragment independently loses BPE merges across macro boundaries and
        can therefore report a false horizon failure.  This routine first
        rolls out the same deterministic surface policy, then tokenizes the
        whole residual surface once and replays the resulting native tokens.

        A partial UTF-8 scalar has no Unicode surface representation yet, so
        that rare case deliberately falls back to the exact incremental byte
        planner below.  Failure here remains UNKNOWN, never UNREACHABLE.
        """

        if state.is_accepting:
            return ()
        if remaining <= 0 or state.pending_utf8:
            return None

        cursor = state.sbn
        surface_parts: list[str] = []
        seen: set[Any] = set()
        # Each deterministic macro consumes non-empty surface text.  The hard
        # cap prevents a buggy representation proposer from looping while
        # remaining generous enough for multi-box/reference repairs.
        max_macro_actions = min(512, max(32, remaining * 8))
        for _ in range(max_macro_actions):
            if cursor.can_end():
                surface = "".join(surface_parts)
                surface_ids = self._encode_completion(surface) if surface else ()
                for eos_token_id in sorted(self.vocabulary.eos_token_ids):
                    candidate = surface_ids + (eos_token_id,)
                    if len(candidate) > remaining:
                        continue
                    replay = state
                    for token_id in candidate:
                        successor = self.controller._compute_transition(
                            replay, token_id
                        )
                        if successor is None:
                            break
                        replay = successor
                    else:
                        if replay.is_accepting:
                            return candidate
                return None

            if cursor in seen:
                return None
            seen.add(cursor)
            fragment = ByT5SBNLogitsProcessor._suggest_completion_suffix(cursor)
            if not fragment:
                return None
            successor = cursor.advance_text(fragment)
            if successor.is_dead:
                return None
            surface_parts.append(fragment)
            cursor = successor
        return None

    def _incremental_completion_token_witness_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> tuple[int, ...] | None:
        """Legacy exact fallback that tokenizes one repair fragment at a time."""

        if state.is_accepting:
            return ()
        if remaining <= 0:
            return None
        cursor = state
        witness: list[int] = []
        planned: list[int] = []
        while len(witness) < remaining:
            if cursor.is_accepting:
                return tuple(witness)
            if cursor.can_end():
                token_id = next(
                    (
                        eos_token_id
                        for eos_token_id in sorted(
                            self.vocabulary.eos_token_ids
                        )
                        if self.controller._compute_transition(
                            cursor, eos_token_id
                        )
                        is not None
                    ),
                    None,
                )
            elif cursor.pending_utf8:
                planned.clear()
                token_id = next(
                    (
                        self.vocabulary.byte_token_ids[byte]
                        for byte in range(0x80, 0xC0)
                        if self.controller._compute_transition(
                            cursor, self.vocabulary.byte_token_ids[byte]
                        )
                        is not None
                    ),
                    None,
                )
            else:
                if not planned:
                    planned.extend(self._completion_token_plan(cursor))
                token_id = planned.pop(0) if planned else None
            if token_id is None:
                return None
            successor = self.controller._compute_transition(cursor, token_id)
            if successor is None:
                return None
            witness.append(token_id)
            cursor = successor
        return tuple(witness) if cursor.is_accepting else None

    def certificate_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> CertificateSearchResult[ByT5ConstraintState]:
        """Expose the DRS constructive planner through the generic API."""

        return certify_constructive_plan(
            state,
            max_steps=remaining,
            propose=self._completion_token_witness_within,
            transition=self.controller.transition,
            is_accepting=lambda current: current.is_accepting,
        )

    def completion_cache_info(self) -> TransitionCacheInfo:
        return self._bounded_finish_cache.cache_info()

    def clear_completion_cache(self) -> None:
        self._bounded_finish_cache.clear()

    def _successor_completion_distance(
        self, successor: ByT5ConstraintState, remaining_after_token: int
    ) -> int | None:
        if successor.is_accepting:
            return 0
        budget = remaining_after_token
        if budget > 0 and self.completion_budget_factor < 1:
            budget = max(1, math.floor(budget * self.completion_budget_factor))
        return self._completion_distance_within(successor, budget)

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return False
        successor = self.controller.transition(state, int(token_id))
        if successor is None:
            return False
        if not self.bounded_completion or self.max_new_tokens is None:
            return True
        generated = len(self.prefix_states.normalize_prefix(token_ids))
        remaining = self.max_new_tokens - generated - 1
        return self._successor_completion_distance(successor, remaining) is not None

    def _ranked_allowed(
        self,
        state: ByT5ConstraintState,
        ranked_token_ids: Iterable[int],
        *,
        generated: int,
        prefix_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        del prefix_token_ids
        accepted: list[int] = []
        if self.bounded_completion:
            self._stats_bounded_rows += 1
        remaining_after_token = (
            None
            if self.max_new_tokens is None
            else self.max_new_tokens - generated - 1
        )
        for token_id_value in ranked_token_ids:
            token_id = int(token_id_value)
            self._stats_candidates_tested += 1
            successor = self.controller.transition(state, token_id)
            if successor is None:
                continue
            self._stats_candidates_accepted += 1
            if self.bounded_completion:
                self._stats_completion_certificates_tested += 1
                if remaining_after_token is None:
                    raise AssertionError("missing completion budget")
                distance = self._successor_completion_distance(
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
        # Sort on the model device, then synchronize once for the whole beam.
        ranked_rows = scores.argsort(dim=-1, descending=True).tolist()
        allowlists: list[list[int] | None] = []
        for prefix, ranked in zip(prefixes, ranked_rows):
            state = self.prefix_states.state_for(prefix)
            if state is not None and state.is_accepting:
                allowlists.append(None)
                continue
            allowed = []
            if state is not None:
                generated = len(self.prefix_states.normalize_prefix(prefix))
                allowed = self._ranked_allowed(
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

    def stats(self) -> SubwordProcessorStats:
        return SubwordProcessorStats(
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
        self._stats_calls = 0
        self._stats_rows = 0
        self._stats_seconds = 0.0
        self._stats_candidates_tested = 0
        self._stats_candidates_accepted = 0
        self._stats_allowed_total = 0
        self._stats_allowed_min: int | None = None
        self._stats_allowed_max: int | None = None
        self._stats_all_masked_rows = 0
        self._stats_bounded_rows = 0
        self._stats_completion_certificates_tested = 0
        self._stats_completion_certificates_accepted = 0
        self._stats_completion_witness_tokens_total = 0
        self._stats_completion_slack_total = 0
        self._stats_completion_slack_min: int | None = None


class WitnessCarryingSubwordSBNLogitsProcessor(SubwordSBNLogitsProcessor):
    """Causal-subword DRS adapter for the shared WCD invariant."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_token_ids: Sequence[int],
        max_new_tokens: int,
        vocabulary: ByteLevelVocabulary | None = None,
        policy: SBNPolicy | None = None,
        retained_tokens_per_row: int = 1,
        max_completion_cache_entries: int = 100_000,
        max_prefix_certificates: int = 20_000,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            vocabulary=vocabulary,
            policy=policy,
            retained_tokens_per_row=retained_tokens_per_row,
            max_new_tokens=max_new_tokens,
            bounded_completion=True,
            completion_budget_factor=1.0,
            max_completion_cache_entries=max_completion_cache_entries,
        )
        self.witnesses = WitnessCarryingController(
            transition=self.controller.transition,
            plan=self.certificate_within,
            max_prefix_certificates=max_prefix_certificates,
        )

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting or self.max_new_tokens is None:
            return False
        prefix = self.prefix_states.normalize_prefix(token_ids)
        return self.witnesses.certify_token(
            prefix,
            state,
            int(token_id),
            remaining_after_token=(
                self.max_new_tokens - len(prefix) - 1
            ),
        ).allowed

    def _ranked_allowed(
        self,
        state: ByT5ConstraintState,
        ranked_token_ids: Iterable[int],
        *,
        generated: int,
        prefix_token_ids: Sequence[int] | None = None,
    ) -> list[int]:
        if self.max_new_tokens is None:
            raise AssertionError("WCD requires max_new_tokens")
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
