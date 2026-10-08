"""Exact ByT5 byte-token lifting for SBN constrained decoding.

ByT5 does not use subwords: ordinary token IDs 3--258 encode the raw byte
values 0--255.  UTF-8 characters can nevertheless span several decoding
steps, so this module keeps an immutable byte buffer beside the immutable SBN
state.  A byte token is admitted only when the resulting byte sequence is a
valid UTF-8 prefix and every completed character preserves an extendable SBN
prefix.

The core classes are dependency-free.  :class:`ByT5SBNLogitsProcessor` follows
the Hugging Face ``LogitsProcessor`` calling convention but intentionally uses
duck typing, so importing this package does not require torch or transformers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import sys
import time
from typing import Any, Iterable, Sequence

from .decoding import CompletionDistanceCache, TransitionCacheInfo
from .sbn import SBNConstraintState, SBNPolicy
from .sbn import spec
from .sbn.state import ParserMode


def _required_non_ascii_kind(state: SBNConstraintState) -> str | None:
    """Return the lexical class a next non-ASCII scalar would need to satisfy."""
    if state.mode is ParserMode.ITEM:
        if not state.current_lexeme or spec.is_synset_prefix(
            state.current_lexeme,
            strict_lemma_case=state.policy.strict_lemma_case,
        ):
            return "lower-printable"
        return None
    if state.mode is ParserMode.SEPARATOR_TARGET:
        return None
    if state.mode is ParserMode.ATTACHMENT_TARGET:
        if spec.quoted_string_is_open(state.current_lexeme):
            return "printable"
        if not state.current_lexeme and state.pending_label not in {
            "Name",
            "Proposition",
        }:
            return "alphabetic"
    return None


@lru_cache(maxsize=1)
def _unicode_prefixes_by_kind() -> dict[str, frozenset[bytes]]:
    """Build exact UTF-8 proper-prefix sets for the three lexical classes."""
    tables: dict[str, set[bytes]] = {
        "printable": set(),
        "alphabetic": set(),
        "lower-printable": set(),
    }
    for codepoint in range(0x80, sys.maxunicode + 1):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        char = chr(codepoint)
        kinds: list[str] = []
        if char.isprintable():
            kinds.append("printable")
            if char == char.lower():
                kinds.append("lower-printable")
        if char.isalpha():
            kinds.append("alphabetic")
        if not kinds:
            continue
        encoded = char.encode("utf-8")
        for length in range(1, len(encoded)):
            prefix = encoded[:length]
            for kind in kinds:
                tables[kind].add(prefix)
    return {kind: frozenset(prefixes) for kind, prefixes in tables.items()}


def _utf8_prefix_has_completion(prefix: bytes, kind: str) -> bool:
    """Whether ``prefix`` can finish as a scalar in the required lexical class."""
    return prefix in _unicode_prefixes_by_kind()[kind]


@dataclass(frozen=True, slots=True)
class ByT5Vocabulary:
    """The fixed raw-byte layout used by ``ByT5Tokenizer``.

    The defaults match both the upstream ByT5 tokenizer and the
    ``XiaoZhang98/byT5-DRS`` checkpoint. IDs 259--383 are the checkpoint's 125
    sentinel tokens and are deliberately unavailable during SBN generation.
    """

    vocab_size: int = 384
    pad_token_id: int = 0
    eos_token_id: int = 1
    unk_token_id: int = 2
    byte_offset: int = 3
    byte_count: int = 256

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        special_ids = {
            self.pad_token_id,
            self.eos_token_id,
            self.unk_token_id,
        }
        if len(special_ids) != 3 or min(special_ids) < 0:
            raise ValueError("pad, EOS, and unknown token IDs must be distinct")
        if self.byte_offset < 0 or self.byte_count != 256:
            raise ValueError("ByT5 requires exactly 256 bytes at a non-negative offset")
        if self.byte_offset + self.byte_count > self.vocab_size:
            raise ValueError("byte token range exceeds vocab_size")
        byte_ids = set(range(self.byte_offset, self.byte_offset + self.byte_count))
        if special_ids & byte_ids:
            raise ValueError("special token IDs overlap the byte token range")
        if max(special_ids) >= self.vocab_size:
            raise ValueError("special token ID exceeds vocab_size")

    def byte_for_token(self, token_id: int) -> int | None:
        byte = token_id - self.byte_offset
        return byte if 0 <= byte < self.byte_count else None

    def token_for_byte(self, byte: int) -> int:
        if not 0 <= byte < self.byte_count:
            raise ValueError("byte must be in the range 0..255")
        return byte + self.byte_offset

    @property
    def byte_token_ids(self) -> range:
        return range(self.byte_offset, self.byte_offset + self.byte_count)


@dataclass(frozen=True, slots=True)
class ByT5ConstraintState:
    """A hashable SBN hypothesis plus an incomplete UTF-8 byte sequence."""

    sbn: SBNConstraintState = SBNConstraintState()
    pending_utf8: bytes = b""

    @classmethod
    def initial(cls, policy: SBNPolicy | None = None) -> "ByT5ConstraintState":
        return cls(sbn=SBNConstraintState.initial(policy))

    @property
    def is_extendable(self) -> bool:
        return self.sbn.is_extendable

    @property
    def is_accepting(self) -> bool:
        return not self.pending_utf8 and self.sbn.is_accepting

    def advance_byte(self, byte: int) -> "ByT5ConstraintState | None":
        """Consume one raw byte, returning ``None`` for an invalid prefix."""
        if not 0 <= byte <= 255:
            raise ValueError("byte must be in the range 0..255")
        if not self.sbn.is_extendable:
            return None

        candidate = self.pending_utf8 + bytes((byte,))
        try:
            text = candidate.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            # A strict decoder reports an incomplete, otherwise valid suffix
            # as "unexpected end of data". It may become one Unicode scalar
            # after one or more continuation-byte tokens.
            kind = _required_non_ascii_kind(self.sbn)
            if (
                error.reason == "unexpected end of data"
                and error.end == len(candidate)
                and len(candidate) < 4
                and kind is not None
                and _utf8_prefix_has_completion(candidate, kind)
            ):
                return replace(self, pending_utf8=candidate)
            return None

        advanced = self.sbn.advance_text(text)
        if not advanced.is_extendable:
            return None
        return type(self)(sbn=advanced)

    def can_end(self) -> bool:
        return not self.pending_utf8 and self.sbn.can_end()

    def finish(self) -> "ByT5ConstraintState | None":
        if self.pending_utf8:
            return None
        finished = self.sbn.finish()
        if not finished.is_accepting:
            return None
        return type(self)(sbn=finished)


@dataclass(frozen=True, slots=True)
class ByT5ProcessorStats:
    calls: int
    rows: int
    seconds: float
    allowed_total: int
    allowed_min: int | None
    allowed_max: int | None
    all_masked_rows: int
    bounded_rows: int
    completion_certificates_tested: int
    completion_certificates_accepted: int
    certified_limit_hits: int
    completion_witness_tokens_total: int
    completion_witness_tokens_min: int | None
    completion_witness_tokens_max: int | None
    completion_slack_total: int
    completion_slack_min: int | None
    completion_slack_max: int | None


class ByT5TokenConstraintController:
    """Cached token transitions for the exact ByT5 byte vocabulary."""

    def __init__(
        self,
        vocabulary: ByT5Vocabulary | None = None,
        *,
        max_cache_entries: int = 100_000,
        max_allowed_cache_entries: int = 10_000,
    ):
        if max_cache_entries < 0 or max_allowed_cache_entries < 0:
            raise ValueError("cache entry limits must be non-negative")
        self.vocabulary = vocabulary or ByT5Vocabulary()
        self.max_cache_entries = max_cache_entries
        self.max_allowed_cache_entries = max_allowed_cache_entries
        self._cache: OrderedDict[
            tuple[ByT5ConstraintState, int], ByT5ConstraintState | None
        ] = OrderedDict()
        self._allowed_cache: OrderedDict[
            ByT5ConstraintState, frozenset[int]
        ] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def transition(
        self, state: ByT5ConstraintState, token_id: int
    ) -> ByT5ConstraintState | None:
        key = (state, token_id)
        if key in self._cache:
            self._hits += 1
            successor = self._cache[key]
            self._cache.move_to_end(key)
            return successor
        self._misses += 1

        successor = self._compute_transition(state, token_id)

        if self.max_cache_entries:
            if len(self._cache) >= self.max_cache_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = successor
        return successor

    def _compute_transition(
        self, state: ByT5ConstraintState, token_id: int
    ) -> ByT5ConstraintState | None:
        if token_id == self.vocabulary.eos_token_id:
            return state.finish()
        byte = self.vocabulary.byte_for_token(token_id)
        return None if byte is None else state.advance_byte(byte)

    def allowed_token_ids(
        self,
        state: ByT5ConstraintState,
        candidate_ids: Iterable[int] | None = None,
    ) -> frozenset[int]:
        # Pad, unknown, sentinels, illegal UTF-8 leads, and impossible byte
        # positions are structurally unavailable. Avoid spending transitions
        # and cache entries on them in the common full-vocabulary path.
        use_allowed_cache = candidate_ids is None
        if use_allowed_cache and state in self._allowed_cache:
            allowed = self._allowed_cache[state]
            self._allowed_cache.move_to_end(state)
            return allowed

        candidates = (
            self._structural_candidate_ids(state)
            if use_allowed_cache
            else candidate_ids
        )
        allowed_ids: set[int] = set()
        for token_id in candidates:
            key = (state, token_id)
            if key in self._cache:
                self._hits += 1
                successor = self._cache[key]
                self._cache.move_to_end(key)
            else:
                # Mask probes are intentionally not stored in the transition
                # cache. Almost all candidate successors are never selected;
                # retaining their full immutable graph states caused beam
                # experiments to grow to many gigabytes. The small allowed-set
                # cache handles repeated beam states, while actual selected
                # transitions are cached by ``transition()``.
                self._misses += 1
                successor = self._compute_transition(state, token_id)
            if successor is not None:
                allowed_ids.add(token_id)
        allowed = frozenset(allowed_ids)
        if use_allowed_cache and self.max_allowed_cache_entries:
            if len(self._allowed_cache) >= self.max_allowed_cache_entries:
                self._allowed_cache.popitem(last=False)
            self._allowed_cache[state] = allowed
        return allowed

    def _structural_candidate_ids(
        self, state: ByT5ConstraintState
    ) -> Iterable[int]:
        if state.pending_utf8:
            return (
                self.vocabulary.token_for_byte(byte)
                for byte in range(0x80, 0xC0)
            )
        # Flat SBN admits only ASCII space or printable scalar values. Avoid
        # constructing and caching doomed transitions for ASCII controls and
        # DEL. Non-ASCII lead bytes are useful only in lexical modes whose
        # current prefix can accept a scalar in one of the supported Unicode
        # character classes.
        scalar_bytes = list(range(0x20, 0x7F))
        if _required_non_ascii_kind(state.sbn) is not None:
            scalar_bytes.extend(range(0xC2, 0xF5))
        byte_ids = (
            self.vocabulary.token_for_byte(byte) for byte in scalar_bytes
        )
        return (self.vocabulary.eos_token_id, *byte_ids)

    def boolean_mask(
        self, state: ByT5ConstraintState, vocab_size: int | None = None
    ) -> tuple[bool, ...]:
        size = self.vocabulary.vocab_size if vocab_size is None else vocab_size
        if size < 0:
            raise ValueError("vocab_size must be non-negative")
        allowed = self.allowed_token_ids(
            state,
            None if size == self.vocabulary.vocab_size else range(size),
        )
        return tuple(token_id in allowed for token_id in range(size))

    def cache_info(self) -> TransitionCacheInfo:
        return TransitionCacheInfo(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            evictions=self._evictions,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._allowed_cache.clear()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def allowed_cache_size(self) -> int:
        return len(self._allowed_cache)


class ByT5PrefixStateCache:
    """Recover constraint states from full decoder prefixes during beam search.

    Hugging Face may reorder beam rows at every step. Caching by the complete
    token prefix is therefore correct under greedy, sampling, and beam search,
    unlike attaching mutable state to a fixed row number.
    """

    def __init__(
        self,
        controller: ByT5TokenConstraintController | None = None,
        *,
        policy: SBNPolicy | None = None,
        decoder_start_token_id: int = 0,
    ):
        self.controller = controller or ByT5TokenConstraintController()
        self.decoder_start_token_id = decoder_start_token_id
        self._initial = ByT5ConstraintState.initial(policy)
        self._states: dict[
            tuple[int, ...], ByT5ConstraintState | None
        ] = {(): self._initial}

    def normalize_prefix(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        prefix = tuple(int(token_id) for token_id in token_ids)
        if prefix and prefix[0] == self.decoder_start_token_id:
            return prefix[1:]
        return prefix

    def state_for(
        self, token_ids: Sequence[int]
    ) -> ByT5ConstraintState | None:
        prefix = self.normalize_prefix(token_ids)
        cached = self._states.get(prefix, ...)
        if cached is not ...:
            return cached

        parent = self.state_for(prefix[:-1])
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

    def clear(self) -> None:
        self._states = {(): self._initial}


class ByT5SBNLogitsProcessor:
    """Hugging Face-compatible logits processor for constrained SBN output.

    Instantiate this class and place it in a Transformers
    ``LogitsProcessorList``. ``__call__`` mutates and returns the provided score
    tensor. No torch symbols are imported here; the tensor only needs the
    standard indexing, ``clone()``, and ``fill_()`` operations.
    """

    def __init__(
        self,
        *,
        vocabulary: ByT5Vocabulary | None = None,
        policy: SBNPolicy | None = None,
        decoder_start_token_id: int = 0,
        max_new_tokens: int | None = None,
        bounded_completion_window: int = 0,
        max_certified_candidates: int | None = None,
        initial_max_certified_candidates: int | None = None,
        completion_budget_factor: float = 1.0,
        max_completion_cache_entries: int = 100_000,
    ):
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if bounded_completion_window < 0:
            raise ValueError("bounded_completion_window must be non-negative")
        if bounded_completion_window and max_new_tokens is None:
            raise ValueError(
                "bounded completion requires a max_new_tokens budget"
            )
        if (
            max_certified_candidates is not None
            and max_certified_candidates <= 0
        ):
            raise ValueError("max_certified_candidates must be positive")
        if (
            initial_max_certified_candidates is not None
            and initial_max_certified_candidates <= 0
        ):
            raise ValueError(
                "initial_max_certified_candidates must be positive"
            )
        if not 0 < completion_budget_factor <= 1:
            raise ValueError("completion_budget_factor must be in (0, 1]")
        if max_completion_cache_entries < 0:
            raise ValueError(
                "max_completion_cache_entries must be non-negative"
            )
        self.controller = ByT5TokenConstraintController(vocabulary)
        self.prefix_states = ByT5PrefixStateCache(
            self.controller,
            policy=policy,
            decoder_start_token_id=decoder_start_token_id,
        )
        self.max_new_tokens = max_new_tokens
        self.bounded_completion_window = bounded_completion_window
        self.max_certified_candidates = max_certified_candidates
        self.initial_max_certified_candidates = (
            initial_max_certified_candidates
        )
        self.completion_budget_factor = completion_budget_factor
        self._bounded_finish_cache = CompletionDistanceCache[
            ByT5ConstraintState
        ](max_completion_cache_entries)
        self.reset_stats()

    def allowed_token_ids_for_prefix(
        self, token_ids: Sequence[int]
    ) -> frozenset[int]:
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return frozenset()
        allowed = self.controller.allowed_token_ids(state)
        # Without model scores there is no principled top-k order, so this
        # diagnostic API returns the complete bounded allowlist even when
        # generation uses lazy score-ranked certification.
        return self._apply_bounded_completion(
            state,
            allowed,
            token_ids,
            candidate_limit=None,
        )

    def token_is_allowed_for_prefix(
        self, token_ids: Sequence[int], token_id: int
    ) -> bool:
        """Audit one proposed token without enumerating the vocabulary.

        This follows the same structural and finite-budget decision as the
        logits mask, but deliberately does not update generation timing stats.
        It is useful for replaying a known valid model path and finding the
        first finite-budget false rejection.
        """
        state = self.prefix_states.state_for(token_ids)
        if state is None or state.is_accepting:
            return False
        successor = self.controller._compute_transition(state, int(token_id))
        if successor is None:
            return False
        if not self.bounded_completion_window or self.max_new_tokens is None:
            return True
        generated = len(self.prefix_states.normalize_prefix(token_ids))
        remaining = self.max_new_tokens - generated
        if remaining > self.bounded_completion_window:
            return True
        if remaining <= 0:
            return False
        return self._successor_has_completion(successor, remaining - 1)

    def _successor_has_completion(
        self, successor: ByT5ConstraintState, remaining_after_token: int
    ) -> bool:
        return (
            self._successor_completion_distance(
                successor, remaining_after_token
            )
            is not None
        )

    def _successor_completion_distance(
        self, successor: ByT5ConstraintState, remaining_after_token: int
    ) -> int | None:
        """Return the checked witness length under the configured reserve."""
        if successor.is_accepting:
            return 0
        certificate_budget = remaining_after_token
        if certificate_budget > 0 and self.completion_budget_factor < 1:
            certificate_budget = max(
                1,
                math.floor(
                    certificate_budget * self.completion_budget_factor
                ),
            )
        return self._completion_distance_within(successor, certificate_budget)

    def _apply_bounded_completion(
        self,
        state: ByT5ConstraintState,
        allowed: Iterable[int],
        token_ids: Sequence[int],
        *,
        ranked_token_ids: Iterable[int] | None = None,
        candidate_limit: int | None = None,
    ) -> frozenset[int]:
        allowed_set = frozenset(allowed)
        if not self.bounded_completion_window or self.max_new_tokens is None:
            return allowed_set
        generated = len(self.prefix_states.normalize_prefix(token_ids))
        remaining = self.max_new_tokens - generated
        if remaining > self.bounded_completion_window:
            return allowed_set
        if remaining <= 0:
            return frozenset()

        self._stats_bounded_rows += 1
        bounded: set[int] = set()
        candidates = (
            ranked_token_ids
            if ranked_token_ids is not None
            else sorted(allowed_set)
        )
        for token_id in candidates:
            if token_id not in allowed_set:
                continue
            self._stats_completion_certificates_tested += 1
            successor = self.controller._compute_transition(state, token_id)
            if successor is None:
                continue
            distance = self._successor_completion_distance(
                successor, remaining - 1
            )
            if distance is not None:
                bounded.add(token_id)
                self._stats_completion_certificates_accepted += 1
                self._stats_completion_witness_tokens_total += distance
                self._stats_completion_witness_tokens_min = (
                    distance
                    if self._stats_completion_witness_tokens_min is None
                    else min(self._stats_completion_witness_tokens_min, distance)
                )
                self._stats_completion_witness_tokens_max = (
                    distance
                    if self._stats_completion_witness_tokens_max is None
                    else max(self._stats_completion_witness_tokens_max, distance)
                )
                slack = remaining - 1 - distance
                self._stats_completion_slack_total += slack
                self._stats_completion_slack_min = (
                    slack
                    if self._stats_completion_slack_min is None
                    else min(self._stats_completion_slack_min, slack)
                )
                self._stats_completion_slack_max = (
                    slack
                    if self._stats_completion_slack_max is None
                    else max(self._stats_completion_slack_max, slack)
                )
                if candidate_limit is not None and len(bounded) >= candidate_limit:
                    self._stats_certified_limit_hits += 1
                    break
        return frozenset(bounded)

    def _can_finish_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> bool:
        return self._completion_distance_within(state, remaining) is not None

    def _completion_distance_within(
        self, state: ByT5ConstraintState, remaining: int
    ) -> int | None:
        """Return the concrete accepting-witness length within ``remaining``.

        This is a certificate-producing oracle rather than an exhaustive
        traversal of the open-class SBN language.  Following one
        syntax-aware repair plan keeps the last-step mask cheap and, most
        importantly, every ``True`` result is witnessed by actual automaton
        transitions ending in EOS.  A failed plan may conservatively remove a
        token that has some other completion; it can never admit a doomed
        token.
        """
        if state.is_accepting:
            return 0
        if remaining <= 0:
            return None
        reusable, cached = self._bounded_finish_cache.lookup(state, remaining)
        if reusable:
            return cached
        if state.can_end():
            self._bounded_finish_cache.store(state, remaining, 1)
            return 1  # EOS consumes one of the remaining token steps.
        cursor = state
        used = 0
        planned_token_ids: list[int] = []
        while used < remaining:
            if cursor.is_accepting:
                self._bounded_finish_cache.store(state, remaining, used)
                return used
            if cursor.can_end():
                # The witness is the EOS transition, which consumes the final
                # token step without changing the decoded surface string.
                distance = used + 1
                self._bounded_finish_cache.store(
                    state, remaining, distance
                )
                return distance

            if cursor.pending_utf8:
                # A pending scalar was introduced by the model candidate, not
                # by an ASCII repair suffix. Complete it before planning at
                # the SBN character level.
                planned_token_ids.clear()
                token_id = self._next_completion_token(cursor)
            else:
                if not planned_token_ids:
                    planned_token_ids.extend(
                        self._completion_token_plan(cursor)
                    )
                token_id = (
                    planned_token_ids.pop(0)
                    if planned_token_ids
                    else None
                )
            if token_id is None:
                self._bounded_finish_cache.store(state, remaining, None)
                return None
            successor = self.controller._compute_transition(cursor, token_id)
            if successor is None:
                self._bounded_finish_cache.store(state, remaining, None)
                return None
            cursor = successor
            used += 1

        result = used if cursor.is_accepting else None
        self._bounded_finish_cache.store(state, remaining, result)
        return result

    def completion_cache_info(self) -> TransitionCacheInfo:
        return self._bounded_finish_cache.cache_info()

    def clear_completion_cache(self) -> None:
        self._bounded_finish_cache.clear()

    def _next_completion_token(
        self, state: ByT5ConstraintState
    ) -> int | None:
        """Choose the next byte of a concrete short completion witness."""
        if state.pending_utf8:
            # At most three continuation bytes are needed.  Every ID returned
            # by this mask preserves both UTF-8 and the SBN lexical class, so a
            # deterministic first choice is a valid constructive witness.
            continuation = sorted(self.controller.allowed_token_ids(state))
            return continuation[0] if continuation else None

        plan = self._completion_token_plan(state)
        return plan[0] if plan else None

    def _completion_token_plan(
        self, state: ByT5ConstraintState
    ) -> tuple[int, ...]:
        """Compile one whole repair fragment without changing goals midway.

        Finite SBN tokens share prefixes (for example, `CONTINUATION` and
        `Cause`). Re-selecting a shortest token after every character can
        silently switch goals. Returning the complete byte plan makes the
        witness follow the token chosen for the originating state.
        """
        if state.pending_utf8:
            return ()

        suffix = self._suggest_completion_suffix(state.sbn)
        if not suffix:
            return ()
        encoded = suffix.encode("utf-8")
        return tuple(
            self.controller.vocabulary.token_for_byte(byte)
            for byte in encoded
        )

    @staticmethod
    def _suggest_completion_suffix(state: SBNConstraintState) -> str:
        token = state.current_lexeme
        if state.mode is ParserMode.ITEM:
            if not token and state.pending_node_edges:
                # Resolve forward concept references in the box where they
                # were requested before opening a Proposition-required box.
                # Moving the target into the new box can make an otherwise
                # completable W prefix violate DRS accessibility (for
                # example, `Time +1` immediately before `CONTINUATION`).
                return "entity.n.00"
            if state.required_box_count > state.box_count:
                # A pending Proposition target requires another context.  Keep
                # following the intended separator after its first character:
                # recomputing the generic shortest finite token at `C` would
                # otherwise switch from `CONTINUATION` to the shorter role
                # `Cause`, attach `?`, and repeat forever without opening the
                # required box.
                separator = "CONTINUATION"
                if separator.startswith(token):
                    suffix = separator[len(token) :]
                    return suffix or " "
            if (
                token
                and state.pending_node_edges
                and "CONJUNCTION".startswith(token)
            ):
                # A conjunction aliases the newly opened context to its
                # attachment context.  That makes it the constructive repair
                # for a partially generated `CO...` separator when a forward
                # concept edge is waiting to materialize; choosing the shorter
                # `CONTRAST` can put the target in a distinct box and create a
                # subordination loop.
                suffix = "CONJUNCTION"[len(token) :]
                return suffix or " "
            if not token and state.active_concept is not None:
                # A trailing delimiter cannot be followed by EOS.  Extending
                # the active concept with the short, general `Of` role is much
                # cheaper than introducing another 11-byte placeholder
                # concept, and its target planner can choose either a constant
                # or a cross-box reference that satisfies W connectedness.
                return "Of"
            if token:
                finite_values = set(spec.SEPARATORS)
                if state.active_concept is not None:
                    finite_values.update(spec.ATTACHMENT_LABELS)
                finite = [
                    value
                    for value in finite_values
                    if value.startswith(token)
                ]
                suffixes = [value[len(token) :] for value in finite]
                if spec.is_synset(token, strict_lemma_case=state.policy.strict_lemma_case):
                    suffixes.append("")
                if spec.is_synset_prefix(
                    token, strict_lemma_case=state.policy.strict_lemma_case
                ):
                    for pos in "nvarx":
                        ending = f".{pos}.00"
                        suffixes.extend(
                            ending[start:]
                            for start in range(len(ending) + 1)
                            if spec.is_synset(
                                token + ending[start:],
                                strict_lemma_case=state.policy.strict_lemma_case,
                            )
                        )
                if suffixes:
                    suffix = min(suffixes, key=lambda value: (len(value), value))
                    return suffix or " "
            return "entity.n.00"

        if state.mode is ParserMode.SEPARATOR_TARGET:
            if token and spec.is_box_reference(token):
                return " "
            # `<1` connects the newly opened box to its predecessor; unlike
            # the special `<0>` boundary it is also safe for W's final
            # connectedness requirement.
            return "<1" if not token else "1"

        if state.pending_label == "Name":
            if spec.quoted_string_is_open(token):
                return '"' if len(token) > 1 else 'x"'
            if token and spec.is_name_constant(token):
                return " "
            return "?"

        if state.pending_label == "Proposition":
            if token and spec.is_box_reference(token):
                return " "
            if token.startswith("<"):
                # A backward Proposition target can be the edge that repairs
                # disconnected boxes after a `<0>` discourse boundary.  The
                # nearest box is not always legal (it may close a
                # subordination cycle), so complete the partial decimal with
                # the nearest target whose committed edge yields an accepting
                # graph.
                magnitude_prefix = token[1:]
                for magnitude in range(1, state.active_box + 1):
                    digits = str(magnitude)
                    if not digits.startswith(magnitude_prefix):
                        continue
                    suffix = digits[len(magnitude_prefix) :]
                    probe = state.advance_text(suffix)
                    if not probe.is_dead and probe.can_end():
                        return suffix or " "
            # A forward box avoids the typed box->concept->box cycle that a
            # reference to the concept's own box would create under B/W.
            return ">1" if not token else "1"

        if token:
            if spec.is_node_reference(token):
                # A decimal prefix may already spell a complete reference
                # whose boundary is semantically invalid, while a longer
                # reference with the same prefix is valid.  For example,
                # `-1` can close a two-node cycle but `-10` can refer to an
                # earlier acyclic concept.  Do not greedily commit the short
                # interpretation in that case: choose the shortest concrete
                # extension whose automaton state can accept EOS.
                boundary = state.advance_text(" ")
                if not boundary.is_dead:
                    return " "
                if (
                    token.startswith("-")
                    and state.active_concept is not None
                ):
                    candidates: list[str] = []
                    for magnitude in range(1, state.active_concept + 1):
                        reference = f"-{magnitude}"
                        if len(reference) <= len(token) or not reference.startswith(
                            token
                        ):
                            continue
                        suffix = reference[len(token) :]
                        probe = state.advance_text(suffix)
                        if not probe.is_dead and probe.can_end():
                            candidates.append(suffix)
                    if candidates:
                        return min(
                            candidates,
                            key=lambda value: (len(value), value),
                        )
                return " "
            if spec.is_constant(token):
                return " "
            if spec.quoted_string_is_open(token):
                return '"' if len(token) > 1 else 'x"'
            finite = [
                value[len(token) :]
                for value in spec.SPECIAL_CONSTANTS
                if value.startswith(token)
            ]
            if finite:
                return min(finite, key=lambda value: (len(value), value)) or " "
            constant_suffix = ByT5SBNLogitsProcessor._constant_completion_suffix(
                token
            )
            if constant_suffix is not None:
                return constant_suffix or " "
            if spec.is_node_reference_prefix(token):
                return "1"
        elif state.active_concept is not None:
            # Prefer the shortest constant whenever it already produces an
            # accepting W graph.  After a special `<0>` discourse boundary the
            # boxes are intentionally not connected by a separator edge, so a
            # constant cannot finish; in that case find a concrete backward
            # concept reference whose committed edge does connect the graph.
            constant_probe = state.advance_text("?")
            if not constant_probe.is_dead and constant_probe.can_end():
                return "?"
            for target in range(state.active_concept - 1, -1, -1):
                reference = f"{target - state.active_concept:+d}"
                probe = state.advance_text(reference)
                if not probe.is_dead and probe.can_end():
                    return reference
        return "?"

    @staticmethod
    def _constant_completion_suffix(token: str) -> str | None:
        """Return one short witness for an incomplete regular constant."""
        candidates: list[str] = []
        if token.isdigit() and len(token) == 2:
            candidates.append(":00")
        if ":" in token:
            _, minute = token.split(":", 1)
            if len(minute) <= 2:
                candidates.append("0" * (2 - len(minute)))
        if token.endswith("."):
            candidates.append("0")
        if token.endswith("/"):
            candidates.append("1")
        if token.endswith("-") and not token.startswith("-"):
            candidates.append("0")
        if token.startswith("'") and (
            len(token) == 1 or not token.endswith("'")
        ):
            body = token[1:]
            if len(body) <= 4:
                candidates.append("X" * (4 - len(body)) + "'")
        valid = [
            suffix
            for suffix in candidates
            if spec.is_constant(token + suffix)
        ]
        return min(valid, key=lambda value: (len(value), value)) if valid else None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        started = time.perf_counter()
        # One device-to-host synchronization for the whole beam is much less
        # expensive on MPS/CUDA than calling ``tolist()`` once per row.
        prefix_rows = (
            input_ids.tolist()
            if hasattr(input_ids, "tolist")
            else [row.tolist() for row in input_ids]
        )
        score_rows = (
            scores.tolist()
            if self.max_certified_candidates is not None
            and hasattr(scores, "tolist")
            else None
        )
        row_allowlists: list[list[int] | None] = []
        for row_index, prefix in enumerate(prefix_rows):
            state = self.prefix_states.state_for(prefix)

            # Generation code may retain completed rows until every batch item
            # has emitted EOS. Those rows will be replaced with padding by the
            # generation loop, so their logits need no further restriction.
            if state is not None and state.is_accepting:
                row_allowlists.append(None)
                continue

            allowed = () if state is None else self.controller.allowed_token_ids(
                state,
                None
                if scores.shape[-1] == self.controller.vocabulary.vocab_size
                else range(scores.shape[-1]),
            )
            if state is not None:
                ranked: list[int] | None = None
                if score_rows is not None:
                    ranked = sorted(
                        allowed,
                        key=lambda token_id: (
                            -float(score_rows[row_index][token_id]),
                            token_id,
                        ),
                    )
                generated_count = len(
                    self.prefix_states.normalize_prefix(prefix)
                )
                candidate_limit = self.max_certified_candidates
                if (
                    generated_count == 0
                    and self.initial_max_certified_candidates is not None
                ):
                    candidate_limit = self.initial_max_certified_candidates
                allowed = sorted(
                    self._apply_bounded_completion(
                        state,
                        allowed,
                        prefix,
                        ranked_token_ids=ranked,
                        candidate_limit=(
                            candidate_limit
                            if ranked is not None
                            else None
                        ),
                    )
                )
            else:
                allowed = []
            allowed_count = len(allowed)
            row_allowlists.append(allowed)
            self._stats_rows += 1
            self._stats_allowed_total += allowed_count
            self._stats_allowed_min = (
                allowed_count
                if self._stats_allowed_min is None
                else min(self._stats_allowed_min, allowed_count)
            )
            self._stats_allowed_max = (
                allowed_count
                if self._stats_allowed_max is None
                else max(self._stats_allowed_max, allowed_count)
            )
            if allowed_count == 0:
                self._stats_all_masked_rows += 1
        if hasattr(scores, "new_full") and hasattr(scores, "reshape"):
            # Build one flattened gather/scatter for the complete beam. This
            # avoids several tiny device operations per row while preserving
            # the original dtype and device. Returning a new score tensor is
            # part of the Hugging Face LogitsProcessor contract.
            vocabulary_size = int(scores.shape[-1])
            positions: list[int] = []
            for row_index, allowed in enumerate(row_allowlists):
                token_ids = range(vocabulary_size) if allowed is None else allowed
                offset = row_index * vocabulary_size
                positions.extend(offset + token_id for token_id in token_ids)
            masked = scores.new_full(scores.shape, float("-inf"))
            if positions:
                indices = input_ids.new_tensor(positions)
                masked_flat = masked.reshape(-1)
                score_flat = scores.reshape(-1)
                masked_flat[indices] = score_flat[indices]
            scores = masked
        else:
            # Dependency-free fallback used by unit tests and simple tensor
            # lookalikes.
            for row_index, allowed in enumerate(row_allowlists):
                if allowed is None:
                    continue
                row = scores[row_index]
                if not allowed:
                    row.fill_(float("-inf"))
                    continue
                kept = row[allowed].clone()
                row.fill_(float("-inf"))
                row[allowed] = kept
        self._stats_calls += 1
        self._stats_seconds += time.perf_counter() - started
        return scores

    def stats(self) -> ByT5ProcessorStats:
        return ByT5ProcessorStats(
            calls=self._stats_calls,
            rows=self._stats_rows,
            seconds=self._stats_seconds,
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
            certified_limit_hits=self._stats_certified_limit_hits,
            completion_witness_tokens_total=(
                self._stats_completion_witness_tokens_total
            ),
            completion_witness_tokens_min=(
                self._stats_completion_witness_tokens_min
            ),
            completion_witness_tokens_max=(
                self._stats_completion_witness_tokens_max
            ),
            completion_slack_total=self._stats_completion_slack_total,
            completion_slack_min=self._stats_completion_slack_min,
            completion_slack_max=self._stats_completion_slack_max,
        )

    def reset_stats(self) -> None:
        self._stats_calls = 0
        self._stats_rows = 0
        self._stats_seconds = 0.0
        self._stats_allowed_total = 0
        self._stats_allowed_min: int | None = None
        self._stats_allowed_max: int | None = None
        self._stats_all_masked_rows = 0
        self._stats_bounded_rows = 0
        self._stats_completion_certificates_tested = 0
        self._stats_completion_certificates_accepted = 0
        self._stats_certified_limit_hits = 0
        self._stats_completion_witness_tokens_total = 0
        self._stats_completion_witness_tokens_min: int | None = None
        self._stats_completion_witness_tokens_max: int | None = None
        self._stats_completion_slack_total = 0
        self._stats_completion_slack_min: int | None = None
        self._stats_completion_slack_max: int | None = None
