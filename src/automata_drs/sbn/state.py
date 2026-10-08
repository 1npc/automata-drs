"""Immutable, character-incremental constraint state for SBN generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from . import spec


class ParserMode(str, Enum):
    ITEM = "item"
    SEPARATOR_TARGET = "separator-target"
    ATTACHMENT_TARGET = "attachment-target"


@dataclass(frozen=True, slots=True)
class SBNPolicy:
    """Strictness switches that remain stable for every beam hypothesis."""

    strict_lemma_case: bool = True
    require_nonempty_final_context: bool = True
    # The reference parser can continue after warning about a cycle, but its
    # data notes classify cyclic SBNs as erroneous: the complete typed SBN
    # graph is intended to be a DAG.
    enforce_graph_acyclicity: bool = True
    enforce_node_reference_range: bool = True
    enforce_node_reference_existence: bool = True
    enforce_proposition_target_range: bool = True
    enforce_proposition_target_existence: bool = True
    # Kept opt-in so the structural B policy and semantic W policy remain
    # separate experiment conditions. The local gold corpus has one upstream-
    # Referee-confirmed semantic exception.
    enforce_referee_wellformedness: bool = False

    @classmethod
    def reference_parser_compatible(cls) -> "SBNPolicy":
        """Mimic the reference parser's cycle-tolerant recovery behavior."""
        return cls(enforce_graph_acyclicity=False)

    @classmethod
    def semantic_referee(cls) -> "SBNPolicy":
        """Enable the full Referee-aligned W configuration."""
        return cls(enforce_referee_wellformedness=True)

    @classmethod
    def for_configuration(cls, configuration: str) -> "SBNPolicy":
        """Return an executable S/R/G/B/W ablation policy."""
        if configuration == "S":
            return cls(
                enforce_graph_acyclicity=False,
                enforce_node_reference_range=False,
                enforce_node_reference_existence=False,
                enforce_proposition_target_range=False,
                enforce_proposition_target_existence=False,
            )
        if configuration == "R":
            return cls(
                enforce_graph_acyclicity=False,
                enforce_proposition_target_range=False,
                enforce_proposition_target_existence=False,
            )
        if configuration == "G":
            return cls(
                enforce_proposition_target_range=False,
                enforce_proposition_target_existence=False,
            )
        if configuration == "B":
            return cls()
        if configuration == "W":
            return cls.semantic_referee()
        raise ValueError(f"unknown SBN constraint configuration {configuration!r}")


@dataclass(frozen=True, slots=True, order=True)
class PendingNodeEdge:
    source: int
    target: int
    label: str


@dataclass(frozen=True, slots=True)
class SBNConstraintState:
    """One hashable constrained-decoding hypothesis.

    The state accepts decoded surface fragments, not model token IDs. This
    keeps SBN semantics independent of a tokenizer and lets a later adapter
    cache ``(state_signature, token_id)`` transitions.
    """

    policy: SBNPolicy = SBNPolicy()
    mode: ParserMode = ParserMode.ITEM
    current_lexeme: str = ""
    pending_label: str | None = None
    active_concept: int | None = None
    concept_count: int = 0
    active_box: int = 0
    box_count: int = 1
    required_box_count: int = 1
    token_count: int = 0
    char_offset: int = 0
    concept_labels: tuple[str, ...] = ()
    concept_boxes: tuple[int, ...] = ()
    node_edges: frozenset[tuple[int, int, str]] = frozenset()
    constant_edges: tuple[tuple[int, str, str], ...] = ()
    pending_node_edges: frozenset[PendingNodeEdge] = frozenset()
    separator_connectors: tuple[tuple[str, str], ...] = ()
    box_edges: tuple[tuple[int, int, str], ...] = ()
    proposition_targets: tuple[tuple[int, int], ...] = ()
    dead_reason: str | None = None
    finished: bool = False

    @classmethod
    def initial(cls, policy: SBNPolicy | None = None) -> "SBNConstraintState":
        return cls(policy=policy or SBNPolicy())

    @property
    def is_dead(self) -> bool:
        return self.dead_reason is not None

    @property
    def is_extendable(self) -> bool:
        return not self.is_dead and not self.finished

    @property
    def is_accepting(self) -> bool:
        return self.finished and not self.is_dead

    def advance_text(self, fragment: str) -> "SBNConstraintState":
        state = self
        if state.finished:
            return state._die("cannot advance an already finished SBN")
        for char in fragment:
            if state.is_dead:
                return state
            next_offset = state.char_offset + 1
            if char == " " and not spec.quoted_string_is_open(state.current_lexeme):
                if not state.current_lexeme:
                    return replace(state, char_offset=next_offset)._die(
                        "leading or repeated whitespace is not canonical SBN"
                    )
                state = state._commit_current_lexeme()
                if state.is_dead:
                    return replace(state, char_offset=next_offset)
                state = replace(
                    state,
                    current_lexeme="",
                    token_count=state.token_count + 1,
                    char_offset=next_offset,
                )
                continue
            if char.isspace() and char != " ":
                return replace(state, char_offset=next_offset)._die(
                    "only ASCII space is allowed between flat SBN lexemes"
                )
            candidate = state.current_lexeme + char
            state = replace(state, current_lexeme=candidate, char_offset=next_offset)
            if not state._current_lexeme_is_prefix():
                return state._die(state._prefix_failure_reason(candidate))
        return state

    def advance_lexeme(self, lexeme: str) -> "SBNConstraintState":
        """Consume one already-delimited lexeme.

        This is the fast path for corpus audits. Model decoding must use
        :meth:`advance_text`, because a model token is not guaranteed to align
        with an SBN lexeme boundary.
        """
        if self.finished:
            return self._die("cannot advance an already finished SBN")
        if self.is_dead:
            return self
        if self.current_lexeme:
            return self._die("advance_lexeme requires an empty lexical buffer")
        if not lexeme or (any(char.isspace() for char in lexeme) and not spec.is_quoted_constant(lexeme)):
            return self._die(f"invalid pre-delimited lexeme {lexeme!r}")
        state = replace(
            self,
            current_lexeme=lexeme,
            char_offset=self.char_offset + len(lexeme),
        )
        if not state._current_lexeme_is_prefix():
            return state._die(state._prefix_failure_reason(lexeme))
        state = state._commit_current_lexeme()
        if state.is_dead:
            return state
        return replace(
            state,
            current_lexeme="",
            token_count=state.token_count + 1,
        )

    def can_end(self) -> bool:
        return self.finish().is_accepting

    def finish(self) -> "SBNConstraintState":
        if self.finished or self.is_dead:
            return self
        if not self.current_lexeme:
            return self._die("SBN is empty or ends with whitespace")
        state = self._commit_current_lexeme()
        if state.is_dead:
            return state
        state = replace(
            state,
            current_lexeme="",
            token_count=state.token_count + 1,
        )
        return state._finish_committed()

    def finish_committed(self) -> "SBNConstraintState":
        """Finish after all inputs were consumed with :meth:`advance_lexeme`."""
        if self.finished or self.is_dead:
            return self
        if self.current_lexeme:
            return self._die("finish_committed requires an empty lexical buffer")
        return self._finish_committed()

    def _finish_committed(self) -> "SBNConstraintState":
        state = self
        if state.mode is not ParserMode.ITEM:
            return state._die(f"SBN ends while expecting {state.mode.value}")
        if state.concept_count == 0:
            return state._die("SBN contains no concept")
        if state.policy.require_nonempty_final_context and state.active_concept is None:
            return state._die("the final context contains no concept")
        if state.policy.enforce_node_reference_existence and state.pending_node_edges:
            target = min(edge.target for edge in state.pending_node_edges)
            return state._die(f"unresolved forward concept reference to index {target}")
        if (
            state.policy.enforce_proposition_target_existence
            and state.required_box_count > state.box_count
        ):
            return state._die(
                "unresolved Proposition target requires "
                f"{state.required_box_count} contexts, found {state.box_count}"
            )
        state = state._apply_referee_guard(require_connected=True)
        if state.is_dead:
            return state
        return replace(state, finished=True)

    def _current_lexeme_is_prefix(self) -> bool:
        token = self.current_lexeme
        if self.mode is ParserMode.ITEM:
            if spec.is_synset_prefix(
                token, strict_lemma_case=self.policy.strict_lemma_case
            ):
                return self._next_concept_is_viable()
            if spec.finite_token_is_prefix(token, spec.SEPARATORS):
                return True
            return self.active_concept is not None and spec.finite_token_is_prefix(
                token, spec.ATTACHMENT_LABELS
            )
        if self.mode is ParserMode.SEPARATOR_TARGET:
            return spec.is_box_reference_prefix(token, backward_only=True)
        if self.mode is ParserMode.ATTACHMENT_TARGET:
            if self.pending_label == "Name":
                return spec.is_name_constant_prefix(token)
            if self.pending_label == "Proposition":
                if spec.is_box_reference_prefix(token):
                    if spec.is_box_reference(token):
                        return self._proposition_prefix_is_extendable(
                            spec.parse_box_reference(token)
                        )
                    if token in {"<", ">"}:
                        return self._proposition_sign_has_completion(token)
                    return True
                return False
            if spec.is_node_reference_prefix(token):
                if spec.is_node_reference(token):
                    return self._node_reference_prefix_is_extendable(
                        spec.parse_node_reference(token)
                    )
                return True
            if spec.is_constant_prefix(token):
                return True
            return False
        return False

    def _prefix_failure_reason(self, token: str) -> str:
        if (
            self.mode is ParserMode.ITEM
            and spec.is_synset_prefix(
                token, strict_lemma_case=self.policy.strict_lemma_case
            )
            and not self._next_concept_is_viable()
        ):
            return (
                f"introducing concept index {self.concept_count} would violate "
                "DRS well-formedness"
            )
        if (
            self.mode is ParserMode.ATTACHMENT_TARGET
            and self.active_concept is not None
            and spec.is_node_reference(token)
        ):
            offset = spec.parse_node_reference(token)
            target = self.active_concept + offset
            if target < 0 and self.policy.enforce_node_reference_range:
                return (
                    f"{self.pending_label} reference {offset:+d} points before "
                    "the first concept"
                )
            if (
                target < self.concept_count
                and self.policy.enforce_graph_acyclicity
                and self._would_create_typed_cycle(
                    ("concept", self.active_concept), ("concept", target)
                )
            ):
                return (
                    f"{self.pending_label} edge {self.active_concept}->{target} "
                    "would make the SBN graph cyclic"
                )
            if (
                target < self.concept_count
                and self.policy.enforce_referee_wellformedness
                and self._node_edge_would_make_subordination_loop(
                    self.active_concept, target, self.pending_label or ""
                )
            ):
                return (
                    f"{self.pending_label} edge {self.active_concept}->{target} "
                    "would make DRS subordination cyclic"
                )
        if (
            self.mode is ParserMode.ATTACHMENT_TARGET
            and self.pending_label == "Proposition"
            and spec.is_box_reference(token)
        ):
            offset = spec.parse_box_reference(token)
            target = self.active_box + offset
            if target < 0 and self.policy.enforce_proposition_target_range:
                return (
                    f"Proposition target {offset:+d} points before the first context"
                )
            if (
                target < self.box_count
                and self.policy.enforce_graph_acyclicity
                and self.active_concept is not None
                and self._would_create_typed_cycle(
                    ("concept", self.active_concept), ("box", target)
                )
            ):
                return (
                    f"Proposition edge concept {self.active_concept}->box {target} "
                    "would make the SBN graph cyclic"
                )
            if (
                target < self.box_count
                and self.policy.enforce_referee_wellformedness
                and self._proposition_edge_would_make_subordination_loop(target)
            ):
                return (
                    f"Proposition edge concept {self.active_concept}->box {target} "
                    "would make DRS subordination cyclic"
                )
        if (
            self.mode is ParserMode.ATTACHMENT_TARGET
            and self.pending_label == "Proposition"
            and token == "<"
        ):
            return (
                "backward Proposition prefix has no viable target: every "
                "completion points before the first context or would make "
                "the SBN graph cyclic"
            )
        if (
            self.mode is ParserMode.ATTACHMENT_TARGET
            and self.pending_label in {"Name", "Proposition"}
        ):
            return f"invalid target prefix {token!r} for {self.pending_label!r}"
        return f"{token!r} cannot complete the expected {self.mode.value}"

    def _node_reference_prefix_is_extendable(self, offset: int) -> bool:
        if self.active_concept is None:
            return False
        target = self.active_concept + offset
        if target < 0:
            # Appending more digits to a negative offset only moves the target
            # farther before the beginning, so no completion can repair it.
            return not self.policy.enforce_node_reference_range
        if target >= self.concept_count:
            return True

        def is_rejected(candidate_target: int) -> bool:
            if self.policy.enforce_graph_acyclicity and self._would_create_typed_cycle(
                ("concept", self.active_concept), ("concept", candidate_target)
            ):
                return True
            return (
                self.policy.enforce_referee_wellformedness
                and self._node_edge_would_make_subordination_loop(
                    self.active_concept,
                    candidate_target,
                    self.pending_label or "",
                )
            )

        if offset == 0:
            # Canonical +0/-0 cannot be extended with another digit, so this
            # self-loop is irreparable at the character-prefix level.
            return not is_rejected(target)
        if not is_rejected(target):
            return True
        if offset > 0:
            # Appending digits selects a future concept and can therefore
            # escape a cycle in the materialized prefix.
            return True

        # A cyclic negative `-m` is repairable only when some longer decimal
        # beginning with `m` still refers to an existing, non-cyclic target.
        # For example, at concept 15, `-1` can become `-10`; at concept 1 it
        # cannot be extended without pointing before the sequence.
        magnitude_prefix = str(-offset)
        for magnitude in range(1, self.active_concept + 1):
            digits = str(magnitude)
            if len(digits) <= len(magnitude_prefix) or not digits.startswith(
                magnitude_prefix
            ):
                continue
            candidate_target = self.active_concept - magnitude
            if not is_rejected(candidate_target):
                return True
        return False

    def _proposition_prefix_is_extendable(self, offset: int) -> bool:
        target = self.active_box + offset
        if target < 0:
            # A larger backward connector remains out of range.
            return not self.policy.enforce_proposition_target_range
        if target >= self.box_count:
            return True

        def is_rejected(candidate_target: int) -> bool:
            if self.active_concept is None:
                return True
            if self.policy.enforce_graph_acyclicity and self._would_create_typed_cycle(
                ("concept", self.active_concept), ("box", candidate_target)
            ):
                return True
            return (
                self.policy.enforce_referee_wellformedness
                and self._proposition_edge_would_make_subordination_loop(
                    candidate_target
                )
            )

        if not is_rejected(target):
            return True
        if offset == 0:
            return False
        if offset > 0:
            # Appending digits selects a future box.
            return True

        magnitude_prefix = str(-offset)
        for magnitude in range(1, self.active_box + 1):
            digits = str(magnitude)
            if len(digits) <= len(magnitude_prefix) or not digits.startswith(
                magnitude_prefix
            ):
                continue
            if not is_rejected(self.active_box - magnitude):
                return True
        return False

    def _proposition_sign_has_completion(self, sign: str) -> bool:
        """Whether a partial box connector has a dynamically viable target.

        ``spec.is_box_reference_prefix`` can only establish lexical liveness.
        At a particular graph state every connector with one sign may already
        be impossible: all backward boxes can form cycles, or every backward
        magnitude can be out of range.  Admitting the bare sign in that case
        creates a state with no model-token successor and eventually an
        all-masked row.  Checking the finite existing backward range is exact;
        for a forward sign, ``>1`` is a sufficient future-box witness whenever
        ``>0`` is not already viable.
        """

        if sign == ">":
            return any(
                self._proposition_prefix_is_extendable(offset)
                for offset in (0, 1)
            )
        if sign == "<":
            if not self.policy.enforce_proposition_target_range:
                return True
            return any(
                self._proposition_prefix_is_extendable(-magnitude)
                for magnitude in range(self.active_box + 1)
            )
        raise ValueError(f"unsupported Proposition connector sign: {sign!r}")

    def _next_concept_is_viable(self) -> bool:
        if not any(
            edge.target == self.concept_count for edge in self.pending_node_edges
        ):
            return True
        # Pending edge validity is independent of the open-class label of the
        # concept that will occupy this absolute index. Probe with a valid
        # placeholder so an irreparable graph/scope cycle is rejected at the
        # first character that commits the hypothesis to introducing it.
        probe = self._introduce_concept("entity.n.01")
        return not probe.is_dead

    def _commit_current_lexeme(self) -> "SBNConstraintState":
        token = self.current_lexeme
        if self.mode is ParserMode.ITEM:
            if spec.is_synset(
                token, strict_lemma_case=self.policy.strict_lemma_case
            ):
                return self._introduce_concept(token)
            if token in spec.SEPARATORS:
                return replace(
                    self,
                    mode=ParserMode.SEPARATOR_TARGET,
                    pending_label=token,
                    active_concept=None,
                )
            if self.active_concept is not None and token in spec.ATTACHMENT_LABELS:
                return replace(
                    self,
                    mode=ParserMode.ATTACHMENT_TARGET,
                    pending_label=token,
                )
            return self._die(f"invalid SBN item {token!r}")

        if self.mode is ParserMode.SEPARATOR_TARGET:
            if not spec.is_box_reference(token) or not token.startswith("<"):
                return self._die(
                    f"separator {self.pending_label!r} requires a backward connector"
                )
            label = self.pending_label
            new_box = self.active_box + 1
            offset = spec.parse_box_reference(token)
            # The reference SBN graph converter resolves a separator `<k>`
            # from the newly introduced box and clamps an overshoot to box 0.
            # `<0>` is a special continuation boundary and adds no box edge.
            box_edges = self.box_edges
            if offset != 0:
                source_box = max(0, new_box + offset)
                box_edges = box_edges + ((source_box, new_box, label or ""),)
            state = replace(
                self,
                mode=ParserMode.ITEM,
                pending_label=None,
                active_box=new_box,
                box_count=self.box_count + 1,
                separator_connectors=self.separator_connectors + ((label or "", token),),
                box_edges=box_edges,
            )
            return state._apply_referee_guard()

        if self.mode is ParserMode.ATTACHMENT_TARGET:
            label = self.pending_label
            if label is None or self.active_concept is None:
                return self._die("attachment target has no active label or concept")
            if label == "Name" and spec.is_name_constant(token):
                state = self._add_constant_edge(label, token)
            elif label == "Proposition" and spec.is_box_reference(token):
                state = self._add_proposition_reference(spec.parse_box_reference(token))
            elif label not in {"Name", "Proposition"} and spec.is_node_reference(token):
                state = self._add_node_reference(label, spec.parse_node_reference(token))
            elif spec.is_constant(token) and label not in {"Name", "Proposition"}:
                state = self._add_constant_edge(label, token)
            else:
                return self._die(f"invalid target {token!r} for {label!r}")
            if state.is_dead:
                return state
            return replace(state, mode=ParserMode.ITEM, pending_label=None)

        return self._die(f"unknown parser mode {self.mode!r}")

    def _introduce_concept(self, label: str) -> "SBNConstraintState":
        new_index = self.concept_count
        state = replace(
            self,
            concept_count=self.concept_count + 1,
            active_concept=new_index,
            concept_labels=self.concept_labels + (label,),
            concept_boxes=self.concept_boxes + (self.active_box,),
        )
        pending = set(state.pending_node_edges)
        for edge in sorted(state.pending_node_edges):
            if edge.target != new_index:
                continue
            state = state._materialize_node_edge(edge.source, edge.target, edge.label)
            if state.is_dead:
                return state
            pending.remove(edge)
        return replace(state, pending_node_edges=frozenset(pending))

    def _add_node_reference(self, label: str, offset: int) -> "SBNConstraintState":
        assert self.active_concept is not None
        target = self.active_concept + offset
        if target < 0 and self.policy.enforce_node_reference_range:
            return self._die(
                f"{label} reference {offset:+d} points before the first concept"
            )
        if target < self.concept_count:
            return self._materialize_node_edge(self.active_concept, target, label)
        edge = PendingNodeEdge(self.active_concept, target, label)
        return replace(
            self,
            pending_node_edges=self.pending_node_edges | frozenset({edge}),
        )

    def _add_constant_edge(self, label: str, target: str) -> "SBNConstraintState":
        assert self.active_concept is not None
        return replace(
            self,
            constant_edges=self.constant_edges
            + ((self.active_concept, label, target),),
        )

    def _materialize_node_edge(
        self, source: int, target: int, label: str
    ) -> "SBNConstraintState":
        if self.policy.enforce_graph_acyclicity and self._would_create_typed_cycle(
            ("concept", source), ("concept", target)
        ):
            return self._die(
                f"{label} edge {source}->{target} would make the SBN graph cyclic"
            )
        state = replace(
            self,
            node_edges=self.node_edges | frozenset({(source, target, label)}),
        )
        return state._apply_referee_guard()

    def _would_create_typed_cycle(
        self,
        source: tuple[str, int],
        target: tuple[str, int],
    ) -> bool:
        if source == target:
            return True
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}

        def add(edge_source: tuple[str, int], edge_target: tuple[str, int]) -> None:
            adjacency.setdefault(edge_source, set()).add(edge_target)

        for concept, box in enumerate(self.concept_boxes):
            add(("box", box), ("concept", concept))
        for edge_source, edge_target, _ in self.node_edges:
            add(("concept", edge_source), ("concept", edge_target))
        for edge_source, edge_target, _ in self.box_edges:
            add(("box", edge_source), ("box", edge_target))
        for concept, box in self.proposition_targets:
            add(("concept", concept), ("box", box))
        frontier = [target]
        visited: set[tuple[str, int]] = set()
        while frontier:
            node = frontier.pop()
            if node == source:
                return True
            if node in visited:
                continue
            visited.add(node)
            frontier.extend(adjacency.get(node, ()))
        return False

    def _add_proposition_reference(self, offset: int) -> "SBNConstraintState":
        assert self.active_concept is not None
        target = self.active_box + offset
        if target < 0 and self.policy.enforce_proposition_target_range:
            return self._die(
                f"Proposition target {offset:+d} points before the first context"
            )
        if self.policy.enforce_graph_acyclicity and self._would_create_typed_cycle(
            ("concept", self.active_concept), ("box", target)
        ):
            return self._die(
                f"Proposition edge concept {self.active_concept}->box {target} "
                "would make the SBN graph cyclic"
            )
        state = replace(
            self,
            required_box_count=max(self.required_box_count, target + 1),
            proposition_targets=self.proposition_targets
            + ((self.active_concept, target),),
        )
        return state._apply_referee_guard()

    def _node_edge_would_make_subordination_loop(
        self,
        source: int,
        target: int,
        label: str,
    ) -> bool:
        candidate = replace(
            self,
            node_edges=self.node_edges | frozenset({(source, target, label)}),
        )
        guarded = candidate._apply_referee_guard()
        return guarded.is_dead and "DRS well-formedness" in (guarded.dead_reason or "")

    def _proposition_edge_would_make_subordination_loop(
        self,
        target_box: int,
    ) -> bool:
        if self.active_concept is None:
            return True
        candidate = replace(
            self,
            proposition_targets=self.proposition_targets
            + ((self.active_concept, target_box),),
        )
        guarded = candidate._apply_referee_guard()
        return guarded.is_dead and "DRS well-formedness" in (guarded.dead_reason or "")

    def _apply_referee_guard(
        self,
        *,
        require_connected: bool = False,
    ) -> "SBNConstraintState":
        if not self.policy.enforce_referee_wellformedness or self.is_dead:
            return self

        # Local imports avoid coupling the tokenizer-independent state module
        # to the optional graph/CLF oracle during module initialization.
        from .graph import SBNGraph
        from .wellformedness import check_sbn_graph_wellformedness

        graph = SBNGraph.from_state(self, require_accepting=False)
        if any(
            edge.target_box >= graph.box_count for edge in graph.proposition_edges
        ):
            graph = replace(
                graph,
                proposition_edges=tuple(
                    edge
                    for edge in graph.proposition_edges
                    if edge.target_box < graph.box_count
                ),
            )
        result = check_sbn_graph_wellformedness(graph)
        if result.family == "subordination-loop" or (
            require_connected and not result.valid
        ):
            return self._die(f"DRS well-formedness: {result.reason}")
        return self

    def _die(self, reason: str) -> "SBNConstraintState":
        if self.dead_reason is not None:
            return self
        return replace(self, dead_reason=reason, finished=False)
