"""Incremental constraints for a small canonical knowledge-graph language.

The module is a third, deliberately independent instantiation of the shared
``IncrementalConstraintState`` contract.  It is not an SBN or PENMAN shim.
Its surface language is line-oriented::

    NODE a Person
    NODE b Book
    EDGE a likes b

All node declarations precede edges.  Node identifiers and edge triples are
strictly lexicographically ordered, node identifiers are unique, every edge
endpoint is already declared, and duplicate edges are forbidden.  At EOS the
optional graph-level policy additionally requires weak connectivity.  This
combines finite record syntax with data-dependent registers and a final graph
invariant while remaining small enough to serve as an executable adapter
contract for future KG representations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class CanonicalTripleMode(str, Enum):
    KEYWORD = "record-keyword"
    NODE_ID = "node-id"
    NODE_LABEL = "node-label"
    EDGE_SOURCE = "edge-source"
    EDGE_RELATION = "edge-relation"
    EDGE_TARGET = "edge-target"


@dataclass(frozen=True, slots=True)
class CanonicalTriplePolicy:
    """Stable graph-level choices for one decoding run."""

    require_connected: bool = True


@dataclass(frozen=True, slots=True)
class CanonicalTripleValidationResult:
    valid: bool
    state: "CanonicalTripleConstraintState"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CanonicalTripleBatchValidationResult:
    """Result from the independent final-surface oracle."""

    valid: bool
    nodes: tuple[tuple[str, str], ...] = ()
    edges: tuple[tuple[str, str, str], ...] = ()
    reason: str | None = None


def _identifier_prefix(text: str) -> bool:
    if not text:
        return True
    if not ("a" <= text[0] <= "z"):
        return False
    return all(
        "a" <= char <= "z"
        or (char.isascii() and char.isdigit())
        or char == "_"
        for char in text[1:]
    )


def _symbol_prefix(text: str) -> bool:
    if not text:
        return True
    if not text[0].isascii() or not text[0].isalpha():
        return False
    return all(
        char.isascii() and (char.isalnum() or char in "_.-")
        for char in text[1:]
    )


@dataclass(frozen=True, slots=True)
class CanonicalTripleConstraintState:
    """One immutable prefix of the canonical triple surface language."""

    policy: CanonicalTriplePolicy = CanonicalTriplePolicy()
    mode: CanonicalTripleMode = CanonicalTripleMode.KEYWORD
    buffer: str = ""
    nodes: tuple[tuple[str, str], ...] = ()
    edges: tuple[tuple[str, str, str], ...] = ()
    edge_phase: bool = False
    pending_node_id: str | None = None
    pending_source: str | None = None
    pending_relation: str | None = None
    char_offset: int = 0
    dead_reason: str | None = None
    finished: bool = False

    @classmethod
    def initial(
        cls, policy: CanonicalTriplePolicy | None = None
    ) -> "CanonicalTripleConstraintState":
        return cls(policy=policy or CanonicalTriplePolicy())

    @property
    def is_dead(self) -> bool:
        return self.dead_reason is not None

    @property
    def is_extendable(self) -> bool:
        return not self.is_dead and not self.finished

    @property
    def is_accepting(self) -> bool:
        return self.finished and not self.is_dead

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(identifier for identifier, _ in self.nodes)

    @staticmethod
    def _open_lexeme_can_reach(text: str, lower_bound: str) -> bool:
        """Whether extending ``text`` can reach or exceed ``lower_bound``."""

        return text >= lower_bound or lower_bound.startswith(text)

    def _edge_source_prefix_is_extendable(self, text: str) -> bool:
        lower_bound = self.edges[-1][0] if self.edges else ""
        return any(
            identifier.startswith(text) and identifier >= lower_bound
            for identifier in self.node_ids
        )

    def _edge_relation_prefix_is_extendable(self, text: str) -> bool:
        if not self.edges or self.pending_source != self.edges[-1][0]:
            return True
        return self._open_lexeme_can_reach(text, self.edges[-1][1])

    def _edge_target_prefix_is_extendable(self, text: str) -> bool:
        assert self.pending_source is not None
        assert self.pending_relation is not None
        return any(
            identifier.startswith(text)
            and (
                not self.edges
                or (self.pending_source, self.pending_relation, identifier)
                > self.edges[-1]
            )
            for identifier in self.node_ids
        )

    def advance_text(self, fragment: str) -> "CanonicalTripleConstraintState":
        state = self
        if state.finished:
            return state._die("cannot advance an already finished canonical graph")
        for char in fragment:
            if state.is_dead:
                return state
            state = state._advance_char(char)
        return state

    def finish(self) -> "CanonicalTripleConstraintState":
        if self.finished or self.is_dead:
            return self
        if self.mode is CanonicalTripleMode.NODE_LABEL:
            state = self._commit_node()
        elif self.mode is CanonicalTripleMode.EDGE_TARGET:
            state = self._commit_edge()
        else:
            return self._die(
                f"canonical graph ends while expecting {self.mode.value}"
            )
        if state.is_dead:
            return state
        if state.policy.require_connected and not state._is_weakly_connected():
            return state._die("canonical graph is not weakly connected")
        return replace(state, finished=True)

    def can_end(self) -> bool:
        return self.finish().is_accepting

    def incomplete_utf8_is_extendable(self, prefix: bytes) -> bool:
        """The registered canonical surface alphabet is deliberately ASCII."""

        return False

    def _advance_char(self, char: str) -> "CanonicalTripleConstraintState":
        state = replace(self, char_offset=self.char_offset + 1)
        mode = state.mode

        if mode is CanonicalTripleMode.KEYWORD:
            if char == " ":
                return state._commit_keyword()
            if char == "\n" or char.isspace():
                return state._die(
                    "canonical graph requires one ASCII space inside records"
                )
            candidate = state.buffer + char
            if state.edge_phase:
                allowed = ("EDGE",)
            elif state.nodes:
                allowed = ("NODE", "EDGE")
            else:
                allowed = ("NODE",)
            if not any(keyword.startswith(candidate) for keyword in allowed):
                return replace(state, buffer=candidate)._die(
                    "invalid canonical graph record keyword"
                )
            return replace(state, buffer=candidate)

        if mode is CanonicalTripleMode.NODE_ID:
            if char == " ":
                return state._commit_node_id()
            if char.isspace():
                return state._die("invalid whitespace in canonical node record")
            candidate = state.buffer + char
            if not _identifier_prefix(candidate) or (
                state.nodes
                and not state._open_lexeme_can_reach(
                    candidate, state.nodes[-1][0]
                )
            ):
                return replace(state, buffer=candidate)._die(
                    "invalid canonical node identifier prefix; identifiers "
                    "cannot remain unique and increasing"
                )
            return replace(state, buffer=candidate)

        if mode is CanonicalTripleMode.NODE_LABEL:
            if char == "\n":
                committed = state._commit_node()
                return (
                    replace(committed, mode=CanonicalTripleMode.KEYWORD)
                    if not committed.is_dead
                    else committed
                )
            if char.isspace():
                return state._die("invalid whitespace in canonical node label")
            candidate = state.buffer + char
            if not _symbol_prefix(candidate):
                return replace(state, buffer=candidate)._die(
                    "invalid canonical node label"
                )
            return replace(state, buffer=candidate)

        if mode is CanonicalTripleMode.EDGE_SOURCE:
            if char == " ":
                return state._commit_edge_source()
            if char.isspace():
                return state._die("invalid whitespace in canonical edge source")
            candidate = state.buffer + char
            if (
                not _identifier_prefix(candidate)
                or not state._edge_source_prefix_is_extendable(candidate)
            ):
                return replace(state, buffer=candidate)._die(
                    "invalid or undeclared canonical edge source prefix; "
                    "edges cannot remain unique and increasing"
                )
            return replace(state, buffer=candidate)

        if mode is CanonicalTripleMode.EDGE_RELATION:
            if char == " ":
                return state._commit_edge_relation()
            if char.isspace():
                return state._die("invalid whitespace in canonical edge relation")
            candidate = state.buffer + char
            if (
                not _symbol_prefix(candidate)
                or not state._edge_relation_prefix_is_extendable(candidate)
            ):
                return replace(state, buffer=candidate)._die(
                    "invalid canonical edge relation prefix; edges cannot "
                    "remain unique and increasing"
                )
            return replace(state, buffer=candidate)

        if mode is CanonicalTripleMode.EDGE_TARGET:
            if char == "\n":
                committed = state._commit_edge()
                return (
                    replace(committed, mode=CanonicalTripleMode.KEYWORD)
                    if not committed.is_dead
                    else committed
                )
            if char.isspace():
                return state._die("invalid whitespace in canonical edge target")
            candidate = state.buffer + char
            if (
                not _identifier_prefix(candidate)
                or not state._edge_target_prefix_is_extendable(candidate)
            ):
                return replace(state, buffer=candidate)._die(
                    "invalid or undeclared canonical edge target prefix; "
                    "edges cannot remain unique and increasing"
                )
            return replace(state, buffer=candidate)

        return state._die(f"unhandled canonical graph mode {mode.value}")

    def _commit_keyword(self) -> "CanonicalTripleConstraintState":
        if self.buffer == "NODE" and not self.edge_phase:
            return replace(self, mode=CanonicalTripleMode.NODE_ID, buffer="")
        if self.buffer == "EDGE" and self.nodes:
            return replace(
                self,
                mode=CanonicalTripleMode.EDGE_SOURCE,
                buffer="",
                edge_phase=True,
            )
        if self.buffer == "EDGE":
            return self._die("an edge cannot precede all node declarations")
        return self._die("incomplete canonical graph record keyword")

    def _commit_node_id(self) -> "CanonicalTripleConstraintState":
        identifier = self.buffer
        if not identifier:
            return self._die("canonical node identifier is empty")
        if self.nodes and identifier <= self.nodes[-1][0]:
            return self._die(
                "canonical node identifiers must be unique and increasing"
            )
        return replace(
            self,
            mode=CanonicalTripleMode.NODE_LABEL,
            buffer="",
            pending_node_id=identifier,
        )

    def _commit_node(self) -> "CanonicalTripleConstraintState":
        identifier = self.pending_node_id
        label = self.buffer
        if not identifier or not label:
            return self._die("canonical node record has an empty field")
        return replace(
            self,
            buffer="",
            nodes=self.nodes + ((identifier, label),),
            pending_node_id=None,
        )

    def _commit_edge_source(self) -> "CanonicalTripleConstraintState":
        source = self.buffer
        if not source:
            return self._die("canonical edge source is empty")
        if source not in self.node_ids:
            return self._die(f"undeclared canonical edge source {source!r}")
        return replace(
            self,
            mode=CanonicalTripleMode.EDGE_RELATION,
            buffer="",
            pending_source=source,
        )

    def _commit_edge_relation(self) -> "CanonicalTripleConstraintState":
        relation = self.buffer
        if not relation:
            return self._die("canonical edge relation is empty")
        return replace(
            self,
            mode=CanonicalTripleMode.EDGE_TARGET,
            buffer="",
            pending_relation=relation,
        )

    def _commit_edge(self) -> "CanonicalTripleConstraintState":
        target = self.buffer
        if not target:
            return self._die("canonical edge target is empty")
        if target not in self.node_ids:
            return self._die(f"undeclared canonical edge target {target!r}")
        assert self.pending_source is not None
        assert self.pending_relation is not None
        edge = (self.pending_source, self.pending_relation, target)
        if self.edges and edge <= self.edges[-1]:
            return self._die("canonical edges must be unique and increasing")
        return replace(
            self,
            buffer="",
            edges=self.edges + (edge,),
            pending_source=None,
            pending_relation=None,
        )

    def _is_weakly_connected(self) -> bool:
        identifiers = self.node_ids
        if not identifiers:
            return False
        reached = {self.nodes[0][0]}
        changed = True
        while changed:
            changed = False
            for source, _, target in self.edges:
                if source in reached and target not in reached:
                    reached.add(target)
                    changed = True
                elif target in reached and source not in reached:
                    reached.add(source)
                    changed = True
        return reached == identifiers

    def _die(self, reason: str) -> "CanonicalTripleConstraintState":
        if self.dead_reason is not None:
            return self
        return replace(self, dead_reason=reason, finished=False)


def validate_canonical_triples(
    text: str,
    policy: CanonicalTriplePolicy | None = None,
) -> CanonicalTripleValidationResult:
    state = CanonicalTripleConstraintState.initial(policy).advance_text(text).finish()
    return CanonicalTripleValidationResult(
        valid=state.is_accepting,
        state=state,
        reason=state.dead_reason,
    )


def validate_canonical_triples_batch(
    text: str,
    policy: CanonicalTriplePolicy | None = None,
) -> CanonicalTripleBatchValidationResult:
    """Validate one complete graph without using the incremental state.

    This intentionally duplicates the final language contract through a
    record-oriented parser. It is used as an independent release oracle and as
    a bounded cross-check for the character-incremental implementation.
    """

    active_policy = policy or CanonicalTriplePolicy()

    def fail(reason: str) -> CanonicalTripleBatchValidationResult:
        return CanonicalTripleBatchValidationResult(False, reason=reason)

    if not text:
        return fail("canonical graph is empty")
    if text.endswith("\n"):
        return fail("canonical graph has a trailing record separator")
    if any(char in "\r\t\f\v" for char in text):
        return fail("canonical graph contains non-canonical whitespace")

    nodes: list[tuple[str, str]] = []
    edges: list[tuple[str, str, str]] = []
    edge_phase = False
    for line in text.split("\n"):
        fields = line.split(" ")
        if any(not field for field in fields):
            return fail("canonical graph spacing is not exact")
        keyword = fields[0]
        if keyword == "NODE":
            if edge_phase:
                return fail("canonical node appears after an edge")
            if len(fields) != 3:
                return fail("canonical node record must have three fields")
            identifier, label = fields[1:]
            if not identifier or not _identifier_prefix(identifier):
                return fail("invalid canonical node identifier")
            if not label or not _symbol_prefix(label):
                return fail("invalid canonical node label")
            if nodes and identifier <= nodes[-1][0]:
                return fail(
                    "canonical node identifiers must be unique and increasing"
                )
            nodes.append((identifier, label))
            continue

        if keyword == "EDGE":
            edge_phase = True
            if len(fields) != 4:
                return fail("canonical edge record must have four fields")
            if not nodes:
                return fail("an edge cannot precede all node declarations")
            source, relation, target = fields[1:]
            node_ids = {identifier for identifier, _ in nodes}
            if source not in node_ids:
                return fail(f"undeclared canonical edge source {source!r}")
            if not relation or not _symbol_prefix(relation):
                return fail("invalid canonical edge relation")
            if target not in node_ids:
                return fail(f"undeclared canonical edge target {target!r}")
            edge = (source, relation, target)
            if edges and edge <= edges[-1]:
                return fail("canonical edges must be unique and increasing")
            edges.append(edge)
            continue

        return fail("invalid canonical graph record keyword")

    if not nodes:
        return fail("canonical graph contains no nodes")
    if active_policy.require_connected:
        identifiers = {identifier for identifier, _ in nodes}
        reached = {nodes[0][0]}
        frontier = list(reached)
        while frontier:
            current = frontier.pop()
            for source, _, target in edges:
                neighbor = (
                    target
                    if source == current
                    else source if target == current else None
                )
                if neighbor is not None and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if reached != identifiers:
            return fail("canonical graph is not weakly connected")

    return CanonicalTripleBatchValidationResult(
        True,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


__all__ = [
    "CanonicalTripleBatchValidationResult",
    "CanonicalTripleConstraintState",
    "CanonicalTripleMode",
    "CanonicalTriplePolicy",
    "CanonicalTripleValidationResult",
    "validate_canonical_triples",
    "validate_canonical_triples_batch",
]
