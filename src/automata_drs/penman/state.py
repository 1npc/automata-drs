"""Character-incremental PENMAN constraints.

This is deliberately a conservative first transfer target rather than a claim
to implement every extension accepted by the ``penman`` package.  It covers
the ordinary AMR form::

    (v / concept :ROLE target ...)

including nested nodes, quoted constants, forward variable references,
reentrancy, and cyclic interpreted graphs.  Surface alignments and comments
are outside the current language.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


_WHITESPACE = frozenset(" \n\t\r\f\v")
_NAME_EXCLUSIONS = frozenset(' \n\t\r\f\v"()/:~')


def _is_name_char(char: str) -> bool:
    return len(char) == 1 and char not in _NAME_EXCLUSIONS


class PENMANMode(str, Enum):
    START = "start"
    NODE_VARIABLE = "node-variable"
    NODE_SLASH = "node-slash"
    NODE_CONCEPT = "node-concept"
    NODE_CONCEPT_STRING = "node-concept-string"
    NODE_BODY = "node-body"
    ROLE = "role"
    TARGET = "target"
    TARGET_SYMBOL = "target-symbol"
    TARGET_STRING = "target-string"
    COMPLETE = "complete"


_NON_ASCII_PREFIX_MODES = frozenset(
    {
        PENMANMode.NODE_VARIABLE,
        PENMANMode.NODE_CONCEPT,
        PENMANMode.NODE_CONCEPT_STRING,
        PENMANMode.ROLE,
        PENMANMode.TARGET,
        PENMANMode.TARGET_SYMBOL,
        PENMANMode.TARGET_STRING,
    }
)


@dataclass(frozen=True, slots=True)
class PENMANPolicy:
    """Stable policy choices for one decoding run.

    Common AMR serializations require every node to carry an instance label
    and every role to have a non-empty label.  The more permissive generic
    PENMAN PEG allows both to be empty; those variants can be enabled for
    corpus-compatibility audits.
    """

    require_concept: bool = True
    require_nonempty_role: bool = True
    reject_duplicate_variables: bool = True


@dataclass(frozen=True, slots=True)
class PENMANValidationResult:
    valid: bool
    state: "PENMANConstraintState"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PENMANConstraintState:
    """One immutable, hashable PENMAN prefix state.

    ``symbol_targets`` are intentionally left ambiguous until more variables
    are declared.  At completion a symbol matching a declared variable is a
    graph edge; otherwise it is a constant attribute, exactly reflecting the
    PENMAN graph interpretation rule.  This admits both backward and forward
    references without imposing SBN's acyclicity constraint on AMR.
    """

    policy: PENMANPolicy = PENMANPolicy()
    mode: PENMANMode = PENMANMode.START
    buffer: str = ""
    stack: tuple[str, ...] = ()
    declared_variables: frozenset[str] = frozenset()
    root_variable: str | None = None
    current_role: str | None = None
    pending_parent: tuple[str, str] | None = None
    nested_edges: tuple[tuple[str, str, str], ...] = ()
    symbol_targets: tuple[tuple[str, str, str], ...] = ()
    string_targets: tuple[tuple[str, str, str], ...] = ()
    concepts: tuple[tuple[str, str], ...] = ()
    string_escape: bool = False
    char_offset: int = 0
    dead_reason: str | None = None
    finished: bool = False

    @classmethod
    def initial(
        cls, policy: PENMANPolicy | None = None
    ) -> "PENMANConstraintState":
        return cls(policy=policy or PENMANPolicy())

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
    def node_edges(self) -> tuple[tuple[str, str, str], ...]:
        symbolic = tuple(
            edge
            for edge in self.symbol_targets
            if edge[2] in self.declared_variables
        )
        return self.nested_edges + symbolic

    @property
    def constant_edges(self) -> tuple[tuple[str, str, str], ...]:
        symbolic = tuple(
            edge
            for edge in self.symbol_targets
            if edge[2] not in self.declared_variables
        )
        return symbolic + self.string_targets

    def advance_text(self, fragment: str) -> "PENMANConstraintState":
        state = self
        if state.finished:
            return state._die("cannot advance an already finished PENMAN graph")
        for char in fragment:
            if state.is_dead:
                return state
            state = state._advance_char(char)
        return state

    def finish(self) -> "PENMANConstraintState":
        if self.finished or self.is_dead:
            return self
        if self.mode is not PENMANMode.COMPLETE:
            return self._die(f"PENMAN graph ends while expecting {self.mode.value}")
        if self.stack:
            return self._die("PENMAN graph ends with unclosed nodes")
        if self.root_variable is None:
            return self._die("PENMAN graph contains no root node")
        return replace(self, finished=True)

    def can_end(self) -> bool:
        return self.finish().is_accepting

    def incomplete_utf8_is_extendable(self, prefix: bytes) -> bool:
        """Whether a valid partial non-ASCII scalar has a legal completion.

        PENMAN delimiters are ASCII. In name and quoted-string positions every
        non-ASCII Unicode scalar is accepted by this conservative grammar; in
        structural positions none is. ``prefix`` has already been validated as
        a proper UTF-8 prefix by the representation-neutral byte lifting layer.
        """

        return bool(
            prefix
            and self.is_extendable
            and self.mode in _NON_ASCII_PREFIX_MODES
        )

    def _advance_char(self, char: str) -> "PENMANConstraintState":
        state = replace(self, char_offset=self.char_offset + 1)
        mode = state.mode

        if mode is PENMANMode.START:
            if char == "(":
                return replace(state, mode=PENMANMode.NODE_VARIABLE)
            return state._die("a PENMAN graph must start with '('")

        if mode is PENMANMode.COMPLETE:
            # Decoder-only AMR checkpoints commonly emit the final closing
            # parenthesis and a transport newline in one BPE token (for
            # example ``b"))\n"``), before emitting an envelope such as
            # ``</amr>``.  Surrounding record whitespace is also accepted by
            # the independent PENMAN parser.  Keeping COMPLETE live across
            # whitespace prevents exact byte-level lifting from rejecting an
            # otherwise valid graph-closing token; any non-whitespace payload
            # remains fail-closed.
            if char in _WHITESPACE:
                return state
            return state._die(
                "non-whitespace characters after the root node are not allowed"
            )

        if mode is PENMANMode.NODE_VARIABLE:
            if char in _WHITESPACE:
                if state.buffer:
                    return replace(state, mode=PENMANMode.NODE_SLASH)
                return state
            if char == "/":
                return state._commit_variable_and_slash()
            if _is_name_char(char):
                return replace(state, buffer=state.buffer + char)
            return state._die("invalid character in PENMAN variable")

        if mode is PENMANMode.NODE_SLASH:
            if char in _WHITESPACE:
                return state
            if char == "/":
                return state._commit_variable_and_slash()
            return state._die("expected '/' after PENMAN variable")

        if mode is PENMANMode.NODE_CONCEPT:
            if char in _WHITESPACE:
                if not state.buffer:
                    return state
                return state._commit_concept()
            if char == '"' and not state.buffer:
                return replace(
                    state,
                    mode=PENMANMode.NODE_CONCEPT_STRING,
                    buffer='"',
                    string_escape=False,
                )
            if char == ":":
                committed = state._commit_concept()
                return committed._begin_role() if not committed.is_dead else committed
            if char == ")":
                committed = state._commit_concept()
                return committed._close_node() if not committed.is_dead else committed
            if _is_name_char(char):
                return replace(state, buffer=state.buffer + char)
            return state._die("invalid character in PENMAN concept")

        if mode is PENMANMode.NODE_CONCEPT_STRING:
            return state._advance_string(char, concept=True)

        if mode is PENMANMode.NODE_BODY:
            if char in _WHITESPACE:
                return state
            if char == ":":
                return state._begin_role()
            if char == ")":
                return state._close_node()
            return state._die("expected a role or ')' in PENMAN node body")

        if mode is PENMANMode.ROLE:
            if char in _WHITESPACE:
                return state._commit_role()
            if char in {'(', '"'}:
                committed = state._commit_role()
                return (
                    committed._begin_target(char)
                    if not committed.is_dead
                    else committed
                )
            if _is_name_char(char):
                return replace(state, buffer=state.buffer + char)
            return state._die("invalid character in PENMAN role")

        if mode is PENMANMode.TARGET:
            if char in _WHITESPACE:
                return state
            return state._begin_target(char)

        if mode is PENMANMode.TARGET_SYMBOL:
            if char in _WHITESPACE:
                return state._commit_symbol_target()
            if char == ":":
                committed = state._commit_symbol_target()
                return committed._begin_role() if not committed.is_dead else committed
            if char == ")":
                committed = state._commit_symbol_target()
                return committed._close_node() if not committed.is_dead else committed
            if _is_name_char(char):
                return replace(state, buffer=state.buffer + char)
            return state._die("invalid character in PENMAN symbol target")

        if mode is PENMANMode.TARGET_STRING:
            return state._advance_string(char, concept=False)

        return state._die(f"unhandled PENMAN parser mode {mode.value}")

    def _commit_variable_and_slash(self) -> "PENMANConstraintState":
        variable = self.buffer
        if not variable:
            return self._die("PENMAN node variable is empty")
        if (
            self.policy.reject_duplicate_variables
            and variable in self.declared_variables
        ):
            return self._die(f"duplicate PENMAN variable declaration {variable!r}")

        nested_edges = self.nested_edges
        if self.pending_parent is not None:
            source, role = self.pending_parent
            nested_edges += ((source, role, variable),)

        return replace(
            self,
            mode=PENMANMode.NODE_CONCEPT,
            buffer="",
            stack=self.stack + (variable,),
            declared_variables=self.declared_variables | {variable},
            root_variable=self.root_variable or variable,
            pending_parent=None,
            nested_edges=nested_edges,
        )

    def _commit_concept(self) -> "PENMANConstraintState":
        if not self.stack:
            return self._die("concept has no active PENMAN node")
        if not self.buffer:
            if self.policy.require_concept:
                return self._die("PENMAN node concept is empty")
            return replace(self, mode=PENMANMode.NODE_BODY)
        return replace(
            self,
            mode=PENMANMode.NODE_BODY,
            concepts=self.concepts + ((self.stack[-1], self.buffer),),
            buffer="",
        )

    def _begin_role(self) -> "PENMANConstraintState":
        if not self.stack:
            return self._die("role has no active PENMAN node")
        return replace(self, mode=PENMANMode.ROLE, buffer="", current_role=None)

    def _commit_role(self) -> "PENMANConstraintState":
        if not self.buffer and self.policy.require_nonempty_role:
            return self._die("PENMAN role label is empty")
        return replace(
            self,
            mode=PENMANMode.TARGET,
            current_role=self.buffer,
            buffer="",
        )

    def _begin_target(self, char: str) -> "PENMANConstraintState":
        if self.current_role is None or not self.stack:
            return self._die("PENMAN target has no active source and role")
        if char == "(":
            return replace(
                self,
                mode=PENMANMode.NODE_VARIABLE,
                buffer="",
                pending_parent=(self.stack[-1], self.current_role),
                current_role=None,
            )
        if char == '"':
            return replace(
                self,
                mode=PENMANMode.TARGET_STRING,
                buffer='"',
                string_escape=False,
            )
        if _is_name_char(char):
            return replace(
                self,
                mode=PENMANMode.TARGET_SYMBOL,
                buffer=char,
            )
        return self._die("invalid start of PENMAN relation target")

    def _commit_symbol_target(self) -> "PENMANConstraintState":
        if not self.buffer or self.current_role is None or not self.stack:
            return self._die("PENMAN relation target is empty")
        return replace(
            self,
            mode=PENMANMode.NODE_BODY,
            symbol_targets=self.symbol_targets
            + ((self.stack[-1], self.current_role, self.buffer),),
            buffer="",
            current_role=None,
        )

    def _advance_string(
        self, char: str, *, concept: bool
    ) -> "PENMANConstraintState":
        if char in "\n\r\f\v":
            return self._die("unescaped line break in PENMAN string")
        if self.string_escape:
            return replace(
                self,
                buffer=self.buffer + char,
                string_escape=False,
            )
        if char == "\\":
            return replace(
                self,
                buffer=self.buffer + char,
                string_escape=True,
            )
        if char != '"':
            return replace(self, buffer=self.buffer + char)

        value = self.buffer + '"'
        if concept:
            if not self.stack:
                return self._die("concept string has no active PENMAN node")
            return replace(
                self,
                mode=PENMANMode.NODE_BODY,
                concepts=self.concepts + ((self.stack[-1], value),),
                buffer="",
                string_escape=False,
            )
        if self.current_role is None or not self.stack:
            return self._die("string target has no active source and role")
        return replace(
            self,
            mode=PENMANMode.NODE_BODY,
            string_targets=self.string_targets
            + ((self.stack[-1], self.current_role, value),),
            current_role=None,
            buffer="",
            string_escape=False,
        )

    def _close_node(self) -> "PENMANConstraintState":
        if not self.stack:
            return self._die("unmatched ')' in PENMAN graph")
        stack = self.stack[:-1]
        return replace(
            self,
            mode=PENMANMode.NODE_BODY if stack else PENMANMode.COMPLETE,
            stack=stack,
            current_role=None,
            buffer="",
        )

    def _die(self, reason: str) -> "PENMANConstraintState":
        if self.dead_reason is not None:
            return self
        return replace(self, dead_reason=reason, finished=False)


def validate_penman(
    text: str, policy: PENMANPolicy | None = None
) -> PENMANValidationResult:
    state = PENMANConstraintState.initial(policy).advance_text(text).finish()
    return PENMANValidationResult(
        valid=state.is_accepting,
        state=state,
        reason=state.dead_reason,
    )
