"""Constructive accepting suffixes for the conservative PENMAN language.

This module supplies positive witnesses, not a complete reachability decision.
Every returned suffix is replayed through the immutable PENMAN state and EOS
before it is exposed. A ``None`` result may be a genuinely impossible state or
an intentionally incomplete construction failure.
"""

from __future__ import annotations

from .state import PENMANConstraintState, PENMANMode


def _fresh_variable(state: PENMANConstraintState) -> str:
    candidate = state.buffer or "x"
    while candidate in state.declared_variables:
        candidate += "0"
    return candidate


def construct_penman_completion_suffix(
    state: PENMANConstraintState,
) -> str | None:
    """Return one locally replayed suffix that makes EOS accepting.

    The suffix contains surface characters only; EOS is supplied by the token
    controller. It preserves already generated variable/concept/role prefixes,
    introduces only a generic ``thing`` concept or target where a required
    field is empty, and closes every open node. A duplicate variable that has
    already crossed into ``NODE_SLASH`` cannot be repaired by appending text
    and therefore fails explicitly.
    """

    if state.is_dead:
        return None
    if state.is_accepting:
        return ""

    mode = state.mode
    closers = ")" * len(state.stack)
    if mode is PENMANMode.START:
        suffix = "(x / thing)"
    elif mode is PENMANMode.COMPLETE:
        suffix = ""
    elif mode is PENMANMode.NODE_VARIABLE:
        variable = _fresh_variable(state)
        suffix = variable[len(state.buffer) :] + " / thing" + closers + ")"
    elif mode is PENMANMode.NODE_SLASH:
        if (
            state.policy.reject_duplicate_variables
            and state.buffer in state.declared_variables
        ):
            return None
        suffix = "/ thing" + closers + ")"
    elif mode is PENMANMode.NODE_CONCEPT:
        suffix = ("thing" if not state.buffer else "") + closers
    elif mode is PENMANMode.NODE_CONCEPT_STRING:
        suffix = ("x\"" if state.string_escape else "\"") + closers
    elif mode is PENMANMode.NODE_BODY:
        suffix = closers
    elif mode is PENMANMode.ROLE:
        suffix = ("mod" if not state.buffer else "") + " thing" + closers
    elif mode is PENMANMode.TARGET:
        suffix = "thing" + closers
    elif mode is PENMANMode.TARGET_SYMBOL:
        suffix = closers
    elif mode is PENMANMode.TARGET_STRING:
        suffix = ("x\"" if state.string_escape else "\"") + closers
    else:  # pragma: no cover - defensive against future enum extensions.
        return None

    replayed = state.advance_text(suffix).finish()
    return suffix if replayed.is_accepting else None


__all__ = ["construct_penman_completion_suffix"]
