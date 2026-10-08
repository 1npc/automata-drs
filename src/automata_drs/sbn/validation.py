"""Convenience validation API built on the incremental SBN state."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .state import SBNConstraintState, SBNPolicy


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    state: SBNConstraintState

    @property
    def reason(self) -> str | None:
        return self.state.dead_reason


def validate_sbn(text: str, policy: SBNPolicy | None = None) -> ValidationResult:
    state = SBNConstraintState.initial(policy).advance_text(text).finish()
    return ValidationResult(valid=state.is_accepting, state=state)


_LEXEME_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')


def validate_sbn_lexemes(
    text: str, policy: SBNPolicy | None = None
) -> ValidationResult:
    """Validate a corpus string through the pre-delimited fast path.

    This deliberately does not check canonical inter-token whitespace. It is
    intended for auditing existing corpora, not for model-token transitions.
    """
    state = SBNConstraintState.initial(policy)
    for lexeme in _LEXEME_RE.findall(text):
        state = state.advance_lexeme(lexeme)
        if state.is_dead:
            break
    state = state.finish_committed()
    return ValidationResult(valid=state.is_accepting, state=state)
