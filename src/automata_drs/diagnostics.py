"""Structured diagnostics for the first rejected SBN prefix event."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .sbn import SBNConstraintState, SBNPolicy


class ConstraintFamily(str, Enum):
    CANONICAL_SURFACE = "canonical-surface"
    LEXICAL_OR_SEQUENCE = "lexical-or-sequence"
    TARGET_SIGNATURE = "target-signature"
    REFERENCE_RANGE = "reference-range"
    REFERENCE_EXISTENCE = "reference-existence"
    GRAPH_ACYCLICITY = "graph-acyclicity"
    SEMANTIC_WELLFORMEDNESS = "semantic-wellformedness"
    CONTEXT_REFERENCE = "context-reference"
    END_STRUCTURE = "end-structure"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class PrefixDiagnostic:
    valid: bool
    family: ConstraintFamily | None
    event: str | None
    char_offset: int
    byte_offset: int
    offending_text: str | None
    reason: str | None
    state: SBNConstraintState


def _classify_reason(reason: str) -> ConstraintFamily:
    if "whitespace" in reason or "ASCII space" in reason:
        return ConstraintFamily.CANONICAL_SURFACE
    if "would make the SBN graph cyclic" in reason:
        return ConstraintFamily.GRAPH_ACYCLICITY
    if (
        "DRS well-formedness" in reason
        or "DRS subordination" in reason
    ):
        return ConstraintFamily.SEMANTIC_WELLFORMEDNESS
    if "unresolved forward concept reference" in reason:
        return ConstraintFamily.REFERENCE_EXISTENCE
    if "points before the first concept" in reason:
        return ConstraintFamily.REFERENCE_RANGE
    if "Proposition" in reason or reason.startswith("separator "):
        return ConstraintFamily.CONTEXT_REFERENCE
    if "invalid target" in reason or "attachment target has no" in reason:
        return ConstraintFamily.TARGET_SIGNATURE
    if (
        "SBN ends while expecting" in reason
        or "SBN contains no concept" in reason
        or "final context contains no concept" in reason
        or "SBN is empty or ends" in reason
    ):
        return ConstraintFamily.END_STRUCTURE
    if (
        "cannot complete the expected" in reason
        or "invalid SBN item" in reason
        or "invalid pre-delimited lexeme" in reason
    ):
        return ConstraintFamily.LEXICAL_OR_SEQUENCE
    return ConstraintFamily.INTERNAL


def diagnose_sbn(
    text: str, policy: SBNPolicy | None = None
) -> PrefixDiagnostic:
    """Locate and classify the first character or EOS rejected by the engine."""
    state = SBNConstraintState.initial(policy).advance_text(text)
    if state.is_dead:
        # `char_offset` is the count after the character which made this
        # prefix dead. Conversion through the original text preserves the
        # corresponding UTF-8 byte offset for a ByT5 experiment report.
        char_offset = state.char_offset
        reason = state.dead_reason or "unknown constraint failure"
        return PrefixDiagnostic(
            valid=False,
            family=_classify_reason(reason),
            event="character",
            char_offset=char_offset,
            byte_offset=len(text[:char_offset].encode("utf-8")),
            offending_text=text[char_offset - 1 : char_offset] or None,
            reason=reason,
            state=state,
        )

    finished = state.finish()
    if finished.is_accepting:
        return PrefixDiagnostic(
            valid=True,
            family=None,
            event=None,
            char_offset=len(text),
            byte_offset=len(text.encode("utf-8")),
            offending_text=None,
            reason=None,
            state=finished,
        )
    reason = finished.dead_reason or "unknown EOS constraint failure"
    return PrefixDiagnostic(
        valid=False,
        family=_classify_reason(reason),
        event="eos",
        char_offset=len(text),
        byte_offset=len(text.encode("utf-8")),
        offending_text="<eos>",
        reason=reason,
        state=finished,
    )
