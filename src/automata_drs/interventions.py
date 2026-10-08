"""Freeze failure-conditioned intervention populations with source hashes."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .diagnostics import diagnose_sbn
from .sbn import SBNPolicy


def build_intervention_plan(
    records: Iterable[dict[str, Any]],
    *,
    input_path: str,
    input_sha256: str,
) -> dict[str, Any]:
    """Select every U row rejected by B or W and record why.

    Stored generation-time validity is independently replayed through the B
    and W policies. A mismatch fails rather than freezing a population from
    stale or incompatible validity fields.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")
    rows = [row for row in records if row.get("configuration") == "U"]
    if not rows:
        raise ValueError("intervention source contains no U rows")
    if any(not isinstance(row.get("id"), str) or not row["id"] for row in rows):
        raise ValueError("every U intervention row must have a non-empty string id")
    identities = [row["id"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate U identifiers in intervention source")

    selected = []
    structural_invalid = 0
    referee_only_invalid = 0
    for row in rows:
        raw_prediction = str(row.get("raw_prediction", row.get("prediction", "")))
        b_diagnostic = diagnose_sbn(
            raw_prediction, SBNPolicy.for_configuration("B")
        )
        w_diagnostic = diagnose_sbn(
            raw_prediction, SBNPolicy.for_configuration("W")
        )
        validity = row.get("validity", {})
        stored_b = validity.get("raw_sbn", validity.get("sbn")) is True
        # Intervention membership is defined on the usable raw model surface.
        # Older artifacts predate the explicit raw field, so retain the same
        # backwards-compatible fallback as the primary paired metric.
        stored_raw_referee = validity.get(
            "raw_clf_referee", validity.get("clf_referee")
        )
        stored_w = stored_b and stored_raw_referee is True
        if b_diagnostic.valid != stored_b:
            raise ValueError(
                f"stored/raw B validity mismatch for {row.get('id')}: "
                f"stored={stored_b}, replay={b_diagnostic.valid}"
            )
        if w_diagnostic.valid != stored_w:
            raise ValueError(
                f"stored/raw W validity mismatch for {row.get('id')}: "
                f"stored={stored_w}, replay={w_diagnostic.valid}"
            )
        if stored_w:
            continue
        if stored_b:
            referee_only_invalid += 1
        else:
            structural_invalid += 1
        identifier = str(row["id"])
        try:
            line_number = int(identifier.rsplit(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"record identifier has no physical line suffix: {identifier}"
            ) from error
        diagnostic = w_diagnostic if stored_b else b_diagnostic
        selected.append(
            {
                "id": identifier,
                "line_number": line_number,
                "input": row.get("input"),
                "reference": row.get("reference"),
                "raw_prediction": raw_prediction,
                "b_valid": b_diagnostic.valid,
                "w_valid": w_diagnostic.valid,
                "stored_validity": {
                    "raw_sbn": validity.get("raw_sbn"),
                    "raw_clf_referee": validity.get("raw_clf_referee"),
                    "sbn": validity.get("sbn"),
                    "clf_referee": validity.get("clf_referee"),
                },
                "failure_family": (
                    None if diagnostic.family is None else diagnostic.family.value
                ),
                "failure_event": diagnostic.event,
                "failure_reason": diagnostic.reason,
                "canonical_tokenization": row.get("generation", {}).get(
                    "canonical_tokenization"
                ),
                "hit_max_new_tokens": row.get("generation", {}).get(
                    "hit_max_new_tokens"
                ),
            }
        )

    return {
        "schema_version": 1,
        "input": input_path,
        "input_sha256": input_sha256,
        "selection_criterion": (
            "configuration == U and "
            "(raw SBN invalid under B or semantic state invalid under W)"
        ),
        "u_records": len(rows),
        "selected_records": len(selected),
        "structural_invalid": structural_invalid,
        "referee_only_invalid": referee_only_invalid,
        "canonical_noncanonical_total": sum(
            row.get("generation", {}).get("canonical_tokenization") is False
            for row in rows
        ),
        "selected": selected,
    }


__all__ = ["build_intervention_plan"]
