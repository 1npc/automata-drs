"""Fail-closed derivation of a post-hoc well-formed beam baseline."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Sequence


U_BEAM_CANDIDATE_CONFIGURATION = "U4"
POSTW_CONFIGURATION = "PostW"


def _finite_score(value: Any, *, identifier: str, rank: int) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"non-numeric beam score for {identifier} rank {rank}: {value!r}"
        ) from error
    if not math.isfinite(score):
        raise ValueError(f"non-finite beam score for {identifier} rank {rank}")
    return score


def _nonnegative_time(value: Any, *, identifier: str, rank: int) -> float:
    try:
        timing = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"non-numeric validation time for {identifier} rank {rank}"
        ) from error
    if not math.isfinite(timing) or timing < 0:
        raise ValueError(
            f"invalid validation time for {identifier} rank {rank}"
        )
    return timing


def _is_w_valid(row: dict[str, Any]) -> bool:
    validity = row.get("validity")
    return (
        isinstance(validity, dict)
        and validity.get("raw_sbn") is True
        and validity.get(
            "raw_clf_referee", validity.get("clf_referee")
        )
        is True
    )


def derive_postw_rows(
    candidate_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> list[dict[str, Any]]:
    """Return U-top1/PostW cells after validating an exact beam candidate grid.

    Candidate rows must be in corpus order and rank order. ``PostW`` chooses
    the first W-valid row because ranks are required to be non-increasing in
    the model's beam sequence score. If no candidate is W-valid, it copies the
    top-1 candidate and records an explicit fallback.
    """

    identifiers = [str(identifier) for identifier in expected_identifiers]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("expected identifiers must be non-empty and unique")
    if num_beams < 2:
        raise ValueError("post-hoc beam selection requires at least two beams")
    expected_rows = len(identifiers) * num_beams
    if len(candidate_rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} beam candidate rows, "
            f"found {len(candidate_rows)}"
        )

    derived: list[dict[str, Any]] = []
    for identifier_index, identifier in enumerate(identifiers):
        start = identifier_index * num_beams
        group = candidate_rows[start : start + num_beams]
        common_input: Any = None
        common_reference: Any = None
        scores: list[float] = []
        validation_times: list[float] = []
        for rank, row in enumerate(group):
            if not isinstance(row, dict):
                raise ValueError(
                    f"beam candidate {identifier} rank {rank} is not an object"
                )
            if str(row.get("id", "")) != identifier:
                raise ValueError(
                    f"beam candidate schedule mismatch at {identifier} rank {rank}"
                )
            if row.get("configuration") != U_BEAM_CANDIDATE_CONFIGURATION:
                raise ValueError(
                    f"beam candidate {identifier} rank {rank} is not U4"
                )
            decode = row.get("decode")
            if not isinstance(decode, dict) or (
                decode.get("strategy") != "beam"
                or decode.get("num_beams") != num_beams
                or decode.get("num_return_sequences") != num_beams
            ):
                raise ValueError(
                    f"beam candidate {identifier} rank {rank} has bad decode metadata"
                )
            beam = row.get("beam_candidate")
            if not isinstance(beam, dict) or beam.get("rank") != rank:
                raise ValueError(
                    f"beam candidate {identifier} has bad rank metadata at {rank}"
                )
            score = _finite_score(
                beam.get("sequence_score"), identifier=identifier, rank=rank
            )
            if scores and score > scores[-1] + 1e-12:
                raise ValueError(
                    f"beam scores are not non-increasing for {identifier}"
                )
            scores.append(score)
            validation_times.append(
                _nonnegative_time(
                    beam.get("validation_ms"),
                    identifier=identifier,
                    rank=rank,
                )
            )
            if rank == 0:
                common_input = row.get("input")
                common_reference = row.get("reference")
            elif (
                row.get("input") != common_input
                or row.get("reference") != common_reference
            ):
                raise ValueError(
                    f"beam candidates disagree on source/reference for {identifier}"
                )

        valid_ranks = [rank for rank, row in enumerate(group) if _is_w_valid(row)]
        selected_rank = valid_ranks[0] if valid_ranks else 0
        fallback = not valid_ranks

        top = deepcopy(group[0])
        top["configuration"] = "U"
        top["beam_selection"] = {
            "source_configuration": U_BEAM_CANDIDATE_CONFIGURATION,
            "selected_candidate_rank": 0,
            "candidate_count": num_beams,
        }
        derived.append(top)

        postw = deepcopy(group[selected_rank])
        postw["configuration"] = POSTW_CONFIGURATION
        postw["beam_selection"] = {
            "method": "highest-sequence-score-w-valid-u-beam",
            "source_configuration": U_BEAM_CANDIDATE_CONFIGURATION,
            "selected_candidate_rank": selected_rank,
            "candidate_count": num_beams,
            "w_valid_candidate_count": len(valid_ranks),
            "w_valid_candidate_ranks": valid_ranks,
            "fallback_to_u_top1": fallback,
            "candidate_sequence_scores": scores,
        }
        timing = postw.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"beam candidate {identifier} has no timing object")
        timing["constraint_ms"] = sum(validation_times)
        derived.append(postw)
    return derived


def validate_beam_candidate_resume_prefix(
    candidate_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> int:
    """Validate whole completed beam groups and return the record count."""

    if len(candidate_rows) % num_beams:
        raise ValueError(
            "beam candidate resume output ends inside a candidate group"
        )
    completed = len(candidate_rows) // num_beams
    if completed > len(expected_identifiers):
        raise ValueError(
            f"beam candidate resume has {completed} records but only "
            f"{len(expected_identifiers)} are expected"
        )
    if completed:
        derive_postw_rows(
            candidate_rows,
            list(expected_identifiers)[:completed],
            num_beams=num_beams,
        )
    return completed


def combine_beam_selection_rows(
    candidate_rows: Sequence[dict[str, Any]],
    online_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> list[dict[str, Any]]:
    """Join derived U/PostW with exact online W/WF cells in fixed order."""

    identifiers = [str(identifier) for identifier in expected_identifiers]
    derived = derive_postw_rows(
        candidate_rows, identifiers, num_beams=num_beams
    )
    if len(online_rows) != 2 * len(identifiers):
        raise ValueError(
            f"expected {2 * len(identifiers)} online beam rows, "
            f"found {len(online_rows)}"
        )
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(online_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"online beam row {row_number} is not an object")
        identifier = str(row.get("id", ""))
        configuration = str(row.get("configuration", ""))
        identity = (identifier, configuration)
        if identifier not in identifiers or configuration not in {"W", "WF"}:
            raise ValueError(f"unexpected online beam cell: {identity}")
        if identity in by_identity:
            raise ValueError(f"duplicate online beam cell: {identity}")
        decode = row.get("decode")
        if not isinstance(decode, dict) or (
            decode.get("strategy") != "beam"
            or decode.get("num_beams") != num_beams
        ):
            raise ValueError(f"online beam cell has bad decode metadata: {identity}")
        by_identity[identity] = row

    combined: list[dict[str, Any]] = []
    for index, identifier in enumerate(identifiers):
        u_row = derived[2 * index]
        postw_row = derived[2 * index + 1]
        for configuration in ("W", "WF"):
            if (identifier, configuration) not in by_identity:
                raise ValueError(
                    f"missing online beam cell: {(identifier, configuration)}"
                )
        for row in (
            u_row,
            postw_row,
            deepcopy(by_identity[(identifier, "W")]),
            deepcopy(by_identity[(identifier, "WF")]),
        ):
            if (
                row.get("input") != u_row.get("input")
                or row.get("reference") != u_row.get("reference")
            ):
                raise ValueError(
                    f"beam study source/reference mismatch for {identifier}"
                )
            combined.append(row)
    return combined


__all__ = [
    "POSTW_CONFIGURATION",
    "U_BEAM_CANDIDATE_CONFIGURATION",
    "derive_postw_rows",
    "combine_beam_selection_rows",
    "validate_beam_candidate_resume_prefix",
]
