"""Fail-closed offline PENMAN selection for the AMR beam diagnostic."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Sequence


U_BEAM_CANDIDATE_CONFIGURATION = "U4"
POSTP_CONFIGURATION = "PostP"
BEAM_RANKING_METHOD = "transformers-final-beam-order-length-penalty-1"


def _finite(value: Any, *, name: str, identifier: str, rank: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is not numeric for {identifier} rank {rank}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite for {identifier} rank {rank}")
    return result


def _nonnegative(value: Any, *, name: str, identifier: str, rank: int) -> float:
    result = _finite(value, name=name, identifier=identifier, rank=rank)
    if result < 0:
        raise ValueError(f"{name} is negative for {identifier} rank {rank}")
    return result


def _raw_penman_valid(row: dict[str, Any]) -> bool:
    validity = row.get("validity")
    return (
        isinstance(validity, dict)
        and validity.get("raw_local_penman") is True
        and validity.get("raw_official_penman") is True
    )


def _validate_candidate_group(
    group: Sequence[dict[str, Any]],
    identifier: str,
    *,
    num_beams: int,
) -> tuple[list[int], list[float | None], list[float]]:
    common_input: Any = None
    common_reference: Any = None
    valid_ranks: list[int] = []
    mean_logprobs: list[float | None] = []
    validation_times: list[float] = []
    for rank, row in enumerate(group):
        if not isinstance(row, dict):
            raise ValueError(f"AMR beam candidate {identifier} rank {rank} is not an object")
        if row.get("task") != "amr":
            raise ValueError(f"AMR beam candidate {identifier} rank {rank} has bad task")
        if str(row.get("id", "")) != identifier:
            raise ValueError(f"AMR beam candidate schedule mismatch at {identifier} rank {rank}")
        if row.get("configuration") != U_BEAM_CANDIDATE_CONFIGURATION:
            raise ValueError(f"AMR beam candidate {identifier} rank {rank} is not U4")
        decode = row.get("decode")
        length_penalty = decode.get("length_penalty") if isinstance(decode, dict) else None
        if not isinstance(decode, dict) or (
            decode.get("strategy") != "beam"
            or decode.get("num_beams") != num_beams
            or decode.get("num_return_sequences") != num_beams
            or decode.get("output_scores") is not False
            or isinstance(length_penalty, bool)
            or not isinstance(length_penalty, (int, float))
            or float(length_penalty) != 1.0
        ):
            raise ValueError(f"AMR beam candidate {identifier} rank {rank} has bad decode metadata")
        beam = row.get("beam_candidate")
        if not isinstance(beam, dict) or (
            beam.get("rank") != rank
            or beam.get("ranking_method") != BEAM_RANKING_METHOD
            or beam.get("sequence_score") is not None
        ):
            raise ValueError(f"AMR beam candidate {identifier} has bad rank metadata at {rank}")
        validation_times.append(
            _nonnegative(
                beam.get("validation_ms"),
                name="validation_ms",
                identifier=identifier,
                rank=rank,
            )
        )
        if rank == 0:
            common_input = row.get("input")
            common_reference = row.get("reference")
        elif row.get("input") != common_input or row.get("reference") != common_reference:
            raise ValueError(f"AMR beam candidates disagree on source/reference for {identifier}")
        if _raw_penman_valid(row):
            valid_ranks.append(rank)

        score = row.get("score")
        if not isinstance(score, dict):
            raise ValueError(f"AMR beam candidate {identifier} rank {rank} has no score object")
        raw_mean = score.get("raw_model_mean_logprob")
        if raw_mean is None:
            if score.get("raw_model_logprob") is not None:
                raise ValueError(f"partial AMR model score for {identifier} rank {rank}")
            mean_logprobs.append(None)
        else:
            mean_logprobs.append(
                _finite(
                    raw_mean,
                    name="raw_model_mean_logprob",
                    identifier=identifier,
                    rank=rank,
                )
            )
            _finite(
                score.get("raw_model_logprob"),
                name="raw_model_logprob",
                identifier=identifier,
                rank=rank,
            )
            scored_tokens = score.get("scored_tokens")
            if (
                isinstance(scored_tokens, bool)
                or not isinstance(scored_tokens, int)
                or scored_tokens <= 0
            ):
                raise ValueError(f"bad scored token count for {identifier} rank {rank}")

    selected_rank = valid_ranks[0] if valid_ranks else 0
    for required_rank in {0, selected_rank}:
        if mean_logprobs[required_rank] is None:
            raise ValueError(
                f"derived AMR candidate {identifier} rank {required_rank} was not model-scored"
            )
    return valid_ranks, mean_logprobs, validation_times


def derive_postp_rows(
    candidate_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> list[dict[str, Any]]:
    """Derive paired U-top1/PostP rows from an exact U beam candidate grid.

    Transformers returns final beam sequences in descending final beam score
    order. Numeric per-step score tensors are deliberately disabled because
    retaining them scales as ``steps * vocabulary``. PostP therefore chooses
    the first raw PENMAN-valid rank; if none is valid it records and returns
    the same top-1 output as U.
    """

    identifiers = [str(value) for value in expected_identifiers]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("expected AMR identifiers must be non-empty and unique")
    if num_beams < 2:
        raise ValueError("AMR PostP requires at least two beams")
    expected_rows = len(identifiers) * num_beams
    if len(candidate_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} AMR candidate rows, found {len(candidate_rows)}")

    result: list[dict[str, Any]] = []
    for index, identifier in enumerate(identifiers):
        group = candidate_rows[index * num_beams : (index + 1) * num_beams]
        valid_ranks, mean_logprobs, validation_times = _validate_candidate_group(
            group, identifier, num_beams=num_beams
        )
        selected_rank = valid_ranks[0] if valid_ranks else 0

        top = deepcopy(group[0])
        top["configuration"] = "U"
        top["beam_selection"] = {
            "source_configuration": U_BEAM_CANDIDATE_CONFIGURATION,
            "selected_candidate_rank": 0,
            "candidate_count": num_beams,
            "ranking_method": BEAM_RANKING_METHOD,
        }
        result.append(top)

        postp = deepcopy(group[selected_rank])
        postp["configuration"] = POSTP_CONFIGURATION
        postp["beam_selection"] = {
            "method": "first-raw-penman-valid-in-final-beam-order",
            "source_configuration": U_BEAM_CANDIDATE_CONFIGURATION,
            "selected_candidate_rank": selected_rank,
            "candidate_count": num_beams,
            "raw_penman_valid_candidate_count": len(valid_ranks),
            "raw_penman_valid_candidate_ranks": valid_ranks,
            "fallback_to_u_top1": not valid_ranks,
            "ranking_method": BEAM_RANKING_METHOD,
            "candidate_raw_model_mean_logprobs": mean_logprobs,
        }
        timing = postp.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"AMR beam candidate {identifier} has no timing object")
        timing["constraint_ms"] = sum(validation_times)
        result.append(postp)
    return result


def validate_amr_beam_candidate_resume_prefix(
    candidate_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> int:
    """Validate complete AMR candidate groups and return completed examples."""

    if len(candidate_rows) % num_beams:
        raise ValueError("AMR candidate resume output ends inside a beam group")
    completed = len(candidate_rows) // num_beams
    if completed > len(expected_identifiers):
        raise ValueError("AMR candidate resume contains more examples than requested")
    if completed:
        derive_postp_rows(
            candidate_rows,
            list(expected_identifiers)[:completed],
            num_beams=num_beams,
        )
    return completed


def combine_amr_beam_rows(
    candidate_rows: Sequence[dict[str, Any]],
    online_p_rows: Sequence[dict[str, Any]],
    expected_identifiers: Sequence[str],
    *,
    num_beams: int = 4,
) -> list[dict[str, Any]]:
    """Join derived U/PostP with exactly paired online-P beam rows."""

    identifiers = [str(value) for value in expected_identifiers]
    derived = derive_postp_rows(candidate_rows, identifiers, num_beams=num_beams)
    if len(online_p_rows) != len(identifiers):
        raise ValueError(f"expected {len(identifiers)} online P rows, found {len(online_p_rows)}")
    online_by_id: dict[str, dict[str, Any]] = {}
    for row in online_p_rows:
        identifier = str(row.get("id", ""))
        decode = row.get("decode")
        if (
            identifier not in identifiers
            or identifier in online_by_id
            or row.get("task") != "amr"
            or row.get("configuration") != "P"
            or not isinstance(decode, dict)
            or decode.get("strategy") != "beam"
            or decode.get("num_beams") != num_beams
        ):
            raise ValueError(f"invalid or duplicate online AMR P row: {identifier}")
        online_by_id[identifier] = row

    combined: list[dict[str, Any]] = []
    for index, identifier in enumerate(identifiers):
        u = derived[2 * index]
        postp = derived[2 * index + 1]
        if identifier not in online_by_id:
            raise ValueError(f"missing online AMR P row: {identifier}")
        online = deepcopy(online_by_id[identifier])
        for row in (postp, online):
            if row.get("input") != u.get("input") or row.get("reference") != u.get("reference"):
                raise ValueError(f"AMR beam source/reference mismatch for {identifier}")
        combined.extend((u, postp, online))
    return combined


__all__ = [
    "BEAM_RANKING_METHOD",
    "POSTP_CONFIGURATION",
    "U_BEAM_CANDIDATE_CONFIGURATION",
    "combine_amr_beam_rows",
    "derive_postp_rows",
    "validate_amr_beam_candidate_resume_prefix",
]
