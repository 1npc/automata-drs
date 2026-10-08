"""Adapters for external semantic graph scorers."""

from __future__ import annotations

import re
from typing import Any, Iterable


_ITEM_SCORE_RE = re.compile(r"^(\d+):([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")


def parse_sbn_smatch_output(output: str, item_count: int) -> list[float]:
    """Parse PMB5 ``sbn_smatch.py`` per-item output.

    The reference script skips malformed pairs while dividing its aggregate by
    the full input size. Missing per-item lines therefore receive score 0.0,
    preserving its official aggregate convention while keeping alignment.
    """
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    scores = [0.0] * item_count
    seen: set[int] = set()
    for raw_line in output.splitlines():
        match = _ITEM_SCORE_RE.fullmatch(raw_line.strip())
        if match is None:
            continue
        index = int(match.group(1))
        if not 0 <= index < item_count:
            raise ValueError(f"SBN Smatch returned out-of-range item index {index}")
        if index in seen:
            raise ValueError(f"SBN Smatch returned duplicate item index {index}")
        seen.add(index)
        scores[index] = float(match.group(2))
    return scores


def _identity(row: dict[str, Any]) -> tuple[str, str, str]:
    decode = row["decode"]
    strategy = (
        "greedy"
        if decode["strategy"] == "greedy"
        else f"beam{decode['num_beams']}"
    )
    return str(row["id"]), str(row["configuration"]), strategy


def merge_semantic_scores(
    records: Iterable[dict[str, Any]],
    score_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for score in score_records:
        identity = _identity(score)
        if identity in score_index:
            raise ValueError(f"duplicate semantic score record: {identity}")
        score_index[identity] = score

    merged: list[dict[str, Any]] = []
    for original in records:
        row = dict(original)
        score = score_index.get(_identity(row))
        if score is not None:
            row["semantic_scores"] = dict(score.get("semantic_scores", {}))
        merged.append(row)
    return merged


def validate_semantic_pair_consistency(
    records: Iterable[dict[str, Any]],
) -> dict[str, int | bool]:
    """Require identical reference/prediction pairs to share exact scores.

    Approximate graph matchers may use randomized search.  Configuration-wise
    scorer invocations can therefore assign different values to byte-identical
    graph pairs and manufacture a paired method effect.  Reviewer-facing
    analyses call this after score merging and fail closed on that condition.
    """

    pair_scores: dict[tuple[str, str], dict[str, Any]] = {}
    records_count = 0
    for row in records:
        records_count += 1
        if "reference" not in row or "prediction" not in row:
            raise ValueError(
                "semantic pair consistency requires reference and prediction"
            )
        scores = row.get("semantic_scores")
        if not isinstance(scores, dict) or not scores:
            raise ValueError(
                "semantic pair consistency requires nonempty semantic_scores"
            )
        pair = (str(row["reference"]), str(row["prediction"]))
        existing = pair_scores.get(pair)
        if existing is None:
            pair_scores[pair] = dict(scores)
        elif existing != scores:
            raise ValueError(
                "identical reference/prediction pair received inconsistent "
                "semantic scores"
            )
    return {
        "identical_pair_score_consistency": True,
        "records": records_count,
        "unique_semantic_pairs": len(pair_scores),
        "reused_score_records": records_count - len(pair_scores),
    }


__all__ = [
    "merge_semantic_scores",
    "parse_sbn_smatch_output",
    "validate_semantic_pair_consistency",
]
