"""Cell-preserving v2 assembly for failure-conditioned interventions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


FULL_CONFIGURATIONS = ("U", "B", "W", "BF", "WF")
GENERATED_CONFIGURATIONS = ("B", "W", "BF", "WF")


def _index(
    rows: Sequence[dict[str, Any]],
    *,
    configurations: tuple[str, ...],
    selected_ids: tuple[str, ...],
    label: str,
    allow_extra_ids: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected = set(selected_ids)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {number} is not an object")
        identifier = str(row.get("id", ""))
        configuration = str(row.get("configuration", ""))
        if configuration not in configurations:
            if allow_extra_ids and identifier not in selected:
                continue
            raise ValueError(
                f"unexpected {label} configuration at row {number}: "
                f"{configuration!r}"
            )
        if identifier not in selected:
            if allow_extra_ids:
                continue
            raise ValueError(f"unexpected {label} id: {identifier!r}")
        identity = (identifier, configuration)
        if identity in index:
            raise ValueError(f"duplicate {label} cell: {identity}")
        index[identity] = row
        counts[configuration] += 1
    expected = Counter(
        {configuration: len(selected_ids) for configuration in configurations}
    )
    if counts != expected:
        raise ValueError(
            f"{label} is not a complete selected matrix: "
            f"observed={dict(counts)}, expected={dict(expected)}"
        )
    return index


def combine_intervention_correction_rows(
    baseline_rows: Sequence[dict[str, Any]],
    constrained_rows: Sequence[dict[str, Any]],
    *,
    selected_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join frozen U cells with fresh B/W/BF/WF cells without mutation."""

    identifiers = tuple(str(value) for value in selected_ids)
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("selected intervention IDs must be nonempty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("selected intervention IDs must be unique")
    baseline = _index(
        baseline_rows,
        configurations=("U",),
        selected_ids=identifiers,
        label="frozen U baseline",
        allow_extra_ids=True,
    )
    constrained = _index(
        constrained_rows,
        configurations=GENERATED_CONFIGURATIONS,
        selected_ids=identifiers,
        label="fresh constrained supplement",
        allow_extra_ids=False,
    )
    cells = {**baseline, **constrained}
    combined: list[dict[str, Any]] = []
    for identifier in identifiers:
        rows = [
            cells[(identifier, configuration)]
            for configuration in FULL_CONFIGURATIONS
        ]
        inputs = {str(row.get("input", "")) for row in rows}
        references = {str(row.get("reference", "")) for row in rows}
        if len(inputs) != 1 or len(references) != 1:
            raise ValueError(
                f"source/reference mismatch across corrected cells: {identifier}"
            )
        combined.extend(rows)
    return combined, {
        "records": len(identifiers),
        "rows": len(combined),
        "identifier_order": list(identifiers),
        "u_source": "frozen-baseline-cell",
        "generated_configurations": list(GENERATED_CONFIGURATIONS),
        "derived_configurations": list(FULL_CONFIGURATIONS),
        "derivation": "cell-preserving-conditioned-intervention-v2",
    }


__all__ = [
    "FULL_CONFIGURATIONS",
    "GENERATED_CONFIGURATIONS",
    "combine_intervention_correction_rows",
]
