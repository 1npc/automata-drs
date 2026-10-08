"""Build a non-duplicative U/B/W/BF/WF layer-ablation matrix."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


PILOT_CONFIGURATIONS = ("U", "BF", "WF")
SUPPLEMENT_CONFIGURATIONS = ("B", "W")
FULL_LADDER = ("U", "B", "W", "BF", "WF")
FINE_SUPPLEMENT_CONFIGURATIONS = ("S", "R", "G")
FINE_FULL_LADDER = ("U", "S", "R", "G", "B", "W", "BF", "WF")


def _index_source(
    rows: Sequence[dict[str, Any]],
    *,
    configurations: tuple[str, ...],
    expected_records: int,
    source_name: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    expected_total = expected_records * len(configurations)
    if len(rows) != expected_total:
        raise ValueError(
            f"{source_name} must contain {expected_total} rows, found {len(rows)}"
        )
    index: dict[tuple[str, str], dict[str, Any]] = {}
    identifier_order: list[str] = []
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{source_name} row {row_number} is not an object")
        identifier = str(row.get("id", ""))
        configuration = str(row.get("configuration", ""))
        if not identifier:
            raise ValueError(f"{source_name} row {row_number} has no id")
        if configuration not in configurations:
            raise ValueError(
                f"unexpected {source_name} configuration at row {row_number}: "
                f"{configuration!r}"
            )
        identity = (identifier, configuration)
        if identity in index:
            raise ValueError(f"duplicate {source_name} cell: {identity}")
        index[identity] = row
        counts[configuration] += 1
        if identifier not in seen_ids:
            seen_ids.add(identifier)
            identifier_order.append(identifier)
    expected_counts = Counter(
        {configuration: expected_records for configuration in configurations}
    )
    if counts != expected_counts or len(identifier_order) != expected_records:
        raise ValueError(
            f"{source_name} is not a complete paired matrix: "
            f"counts={dict(counts)}, ids={len(identifier_order)}"
        )
    expected_cells = {
        (identifier, configuration)
        for identifier in identifier_order
        for configuration in configurations
    }
    if set(index) != expected_cells:
        raise ValueError(f"{source_name} configurations do not share one ID set")
    return index, identifier_order


def combine_layer_ablation_rows(
    pilot_rows: Sequence[dict[str, Any]],
    supplement_rows: Sequence[dict[str, Any]],
    *,
    expected_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Combine pilot and B/W cells without changing any source row content."""

    if expected_records <= 0:
        raise ValueError("expected_records must be positive")
    pilot, identifier_order = _index_source(
        pilot_rows,
        configurations=PILOT_CONFIGURATIONS,
        expected_records=expected_records,
        source_name="pilot",
    )
    supplement, supplement_order = _index_source(
        supplement_rows,
        configurations=SUPPLEMENT_CONFIGURATIONS,
        expected_records=expected_records,
        source_name="B/W supplement",
    )
    if set(identifier_order) != set(supplement_order):
        missing = sorted(set(identifier_order) - set(supplement_order))
        extra = sorted(set(supplement_order) - set(identifier_order))
        raise ValueError(
            f"pilot and B/W populations differ: missing={missing}, extra={extra}"
        )

    combined: list[dict[str, Any]] = []
    cells = {**pilot, **supplement}
    for identifier in identifier_order:
        rows = [cells[(identifier, configuration)] for configuration in FULL_LADDER]
        inputs = {str(row.get("input", "")) for row in rows}
        references = {str(row.get("reference", "")) for row in rows}
        if len(inputs) != 1 or len(references) != 1:
            raise ValueError(
                f"source/reference mismatch across layer cells: {identifier}"
            )
        combined.extend(rows)
    return combined, {
        "records": expected_records,
        "rows": len(combined),
        "identifier_order": identifier_order,
        "pilot_configurations": list(PILOT_CONFIGURATIONS),
        "supplement_configurations": list(SUPPLEMENT_CONFIGURATIONS),
        "derived_full_ladder": list(FULL_LADDER),
        "derivation": "cell-preserving-join-by-id",
    }


def combine_fine_grained_ablation_rows(
    layer_rows: Sequence[dict[str, Any]],
    supplement_rows: Sequence[dict[str, Any]],
    *,
    expected_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join S/R/G with an existing U/B/W/BF/WF artifact without mutation."""

    if expected_records <= 0:
        raise ValueError("expected_records must be positive")
    layer, identifier_order = _index_source(
        layer_rows,
        configurations=FULL_LADDER,
        expected_records=expected_records,
        source_name="U/B/W/BF/WF layer artifact",
    )
    supplement, supplement_order = _index_source(
        supplement_rows,
        configurations=FINE_SUPPLEMENT_CONFIGURATIONS,
        expected_records=expected_records,
        source_name="S/R/G supplement",
    )
    if set(identifier_order) != set(supplement_order):
        missing = sorted(set(identifier_order) - set(supplement_order))
        extra = sorted(set(supplement_order) - set(identifier_order))
        raise ValueError(
            "layer artifact and S/R/G populations differ: "
            f"missing={missing}, extra={extra}"
        )

    combined: list[dict[str, Any]] = []
    cells = {**layer, **supplement}
    for identifier in identifier_order:
        rows = [
            cells[(identifier, configuration)]
            for configuration in FINE_FULL_LADDER
        ]
        inputs = {str(row.get("input", "")) for row in rows}
        references = {str(row.get("reference", "")) for row in rows}
        if len(inputs) != 1 or len(references) != 1:
            raise ValueError(
                f"source/reference mismatch across fine-grained cells: {identifier}"
            )
        combined.extend(rows)
    return combined, {
        "records": expected_records,
        "rows": len(combined),
        "identifier_order": identifier_order,
        "source_ladder": list(FULL_LADDER),
        "supplement_configurations": list(FINE_SUPPLEMENT_CONFIGURATIONS),
        "derived_full_ladder": list(FINE_FULL_LADDER),
        "derivation": "cell-preserving-fine-grained-join-by-id",
    }


__all__ = [
    "FINE_FULL_LADDER",
    "FINE_SUPPLEMENT_CONFIGURATIONS",
    "FULL_LADDER",
    "PILOT_CONFIGURATIONS",
    "SUPPLEMENT_CONFIGURATIONS",
    "combine_layer_ablation_rows",
    "combine_fine_grained_ablation_rows",
]
