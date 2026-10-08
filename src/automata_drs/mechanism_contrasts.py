"""Descriptive paired contrasts between cumulative decoding mechanisms.

The confirmatory experiment report compares every constrained condition with
U.  That is the correct population baseline, but it does not directly answer
the mechanism questions in the experiment plan: S -> R, B -> BF, W -> WF,
and the other cumulative-layer increments.  This module derives those paired
contrasts from an already frozen, fully scored matrix.  It never selects a
population, changes a generation row, or applies a performance threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from typing import Any

from .paired_analysis import analyze_paired_experiment


DEFAULT_DRS_CONTRASTS = (
    ("U", "S"),
    ("S", "R"),
    ("R", "G"),
    ("G", "B"),
    ("B", "W"),
    ("B", "BF"),
    ("W", "WF"),
)


def _holm_adjust(comparisons: list[dict[str, Any]]) -> None:
    """Adjust binary p-values across the complete registered contrast family."""

    if not comparisons:
        return
    metric_names = tuple(comparisons[0]["metrics"])
    for name in metric_names:
        metrics = [comparison["metrics"][name] for comparison in comparisons]
        if not metrics or metrics[0].get("kind") != "binary":
            continue
        family = sorted(
            (
                (float(metric["mcnemar_exact_p"]), index)
                for index, metric in enumerate(metrics)
            ),
            key=lambda item: (item[0], item[1]),
        )
        running = 0.0
        total = len(family)
        for rank, (p_value, comparison_index) in enumerate(family):
            adjusted = min(1.0, (total - rank) * p_value)
            running = max(running, adjusted)
            comparisons[comparison_index]["metrics"][name][
                "mcnemar_exact_p_holm"
            ] = running


def analyze_mechanism_contrasts(
    records: Iterable[dict[str, Any]],
    *,
    contrasts: Sequence[tuple[str, str]] = DEFAULT_DRS_CONTRASTS,
    bootstrap_resamples: int = 5_000,
    confidence: float = 0.95,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Return paired right-minus-left contrasts from one immutable matrix.

    Each contrast is analyzed through the same metric extraction, paired
    bootstrap, and exact McNemar implementation as the primary U comparisons.
    Rows are copied before their configuration labels are normalized, so the
    source matrix remains byte-for-byte untouched.
    """

    rows = list(records)
    if not rows:
        raise ValueError("mechanism contrasts require a nonempty matrix")
    normalized = [(str(left), str(right)) for left, right in contrasts]
    if any(not left or not right or left == right for left, right in normalized):
        raise ValueError("contrast endpoints must be distinct nonempty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("mechanism contrasts must be unique")

    comparisons: list[dict[str, Any]] = []
    for left, right in normalized:
        selected: list[dict[str, Any]] = []
        for row in rows:
            configuration = str(row.get("configuration", ""))
            if configuration not in {left, right}:
                continue
            # JSON round-tripping is deliberate: nested experiment fields must
            # not alias the immutable source records through the analysis.
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            copied["configuration"] = "U" if configuration == left else right
            selected.append(copied)
        report = analyze_paired_experiment(
            selected,
            configurations=(right,),
            bootstrap_resamples=bootstrap_resamples,
            confidence=confidence,
            seed=seed,
        )
        if len(report["comparisons"]) != 1:
            raise ValueError(f"unexpected comparison count for {left}->{right}")
        comparison = report["comparisons"][0]
        comparison.update(
            {
                "baseline_configuration": left,
                "configuration": right,
                "contrast": f"{right}-minus-{left}",
            }
        )
        comparisons.append(comparison)

    _holm_adjust(comparisons)
    population_hashes = {
        str(comparison["population_ids_sha256"]) for comparison in comparisons
    }
    return {
        "schema_version": 1,
        "analysis": "paired-mechanism-contrasts",
        "records": len(rows),
        "contrasts": [
            {"left": left, "right": right} for left, right in normalized
        ],
        "comparisons": comparisons,
        "parameters": {
            "paired_unit": "example",
            "delta_direction": "right-minus-left",
            "bootstrap": "paired-percentile",
            "bootstrap_resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
            "mcnemar": "two-sided-exact-conditional-binomial",
            "multiplicity": (
                "Holm within each binary metric across registered contrasts"
            ),
        },
        "diagnostics": {
            "shared_population_across_contrasts": len(population_hashes) == 1,
            "population_ids_sha256": (
                next(iter(population_hashes))
                if len(population_hashes) == 1
                else None
            ),
        },
        "claim_boundary": {
            "generation_rows_modified": False,
            "population_selected_from_outcomes": False,
            "performance_threshold_used": False,
            "confirmatory_status": "descriptive-post-primary",
            "note": (
                "The mechanism questions predate the frozen outputs, but this "
                "direct-contrast implementation was added after development "
                "results were visible and is therefore reported descriptively."
            ),
        },
    }


__all__ = ["DEFAULT_DRS_CONTRASTS", "analyze_mechanism_contrasts"]
