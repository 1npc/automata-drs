"""Fail-closed gate for the post-primary real-logit ranking audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .causal_lm import (
    iter_sbn_records,
    scheduled_generation_identities,
    select_sbn_records_by_line,
    validate_generation_resume_prefix,
)


MODEL = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def _adapter_hashes(adapter: Path) -> dict[str, str]:
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    return {name: _sha256(adapter / name) for name in names}


def _nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def evaluate_adaptive_ranking_integration(
    outputs: Sequence[Path],
    *,
    selection_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate the registered greedy and beam4 same-logit audit cells."""

    if len(outputs) != 2 or len(set(outputs)) != 2:
        raise ValueError("integration gate requires two distinct outputs")
    selection = _read_object(selection_path)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    registered = selection.get("registered_audit")
    selected = selection.get("selected")
    counts = selection.get("counts")
    expected_registered = {
        "configurations": ["BF", "WF"],
        "decode_cells": [
            {"strategy": "greedy", "num_beams": 1, "retained": 1},
            {"strategy": "beam", "num_beams": 4, "retained": 8},
        ],
        "max_new_tokens": 256,
        "completion_budget_factor": 1.0,
        "adaptive_initial_k": 32,
        "same_logits_three_way_shadow": True,
        "returned_generation_path": "frozen_unstable_full_sort",
        "stable_adaptive_mismatches_required": 0,
        "frozen_stable_mismatches_required_for_drop_in_claim": 0,
        "performance_threshold_used": False,
    }
    selection_ok = (
        selection.get("schema_version") == 1
        and selection.get("kind")
        == "adaptive-ranking-real-logit-audit-selection"
        and isinstance(selected, list)
        and len(selected) == 32
        and all(
            isinstance(item, dict)
            and _nonnegative_integer(item.get("line_number"))
            and item.get("line_number") > 0
            for item in selected
        )
        and isinstance(counts, dict)
        and counts.get("u_failure_pressure") == 21
        and counts.get("length_stratified_valid_controls") == 11
        and counts.get("selected") == 32
        and counts.get("generation_cells") == 128
        and sum(
            item.get("group") == "u_failure_pressure" for item in selected
        )
        == 21
        and sum(
            item.get("group") == "length_stratified_valid_control"
            for item in selected
        )
        == 11
        and registered == expected_registered
    )
    check(
        "registered_selection",
        selection_ok,
        {"counts": counts, "registered_audit": registered},
        "32 rows (21 pressure + 11 controls), 128 registered cells",
    )
    if not selection_ok:
        return {
            "schema_version": 1,
            "kind": "adaptive-ranking-real-logit-integration-gate",
            "passed": False,
            "drop_in_optimization_supported": False,
            "performance_threshold_used": False,
            "checks": checks,
            "cells": [],
        }

    line_numbers = [int(item["line_number"]) for item in selected]
    data_path = Path(str(selection.get("data", "")))
    data_ok = (
        data_path.is_file()
        and _sha256(data_path) == selection.get("data_sha256")
        and len(set(line_numbers)) == 32
        and all(number > 0 for number in line_numbers)
    )
    check(
        "selection_data_binding",
        data_ok,
        {
            "data": str(data_path),
            "sha256": _sha256(data_path) if data_path.is_file() else None,
            "unique_lines": len(set(line_numbers)),
        },
        {
            "sha256": selection.get("data_sha256"),
            "unique_lines": 32,
        },
    )
    if not data_ok:
        return {
            "schema_version": 1,
            "kind": "adaptive-ranking-real-logit-integration-gate",
            "passed": False,
            "drop_in_optimization_supported": False,
            "performance_threshold_used": False,
            "checks": checks,
            "cells": [],
        }
    records = select_sbn_records_by_line(
        iter_sbn_records(data_path), line_numbers
    )
    expected = scheduled_generation_identities(
        records, ("BF", "WF"), schedule="rotate"
    )

    implementation = {
        "audit_runner_sha256": _sha256(
            repository_root / "scripts" / "audit_adaptive_ranking_integration.py"
        ),
        "adaptive_subword_sha256": _sha256(
            repository_root / "src" / "automata_drs" / "adaptive_subword.py"
        ),
        "ranking_sha256": _sha256(
            repository_root / "src" / "automata_drs" / "ranking.py"
        ),
        "frozen_subword_sha256": _sha256(
            repository_root / "src" / "automata_drs" / "subword.py"
        ),
    }
    cell_map: dict[int, Path] = {}
    invalid_cells: list[Any] = []
    manifests: dict[int, dict[str, Any]] = {}
    rows_by_cell: dict[int, list[dict[str, Any]]] = {}
    for output in outputs:
        manifest_path = Path(str(output) + ".manifest.json")
        manifest = _read_object(manifest_path)
        beams = manifest.get("num_beams")
        if beams not in {1, 4} or beams in cell_map:
            invalid_cells.append(beams)
            continue
        cell_map[int(beams)] = output
        manifests[int(beams)] = manifest
        rows_by_cell[int(beams)] = _read_rows(output)
    check(
        "registered_decode_cells",
        not invalid_cells and set(cell_map) == {1, 4},
        {"num_beams": sorted(cell_map), "invalid": invalid_cells},
        {"num_beams": [1, 4], "invalid": []},
    )

    cell_diagnostics: list[dict[str, Any]] = []
    total_frozen_stable = 0
    total_stable_adaptive = 0
    for beams in (1, 4):
        if beams not in cell_map:
            continue
        output = cell_map[beams]
        manifest = manifests[beams]
        rows = rows_by_cell[beams]
        prefix = "greedy" if beams == 1 else "beam4"
        adapter = Path(str(manifest.get("adapter", "")))
        manifest_ok = (
            manifest.get("schema_version") == 1
            and manifest.get("kind")
            == "adaptive-ranking-real-logit-integration-audit"
            and manifest.get("status") == "complete"
            and manifest.get("selection") == str(selection_path)
            and manifest.get("selection_sha256") == _sha256(selection_path)
            and manifest.get("data") == str(data_path)
            and manifest.get("data_sha256") == selection.get("data_sha256")
            and manifest.get("model") == MODEL
            and manifest.get("revision") == REVISION
            and manifest.get("model_loader") == "multimodal-text-only"
            and manifest.get("configurations") == ["BF", "WF"]
            and manifest.get("schedule") == "rotate"
            and manifest.get("max_input_tokens") == 512
            and manifest.get("max_new_tokens") == 256
            and manifest.get("completion_budget_factor") == 1.0
            and manifest.get("adaptive_initial_k") == 32
            and manifest.get("expected_examples") == 32
            and manifest.get("expected_rows") == 64
            and manifest.get("rows") == 64
            and manifest.get("same_logits_three_way_shadow") is True
            and manifest.get("returned_generation_path")
            == "frozen_unstable_full_sort"
            and manifest.get("performance_threshold_used") is False
            and manifest.get("physical_gpu_index") == "0"
            and manifest.get("cpu_thread_limits") == THREAD_LIMITS
            and manifest.get("implementation") == implementation
            and adapter.is_dir()
            and manifest.get("adapter_hashes") == _adapter_hashes(adapter)
            and manifest.get("output_sha256") == _sha256(output)
            and isinstance(manifest.get("completed_at_utc"), str)
        )
        check(
            f"{prefix}_manifest_contract",
            manifest_ok,
            manifest,
            "complete pinned Qwen/A40 audit manifest with current hashes",
        )
        try:
            completed = validate_generation_resume_prefix(
                rows, expected, num_beams=beams
            )
            population_ok = completed == 64
        except ValueError as error:
            completed = str(error)
            population_ok = False
        check(
            f"{prefix}_population",
            population_ok,
            completed,
            64,
        )

        row_failures: list[dict[str, Any]] = []
        frozen_stable = 0
        stable_adaptive = 0
        for index, row in enumerate(rows):
            equivalence = row.get("equivalence")
            processors = row.get("processors")
            runtime = row.get("runtime")
            validity = row.get("validity")
            if not all(
                isinstance(value, dict)
                for value in (equivalence, processors, runtime, validity)
            ):
                row_failures.append({"row": index + 1, "reason": "missing objects"})
                continue
            assert isinstance(equivalence, dict)
            assert isinstance(processors, dict)
            assert isinstance(runtime, dict)
            assert isinstance(validity, dict)
            fs = equivalence.get("frozen_stable_mismatches")
            sa = equivalence.get("stable_adaptive_mismatches")
            frozen_stable += int(fs) if _nonnegative_integer(fs) else 0
            stable_adaptive += int(sa) if _nonnegative_integer(sa) else 0
            triplet = [processors.get(name) for name in ("frozen", "stable", "adaptive")]
            ranking = processors.get("adaptive_ranking")
            stats_ok = all(
                isinstance(value, dict) for value in triplet
            ) and isinstance(ranking, dict)
            if stats_ok:
                frozen, stable, adaptive = triplet
                assert isinstance(frozen, dict)
                assert isinstance(stable, dict)
                assert isinstance(adaptive, dict)
                assert isinstance(ranking, dict)
                shared = ("calls", "rows", "allowed_total", "allowed_min", "allowed_max")
                stats_ok = (
                    all(
                        frozen.get(name)
                        == stable.get(name)
                        == adaptive.get(name)
                        for name in shared
                    )
                    and all(
                        item.get("all_masked_rows") == 0
                        for item in (frozen, stable, adaptive)
                    )
                    and ranking.get("ranked_rows") == adaptive.get("rows")
                    and ranking.get("certificate_tests")
                    == adaptive.get("candidates_tested")
                    and _nonnegative_integer(ranking.get("cumulative_candidate_pool"))
                    and ranking.get("python_scalar_values_materialized")
                    == 2 * ranking.get("cumulative_candidate_pool", -1)
                )
            row_ok = (
                _nonnegative_integer(equivalence.get("calls"))
                and equivalence.get("calls") > 0
                and _nonnegative_integer(equivalence.get("rows"))
                and equivalence.get("rows") > 0
                and fs == 0
                and sa == 0
                and equivalence.get("frozen_stable_examples") == []
                and stats_ok
                and validity.get("sbn") is True
                and (
                    row.get("configuration") != "WF"
                    or validity.get("clf_referee") is True
                )
                and runtime.get("cuda_device_name") == "NVIDIA A40"
                and isinstance(runtime.get("cuda_device_total_bytes"), int)
                and 40 * 1024**3 <= runtime["cuda_device_total_bytes"] <= 50 * 1024**3
                and isinstance(row.get("raw_prediction"), str)
                and isinstance(row.get("surface_token_ids"), list)
            )
            if not row_ok:
                row_failures.append({"row": index + 1, "reason": "contract/equivalence"})
        total_frozen_stable += frozen_stable
        total_stable_adaptive += stable_adaptive
        check(
            f"{prefix}_same_logit_equivalence",
            population_ok and not row_failures,
            {
                "row_failures": row_failures[:20],
                "frozen_stable_mismatches": frozen_stable,
                "stable_adaptive_mismatches": stable_adaptive,
            },
            {
                "row_failures": [],
                "frozen_stable_mismatches": 0,
                "stable_adaptive_mismatches": 0,
            },
        )
        cell_diagnostics.append(
            {
                "num_beams": beams,
                "output": str(output),
                "sha256": _sha256(output),
                "rows": len(rows),
                "frozen_stable_mismatches": frozen_stable,
                "stable_adaptive_mismatches": stable_adaptive,
            }
        )

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "kind": "adaptive-ranking-real-logit-integration-gate",
        "passed": passed,
        "drop_in_optimization_supported": (
            passed
            and total_frozen_stable == 0
            and total_stable_adaptive == 0
        ),
        "performance_threshold_used": False,
        "selection": str(selection_path),
        "selection_sha256": _sha256(selection_path),
        "checks": checks,
        "cells": cell_diagnostics,
        "implementation": implementation,
    }


def verify_frozen_adaptive_ranking_integration(
    current: dict[str, Any], stored: dict[str, Any]
) -> None:
    if current.get("passed") is not True:
        raise ValueError("current adaptive-ranking integration evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored adaptive-ranking integration gate is unsupported")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored adaptive-ranking integration gate has no timestamp")
    observed = {key: value for key, value in stored.items() if key != "checked_at_utc"}
    if observed != current:
        raise ValueError("stored adaptive-ranking integration gate differs from current evidence")


def freeze_adaptive_ranking_integration(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("passed") is not True:
        raise ValueError("cannot freeze failed adaptive-ranking integration evidence")
    return {**report, "checked_at_utc": datetime.now(timezone.utc).isoformat()}


__all__ = [
    "evaluate_adaptive_ranking_integration",
    "freeze_adaptive_ranking_integration",
    "verify_frozen_adaptive_ranking_integration",
]
