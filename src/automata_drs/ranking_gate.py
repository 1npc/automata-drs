"""Fail-closed integrity gate for the post-primary CUDA ranking study."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SOURCE_CONTRACT = {
    "model": "Qwen/Qwen3.5-4B-Base",
    "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "config_sha256": (
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
    ),
    "tokenizer_sha256": (
        "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
    ),
    "tokenizer_config_sha256": (
        "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c"
    ),
}
COMMON_PARAMETERS = {
    "vocabulary_size": 248_320,
    "mapped_token_count": 248_077,
    "unmapped_logit_positions": 243,
    "initial_k": 32,
    "certificate_modulus": 97,
    "certificate_residues": 1,
    "repeats": 20,
    "warmup": 3,
    "seed": 20260826,
    "device": "cuda",
    "dtype": "bfloat16",
}
REQUIRED_CELLS = frozenset({1, 8})


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


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _close(observed: Any, expected: float) -> bool:
    value = _number(observed)
    return value is not None and math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _argsort_stability(path: Path) -> bool:
    """Return the unique explicit/default stability of argsort calls.

    The frozen subword processor has one score-ranking call.  Treat a missing
    keyword as PyTorch's documented default ``stable=False`` and reject mixed
    or non-literal forms so the report boundary cannot silently go stale.
    """

    observed: list[bool] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "argsort"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        stable = keywords.get("stable")
        if stable is None:
            observed.append(False)
        elif isinstance(stable, ast.Constant) and isinstance(stable.value, bool):
            observed.append(stable.value)
        else:
            raise ValueError(f"non-literal argsort stability in {path}")
    if not observed or len(set(observed)) != 1:
        raise ValueError(f"expected one uniform argsort stability in {path}")
    return observed[0]


def evaluate_adaptive_ranking_benchmarks(
    report_paths: Sequence[Path],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate both registered CUDA cells without imposing a speed threshold."""

    if len(report_paths) != 2 or len(set(report_paths)) != 2:
        raise ValueError("ranking gate requires two distinct report paths")
    reports = [(path, _read_object(path)) for path in report_paths]
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

    cell_map: dict[int, tuple[Path, dict[str, Any]]] = {}
    invalid_required: list[Any] = []
    for path, report in reports:
        required = report.get("parameters", {}).get("required")
        if (
            not isinstance(required, int)
            or isinstance(required, bool)
            or required in cell_map
        ):
            invalid_required.append(required)
            continue
        cell_map[required] = (path, report)
    check(
        "registered_cells",
        not invalid_required and set(cell_map) == REQUIRED_CELLS,
        {"required": sorted(cell_map), "invalid": invalid_required},
        {"required": sorted(REQUIRED_CELLS), "invalid": []},
    )

    ranking_path = repository_root / "src" / "automata_drs" / "ranking.py"
    benchmark_path = repository_root / "scripts" / "benchmark_adaptive_ranking.py"
    publication_processor_path = (
        repository_root / "src" / "automata_drs" / "subword.py"
    )
    publication_generator_path = (
        repository_root / "scripts" / "generate_causal_lm_baseline.py"
    )
    implementation = {
        "ranking_sha256": _sha256(ranking_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "publication_processor_sha256": _sha256(publication_processor_path),
        "publication_generator_sha256": _sha256(publication_generator_path),
    }
    publication_argsort_stable = _argsort_stability(
        publication_processor_path
    )
    check(
        "publication_processor_tie_semantics",
        publication_argsort_stable is False,
        {"argsort_stable": publication_argsort_stable},
        {"argsort_stable": False},
    )
    cell_diagnostics: list[dict[str, Any]] = []
    for required in sorted(REQUIRED_CELLS):
        if required not in cell_map:
            continue
        path, report = cell_map[required]
        prefix = f"required{required}"
        parameters = report.get("parameters")
        expected_parameters = {**COMMON_PARAMETERS, "required": required}
        scope = report.get("scope")
        expected_scope = {
            "synthetic_scores_and_certificate_predicate": True,
            "end_to_end_generation_speed_claim": False,
            "integrated_into_frozen_publication_generator": False,
        }
        check(
            f"{prefix}_schema",
            report.get("schema_version") == 4
            and report.get("benchmark")
            == "tie-closed-adaptive-token-ranking",
            {
                "schema_version": report.get("schema_version"),
                "benchmark": report.get("benchmark"),
            },
            {
                "schema_version": 4,
                "benchmark": "tie-closed-adaptive-token-ranking",
            },
        )
        check(
            f"{prefix}_source_contract",
            report.get("source_contract") == SOURCE_CONTRACT,
            report.get("source_contract"),
            SOURCE_CONTRACT,
        )
        check(
            f"{prefix}_parameters",
            parameters == expected_parameters,
            parameters,
            expected_parameters,
        )
        check(
            f"{prefix}_scope",
            scope == expected_scope,
            scope,
            expected_scope,
        )
        expected_ranking_order = {
            "primary": "descending_score",
            "tie_break": "ascending_token_id",
            "full_reference_argsort_stable": True,
            "frozen_publication_argsort_stable": False,
            "frozen_publication_equivalence_claim": False,
        }
        check(
            f"{prefix}_ranking_order_boundary",
            report.get("ranking_order") == expected_ranking_order,
            report.get("ranking_order"),
            expected_ranking_order,
        )

        environment = report.get("environment")
        environment = environment if isinstance(environment, dict) else {}
        threads = environment.get("cpu_thread_limits")
        expected_threads = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        environment_ok = (
            environment.get("device_name") == "NVIDIA A40"
            and environment.get("device_compute_capability") == [8, 6]
            and isinstance(environment.get("device_total_memory_bytes"), int)
            and not isinstance(environment.get("device_total_memory_bytes"), bool)
            and 40 * 1024**3
            <= environment["device_total_memory_bytes"]
            <= 50 * 1024**3
            and isinstance(environment.get("torch"), str)
            and isinstance(environment.get("torch_cuda_runtime"), str)
            and environment.get("cuda_visible_devices") == "0"
            and threads == expected_threads
        )
        check(
            f"{prefix}_a40_environment",
            environment_ok,
            environment,
            {
                "device_name": "NVIDIA A40",
                "device_compute_capability": [8, 6],
                "device_total_memory_bytes": "40-50 GiB",
                "cuda_visible_devices": "0",
                "cpu_thread_limits": expected_threads,
                "torch_and_cuda_runtime": "present",
            },
        )

        correctness = report.get("correctness")
        correctness = correctness if isinstance(correctness, dict) else {}
        selected = correctness.get("selected_token_ids")
        correctness_ok = (
            correctness.get("mismatches") == 0
            and correctness.get("examples") == []
            and isinstance(selected, list)
            and len(selected) == required
            and all(
                isinstance(token_id, int)
                and not isinstance(token_id, bool)
                and 0 <= token_id < COMMON_PARAMETERS["mapped_token_count"]
                for token_id in selected
            )
        )
        check(
            f"{prefix}_ranking_equivalence",
            correctness_ok,
            correctness,
            {
                "mismatches": 0,
                "examples": [],
                "selected_count": required,
                "selected_ids_below": COMMON_PARAMETERS["mapped_token_count"],
            },
        )

        full = report.get("full_ranking")
        adaptive = report.get("adaptive")
        comparison = report.get("comparison")
        full = full if isinstance(full, dict) else {}
        adaptive = adaptive if isinstance(adaptive, dict) else {}
        comparison = comparison if isinstance(comparison, dict) else {}
        full_seconds = _number(full.get("seconds_median"))
        full_p95 = _number(full.get("seconds_p95"))
        full_tests = _number(full.get("certificate_tests_median"))
        adaptive_seconds = _number(adaptive.get("seconds_median"))
        adaptive_p95 = _number(adaptive.get("seconds_p95"))
        adaptive_tests = _number(adaptive.get("certificate_tests_median"))
        rounds = _number(adaptive.get("topk_rounds_median"))
        peak_max = _number(adaptive.get("peak_candidate_pool_max"))
        boundary_ties = _number(adaptive.get("boundary_ties_added_max"))
        peak = _number(adaptive.get("peak_candidate_pool_median"))
        cumulative = _number(
            adaptive.get("cumulative_candidate_pool_median")
        )
        materialized = _number(
            adaptive.get("python_scalar_values_materialized_median")
        )
        counts_ok = (
            full.get("python_scalar_values_materialized_per_repeat")
            == COMMON_PARAMETERS["vocabulary_size"]
            and full.get("materialization_contents") == "token_ids"
            and adaptive.get("materialization_contents")
            == "candidate_ids_and_scores_each_round"
            and peak is not None
            and peak_max is not None
            and cumulative is not None
            and materialized is not None
            and 0 < peak <= COMMON_PARAMETERS["vocabulary_size"]
            and peak <= peak_max <= COMMON_PARAMETERS["vocabulary_size"]
            and cumulative >= peak
            and materialized == 2 * cumulative
            and full_tests is not None
            and 0 < full_tests <= COMMON_PARAMETERS["vocabulary_size"]
            and adaptive_tests is not None
            and 0 < adaptive_tests <= COMMON_PARAMETERS["vocabulary_size"]
            and rounds is not None
            and 1 <= rounds <= math.ceil(
                math.log2(
                    COMMON_PARAMETERS["vocabulary_size"]
                    / COMMON_PARAMETERS["initial_k"]
                )
            )
            + 1
            and boundary_ties is not None
            and 0 <= boundary_ties <= COMMON_PARAMETERS["vocabulary_size"]
            and isinstance(adaptive.get("full_vocabulary_fallbacks"), int)
            and not isinstance(adaptive.get("full_vocabulary_fallbacks"), bool)
            and 0
            <= adaptive["full_vocabulary_fallbacks"]
            <= COMMON_PARAMETERS["repeats"]
            and full_seconds is not None
            and full_seconds > 0
            and full_p95 is not None
            and full_p95 >= full_seconds
            and adaptive_seconds is not None
            and adaptive_seconds > 0
            and adaptive_p95 is not None
            and adaptive_p95 >= adaptive_seconds
        )
        check(
            f"{prefix}_measurement_counts",
            counts_ok,
            {"full": full, "adaptive": adaptive},
            "finite timings and internally consistent cumulative counts",
        )

        comparison_ok = False
        if counts_ok:
            assert full_seconds is not None
            assert adaptive_seconds is not None
            assert cumulative is not None
            assert materialized is not None
            vocabulary_size = COMMON_PARAMETERS["vocabulary_size"]
            comparison_ok = (
                _close(
                    comparison.get("median_wall_time_speedup"),
                    full_seconds / adaptive_seconds,
                )
                and _close(
                    comparison.get("candidate_id_transfer_reduction"),
                    1.0 - cumulative / vocabulary_size,
                )
                and _close(
                    comparison.get("python_scalar_materialization_reduction"),
                    1.0 - materialized / vocabulary_size,
                )
                and _close(
                    comparison.get("device_to_host_scalar_transfer_reduction"),
                    1.0 - materialized / vocabulary_size,
                )
                and comparison.get("device_to_host_transfer_applies") is True
            )
        check(
            f"{prefix}_derived_comparison",
            comparison_ok,
            comparison,
            "values recomputed from recorded medians; CUDA transfer applies",
        )
        check(
            f"{prefix}_implementation_hashes",
            report.get("implementation") == implementation,
            report.get("implementation"),
            implementation,
        )
        cell_diagnostics.append(
            {
                "required": required,
                "path": str(path),
                "sha256": _sha256(path),
                "full_seconds_median": full_seconds,
                "adaptive_seconds_median": adaptive_seconds,
                "median_wall_time_speedup": comparison.get(
                    "median_wall_time_speedup"
                ),
                "performance_threshold_used": False,
            }
        )

    return {
        "schema_version": 1,
        "kind": "adaptive-ranking-cuda-benchmark-gate",
        "passed": all(item["passed"] for item in checks),
        "performance_threshold_used": False,
        "checks": checks,
        "cells": cell_diagnostics,
        "implementation": implementation,
    }


def verify_frozen_adaptive_ranking_benchmarks(
    current: dict[str, Any], stored: dict[str, Any]
) -> None:
    if current.get("passed") is not True:
        raise ValueError("current adaptive-ranking benchmark evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored adaptive-ranking benchmark gate is unsupported or failed")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored adaptive-ranking benchmark gate has no timestamp")
    observed = {key: value for key, value in stored.items() if key != "checked_at_utc"}
    if observed != current:
        differing = sorted(
            key
            for key in set(observed) | set(current)
            if observed.get(key) != current.get(key)
        )
        raise ValueError(
            "stored adaptive-ranking benchmark gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "COMMON_PARAMETERS",
    "SOURCE_CONTRACT",
    "evaluate_adaptive_ranking_benchmarks",
    "verify_frozen_adaptive_ranking_benchmarks",
]
