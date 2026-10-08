#!/usr/bin/env python3
"""Check that manuscript result claims remain locked to final suite evidence.

This checker never edits manuscript text. It reports whether DRS and AMR
result slots may be filled from their canonical, collector-produced JSON
artifacts. Partial matrices, logs, and individual summaries are intentionally
outside the accepted evidence interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.release import (  # noqa: E402
    PENMAN_RELEASE_GUARANTEE,
    has_complete_selective_release_accounting,
)


MANUSCRIPT_PENDING_MINIMUMS = {
    "docs/manuscript.md": 34,
    "docs/manuscript-abstract.md": 8,
    "docs/manuscript-scaffold.md": 2,
    "docs/manuscript-experiments.md": 20,
    "docs/manuscript-conclusion.md": 4,
}
EXPECTED_DRS_STAGES = {
    "pilot-dev",
    "layer-supplement",
    "fine-supplement",
    "beam-candidates-and-online",
    "publication-dev",
    "publication-test",
    "publication-tax",
    "conditioned-dev",
    "conditioned-test",
    "conditioned-tax",
}
EXPECTED_DRS_RELEASE_SPLITS = {"dev", "test", "tax"}
EXPECTED_DRS_EVALUATIONS = {
    "dev400",
    "dev400-fine",
    "dev400-beam4",
    "dev1132",
    "test1132",
    "tax117",
}
EXPECTED_DRS_COMPARISONS = {
    "dev400": {"BF", "WF"},
    "dev400-fine": {"S", "R", "G", "B", "W"},
    "dev400-beam4": {"PostW", "W", "WF"},
    "dev1132": {"BF", "WF"},
    "test1132": {"BF", "WF"},
    "tax117": {"BF", "WF"},
}
EXPECTED_AMR_BEAM_COMPARISONS = {
    "postp_minus_u",
    "p_minus_u",
    "p_minus_postp",
}
DRS_RELEASE_CLAIM_BOUNDARY = (
    "policy-relative structural/well-formedness release only; "
    "not task-semantic correctness"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_table_is_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(path, str)
            and bool(path)
            and _is_sha256(digest)
            for path, digest in value.items()
        )
    )


def _path_is_hash_bound(path: Any, evidence_hashes: Any) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and isinstance(evidence_hashes, dict)
        and _is_sha256(evidence_hashes.get(path))
    )


def _digest_is_hash_bound(digest: Any, evidence_hashes: Any) -> bool:
    return (
        _is_sha256(digest)
        and isinstance(evidence_hashes, dict)
        and digest in evidence_hashes.values()
    )


def _nonempty_object(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _manuscript_lock() -> tuple[dict[str, Any], bool]:
    files: dict[str, Any] = {}
    intact = True
    for relative, minimum in MANUSCRIPT_PENDING_MINIMUMS.items():
        path = REPOSITORY_ROOT / relative
        text = path.read_text(encoding="utf-8")
        count = text.count("RESULT PENDING")
        passed = count >= minimum
        intact = intact and passed
        files[relative] = {
            "sha256": _sha256(path),
            "result_pending_occurrences": count,
            "minimum_while_locked": minimum,
            "passed": passed,
        }
    return files, intact


def _drs_evidence(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    checks: dict[str, bool] = {"exists": value is not None}
    if value is None:
        return {"path": str(path), "ready": False, "checks": checks}

    selection = value.get("host_selection", {})
    if not isinstance(selection, dict):
        selection = {}
    gates = value.get("method_gates", [])
    releases = value.get("certified_releases", [])
    models = value.get("models", [])
    if not isinstance(gates, list):
        gates = []
    if not isinstance(releases, list):
        releases = []
    if not isinstance(models, list):
        models = []
    evidence = value.get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}
    evidence_hashes = evidence.get("input_sha256", {})
    if not isinstance(evidence_hashes, dict):
        evidence_hashes = {}
    selected_key = selection.get("selected_key")
    selected_models = [
        row
        for row in models
        if isinstance(row, dict) and row.get("key") == selected_key
    ]
    observed_stages = {
        row.get("stage")
        for row in gates
        if isinstance(row, dict) and row.get("passed") is True
    }
    method_gates_valid = (
        len(gates) == len(EXPECTED_DRS_STAGES)
        and observed_stages == EXPECTED_DRS_STAGES
        and all(
            isinstance(row, dict)
            and row.get("passed") is True
            and _path_is_hash_bound(row.get("path"), evidence_hashes)
            for row in gates
        )
    )
    observed_release_splits: set[str] = set()
    releases_valid = len(releases) == 3
    for row in releases:
        if not isinstance(row, dict):
            releases_valid = False
            continue
        audit = row.get("audit", {})
        if not isinstance(audit, dict):
            releases_valid = False
            continue
        population = audit.get("population", {})
        release = audit.get("release", {})
        audit_input = audit.get("input", {})
        implementation = audit.get("implementation", {})
        if not all(
            isinstance(item, dict)
            for item in (population, release, audit_input, implementation)
        ):
            releases_valid = False
            continue
        split = row.get("split")
        observed_release_splits.add(split)
        examples = population.get("examples")
        expected_rows = examples * 2 if isinstance(examples, int) else None
        expected_guarantees = (
            {
                "finite-horizon-structural-and-referee-policy-membership": examples,
                "finite-horizon-structural-policy-membership": examples,
            }
            if isinstance(examples, int) and examples > 0
            else None
        )
        completion_factor = audit_input.get("completion_budget_factor")
        releases_valid = releases_valid and (
            _path_is_hash_bound(row.get("path"), evidence_hashes)
            and audit.get("schema_version") == 2
            and audit.get("audit") == "sbn-certified-release"
            and audit.get("claim_boundary") == DRS_RELEASE_CLAIM_BOUNDARY
            and audit.get("gate", {}).get("passed") is True
            and audit.get("gate", {}).get("performance_threshold_used") is False
            and isinstance(examples, int)
            and not isinstance(examples, bool)
            and examples > 0
            and population.get("rows") == expected_rows
            and population.get("unique_identities") == expected_rows
            and release.get("releasable_rows") == expected_rows
            and release.get("refused_rows") == 0
            and release.get("failures") == []
            and release.get("reason_counts") == {}
            and release.get("guarantee_counts") == expected_guarantees
            and audit_input.get("configurations") == ["BF", "WF"]
            and isinstance(completion_factor, (int, float))
            and not isinstance(completion_factor, bool)
            and float(completion_factor) == 1.0
            and _digest_is_hash_bound(
                audit_input.get("predictions_sha256"), evidence_hashes
            )
            and _digest_is_hash_bound(
                audit_input.get("manifest_sha256"), evidence_hashes
            )
            and _is_sha256(implementation.get("audit_sha256"))
            and _is_sha256(implementation.get("release_sha256"))
        )
    comparisons_valid = False
    method_baselines_valid = False
    if len(selected_models) == 1:
        method_baselines = selected_models[0].get("method_baselines", [])
        if isinstance(method_baselines, list):
            baseline_evaluations: set[str] = set()
            method_baselines_valid = len(method_baselines) == len(
                EXPECTED_DRS_EVALUATIONS
            )
            for row in method_baselines:
                if not isinstance(row, dict):
                    method_baselines_valid = False
                    continue
                evaluation = row.get("evaluation")
                expected_strategy = (
                    "beam4" if evaluation == "dev400-beam4" else "greedy"
                )
                row_valid = (
                    isinstance(evaluation, str)
                    and evaluation in EXPECTED_DRS_EVALUATIONS
                    and evaluation not in baseline_evaluations
                    and row.get("configuration") == "U"
                    and row.get("strategy") == expected_strategy
                    and _nonempty_object(row.get("group"))
                    and _path_is_hash_bound(row.get("path"), evidence_hashes)
                )
                method_baselines_valid = method_baselines_valid and row_valid
                if isinstance(evaluation, str):
                    baseline_evaluations.add(evaluation)
            method_baselines_valid = (
                method_baselines_valid
                and baseline_evaluations == EXPECTED_DRS_EVALUATIONS
            )
        comparisons = selected_models[0].get("comparisons", [])
        if isinstance(comparisons, list):
            observed: dict[str, set[str]] = {
                evaluation: set() for evaluation in EXPECTED_DRS_EVALUATIONS
            }
            seen_cells: set[tuple[str, str, str]] = set()
            comparisons_valid = len(comparisons) == sum(
                len(configurations)
                for configurations in EXPECTED_DRS_COMPARISONS.values()
            )
            for row in comparisons:
                if not isinstance(row, dict):
                    comparisons_valid = False
                    continue
                evaluation = row.get("evaluation")
                configuration = row.get("configuration")
                strategy = row.get("strategy")
                expected_strategy = (
                    "beam4" if evaluation == "dev400-beam4" else "greedy"
                )
                cell = (str(evaluation), str(configuration), str(strategy))
                row_valid = (
                    isinstance(evaluation, str)
                    and isinstance(configuration, str)
                    and evaluation in EXPECTED_DRS_COMPARISONS
                    and configuration in EXPECTED_DRS_COMPARISONS[evaluation]
                    and strategy == expected_strategy
                    and cell not in seen_cells
                    and _nonempty_object(row.get("group"))
                    and _nonempty_object(row.get("paired"))
                    and _nonempty_object(row.get("statistics"))
                    and _path_is_hash_bound(row.get("path"), evidence_hashes)
                    and _path_is_hash_bound(
                        row.get("statistics_path"), evidence_hashes
                    )
                )
                comparisons_valid = comparisons_valid and row_valid
                seen_cells.add(cell)
                if (
                    isinstance(evaluation, str)
                    and evaluation in observed
                    and isinstance(configuration, str)
                ):
                    observed[evaluation].add(configuration)
            comparisons_valid = (
                comparisons_valid and observed == EXPECTED_DRS_COMPARISONS
            )
    cross_representation = value.get("cross_representation", {})
    cross_audit = (
        cross_representation.get("audit", {})
        if isinstance(cross_representation, dict)
        else {}
    )
    cross_coverage = cross_audit.get("coverage", {})
    cross_source = cross_audit.get("source", {})
    cross_tokenizer = cross_audit.get("tokenizer", {})
    selected_model = selected_models[0] if len(selected_models) == 1 else {}
    cross_representation_valid = (
        isinstance(cross_representation, dict)
        and _path_is_hash_bound(
            cross_representation.get("path"), evidence_hashes
        )
        and cross_audit.get("audit") == "penman-bytelevel-token-lifting"
        and cross_audit.get("gate", {}).get("passed") is True
        and cross_source.get("records") == 1471
        and cross_coverage.get("denominator") == 1471
        and cross_coverage.get("exact_byte_roundtrip") == 1471
        and cross_coverage.get("accepted_with_eos") == 1471
        and cross_tokenizer.get("model") == selected_model.get("model")
        and cross_tokenizer.get("requested_revision")
        == selected_model.get("revision")
    )
    resource_plan = value.get("method_resource_plan", {})
    resource_plan_valid = (
        isinstance(resource_plan, dict)
        and resource_plan.get("kind") == "selected-lora-method-resource-plan"
        and resource_plan.get("selected_key") == selected_key
        and resource_plan.get("gate", {}).get("passed") is True
        and any(
            Path(source_path).name == "selected-method-resource-plan.json"
            for source_path in evidence_hashes
        )
    )
    checks.update(
        {
            "collector_schema": (
                value.get("schema_version") == 7 and value.get("scope") == "full"
            ),
            "frozen_selected_host": (
                selection.get("status") == "selected"
                and isinstance(selected_key, str)
                and bool(selected_key)
                and len(selected_models) == 1
            ),
            "resource_plan": resource_plan_valid,
            "all_method_stages": method_gates_valid,
            "all_publication_releases": (
                releases_valid
                and observed_release_splits == EXPECTED_DRS_RELEASE_SPLITS
            ),
            "all_comparison_cells": comparisons_valid,
            "all_evaluation_baselines": method_baselines_valid,
            "cross_representation": cross_representation_valid,
            "hash_bound_inputs": (
                _is_sha256(evidence.get("config_sha256"))
                and _hash_table_is_valid(evidence_hashes)
            ),
        }
    )
    ready = all(checks.values())
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "ready": ready,
        "checks": checks,
    }


def _amr_evidence(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    checks: dict[str, bool] = {"exists": value is not None}
    if value is None:
        return {"path": str(path), "ready": False, "checks": checks}
    matrices = value.get("matrices", [])
    if not isinstance(matrices, list):
        matrices = []
    evidence_hashes = value.get("evidence", {})
    if not isinstance(evidence_hashes, dict):
        evidence_hashes = {}
    matrix_splits = {
        row.get("split") for row in matrices if isinstance(row, dict)
    }
    matrices_valid = len(matrices) == 3
    dev_attempts: int | None = None
    for row in matrices:
        if not isinstance(row, dict):
            matrices_valid = False
            continue
        attempts = row.get("examples")
        release = row.get("release", {})
        if (
            row.get("split") == "dev"
            and isinstance(attempts, int)
            and not isinstance(attempts, bool)
        ):
            dev_attempts = attempts
        matrices_valid = matrices_valid and (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts > 0
            and _path_is_hash_bound(row.get("gate"), evidence_hashes)
            and _nonempty_object(row.get("u"))
            and _nonempty_object(row.get("p"))
            and _nonempty_object(row.get("paired"))
            and has_complete_selective_release_accounting(
                release,
                attempts=attempts,
                guarantee=PENMAN_RELEASE_GUARANTEE,
            )
        )
    certification = value.get("certification", {})
    if not isinstance(certification, dict):
        certification = {}
    beam = value.get("beam", {})
    if not isinstance(beam, dict):
        beam = {}
    beam_release = beam.get("release", {})
    beam_comparisons = beam.get("comparisons", {})
    beam_groups = beam.get("groups", {})
    beam_evidence_valid = (
        _path_is_hash_bound(beam.get("report_path"), evidence_hashes)
        and isinstance(beam_comparisons, dict)
        and set(beam_comparisons) == EXPECTED_AMR_BEAM_COMPARISONS
        and all(
            _nonempty_object(comparison)
            for comparison in beam_comparisons.values()
        )
        and isinstance(beam_groups, dict)
        and set(beam_groups) == {"U", "PostP", "P"}
        and all(_nonempty_object(group) for group in beam_groups.values())
        and _nonempty_object(beam.get("selection"))
        and _nonempty_object(beam.get("search"))
        and isinstance(dev_attempts, int)
        and has_complete_selective_release_accounting(
            beam_release,
            attempts=dev_attempts,
            guarantee=PENMAN_RELEASE_GUARANTEE,
        )
    )
    resource_plan = value.get("resource_plan", {})
    resource_plan_valid = (
        isinstance(resource_plan, dict)
        and resource_plan.get("audit") == "amr-full-run-resource-plan"
        and resource_plan.get("gate", {}).get("passed") is True
        and resource_plan.get("gate", {}).get("performance_threshold_used") is False
    )
    checks.update(
        {
            "collector_schema": (
                value.get("schema_version") == 2
                and value.get("audit") == "amr-transfer-suite"
            ),
            "suite_gate": (
                value.get("gate", {}).get("passed") is True
                and value.get("gate", {}).get("performance_threshold_used") is False
            ),
            "exact_matrix_chain": (
                matrices_valid and matrix_splits == {"pilot", "dev", "test"}
            ),
            "beam_evidence": beam_evidence_valid,
            "resource_plan": resource_plan_valid,
            "training_evidence": _nonempty_object(value.get("training")),
            "certification_boundary": (
                certification.get("orthogonal_to_deterministic_search") is True
                and "AMR task-semantic correctness"
                in certification.get("non_guarantees", [])
                and "availability of a releasable graph for every decoding attempt"
                in certification.get("non_guarantees", [])
            ),
            "hash_bound_inputs": _hash_table_is_valid(evidence_hashes),
        }
    )
    ready = all(checks.values())
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "ready": ready,
        "checks": checks,
    }


def _result_package(
    path: Path | None,
    drs_suite: Path,
    amr_suite: Path,
    package_builder: Any,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "ready": False, "checks": {"exists": False}}
    value = _read_object(path)
    checks: dict[str, bool] = {"exists": value is not None}
    if value is None:
        return {"path": str(path), "ready": False, "checks": checks}
    sources = value.get("sources", {})
    drs = value.get("drs", {})
    amr = value.get("amr", {})
    boundary = value.get("claim_boundary", {})
    expected = package_builder(drs_suite, amr_suite)
    drs_publication = drs.get("publication", [])
    drs_mechanism = drs.get("mechanism_ladder", [])
    drs_beam = drs.get("beam_diagnostic", [])
    amr_matrices = amr.get("matrices", [])
    amr_beam = amr.get("beam", {}).get("comparisons", [])
    checks.update(
        {
            "package_schema": (
                value.get("schema_version") == 1
                and value.get("audit") == "manuscript-result-package"
            ),
            "suite_hashes": (
                sources.get("drs", {}).get("sha256") == _sha256(drs_suite)
                and sources.get("amr", {}).get("sha256") == _sha256(amr_suite)
            ),
            "exact_drs_tables": (
                isinstance(drs_publication, list)
                and len(drs_publication) == 9
                and isinstance(drs_mechanism, list)
                and len(drs_mechanism) == 8
                and isinstance(drs_beam, list)
                and len(drs_beam) == 4
            ),
            "exact_amr_tables": (
                isinstance(amr_matrices, list)
                and len(amr_matrices) == 3
                and {row.get("split") for row in amr_matrices if isinstance(row, dict)}
                == {"pilot", "dev", "test"}
                and isinstance(amr_beam, list)
                and len(amr_beam) == 3
            ),
            "claim_boundary": (
                boundary.get(
                    "certified_policy_validity_is_conditional_on_release"
                )
                is True
                and boundary.get("release_coverage_is_reported_over_attempts")
                is True
                and boundary.get(
                    "structural_certification_implies_task_semantic_correctness"
                )
                is False
                and boundary.get("performance_threshold_used") is False
                and boundary.get("automatic_paper_framing_used") is False
            ),
            "package_gate": (
                value.get("gate", {}).get("passed") is True
                and value.get("gate", {}).get("performance_threshold_used") is False
            ),
            "exact_recomputation": value == expected,
        }
    )
    ready = all(checks.values())
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "ready": ready,
        "checks": checks,
    }


def build_report(
    drs_suite: Path,
    amr_suite: Path,
    result_package: Path | None = None,
    result_package_builder: Any = None,
) -> dict[str, Any]:
    manuscript_files, lock_intact = _manuscript_lock()
    drs = _drs_evidence(drs_suite)
    amr = _amr_evidence(amr_suite)
    if result_package_builder is None:
        package_script = REPOSITORY_ROOT / "scripts" / "build_manuscript_result_package.py"
        spec = importlib.util.spec_from_file_location(
            "result_lock_package_builder", package_script
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load manuscript result package builder: {package_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def result_package_builder(drs: Path, amr: Path) -> dict[str, Any]:
            return module.build_package(
                drs,
                amr,
                lock_builder=lambda _drs, _amr: {
                    "drs": {"ready": True},
                    "amr": {"ready": True},
                },
            )

    package = _result_package(
        result_package,
        drs_suite,
        amr_suite,
        result_package_builder,
    )
    if drs["ready"] and amr["ready"] and package["ready"]:
        state = "all-suite-evidence-and-result-package-ready-for-integration"
    elif drs["ready"] and amr["ready"]:
        state = "all-suite-evidence-ready-for-result-package"
    elif drs["ready"] or amr["ready"]:
        state = "partial-suite-evidence-ready"
    else:
        state = "locked-waiting-for-final-suite-evidence"
    return {
        "schema_version": 1,
        "audit": "manuscript-result-lock",
        "state": state,
        "manuscript_lock": {
            "passed": lock_intact,
            "manual_log_or_partial_jsonl_numbers_allowed": False,
            "files": manuscript_files,
        },
        "drs": drs,
        "amr": amr,
        "result_package": package,
        "can_build_result_package": drs["ready"] and amr["ready"] and lock_intact,
        "can_integrate": {
            "drs_results": package["ready"] and lock_intact,
            "amr_results": package["ready"] and lock_intact,
        },
        "gate": {"passed": lock_intact, "performance_threshold_used": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drs-suite",
        type=Path,
        default=Path("runs/lora/recent-model-and-method-results-v3.json"),
    )
    parser.add_argument(
        "--amr-suite",
        type=Path,
        default=Path("runs/amr/qwen35-4b-base-transfer-suite-v1.json"),
    )
    parser.add_argument(
        "--result-package",
        type=Path,
        default=Path("runs/paper/manuscript-result-package-v1.json"),
    )
    args = parser.parse_args()
    try:
        report = build_report(args.drs_suite, args.amr_suite, args.result_package)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
