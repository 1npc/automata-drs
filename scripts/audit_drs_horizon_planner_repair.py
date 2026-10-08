#!/usr/bin/env python3
"""Verify a registered subword planner repair on all frozen W paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import audit_drs_horizon_false_negatives as pre_fix_audit  # noqa: E402
from automata_drs.subword import ByteLevelVocabulary  # noqa: E402


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs/wcd_drs_horizon_planner_repair_v1.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"registered path must be repository-relative: {value}")
    resolved = (REPOSITORY_ROOT / path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"registered path escapes repository: {value}") from error
    return resolved


def load_contract(config_path: Path) -> dict[str, Any]:
    contract = json.loads(config_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported planner repair audit schema")
    if contract.get("kind") != "wcd-drs-horizon-planner-repair-audit":
        raise ValueError("unexpected planner repair audit kind")
    if contract.get("status") != "registered-development-audit":
        raise ValueError("planner repair audit must remain a development audit")
    source = contract.get("source") or {}
    if source.get("records") != 128 or source.get("max_new_tokens") != 48:
        raise ValueError("planner repair audit population or horizon changed")
    if source.get("known_false_negative_rows") != [
        "data/en/test/standard.sbn:76",
        "data/en/test/standard.sbn:96",
    ]:
        raise ValueError("known false-negative row identities changed")
    if source.get("known_false_negative_events") != 3:
        raise ValueError("known false-negative event count changed")
    repair = contract.get("repair") or {}
    if any(
        repair.get(name) is not False
        for name in (
            "trusted_core_changed",
            "state_transition_changed",
            "release_policy_changed",
            "new_semantic_rules_added",
        )
    ):
        raise ValueError("registered repair scope changed")
    if contract.get("audit", {}).get("performance_conditioned_selection") is not False:
        raise ValueError("repair audit must retain all W rows")
    resources = contract.get("resource_contract") or {}
    if resources.get("model_weights_loaded") is not False:
        raise ValueError("repair audit must remain model-free")
    if resources.get("gpu_used") is not False:
        raise ValueError("repair audit must remain CPU-only")
    if resources.get("cpu_math_threads") != 1:
        raise ValueError("repair audit must remain one-threaded")
    return contract


def verify_registered_path(evidence: dict[str, str]) -> Path:
    path = resolve_repository_path(evidence["path"])
    if not path.is_file():
        raise FileNotFoundError(f"missing registered evidence: {path}")
    if sha256_path(path) != evidence["sha256"]:
        raise ValueError(f"registered evidence hash changed: {path}")
    return path


def summarize_outcomes(
    outcomes: list[dict[str, Any]], known_rows: set[str]
) -> dict[str, Any]:
    repaired = {
        outcome["id"]
        for outcome in outcomes
        if outcome["id"] in known_rows
        and outcome["planner_accepts_entire_path"] is True
    }
    return {
        "rows": len(outcomes),
        "auditable_rows": sum(int(outcome["auditable"]) for outcome in outcomes),
        "source_path_replay_accepting_rows": sum(
            int(outcome.get("source_path_replay_accepting") is True)
            for outcome in outcomes
        ),
        "planner_accepts_entire_path_rows": sum(
            int(outcome["planner_accepts_entire_path"] is True)
            for outcome in outcomes
        ),
        "rows_with_constructive_planner_false_negative": sum(
            int(bool(outcome["false_negative_events"])) for outcome in outcomes
        ),
        "constructive_planner_false_negative_events": sum(
            len(outcome["false_negative_events"]) for outcome in outcomes
        ),
        "other_rejection_events": sum(
            len(outcome["other_rejection_events"]) for outcome in outcomes
        ),
        "known_false_negative_rows_repaired": sorted(repaired),
    }


def build_report(
    *,
    contract: dict[str, Any],
    config_path: Path,
    tokenizer: Any,
    transformers_version: str,
) -> dict[str, Any]:
    source = contract["source"]
    manifest_path = verify_registered_path(source["pre_fix_audit_manifest"])
    pre_contract_path = verify_registered_path(source["pre_fix_audit_contract"])
    pre_report_path = verify_registered_path(source["pre_fix_audit_report"])
    pre_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pre_report = json.loads(pre_report_path.read_text(encoding="utf-8"))
    expected_pre = {
        "rows": 128,
        "planner_accepts_entire_path_rows": 126,
        "rows_with_constructive_planner_false_negative": 2,
        "constructive_planner_false_negative_events": 3,
        "other_rejection_events": 0,
    }
    for key, value in expected_pre.items():
        if pre_report.get("summary", {}).get(key) != value:
            raise ValueError(f"pre-fix summary changed at {key}")
    if pre_manifest.get("observed_summary", {}).get(
        "constructive_planner_false_negative_events"
    ) != 3:
        raise ValueError("pre-fix implementation manifest is inconsistent")

    pre_contract = pre_fix_audit.load_contract(pre_contract_path)
    paths = pre_fix_audit.verify_registered_evidence(pre_contract)
    rows, _ = pre_fix_audit.validate_source(pre_contract, paths)
    vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
    w_rows = [row for row in rows if row["configuration"] == "W"]
    outcomes = [
        pre_fix_audit.audit_w_path(
            row,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            horizon=int(source["max_new_tokens"]),
        )
        for row in w_rows
    ]
    known_rows = set(source["known_false_negative_rows"])
    summary = summarize_outcomes(outcomes, known_rows)
    expected_gate = contract["gate"]
    observed_gate = {
        "all_128_W_rows_auditable": summary["auditable_rows"] == 128,
        "all_128_source_paths_replay_accepting": (
            summary["source_path_replay_accepting_rows"] == 128
        ),
        "planner_accepts_entire_path_rows": summary[
            "planner_accepts_entire_path_rows"
        ],
        "constructive_planner_false_negative_rows": summary[
            "rows_with_constructive_planner_false_negative"
        ],
        "constructive_planner_false_negative_events": summary[
            "constructive_planner_false_negative_events"
        ],
        "other_rejection_events": summary["other_rejection_events"],
        "both_known_failure_rows_repaired": set(
            summary["known_false_negative_rows_repaired"]
        )
        == known_rows,
    }
    failures = {
        key: {"expected": value, "observed": observed_gate.get(key)}
        for key, value in expected_gate.items()
        if observed_gate.get(key) != value
    }
    repair_path = resolve_repository_path(contract["repair"]["implementation"])
    test_path = resolve_repository_path(contract["repair"]["unit_test"])
    if sha256_path(repair_path) != contract["repair"][
        "implementation_sha256_at_registration"
    ]:
        failures["registered_repair_implementation"] = {
            "expected": contract["repair"]["implementation_sha256_at_registration"],
            "observed": sha256_path(repair_path),
        }
    if sha256_path(test_path) != contract["repair"][
        "unit_test_sha256_at_registration"
    ]:
        failures["registered_repair_unit_test"] = {
            "expected": contract["repair"]["unit_test_sha256_at_registration"],
            "observed": sha256_path(test_path),
        }
    known_cases = [
        outcome for outcome in outcomes if outcome["id"] in known_rows
    ]
    return {
        "schema_version": 1,
        "kind": "wcd-drs-horizon-planner-repair-audit-report",
        "status": "passed" if not failures else "failed",
        "contract": {
            "path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": sha256_path(config_path),
            "registration_boundary": contract["registration_boundary"],
        },
        "implementation": {
            "audit": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "audit_sha256": sha256_path(Path(__file__).resolve()),
            "pre_fix_audit_helper": str(
                Path(pre_fix_audit.__file__).resolve().relative_to(REPOSITORY_ROOT)
            ),
            "pre_fix_audit_helper_sha256": sha256_path(
                Path(pre_fix_audit.__file__).resolve()
            ),
            "planner": str(repair_path.relative_to(REPOSITORY_ROOT)),
            "planner_sha256": sha256_path(repair_path),
            "unit_test": str(test_path.relative_to(REPOSITORY_ROOT)),
            "unit_test_sha256": sha256_path(test_path),
        },
        "tokenizer": {
            "adapter": str(paths["adapter"].relative_to(REPOSITORY_ROOT)),
            "transformers": transformers_version,
            "vocabulary_fragments": len(vocabulary.fragments),
            "raw_byte_fallbacks": len(vocabulary.byte_token_ids),
        },
        "pre_fix_summary": pre_report["summary"],
        "post_fix_summary": summary,
        "known_cases": known_cases,
        "gate": {
            "passed": not failures,
            "failures": failures,
            "observed": observed_gate,
        },
        "integrity": {
            "all_W_rows_retained": len(outcomes) == 128,
            "performance_conditioned_selection": False,
            "model_weights_loaded": False,
            "gpu_used": False,
            "development_not_confirmatory": True,
            "claim_boundary": contract["claim_boundary"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config_path = args.config.resolve()
    contract = load_contract(config_path)
    pre_contract_path = verify_registered_path(
        contract["source"]["pre_fix_audit_contract"]
    )
    pre_contract = pre_fix_audit.load_contract(pre_contract_path)
    paths = pre_fix_audit.verify_registered_evidence(pre_contract)
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error
    tokenizer = AutoTokenizer.from_pretrained(paths["adapter"], local_files_only=True)
    report = build_report(
        contract=contract,
        config_path=config_path,
        tokenizer=tokenizer,
        transformers_version=transformers.__version__,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if len(rendered.encode("utf-8")) > int(
        contract["resource_contract"]["maximum_output_bytes"]
    ):
        raise ValueError("planner repair audit report exceeds its output ceiling")
    if args.verify_existing is not None:
        if args.verify_existing.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"stored repair audit differs: {args.verify_existing}")
        print(f"verified planner repair audit: {args.verify_existing}")
        return 0
    assert args.output is not None
    if args.output.exists():
        parser.error(f"refusing to overwrite existing report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["post_fix_summary"], ensure_ascii=False, sort_keys=True))
    print(f"planner repair gate passed={report['gate']['passed']}")
    if not report["gate"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
