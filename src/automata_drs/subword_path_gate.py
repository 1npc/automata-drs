"""Integrity gate for selected-host gold subword constraint-path audits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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


def evaluate_selected_subword_path_audit(
    config: dict[str, Any],
    model_row: dict[str, Any],
    *,
    split: str,
    corpus_path: Path,
    audit_path: Path,
    runs_root: Path,
) -> dict[str, Any]:
    """Check coverage, provenance, activation, and bounded memory evidence."""

    if split not in {"train", "dev", "test"}:
        raise ValueError("selected-host path gate supports train, dev, or test")
    if not corpus_path.is_file():
        raise ValueError(f"missing path-audit corpus: {corpus_path}")
    if not audit_path.is_file():
        raise ValueError(f"missing selected-host path audit: {audit_path}")
    audit = _read_object(audit_path)
    pilot = config.get("method_experiment", {}).get("pilot", {})
    key = str(model_row["key"])
    training_manifest_path: Path | None = None
    if split == "train":
        training_manifest_path = (
            runs_root / f"{key}-r16-seed42" / "train_manifest.json"
        )
        training_manifest = _read_object(training_manifest_path)
        training_population = training_manifest.get("train")
        if not isinstance(training_population, dict):
            raise ValueError("selected-host training manifest has no train population")
        expected_records = int(training_population.get("records", 0))
        if training_manifest.get("train_path") != str(corpus_path):
            raise ValueError(
                "selected-host training manifest is not bound to the audit corpus"
            )
    elif split == "dev":
        expected_records = int(
            config.get("host_selection", {}).get("required_records", 0)
        )
    else:
        matrices = [
            row
            for row in config.get("method_experiment", {}).get(
                "publication_matrices", []
            )
            if row.get("split") == split
        ]
        if len(matrices) != 1:
            raise ValueError(f"missing or duplicate publication matrix: {split}")
        expected_records = int(matrices[0].get("records", 0))
    expected_max_new_tokens = int(pilot.get("max_new_tokens", 0))
    if expected_records <= 0 or expected_max_new_tokens <= 0:
        raise ValueError("configuration has no valid path-audit contract")
    expected_tokenizer = str(runs_root / f"{key}-r16-seed42" / "adapter")
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

    check("corpus_path", audit.get("path") == str(corpus_path), audit.get("path"), str(corpus_path))
    corpus_sha = _sha256(corpus_path)
    check("corpus_sha256", audit.get("input_sha256") == corpus_sha, audit.get("input_sha256"), corpus_sha)
    check("selected_tokenizer", audit.get("tokenizer") == expected_tokenizer, audit.get("tokenizer"), expected_tokenizer)
    check(
        "generation_budget",
        audit.get("max_new_tokens") == expected_max_new_tokens,
        audit.get("max_new_tokens"),
        expected_max_new_tokens,
    )
    check(
        "record_contract",
        audit.get("expected_records") == expected_records,
        audit.get("expected_records"),
        expected_records,
    )
    check("raw_byte_fallbacks", audit.get("raw_byte_fallbacks") == 256, audit.get("raw_byte_fallbacks"), 256)

    reports = audit.get("reports")
    report_map = {
        str(report.get("configuration")): report
        for report in reports
        if isinstance(report, dict)
    } if isinstance(reports, list) else {}
    check("configuration_matrix", set(report_map) == {"W", "WF"}, sorted(report_map), ["W", "WF"])
    coverage_failures: list[dict[str, Any]] = []
    memory_failures: list[dict[str, Any]] = []
    for configuration in ("W", "WF"):
        report = report_map.get(configuration)
        if report is None:
            coverage_failures.append({"configuration": configuration, "reason": "missing"})
            continue
        eligible = report.get("eligible")
        accepted = report.get("accepted")
        false_rejections = report.get("false_rejection_count")
        if (
            report.get("records") != expected_records
            or not isinstance(eligible, int)
            or isinstance(eligible, bool)
            or eligible <= 0
            or accepted != eligible
            or false_rejections != 0
            or report.get("false_rejection_examples") != []
        ):
            coverage_failures.append(
                {
                    "configuration": configuration,
                    "records": report.get("records"),
                    "eligible": eligible,
                    "accepted": accepted,
                    "false_rejection_count": false_rejections,
                }
            )
        peak = report.get("peak_prefix_cache_entries")
        transition = report.get("transition_cache")
        completion = report.get("completion_cache")
        memory_ok = (
            isinstance(peak, int)
            and not isinstance(peak, bool)
            and 0 < peak <= expected_max_new_tokens + 1
            and isinstance(transition, dict)
            and isinstance(transition.get("size"), int)
            and 0 <= transition["size"] <= 100_000
            and isinstance(completion, dict)
            and isinstance(completion.get("size"), int)
            and 0 <= completion["size"] <= 100_000
        )
        if not memory_ok:
            memory_failures.append(
                {
                    "configuration": configuration,
                    "peak_prefix_cache_entries": peak,
                    "transition_cache": transition,
                    "completion_cache": completion,
                }
            )
    check("zero_gold_path_false_rejections", not coverage_failures, coverage_failures, [])
    check("bounded_audit_memory", not memory_failures, memory_failures, [])
    wf_completion = report_map.get("WF", {}).get("completion_cache", {})
    check(
        "wf_certificate_planner_activated",
        isinstance(wf_completion, dict)
        and isinstance(wf_completion.get("misses"), int)
        and wf_completion.get("misses", 0) > 0
        and wf_completion.get("size", 0) > 0,
        wf_completion,
        "positive completion-cache misses and entries",
    )
    gate = audit.get("gate", {})
    check(
        "audit_fail_closed_gate",
        isinstance(gate, dict)
        and gate.get("require_zero_false_rejections") is True
        and gate.get("passed") is True,
        gate,
        {"require_zero_false_rejections": True, "passed": True},
    )
    evidence = {
        "audit_path": str(audit_path),
        "audit_sha256": _sha256(audit_path),
        "corpus_sha256": corpus_sha,
        "records": expected_records,
        "max_new_tokens": expected_max_new_tokens,
    }
    if training_manifest_path is not None:
        evidence["training_manifest_path"] = str(training_manifest_path)
        evidence["training_manifest_sha256"] = _sha256(training_manifest_path)
    return {
        "schema_version": 1,
        "kind": "selected-subword-path-audit-gate",
        "model_key": key,
        "model": model_row["repo_id"],
        "revision": model_row["revision"],
        "split": split,
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "evidence": evidence,
    }


def verify_frozen_selected_subword_path_audit(
    current: dict[str, Any],
    stored: dict[str, Any],
    *,
    config_sha256: str,
    selection_sha256: str,
) -> None:
    """Require a stored gate to equal recomputed path-audit evidence."""

    if current.get("passed") is not True:
        raise ValueError("current selected-host path-audit evidence does not pass")
    if stored.get("schema_version") != 1 or stored.get("passed") is not True:
        raise ValueError("stored selected-host path-audit gate is unsupported or failed")
    if not isinstance(stored.get("checked_at_utc"), str):
        raise ValueError("stored selected-host path-audit gate has no timestamp")
    expected = json.loads(json.dumps(current, ensure_ascii=False))
    expected["config_sha256"] = config_sha256
    expected["selection_sha256"] = selection_sha256
    observed = {key: value for key, value in stored.items() if key != "checked_at_utc"}
    if observed != expected:
        differing = sorted(
            key
            for key in set(observed) | set(expected)
            if observed.get(key) != expected.get(key)
        )
        raise ValueError(
            "stored selected-host path-audit gate differs from current evidence: "
            + ", ".join(differing)
        )


__all__ = [
    "evaluate_selected_subword_path_audit",
    "verify_frozen_selected_subword_path_audit",
]
