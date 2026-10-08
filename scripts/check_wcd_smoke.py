#!/usr/bin/env python3
"""Fail-closed smoke gate for WCD generation on DRS or AMR JSONL rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "configs" / "wcd_experiment_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def raw_validity(row: dict[str, Any]) -> bool:
    validity = row.get("validity")
    if not isinstance(validity, dict):
        return False
    if row.get("task") == "amr" or "raw_local_penman" in validity:
        return (
            validity.get("raw_local_penman") is True
            and validity.get("raw_official_penman") is True
        )
    valid = validity.get("raw_sbn") is True
    if row.get("configuration") == "WWCD":
        valid = valid and validity.get("raw_clf_referee") is True
    return valid


def build_report(
    predictions: Path,
    *,
    configurations: tuple[str, ...],
    expected_records_per_configuration: int,
    score_report: Path | None = None,
    sbn_scores: Path | None = None,
    expected_sbn_scorer_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_records_per_configuration <= 0:
        raise ValueError("expected record count must be positive")
    if not configurations or len(set(configurations)) != len(configurations):
        raise ValueError("configurations must be non-empty and unique")
    rows = load_jsonl(predictions)
    predictions_digest = sha256_file(predictions)
    expected = set(configurations)
    observed = {str(row.get("configuration")) for row in rows}
    failures: list[str] = []
    manifest_path = Path(str(predictions) + ".manifest.json")
    manifest: dict[str, Any] | None = None
    if not manifest_path.is_file():
        failures.append("missing-generation-manifest")
    else:
        manifest = load_object(manifest_path)
        if manifest.get("status") != "complete":
            failures.append("generation-manifest-not-complete")
        if manifest.get("output_sha256") != predictions_digest:
            failures.append("generation-manifest-output-hash")
        if manifest.get("rows") != len(rows):
            failures.append("generation-manifest-row-count")
        manifest_configurations = manifest.get("configurations")
        if not isinstance(manifest_configurations, list) or (
            set(manifest_configurations) != expected
        ):
            failures.append("generation-manifest-configurations")
    if observed != expected:
        failures.append("configuration-set-mismatch")

    identities: set[tuple[str, str]] = set()
    summaries: dict[str, dict[str, int]] = {}
    for configuration in configurations:
        selected = [
            row for row in rows if row.get("configuration") == configuration
        ]
        summary = {
            "rows": len(selected),
            "raw_valid_rows": 0,
            "processor_rows": 0,
            "all_masked_rows": 0,
            "prefix_rows_certified": 0,
            "prefix_rows_unavailable": 0,
            "carried_tokens": 0,
            "cached_tokens": 0,
            "replanned_tokens": 0,
            "rejected_without_certificate": 0,
            "allowed_total": 0,
        }
        if len(selected) != expected_records_per_configuration:
            failures.append(f"{configuration}:row-count")
        for row in selected:
            identity = (configuration, str(row.get("id")))
            if identity in identities:
                failures.append(f"{configuration}:duplicate-id")
            identities.add(identity)
            if raw_validity(row):
                summary["raw_valid_rows"] += 1
            constraint = row.get("constraint")
            if not isinstance(constraint, dict):
                failures.append(f"{configuration}:missing-constraint")
                continue
            for key in (
                "rows",
                "all_masked_rows",
                "prefix_rows_certified",
                "prefix_rows_unavailable",
                "carried_tokens",
                "cached_tokens",
                "replanned_tokens",
                "rejected_without_certificate",
                "allowed_total",
            ):
                value = constraint.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    failures.append(f"{configuration}:invalid-{key}")
                    continue
                target = "processor_rows" if key == "rows" else key
                summary[target] += value

        if summary["raw_valid_rows"] != len(selected):
            failures.append(f"{configuration}:raw-validity")
        if summary["all_masked_rows"] != 0:
            failures.append(f"{configuration}:all-masked")
        if summary["prefix_rows_unavailable"] != 0:
            failures.append(f"{configuration}:prefix-unavailable")
        if summary["prefix_rows_certified"] != summary["processor_rows"]:
            failures.append(f"{configuration}:prefix-accounting")
        admitted = (
            summary["carried_tokens"]
            + summary["cached_tokens"]
            + summary["replanned_tokens"]
        )
        if admitted != summary["allowed_total"]:
            failures.append(f"{configuration}:decision-accounting")
        if selected and admitted == 0:
            failures.append(f"{configuration}:no-wcd-activity")
        summaries[configuration] = summary

    score_binding: dict[str, Any] | None = None
    if score_report is not None:
        score = load_object(score_report)
        scorer = score.get("scorer")
        score_gate = score.get("gate")
        score_predictions = score.get("predictions")
        score_groups = score.get("groups")
        if not isinstance(scorer, dict) or (
            scorer.get("name") != "smatchpp"
            or scorer.get("version") != "1.8.0"
            or scorer.get("solver") != "ilp"
            or scorer.get("optimality_required") is not True
        ):
            failures.append("score-report-not-frozen-smatchpp")
        if not isinstance(score_gate, dict) or (
            score_gate.get("passed") is not True
            or score_gate.get("all_alignments_optimal") is not True
        ):
            failures.append("score-report-gate")
        if not isinstance(score_predictions, dict) or (
            score_predictions.get("sha256") != predictions_digest
            or score_predictions.get("rows") != len(rows)
        ):
            failures.append("score-report-predictions-binding")
        if not isinstance(score_groups, dict) or set(score_groups) != expected:
            failures.append("score-report-configurations")
        score_binding = {
            "path": str(score_report),
            "sha256": sha256_file(score_report),
            "scorer": None if not isinstance(scorer, dict) else scorer,
        }

    sbn_score_binding: dict[str, Any] | None = None
    semantic_scores: dict[str, float] | None = None
    if sbn_scores is not None:
        if not expected_sbn_scorer_sha256:
            failures.append("missing-expected-sbn-scorer-sha256")
        score_rows = load_jsonl(sbn_scores)
        score_manifest_path = Path(str(sbn_scores) + ".manifest.json")
        if not score_manifest_path.is_file():
            failures.append("missing-sbn-score-manifest")
            score_manifest: dict[str, Any] = {}
        else:
            score_manifest = load_object(score_manifest_path)
        score_identities = [
            (str(row.get("configuration")), str(row.get("id")))
            for row in score_rows
        ]
        if len(set(score_identities)) != len(score_identities) or (
            set(score_identities) != identities
        ):
            failures.append("sbn-score-identities")
        if score_manifest.get("input_sha256") != predictions_digest:
            failures.append("sbn-score-predictions-binding")
        if score_manifest.get("output_sha256") != sha256_file(sbn_scores):
            failures.append("sbn-score-output-binding")
        if score_manifest.get("records") != len(rows):
            failures.append("sbn-score-row-count")
        if (
            score_manifest.get("scoring_protocol")
            != "seeded-deduplicated-semantic-pairs-v2"
            or score_manifest.get("seed") != 42
            or score_manifest.get("identical_pair_score_consistency") is not True
        ):
            failures.append("sbn-score-protocol")
        if score_manifest.get("scorer_sha256") != expected_sbn_scorer_sha256:
            failures.append("sbn-scorer-hash")
        grouped_values: dict[str, list[float]] = {
            configuration: [] for configuration in configurations
        }
        for row in score_rows:
            value = row.get("semantic_scores", {}).get("sbn_smatch")
            configuration = str(row.get("configuration"))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                or configuration not in grouped_values
            ):
                failures.append("invalid-sbn-score-value")
                continue
            grouped_values[configuration].append(float(value))
        semantic_scores = {
            configuration: sum(values) / len(values)
            for configuration, values in grouped_values.items()
            if values
        }
        if set(semantic_scores) != expected:
            failures.append("sbn-score-configurations")
        sbn_score_binding = {
            "path": str(sbn_scores),
            "sha256": sha256_file(sbn_scores),
            "manifest": str(score_manifest_path),
            "manifest_sha256": (
                sha256_file(score_manifest_path)
                if score_manifest_path.is_file()
                else None
            ),
            "scorer_sha256": score_manifest.get("scorer_sha256"),
            "protocol": score_manifest.get("scoring_protocol"),
            "seed": score_manifest.get("seed"),
        }

    inputs = {
        "predictions": str(predictions),
        "predictions_sha256": predictions_digest,
        "manifest": str(manifest_path),
        "manifest_sha256": (
            None if manifest is None else sha256_file(manifest_path)
        ),
        "registration": str(REGISTRATION.relative_to(ROOT)),
        "registration_sha256": sha256_file(REGISTRATION),
        "configurations": list(configurations),
        "expected_records_per_configuration": (
            expected_records_per_configuration
        ),
        "score_report": score_binding,
    }
    if sbn_scores is not None:
        inputs["sbn_scores"] = sbn_score_binding
        inputs["expected_sbn_scorer_sha256"] = expected_sbn_scorer_sha256

    report = {
        "schema_version": 1,
        "kind": "wcd-smoke-gate",
        "status": "passed" if not failures else "failed",
        "gate": {"passed": not failures, "failures": sorted(set(failures))},
        "inputs": inputs,
        "summaries": summaries,
    }
    if semantic_scores is not None:
        report["semantic_scores"] = {
            "metric": "sbn_smatch",
            "means": semantic_scores,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--configurations", nargs="+", required=True)
    parser.add_argument(
        "--expected-records-per-configuration", type=int, required=True
    )
    parser.add_argument("--score-report", type=Path)
    parser.add_argument("--sbn-scores", type=Path)
    parser.add_argument("--expected-sbn-scorer-sha256")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.predictions,
        configurations=tuple(args.configurations),
        expected_records_per_configuration=(
            args.expected_records_per_configuration
        ),
        score_report=args.score_report,
        sbn_scores=args.sbn_scores,
        expected_sbn_scorer_sha256=args.expected_sbn_scorer_sha256,
    )
    if args.verify_existing is not None:
        existing = json.loads(args.verify_existing.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit("existing WCD smoke gate is not reproducible")
    else:
        assert args.output is not None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as sink:
            json.dump(report, sink, ensure_ascii=False, indent=2, sort_keys=True)
            sink.write("\n")
    if not report["gate"]["passed"]:
        raise SystemExit(
            "WCD smoke gate failed: "
            + ", ".join(report["gate"]["failures"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
