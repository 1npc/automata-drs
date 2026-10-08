#!/usr/bin/env python3
"""Freeze cross-representation WCD smoke evidence with immutable hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSON objects: {path}")
    return rows


def bind_generation(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(path)
    digest = sha256_file(path)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = load_object(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"generation manifest is not complete: {manifest_path}")
    if manifest.get("output_sha256") != digest or manifest.get("rows") != len(rows):
        raise ValueError(f"generation manifest does not bind output: {manifest_path}")
    return rows, {
        "path": str(path),
        "sha256": digest,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": len(rows),
    }


def bind_gate(path: Path, predictions_sha256: str) -> dict[str, Any]:
    gate = load_object(path)
    if gate.get("gate", {}).get("passed") is not True:
        raise ValueError(f"WCD gate did not pass: {path}")
    if gate.get("inputs", {}).get("predictions_sha256") != predictions_sha256:
        raise ValueError(f"WCD gate does not bind predictions: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def bind_sbn_scores(
    path: Path,
    *,
    predictions_sha256: str,
    expected_identities: set[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, Any]]:
    rows = load_jsonl(path)
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = load_object(manifest_path)
    identities = [(str(row.get("id")), str(row.get("configuration"))) for row in rows]
    if len(set(identities)) != len(identities) or set(identities) != expected_identities:
        raise ValueError("SBN score identities do not match predictions")
    if manifest.get("input_sha256") != predictions_sha256:
        raise ValueError("SBN score manifest does not bind predictions")
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError("SBN score manifest does not bind score output")
    if manifest.get("records") != len(rows):
        raise ValueError("SBN score manifest record count differs")
    if manifest.get("scoring_protocol") != "seeded-deduplicated-semantic-pairs-v2":
        raise ValueError("SBN scores do not use the frozen deduplicated protocol")
    if manifest.get("seed") != 42 or manifest.get("identical_pair_score_consistency") is not True:
        raise ValueError("SBN score seed or duplicate consistency differs")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("semantic_scores", {}).get("sbn_smatch")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("invalid SBN score value")
        grouped[str(row["configuration"])].append(float(value))
    means = {name: sum(values) / len(values) for name, values in grouped.items()}
    return means, {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scorer_sha256": manifest.get("scorer_sha256"),
        "protocol": manifest.get("scoring_protocol"),
        "seed": manifest.get("seed"),
    }


def bind_amr_scores(
    path: Path, predictions_sha256: str
) -> tuple[dict[str, float], dict[str, Any]]:
    report = load_object(path)
    scorer = report.get("scorer", {})
    gate = report.get("gate", {})
    if (
        scorer.get("name") != "smatchpp"
        or scorer.get("version") != "1.8.0"
        or scorer.get("solver") != "ilp"
        or scorer.get("optimality_required") is not True
    ):
        raise ValueError("AMR scores do not use frozen Smatch++ ILP")
    if gate.get("passed") is not True or gate.get("all_alignments_optimal") is not True:
        raise ValueError("AMR scoring gate or alignment optimality failed")
    if report.get("predictions", {}).get("sha256") != predictions_sha256:
        raise ValueError("AMR score report does not bind predictions")
    means = {
        name: float(group["raw_primary_smatch"]["f1"])
        for name, group in report.get("groups", {}).items()
    }
    return means, {
        "path": str(path),
        "sha256": sha256_file(path),
        "scorer": scorer,
        "all_alignments_optimal": True,
    }


def group_summary(
    rows: list[dict[str, Any]],
    predecessor_rows: list[dict[str, Any]],
    *,
    configuration: str,
    predecessor_configuration: str,
    semantic_score: float,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("configuration") == configuration]
    predecessor = {
        (str(row.get("id")), str(row.get("configuration"))): row
        for row in predecessor_rows
    }
    counter_keys = (
        "rows",
        "all_masked_rows",
        "prefix_rows_certified",
        "prefix_rows_unavailable",
        "carried_tokens",
        "cached_tokens",
        "replanned_tokens",
        "rejected_without_certificate",
        "allowed_total",
    )
    counters = {
        key: sum(int(row["constraint"][key]) for row in selected)
        for key in counter_keys
    }
    exact = 0
    predecessor_generation_ms = 0.0
    predecessor_constraint_ms = 0.0
    for row in selected:
        key = (str(row["id"]), predecessor_configuration)
        if key not in predecessor:
            raise ValueError(f"missing paired predecessor row: {key}")
        paired = predecessor[key]
        exact += row.get("raw_prediction") == paired.get("raw_prediction")
        predecessor_generation_ms += float(paired["timing"]["generation_ms"])
        predecessor_constraint_ms += float(paired["timing"]["constraint_ms"])
    return {
        "configuration": configuration,
        "predecessor_configuration": predecessor_configuration,
        "records": len(selected),
        "raw_valid_records": sum(
            row.get("validity", {}).get("raw_sbn", row.get("validity", {}).get("raw_official_penman"))
            is True
            for row in selected
        ),
        "exact_raw_prediction_matches_predecessor": exact,
        "semantic_score": semantic_score,
        "generation_seconds": sum(float(row["timing"]["generation_ms"]) for row in selected) / 1000,
        "predecessor_generation_seconds": predecessor_generation_ms / 1000,
        "constraint_seconds": sum(float(row["timing"]["constraint_ms"]) for row in selected) / 1000,
        "predecessor_constraint_seconds": predecessor_constraint_ms / 1000,
        "constraint": counters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "drs_predictions", "drs_scores", "drs_gate", "drs_predecessor",
        "amr_predictions", "amr_scores", "amr_gate", "amr_predecessor",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite evidence: {args.output}")

    drs_rows, drs_input = bind_generation(args.drs_predictions)
    drs_predecessor_rows, drs_predecessor_input = bind_generation(args.drs_predecessor)
    drs_gate = bind_gate(args.drs_gate, drs_input["sha256"])
    identities = {(str(row["id"]), str(row["configuration"])) for row in drs_rows}
    drs_means, drs_scores = bind_sbn_scores(
        args.drs_scores,
        predictions_sha256=drs_input["sha256"],
        expected_identities=identities,
    )

    amr_rows, amr_input = bind_generation(args.amr_predictions)
    amr_predecessor_rows, amr_predecessor_input = bind_generation(args.amr_predecessor)
    amr_gate = bind_gate(args.amr_gate, amr_input["sha256"])
    amr_means, amr_scores = bind_amr_scores(args.amr_scores, amr_input["sha256"])

    drs_groups = [
        group_summary(drs_rows, drs_predecessor_rows, configuration="BWCD", predecessor_configuration="BF", semantic_score=drs_means["BWCD"]),
        group_summary(drs_rows, drs_predecessor_rows, configuration="WWCD", predecessor_configuration="WF", semantic_score=drs_means["WWCD"]),
    ]
    amr_groups = [
        group_summary(amr_rows, amr_predecessor_rows, configuration="WCD", predecessor_configuration="PF", semantic_score=amr_means["WCD"])
    ]
    all_groups = drs_groups + amr_groups
    structural_pass = all(
        group["raw_valid_records"] == group["records"]
        and group["constraint"]["all_masked_rows"] == 0
        and group["constraint"]["prefix_rows_unavailable"] == 0
        and group["constraint"]["prefix_rows_certified"] == group["constraint"]["rows"]
        for group in all_groups
    )
    behavioral_equivalence = all(
        group["exact_raw_prediction_matches_predecessor"] == group["records"]
        for group in all_groups
    )
    report = {
        "schema_version": 1,
        "kind": "wcd-cross-representation-smoke-evidence",
        "inputs": {
            "drs_predictions": drs_input,
            "drs_predecessor": drs_predecessor_input,
            "drs_gate": drs_gate,
            "drs_scores": drs_scores,
            "amr_predictions": amr_input,
            "amr_predecessor": amr_predecessor_input,
            "amr_gate": amr_gate,
            "amr_scores": amr_scores,
        },
        "drs": {"groups": drs_groups},
        "amr": {"groups": amr_groups},
        "promotion_gate": {
            "structural_pass": structural_pass,
            "behaviorally_equivalent_to_frozen_predecessors": behavioral_equivalence,
            "passed": structural_pass and behavioral_equivalence,
        },
        "claim_boundary": {
            "supported": "The shared WCD controller preserved an accepting certificate at every observed decoding row and matched frozen predecessor outputs on both representations.",
            "not_supported": "These smoke samples do not establish a semantic-quality improvement or population-level efficiency improvement.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        json.dump(report, sink, ensure_ascii=False, indent=2, sort_keys=True)
        sink.write("\n")
    print(f"promotion gate passed: {report['promotion_gate']['passed']}")
    return 0 if report["promotion_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
