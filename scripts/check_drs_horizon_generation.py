#!/usr/bin/env python3
"""Fail-closed gate for the held-out DRS horizon generation matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


CONFIGURATIONS = ("W", "WF", "WWCD")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def numeric(row: dict[str, Any], *keys: str) -> float | None:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def integer(row: dict[str, Any], *keys: str) -> int | None:
    value = numeric(row, *keys)
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def build_report(
    predictions_path: Path,
    *,
    config_path: Path,
    score_path: Path,
) -> dict[str, Any]:
    contract = load_object(config_path)
    if contract.get("schema_version") != 1 or contract.get("kind") != (
        "wcd-drs-heldout-horizon-generation"
    ):
        raise ValueError("unexpected held-out horizon contract")
    generation_contract = contract.get("generation") or {}
    data_contract = contract.get("data") or {}
    if generation_contract.get("configurations") != list(CONFIGURATIONS):
        raise ValueError("held-out horizon configurations changed")
    if generation_contract.get("max_new_tokens") != 48:
        raise ValueError("held-out horizon budget changed")
    selection = data_contract.get("selection")
    if selection not in {
        "first-128-source-order",
        "fixed-128-source-order-window",
    }:
        raise ValueError("held-out population selection changed")
    if data_contract.get("selection_performance_conditioned") is not False:
        raise ValueError("held-out population is performance-conditioned")
    start = data_contract.get("start")
    limit = data_contract.get("limit")
    if not isinstance(start, int) or start < 0 or limit != 128:
        raise ValueError("held-out source-order window changed")
    if selection == "first-128-source-order" and start != 0:
        raise ValueError("first-128 selection must start at zero")

    failures: list[str] = []
    predictions = load_jsonl(predictions_path)
    prediction_sha = sha256_path(predictions_path)
    manifest_path = Path(str(predictions_path) + ".manifest.json")
    if not manifest_path.is_file():
        failures.append("missing-generation-manifest")
        manifest: dict[str, Any] = {}
    else:
        manifest = load_object(manifest_path)
    expected_manifest = {
        "status": "complete",
        "model": contract["model"]["repo_id"],
        "requested_revision": contract["model"]["revision"],
        "resolved_revision": contract["model"]["revision"],
        "start": start,
        "limit": limit,
        "max_input_tokens": 512,
        "max_new_tokens": 48,
        "num_beams": 1,
        "batch_size": 1,
        "configurations": list(CONFIGURATIONS),
        "schedule": "rotate",
        "expected_examples": 128,
        "expected_rows": 384,
        "examples": 128,
        "rows": 384,
        "output_sha256": prediction_sha,
    }
    manifest_mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if manifest_mismatches:
        failures.append("generation-manifest-contract")
    if manifest.get("physical_gpu_index") not in {"0", "1"}:
        failures.append("generation-physical-gpu")
    if len(predictions) != 384:
        failures.append("prediction-row-count")

    counts = Counter(str(row.get("configuration")) for row in predictions)
    if counts != Counter({name: 128 for name in CONFIGURATIONS}):
        failures.append("prediction-configuration-counts")
    identities = [(str(row.get("configuration")), str(row.get("id"))) for row in predictions]
    if len(set(identities)) != len(identities):
        failures.append("duplicate-prediction-identity")
    id_lists = {
        configuration: [
            str(row.get("id"))
            for row in predictions
            if row.get("configuration") == configuration
        ]
        for configuration in CONFIGURATIONS
    }
    if any(ids != id_lists[CONFIGURATIONS[0]] for ids in id_lists.values()):
        failures.append("unpaired-or-reordered-identifiers")

    summaries: dict[str, dict[str, Any]] = {}
    by_identity = {
        (str(row.get("id")), str(row.get("configuration"))): row
        for row in predictions
    }
    for configuration in CONFIGURATIONS:
        selected = [
            row for row in predictions if row.get("configuration") == configuration
        ]
        raw_sbn = sum(
            row.get("validity", {}).get("raw_sbn") is True for row in selected
        )
        raw_referee = sum(
            row.get("validity", {}).get("raw_clf_referee") is True
            for row in selected
        )
        ended_eos = sum(
            row.get("generation", {}).get("ended_with_eos") is True
            for row in selected
        )
        hit_max = sum(
            row.get("generation", {}).get("hit_max_new_tokens") is True
            for row in selected
        )
        all_masked = sum(
            integer(row, "constraint", "all_masked_rows") or 0
            for row in selected
        )
        processor_rows = sum(
            integer(row, "constraint", "rows") or 0 for row in selected
        )
        generation_values = [
            value
            for row in selected
            if (value := numeric(row, "timing", "generation_ms")) is not None
        ]
        constraint_values = [
            value
            for row in selected
            if (value := numeric(row, "timing", "constraint_ms")) is not None
        ]
        output_tokens = [
            value
            for row in selected
            if (
                value := integer(
                    row, "generation", "output_tokens_including_eos"
                )
            )
            is not None
        ]
        if not (
            len(generation_values) == len(constraint_values) == len(output_tokens) == len(selected)
        ):
            failures.append(f"{configuration}:missing-timing-or-token-count")
        summary: dict[str, Any] = {
            "rows": len(selected),
            "raw_sbn_valid": raw_sbn,
            "raw_referee_valid": raw_referee,
            "ended_with_eos": ended_eos,
            "hit_max_new_tokens": hit_max,
            "all_masked_rows": all_masked,
            "processor_rows": processor_rows,
            "generation_ms_mean": mean(generation_values) if generation_values else None,
            "constraint_ms_mean": mean(constraint_values) if constraint_values else None,
            "output_tokens_mean": mean(output_tokens) if output_tokens else None,
        }
        if configuration == "WF":
            summary["completion_certificates_tested"] = sum(
                integer(row, "constraint", "completion_certificates_tested") or 0
                for row in selected
            )
            summary["completion_certificates_accepted"] = sum(
                integer(row, "constraint", "completion_certificates_accepted") or 0
                for row in selected
            )
        if configuration == "WWCD":
            for key in (
                "prefix_rows_certified",
                "prefix_rows_unavailable",
                "carried_tokens",
                "cached_tokens",
                "replanned_tokens",
                "rejected_without_certificate",
                "allowed_total",
            ):
                summary[key] = sum(
                    integer(row, "constraint", key) or 0 for row in selected
                )
        summaries[configuration] = summary

    for configuration in ("WF", "WWCD"):
        summary = summaries[configuration]
        if summary["rows"] != 128:
            failures.append(f"{configuration}:row-count")
        if summary["raw_sbn_valid"] != 128:
            failures.append(f"{configuration}:raw-sbn-release")
        if summary["raw_referee_valid"] != 128:
            failures.append(f"{configuration}:raw-referee-release")
        if summary["ended_with_eos"] != 128 or summary["hit_max_new_tokens"] != 0:
            failures.append(f"{configuration}:bounded-eos-release")
        if summary["all_masked_rows"] != 0:
            failures.append(f"{configuration}:all-masked")
    wcd = summaries["WWCD"]
    if wcd.get("prefix_rows_unavailable") != 0:
        failures.append("WWCD:prefix-unavailable")
    if wcd.get("prefix_rows_certified") != wcd.get("processor_rows"):
        failures.append("WWCD:prefix-accounting")
    admitted = sum(
        int(wcd.get(key, 0))
        for key in ("carried_tokens", "cached_tokens", "replanned_tokens")
    )
    if admitted != wcd.get("allowed_total"):
        failures.append("WWCD:decision-accounting")

    paired: dict[str, dict[str, int]] = {}
    for left, right in (("W", "WF"), ("W", "WWCD"), ("WF", "WWCD")):
        changed = exact = raw_fixed = eos_rescued = 0
        for identifier in id_lists[CONFIGURATIONS[0]]:
            a = by_identity[(identifier, left)]
            b = by_identity[(identifier, right)]
            same = a.get("prediction") == b.get("prediction")
            exact += int(same)
            changed += int(not same)
            raw_fixed += int(
                a.get("validity", {}).get("raw_sbn") is not True
                and b.get("validity", {}).get("raw_sbn") is True
            )
            eos_rescued += int(
                a.get("generation", {}).get("ended_with_eos") is not True
                and b.get("generation", {}).get("ended_with_eos") is True
            )
        paired[f"{right}-{left}"] = {
            "paired": 128,
            "prediction_exact": exact,
            "prediction_changed": changed,
            "raw_sbn_fixed": raw_fixed,
            "accepting_eos_rescued": eos_rescued,
        }

    scores = load_jsonl(score_path)
    score_sha = sha256_path(score_path)
    score_manifest_path = Path(str(score_path) + ".manifest.json")
    if not score_manifest_path.is_file():
        failures.append("missing-score-manifest")
        score_manifest: dict[str, Any] = {}
    else:
        score_manifest = load_object(score_manifest_path)
    score_identities = [
        (str(row.get("configuration")), str(row.get("id"))) for row in scores
    ]
    if len(set(score_identities)) != len(score_identities) or set(score_identities) != set(identities):
        failures.append("score-identities")
    scoring_contract = contract["scoring"]
    expected_score_manifest = {
        "input_sha256": prediction_sha,
        "output_sha256": score_sha,
        "records": 384,
        "scoring_protocol": scoring_contract["protocol"],
        "seed": scoring_contract["seed"],
        "scorer_sha256": scoring_contract["scorer_sha256"],
        "identical_pair_score_consistency": True,
    }
    score_manifest_mismatches = {
        key: {"expected": expected, "observed": score_manifest.get(key)}
        for key, expected in expected_score_manifest.items()
        if score_manifest.get(key) != expected
    }
    if score_manifest_mismatches:
        failures.append("score-manifest-contract")
    score_values: dict[str, list[float]] = {name: [] for name in CONFIGURATIONS}
    for row in scores:
        configuration = str(row.get("configuration"))
        value = numeric(row, "semantic_scores", "sbn_smatch")
        if configuration not in score_values or value is None or not 0 <= value <= 1:
            failures.append("invalid-score-row")
            continue
        score_values[configuration].append(value)
    semantic_means = {
        name: mean(values) if len(values) == 128 else None
        for name, values in score_values.items()
    }
    if any(value is None for value in semantic_means.values()):
        failures.append("score-group-count")

    return {
        "schema_version": 1,
        "kind": "drs-heldout-horizon-generation-gate",
        "status": "passed" if not failures else "failed",
        "gate": {"passed": not failures, "failures": sorted(set(failures))},
        "contract": {
            "path": str(config_path),
            "sha256": sha256_path(config_path),
        },
        "predictions": {
            "path": str(predictions_path),
            "sha256": prediction_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": (
                sha256_path(manifest_path) if manifest_path.is_file() else None
            ),
            "manifest_mismatches": manifest_mismatches,
        },
        "scores": {
            "path": str(score_path),
            "sha256": score_sha,
            "manifest": str(score_manifest_path),
            "manifest_sha256": (
                sha256_path(score_manifest_path)
                if score_manifest_path.is_file()
                else None
            ),
            "manifest_mismatches": score_manifest_mismatches,
            "means": semantic_means,
        },
        "summaries": summaries,
        "paired": paired,
        "claim_boundary": contract["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(
            args.predictions, config_path=args.config, score_path=args.scores
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing is not None:
        if load_object(args.verify_existing) != report:
            parser.error("stored held-out horizon gate differs from current evidence")
    else:
        assert args.output is not None
        if args.output.exists():
            parser.error(f"refusing to overwrite gate: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    if not report["gate"]["passed"]:
        raise SystemExit(
            "held-out horizon gate failed: "
            + ", ".join(report["gate"]["failures"])
        )
    print("held-out DRS horizon gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
