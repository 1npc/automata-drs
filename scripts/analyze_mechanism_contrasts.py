#!/usr/bin/env python3
"""Write a hash-bound descriptive mechanism-contrast sidecar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.mechanism_contrasts import (  # noqa: E402
    DEFAULT_DRS_CONTRASTS,
    analyze_mechanism_contrasts,
)
from automata_drs.paired_analysis import experiment_identity  # noqa: E402
from automata_drs.scoring import (  # noqa: E402
    merge_semantic_scores,
    validate_semantic_pair_consistency,
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def _parse_contrast(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise argparse.ArgumentTypeError("contrast must be LEFT:RIGHT")
    return parts[0], parts[1]


def _identities(rows: list[dict[str, Any]], label: str) -> set[tuple[str, str, str]]:
    values = [experiment_identity(row) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} identities")
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "lora_baselines.json",
    )
    parser.add_argument(
        "--contrast",
        action="append",
        type=_parse_contrast,
        dest="contrasts",
        help="paired LEFT:RIGHT contrast; repeat to override the DRS ladder",
    )
    args = parser.parse_args()

    prediction_manifest_path = Path(str(args.predictions) + ".manifest.json")
    score_manifest_path = Path(str(args.scores) + ".manifest.json")
    required = (
        args.predictions,
        prediction_manifest_path,
        args.scores,
        score_manifest_path,
        args.config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing mechanism-contrast inputs: {missing}")
    if args.output.exists():
        parser.error(f"refusing to overwrite mechanism contrasts: {args.output}")

    config = _read_object(args.config)
    parameters = config.get("method_experiment", {}).get("paired_statistics")
    if not isinstance(parameters, dict):
        parser.error("config has no method_experiment.paired_statistics object")

    prediction_manifest = _read_object(prediction_manifest_path)
    score_manifest = _read_object(score_manifest_path)
    if prediction_manifest.get("status") != "complete":
        parser.error("prediction manifest is not complete")
    predictions = _read_jsonl(args.predictions)
    scores = _read_jsonl(args.scores)
    if prediction_manifest.get("rows") != len(predictions):
        parser.error("prediction manifest row count mismatch")
    if (
        "expected_rows" in prediction_manifest
        and prediction_manifest.get("expected_rows") != len(predictions)
    ):
        parser.error("prediction manifest expected row count mismatch")
    if prediction_manifest.get("output_sha256") != _sha256(args.predictions):
        parser.error("prediction manifest output hash mismatch")
    if _identities(predictions, "prediction") != _identities(scores, "score"):
        parser.error("semantic score cells do not match prediction cells")
    if score_manifest.get("input_sha256") != _sha256(args.predictions):
        parser.error("score manifest input hash mismatch")
    if score_manifest.get("output_sha256") != _sha256(args.scores):
        parser.error("score manifest output hash mismatch")
    if score_manifest.get("records") != len(predictions):
        parser.error("score manifest record count mismatch")
    if (
        score_manifest.get("scoring_protocol")
        != "seeded-deduplicated-semantic-pairs-v2"
        or score_manifest.get("identical_pair_score_consistency") is not True
    ):
        parser.error(
            "mechanism contrasts require seeded deduplicated semantic scores"
        )

    contrasts = tuple(args.contrasts or DEFAULT_DRS_CONTRASTS)
    merged = merge_semantic_scores(predictions, scores)
    try:
        pair_diagnostics = validate_semantic_pair_consistency(merged)
    except ValueError as error:
        parser.error(str(error))
    for name in ("unique_semantic_pairs", "reused_score_records"):
        if score_manifest.get(name) != pair_diagnostics[name]:
            parser.error(f"score manifest {name} mismatch")
    report = analyze_mechanism_contrasts(
        merged,
        contrasts=contrasts,
        bootstrap_resamples=int(parameters["bootstrap_resamples"]),
        confidence=float(parameters["confidence"]),
        seed=int(parameters["seed"]),
    )
    report.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "predictions": str(args.predictions),
                "predictions_sha256": _sha256(args.predictions),
                "prediction_manifest": str(prediction_manifest_path),
                "prediction_manifest_sha256": _sha256(prediction_manifest_path),
                "scores": str(args.scores),
                "scores_sha256": _sha256(args.scores),
                "score_manifest": str(score_manifest_path),
                "score_manifest_sha256": _sha256(score_manifest_path),
                "config": str(args.config),
                "config_sha256": _sha256(args.config),
            },
            "implementation": {
                "analysis_sha256": _sha256(
                    REPOSITORY_ROOT
                    / "src"
                    / "automata_drs"
                    / "mechanism_contrasts.py"
                ),
                "script_sha256": _sha256(Path(__file__)),
            },
            "gate": {
                "passed": True,
                "complete_pairing": True,
                "score_identity_match": True,
                "identical_pair_score_consistency": True,
                "source_hashes_verified": True,
                "performance_threshold_used": False,
            },
            "semantic_pair_diagnostics": pair_diagnostics,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(f"mechanism contrasts: {args.output}")
    print(f"comparisons: {len(report['comparisons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
