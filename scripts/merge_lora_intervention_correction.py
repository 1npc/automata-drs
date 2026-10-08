#!/usr/bin/env python3
"""Assemble immutable U-baseline plus fresh constrained intervention cells."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.intervention_correction import (  # noqa: E402
    FULL_CONFIGURATIONS,
    GENERATED_CONFIGURATIONS,
    combine_intervention_correction_rows,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--correction-config",
        type=Path,
        default=Path("configs/qwen_drs_intervention_correction_v2.json"),
    )
    args = parser.parse_args()
    output_manifest = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or output_manifest.exists():
        parser.error(f"refusing to overwrite corrected matrix: {args.output}")
    required = (
        args.baseline,
        Path(str(args.baseline) + ".manifest.json"),
        args.supplement,
        Path(str(args.supplement) + ".manifest.json"),
        args.selection,
        args.correction_config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"missing correction inputs: {missing}")
    try:
        correction = _object(args.correction_config)
        if correction.get("kind") != "qwen-drs-conditioned-intervention-correction-v2":
            raise ValueError("unsupported intervention correction config")
        matrix = correction.get("matrix", {})
        if matrix.get("configurations") != list(FULL_CONFIGURATIONS):
            raise ValueError("corrected full configuration ladder changed")
        if matrix.get("generated_configurations") != list(
            GENERATED_CONFIGURATIONS
        ):
            raise ValueError("corrected generated configurations changed")
        if matrix.get("performance_threshold_used") is not False:
            raise ValueError("correction cannot use a performance threshold")
        selection = _object(args.selection)
        if selection.get("input") != str(args.baseline):
            raise ValueError("selection plan does not name the frozen baseline")
        if selection.get("input_sha256") != _sha(args.baseline):
            raise ValueError("selection plan baseline hash mismatch")
        selected = selection.get("selected")
        if not isinstance(selected, list):
            raise ValueError("selection plan has no selected list")
        selected_ids = [str(row.get("id", "")) for row in selected]
        selected_lines = [int(row.get("line_number", 0)) for row in selected]
        if selection.get("selected_records") != len(selected_ids):
            raise ValueError("selection plan count mismatch")
        supplement_manifest = _object(
            Path(str(args.supplement) + ".manifest.json")
        )
        if supplement_manifest.get("status") != "complete":
            raise ValueError("constrained supplement manifest is not complete")
        if supplement_manifest.get("output_sha256") != _sha(args.supplement):
            raise ValueError("constrained supplement manifest hash mismatch")
        expected_supplement = {
            "line_numbers": selected_lines,
            "max_new_tokens": int(matrix["max_new_tokens"]),
            "num_beams": 1,
            "batch_size": 1,
            "configurations": list(GENERATED_CONFIGURATIONS),
            "score_model_logprob": True,
            "schedule": "rotate",
        }
        mismatches = {
            name: {"observed": supplement_manifest.get(name), "expected": value}
            for name, value in expected_supplement.items()
            if supplement_manifest.get(name) != value
        }
        if mismatches:
            raise ValueError(f"constrained supplement contract changed: {mismatches}")
        rows, evidence = combine_intervention_correction_rows(
            _jsonl(args.baseline),
            _jsonl(args.supplement),
            selected_ids=selected_ids,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    baseline_manifest = Path(str(args.baseline) + ".manifest.json")
    supplement_manifest_path = Path(str(args.supplement) + ".manifest.json")
    output_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "kind": "derived-lora-conditioned-intervention-v2",
                "output": str(args.output),
                "output_sha256": _sha(args.output),
                "rows": len(rows),
                "expected_rows": len(rows),
                "model": supplement_manifest.get("model"),
                "requested_revision": supplement_manifest.get(
                    "requested_revision"
                ),
                "line_numbers": selected_lines,
                "max_new_tokens": int(matrix["max_new_tokens"]),
                "num_beams": 1,
                "batch_size": 1,
                "configurations": list(FULL_CONFIGURATIONS),
                "score_model_logprob": True,
                "schedule": matrix["derived_schedule"],
                "sources": {
                    "U": {
                        "path": str(args.baseline),
                        "sha256": _sha(args.baseline),
                        "manifest_path": str(baseline_manifest),
                        "manifest_sha256": _sha(baseline_manifest),
                    },
                    "B/W/BF/WF": {
                        "path": str(args.supplement),
                        "sha256": _sha(args.supplement),
                        "manifest_path": str(supplement_manifest_path),
                        "manifest_sha256": _sha(supplement_manifest_path),
                    },
                    "selection": {
                        "path": str(args.selection),
                        "sha256": _sha(args.selection),
                    },
                    "correction_config": {
                        "path": str(args.correction_config),
                        "sha256": _sha(args.correction_config),
                    },
                },
                **evidence,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"merged {len(rows)} corrected immutable cells into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
