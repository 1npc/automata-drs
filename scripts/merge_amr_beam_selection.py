#!/usr/bin/env python3
"""Freeze a paired AMR U/PostP/online-P beam comparison matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.amr_beam_selection import combine_amr_beam_rows  # noqa: E402
from automata_drs.amr_causal_lm import iter_amr_task_records  # noqa: E402
from automata_drs.generation_manifest import (  # noqa: E402
    KIND as GENERATION_KIND,
    SCHEMA_VERSION as GENERATION_SCHEMA_VERSION,
    complete_generation_manifest,
    sha256_file,
)


CANDIDATE_KIND = "amr-unconstrained-beam-candidates"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--online-p", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/amr_transfer_experiment.json")
    )
    args = parser.parse_args()

    output_path = args.output or args.verify_existing
    assert output_path is not None
    output_manifest_path = Path(str(output_path) + ".manifest.json")
    if args.output is not None and (args.output.exists() or output_manifest_path.exists()):
        parser.error("refusing to overwrite AMR beam matrix or manifest")
    candidate_manifest_path = Path(str(args.candidates) + ".manifest.json")
    online_manifest_path = Path(str(args.online_p) + ".manifest.json")
    try:
        config = _read_object(args.config)
        candidate_manifest = _read_object(candidate_manifest_path)
        online_manifest = complete_generation_manifest(
            args.online_p, online_manifest_path
        )
        if candidate_manifest.get("status") != "complete":
            raise ValueError("AMR candidate manifest is not complete")
        if candidate_manifest.get("kind") != CANDIDATE_KIND:
            raise ValueError("unexpected AMR candidate manifest kind")
        if candidate_manifest.get("task") != "amr":
            raise ValueError("candidate manifest is not AMR")
        if candidate_manifest.get("configuration") != "U4":
            raise ValueError("candidate manifest is not U4")
        if candidate_manifest.get("output_scores") is not False:
            raise ValueError("candidate generation retained stepwise score tensors")
        if candidate_manifest.get("score_policy") != (
            "chunked-teacher-forcing-for-u-top1-and-postp-only"
        ):
            raise ValueError("candidate score policy is not memory-bounded")
        if candidate_manifest.get("output_sha256") != sha256_file(args.candidates):
            raise ValueError("AMR candidate output hash differs from its manifest")
        if candidate_manifest.get("experiment_config_sha256") != sha256_file(args.config):
            raise ValueError("candidate manifest does not bind the frozen AMR config")
        if online_manifest.get("task") != "amr" or online_manifest.get(
            "configurations"
        ) != ["P"]:
            raise ValueError("online matrix must contain only AMR P")
        if online_manifest.get("score_model_logprob") is not True:
            raise ValueError("online P rows must include raw-model log probabilities")
        if online_manifest.get("experiment_config_sha256") != sha256_file(args.config):
            raise ValueError("online P manifest does not bind the frozen AMR config")
        num_beams = int(candidate_manifest["num_beams"])
        if num_beams < 2 or int(candidate_manifest["num_return_sequences"]) != num_beams:
            raise ValueError("candidate beam dimensions are invalid")
        if int(online_manifest.get("num_beams", -1)) != num_beams:
            raise ValueError("candidate and online P beam widths differ")
        shared_fields = (
            "model",
            "model_loader",
            "requested_revision",
            "resolved_revision",
            "adapter",
            "path",
            "start",
            "limit",
            "max_input_tokens",
            "max_new_tokens",
            "source_sha256",
            "experiment_config_sha256",
        )
        differences = [
            field
            for field in shared_fields
            if candidate_manifest.get(field) != online_manifest.get(field)
        ]
        if differences:
            raise ValueError(
                "candidate and online P manifests differ: " + ", ".join(differences)
            )
        if candidate_manifest.get("model") != config["host"]["repo_id"]:
            raise ValueError("beam matrix model differs from the frozen AMR host")
        if candidate_manifest.get("requested_revision") != config["host"]["revision"]:
            raise ValueError("beam matrix revision differs from the frozen AMR host")

        records = list(
            iter_amr_task_records(
                Path(str(candidate_manifest["path"])),
                start=int(candidate_manifest["start"]),
                limit=int(candidate_manifest["limit"]),
            )
        )
        identifiers = [record.identifier for record in records]
        if len(records) != int(candidate_manifest["expected_examples"]):
            raise ValueError("candidate manifest example count differs from corpus")
        candidate_rows = _read_rows(args.candidates)
        online_rows = _read_rows(args.online_p)
        combined = combine_amr_beam_rows(
            candidate_rows, online_rows, identifiers, num_beams=num_beams
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))

    manifest = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "kind": GENERATION_KIND,
        "task": "amr",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": candidate_manifest["model"],
        "model_loader": candidate_manifest["model_loader"],
        "requested_revision": candidate_manifest["requested_revision"],
        "resolved_revision": candidate_manifest.get("resolved_revision"),
        "adapter": candidate_manifest["adapter"],
        "path": candidate_manifest["path"],
        "start": candidate_manifest["start"],
        "limit": candidate_manifest["limit"],
        "line_numbers": None,
        "max_input_tokens": candidate_manifest["max_input_tokens"],
        "max_new_tokens": candidate_manifest["max_new_tokens"],
        "num_beams": num_beams,
        "batch_size": 1,
        "configurations": ["U", "PostP", "P"],
        "schedule": "fixed",
        "source_sha256": candidate_manifest["source_sha256"],
        "experiment_config_sha256": candidate_manifest[
            "experiment_config_sha256"
        ],
        "expected_examples": len(records),
        "expected_rows": len(combined),
        "derivation": "u-top1-and-first-raw-penman-valid-u-beam-plus-online-p",
        "candidate_path": str(args.candidates),
        "candidate_sha256": sha256_file(args.candidates),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "online_p_path": str(args.online_p),
        "online_p_sha256": sha256_file(args.online_p),
        "online_p_manifest_sha256": sha256_file(online_manifest_path),
    }
    if args.verify_existing is not None:
        try:
            stored_rows = _read_rows(args.verify_existing)
            stored_manifest = _read_object(output_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if stored_rows != combined:
            parser.error("stored AMR beam matrix differs from current derivation")
        expected_fields = {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at_utc", "status"}
        }
        if any(stored_manifest.get(key) != value for key, value in expected_fields.items()):
            parser.error("stored AMR beam manifest differs from current derivation")
        try:
            complete_generation_manifest(args.verify_existing, output_manifest_path)
        except ValueError as error:
            parser.error(f"stored AMR beam matrix failed completion: {error}")
        print(f"verified merged AMR beam matrix: {len(records)} examples")
        return 0

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        args.output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined),
    )
    _write_atomic(
        output_manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    try:
        complete_generation_manifest(args.output, output_manifest_path)
    except ValueError as error:
        parser.error(f"derived AMR beam matrix failed completion: {error}")
    print(f"merged AMR beam matrix: {len(records)} examples, {len(combined)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
