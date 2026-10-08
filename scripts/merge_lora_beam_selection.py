#!/usr/bin/env python3
"""Derive and freeze a U/PostW/W/WF beam4 comparison matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.beam_selection import (  # noqa: E402
    combine_beam_selection_rows,
)
from automata_drs.causal_lm import iter_sbn_records  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lora_baselines.json")
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/en/dev/standard.sbn")
    )
    args = parser.parse_args()

    output_manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or output_manifest_path.exists():
        parser.error("refusing to overwrite beam study matrix or manifest")
    candidate_manifest_path = Path(str(args.candidates) + ".manifest.json")
    online_manifest_path = Path(str(args.online) + ".manifest.json")
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes.decode("utf-8"))
        study = config["method_experiment"]["beam_selection_study"]
        records = int(study["records"])
        num_beams = int(study["num_beams"])
        max_new_tokens = int(study["max_new_tokens"])
        if study["generated_conditions"] != ["U4-candidates", "W", "WF"]:
            raise ValueError("frozen beam generated conditions are invalid")
        if study["derived_comparison"] != ["U", "PostW", "W", "WF"]:
            raise ValueError("frozen beam derived comparison is invalid")
        if int(study["u_num_return_sequences"]) != num_beams:
            raise ValueError("frozen U beam return count differs from beam width")
        expected_identifiers = [
            record.identifier
            for record in iter_sbn_records(args.data, limit=records)
        ]
        if len(expected_identifiers) != records:
            raise ValueError("beam study data has fewer records than frozen")

        candidate_manifest = read_object(candidate_manifest_path)
        online_manifest = read_object(online_manifest_path)
        if (
            candidate_manifest.get("status") != "complete"
            or candidate_manifest.get("kind") != "unconstrained-beam-candidates"
            or candidate_manifest.get("configuration") != "U4"
            or int(candidate_manifest.get("num_beams", -1)) != num_beams
            or int(candidate_manifest.get("num_return_sequences", -1))
            != num_beams
            or int(candidate_manifest.get("limit", -1)) != records
            or int(candidate_manifest.get("max_new_tokens", -1))
            != max_new_tokens
        ):
            raise ValueError("candidate manifest differs from frozen beam study")
        if candidate_manifest.get("output_sha256") != sha256(args.candidates):
            raise ValueError("candidate output hash differs from its manifest")
        if (
            online_manifest.get("configurations") != ["W", "WF"]
            or int(online_manifest.get("num_beams", -1)) != num_beams
            or int(online_manifest.get("limit", -1)) != records
            or int(online_manifest.get("max_new_tokens", -1)) != max_new_tokens
        ):
            raise ValueError("online manifest differs from frozen beam study")
        shared_manifest_fields = (
            "model",
            "model_loader",
            "requested_revision",
            "adapter",
            "path",
            "start",
        )
        differences = [
            field
            for field in shared_manifest_fields
            if candidate_manifest.get(field) != online_manifest.get(field)
        ]
        if differences:
            raise ValueError(
                "candidate and online manifests differ: " + ", ".join(differences)
            )
        candidate_rows = read_jsonl(args.candidates)
        online_rows = read_jsonl(args.online)
        combined = combine_beam_selection_rows(
            candidate_rows,
            online_rows,
            expected_identifiers,
            num_beams=num_beams,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as sink:
        for row in combined:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_manifest = {
        "schema_version": 1,
        "kind": "derived-beam-selection-matrix",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "study_contract": study,
        "candidate_path": str(args.candidates),
        "candidate_sha256": sha256(args.candidates),
        "candidate_manifest_sha256": sha256(candidate_manifest_path),
        "online_path": str(args.online),
        "online_sha256": sha256(args.online),
        "online_manifest_sha256": sha256(online_manifest_path),
        "output_path": str(args.output),
        "output_sha256": sha256(args.output),
        "records": records,
        "rows": len(combined),
        "configurations": ["U", "PostW", "W", "WF"],
        "num_beams": num_beams,
    }
    output_manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"merged beam study matrix: {records} records, {len(combined)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
