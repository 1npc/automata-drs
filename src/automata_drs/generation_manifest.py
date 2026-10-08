"""Fail-closed completion records for causal-LM generation matrices.

Generation JSONL files are append-only while a run is in progress.  A
manifest may be promoted to ``complete`` only after the output is proven to be
the exact deterministic schedule described by that manifest.  This module is
Torch-free so an interrupted run can be finalized without loading a model or
claiming a GPU.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from .causal_lm import (
    iter_sbn_records,
    scheduled_generation_identities,
    select_sbn_records_by_line,
    validate_generation_resume_prefix,
)
from .amr_causal_lm import iter_amr_task_records


SCHEMA_VERSION = 1
KIND = "causal-lm-generation-matrix"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise ValueError(f"blank generation row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid generation JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"generation row {line_number} is not a JSON object"
                )
            rows.append(row)
    return rows


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"generation manifest {name} must be an integer")
    return value


def _expected_schedule(
    manifest: dict[str, Any],
) -> tuple[list[tuple[str, str]], int, str]:
    source_value = manifest.get("path")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("generation manifest has no corpus path")
    source_path = Path(source_value)
    if not source_path.is_file():
        raise ValueError(f"generation corpus is missing: {source_path}")

    start = _integer(manifest.get("start"), "start")
    limit = _integer(manifest.get("limit"), "limit")
    task = manifest.get("task", "sbn")
    line_numbers = manifest.get("line_numbers")
    if task == "amr":
        if line_numbers is not None:
            raise ValueError("AMR generation manifests do not use TSV line_numbers")
        records = list(iter_amr_task_records(source_path, start=start, limit=limit))
    elif task == "sbn" and line_numbers is None:
        records = list(iter_sbn_records(source_path, start=start, limit=limit))
    elif task == "sbn":
        if not isinstance(line_numbers, list):
            raise ValueError("generation manifest line_numbers must be a list or null")
        if start != 0 or limit != -1:
            raise ValueError(
                "generation manifest combines line_numbers with start or limit"
            )
        records = select_sbn_records_by_line(
            iter_sbn_records(source_path),
            [_integer(value, "line_numbers item") for value in line_numbers],
        )
    else:
        raise ValueError(f"unsupported generation manifest task: {task}")
    if not records:
        raise ValueError("generation manifest selects no corpus records")

    configurations = manifest.get("configurations")
    if not isinstance(configurations, list) or not all(
        isinstance(value, str) and value for value in configurations
    ):
        raise ValueError(
            "generation manifest configurations must be non-empty strings"
        )
    schedule = manifest.get("schedule")
    if not isinstance(schedule, str):
        raise ValueError("generation manifest has no schedule")
    expected = scheduled_generation_identities(
        records, configurations, schedule=schedule
    )
    return expected, len(records), sha256_file(source_path)


def complete_generation_manifest(
    predictions: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a full matrix and atomically promote its manifest.

    A legacy or ``running`` manifest is repairable.  A manifest already marked
    complete is immutable: all completion counters and hashes must still match
    or the function fails instead of silently blessing changed evidence.
    """

    if manifest_path is None:
        manifest_path = Path(str(predictions) + ".manifest.json")
    if not predictions.is_file():
        raise ValueError(f"generation output is missing: {predictions}")
    if not manifest_path.is_file():
        raise ValueError(f"generation manifest is missing: {manifest_path}")

    manifest = _read_object(manifest_path)
    schema_version = manifest.get("schema_version")
    if schema_version not in {None, SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported generation manifest schema: {schema_version}"
        )
    kind = manifest.get("kind")
    if kind not in {None, KIND}:
        raise ValueError(f"unsupported generation manifest kind: {kind}")
    status = manifest.get("status")
    if status not in {None, "running", "complete"}:
        raise ValueError(f"unsupported generation manifest status: {status}")
    num_beams = _integer(manifest.get("num_beams"), "num_beams")
    if num_beams <= 0:
        raise ValueError("generation manifest num_beams must be positive")

    expected, example_count, source_digest = _expected_schedule(manifest)
    rows = _read_rows(predictions)
    completed = validate_generation_resume_prefix(
        rows, expected, num_beams=num_beams
    )
    if completed != len(expected):
        raise ValueError(
            "generation output is incomplete: "
            f"{completed}/{len(expected)} scheduled rows"
        )
    output_digest = sha256_file(predictions)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "status": "complete",
        "source_sha256": source_digest,
        "expected_examples": example_count,
        "expected_rows": len(expected),
        "examples": example_count,
        "rows": len(rows),
        "output_sha256": output_digest,
    }

    if status == "complete":
        mismatches = {
            name: {"observed": manifest.get(name), "expected": expected_value}
            for name, expected_value in completion.items()
            if manifest.get(name) != expected_value
        }
        if mismatches:
            raise ValueError(
                f"completed generation manifest no longer matches evidence: {mismatches}"
            )
        if not isinstance(manifest.get("completed_at_utc"), str):
            raise ValueError("completed generation manifest has no timestamp")
        return manifest

    for name, expected_value in completion.items():
        if name == "status":
            continue
        if name in manifest and manifest[name] != expected_value:
            raise ValueError(
                f"generation manifest {name} conflicts with completed evidence"
            )
    completed_manifest = {
        **manifest,
        **completion,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(completed_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return completed_manifest
