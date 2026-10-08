#!/usr/bin/env python3
"""Audit deterministic PENMAN completion of horizon-truncated AMR prefixes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import (  # noqa: E402
    PENMANConstraintState,
    construct_penman_completion_suffix,
    validate_penman,
)
from automata_drs.amr_scoring import (  # noqa: E402
    aggregate_amr_scores,
    create_amr_smatchpp,
    score_amr_pair,
    smatchpp_scoreable,
)
from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
    sha256_file,
)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object")
            rows.append(value)
    return rows


def complete_penman_prefix(prefix: str) -> tuple[str | None, str | None]:
    """Return a locally certified completion and the appended suffix."""

    state = PENMANConstraintState.initial().advance_text(prefix)
    suffix = construct_penman_completion_suffix(state)
    if suffix is None:
        return None, None
    completed = prefix + suffix
    return (completed, suffix) if validate_penman(completed).valid else (None, suffix)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--configuration", default="P")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-solver-seconds", type=int, default=240)
    args = parser.parse_args()
    if args.max_solver_seconds <= 0:
        parser.error("--max-solver-seconds must be positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite report: {args.output}")

    try:
        import penman
        import smatchpp  # noqa: F401
    except ImportError as error:
        parser.error(f"AMR audit dependencies are unavailable: {error}")
    measure = create_amr_smatchpp(max_solver_seconds=args.max_solver_seconds)

    manifest_path = Path(str(args.predictions) + ".manifest.json")
    try:
        manifest = complete_generation_manifest(args.predictions, manifest_path)
        rows = _read_rows(args.predictions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    selected = [
        row for row in rows if row.get("configuration") == args.configuration
    ]
    if not selected:
        parser.error(f"no {args.configuration} rows in generation matrix")
    if len(selected) != manifest.get("expected_examples"):
        parser.error("selected configuration does not cover every example")

    detailed: list[dict[str, Any]] = []
    suffix_lengths: list[int] = []
    pair_scores: list[dict[str, Any]] = []
    for row in selected:
        identifier = str(row.get("id", ""))
        prefix = str(row.get("raw_prediction", ""))
        reference = str(row.get("reference", ""))
        completed, suffix = complete_penman_prefix(prefix)
        official = False
        if completed is not None:
            try:
                penman.decode(completed)
                official = True
            except Exception:
                official = False
        scoreable = (
            completed is not None
            and official
            and smatchpp_scoreable(measure, completed)
        )
        scored_surface = completed if scoreable else None
        pair_score = score_amr_pair(measure, scored_surface, reference)
        pair_scores.append(pair_score)
        suffix_length = None if suffix is None else len(suffix)
        if suffix_length is not None:
            suffix_lengths.append(suffix_length)
        detailed.append(
            {
                "id": identifier,
                "prefix_characters": len(prefix),
                "suffix": suffix,
                "suffix_characters": suffix_length,
                "completion_constructed": completed is not None,
                "local_penman": (
                    completed is not None and validate_penman(completed).valid
                ),
                "official_penman": official,
                "smatchpp_scoreable": scoreable,
                "smatchpp": pair_score,
            }
        )
    corpus_score = aggregate_amr_scores(pair_scores)
    constructed = sum(item["completion_constructed"] for item in detailed)
    local_valid = sum(item["local_penman"] for item in detailed)
    official_valid = sum(item["official_penman"] for item in detailed)
    scoreable = sum(item["smatchpp_scoreable"] for item in detailed)
    count = len(detailed)
    gate_passed = (
        constructed == local_valid == official_valid == scoreable == count
        and corpus_score["all_alignments_optimal"]
    )
    report = {
        "schema_version": 1,
        "audit": "amr-deterministic-horizon-completion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "configuration": args.configuration,
            "records": count,
        },
        "method": {
            "completion": "construct_penman_completion_suffix",
            "trigger": "post-horizon feasibility audit",
            "model_tokens_added": 0,
            "claim_scope": (
                "constructive availability witness; not yet an in-budget "
                "decoding policy"
            ),
        },
        "summary": {
            "completion_constructed": constructed,
            "local_penman": local_valid,
            "official_penman": official_valid,
            "smatchpp_scoreable": scoreable,
            "suffix_characters_mean": (
                statistics.fmean(suffix_lengths) if suffix_lengths else None
            ),
            "suffix_characters_max": max(suffix_lengths, default=None),
            "completed_corpus_smatchpp": corpus_score,
        },
        "rows": detailed,
        "gate": {
            "passed": gate_passed,
            "all_prefixes_constructively_completable": constructed == count,
            "all_completed_surfaces_dual_valid": (
                local_valid == official_valid == count
            ),
            "all_completed_surfaces_smatchpp_scoreable": scoreable == count,
            "all_alignments_optimal": corpus_score["all_alignments_optimal"],
            "performance_threshold_used": False,
        },
    }
    try:
        _write_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to overwrite report: {args.output}")
    print(f"report: {args.output}")
    print(
        f"completed={constructed}/{count} dual-valid={official_valid}/{count} "
        f"Smatch++={corpus_score['f1']:.6f} gate={gate_passed}"
    )
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
