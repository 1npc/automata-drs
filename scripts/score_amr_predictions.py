#!/usr/bin/env python3
"""Score complete paired AMR matrices with exact Smatch++ ILP alignment."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

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


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    parser.add_argument("--max-solver-seconds", type=int, default=240)
    args = parser.parse_args()
    if args.max_solver_seconds <= 0:
        parser.error("--max-solver-seconds must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite score report: {args.output}")

    try:
        import penman
        import smatchpp
    except ImportError as error:
        parser.error(f"AMR scoring dependencies are unavailable: {error}")
    measure = create_amr_smatchpp(max_solver_seconds=args.max_solver_seconds)

    manifest_path = Path(str(args.predictions) + ".manifest.json")
    try:
        manifest = complete_generation_manifest(args.predictions, manifest_path)
    except ValueError as error:
        parser.error(str(error))
    if manifest.get("task") != "amr":
        parser.error("generation manifest is not an AMR task")

    try:
        rows = _read_rows(args.predictions)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_configuration[str(row.get("configuration", ""))].append(row)
    configurations = manifest["configurations"]
    if set(by_configuration) != set(configurations):
        parser.error("prediction configurations differ from the manifest")

    reference_by_id: dict[str, str] = {}
    group_reports: dict[str, dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    integrity_passed = True
    pair_cache: dict[tuple[str | None, str], dict[str, Any]] = {}

    for configuration in configurations:
        group = by_configuration[configuration]
        if len(group) != manifest["expected_examples"]:
            parser.error(
                f"{configuration} has {len(group)} rows; expected "
                f"{manifest['expected_examples']}"
            )
        raw_valid_count = 0
        extracted_valid_count = 0
        raw_smatchpp_scoreable_count = 0
        extracted_smatchpp_scoreable_count = 0
        raw_pair_scores: list[dict[str, Any]] = []
        extracted_pair_scores: list[dict[str, Any]] = []

        for index, row in enumerate(group):
            identifier = str(row.get("id", ""))
            reference = str(row.get("reference", ""))
            raw = str(row.get("raw_prediction", ""))
            extracted = str(row.get("prediction", ""))
            try:
                penman.decode(reference)
            except Exception as error:
                parser.error(f"invalid reference for {identifier}: {error}")
            previous_reference = reference_by_id.setdefault(identifier, reference)
            if previous_reference != reference:
                parser.error(f"reference changed across conditions for {identifier}")

            try:
                penman.decode(raw)
                raw_valid = True
            except Exception:
                raw_valid = False
            try:
                penman.decode(extracted)
                extracted_valid = True
            except Exception:
                extracted_valid = False
            raw_smatchpp_scoreable = raw_valid and smatchpp_scoreable(measure, raw)
            extracted_smatchpp_scoreable = (
                extracted_valid and smatchpp_scoreable(measure, extracted)
            )
            raw_valid_count += int(raw_valid)
            extracted_valid_count += int(extracted_valid)
            raw_smatchpp_scoreable_count += int(raw_smatchpp_scoreable)
            extracted_smatchpp_scoreable_count += int(extracted_smatchpp_scoreable)
            raw_scored = raw if raw_smatchpp_scoreable else None
            extracted_scored = extracted if extracted_smatchpp_scoreable else None
            raw_key = (raw_scored, reference)
            extracted_key = (extracted_scored, reference)
            if raw_key not in pair_cache:
                pair_cache[raw_key] = score_amr_pair(
                    measure, raw_scored, reference
                )
            if extracted_key not in pair_cache:
                pair_cache[extracted_key] = score_amr_pair(
                    measure, extracted_scored, reference
                )
            raw_pair = pair_cache[raw_key]
            extracted_pair = pair_cache[extracted_key]
            raw_pair_scores.append(raw_pair)
            extracted_pair_scores.append(extracted_pair)
            paired_rows.append(
                {
                    "id": identifier,
                    "configuration": configuration,
                    "raw_official_penman": raw_valid,
                    "raw_smatchpp_scoreable": raw_smatchpp_scoreable,
                    "prediction_official_penman": extracted_valid,
                    "prediction_smatchpp_scoreable": extracted_smatchpp_scoreable,
                    "raw_smatch": raw_pair,
                    "prediction_smatch": extracted_pair,
                }
            )
        raw_corpus = aggregate_amr_scores(raw_pair_scores)
        extracted_corpus = aggregate_amr_scores(extracted_pair_scores)
        group_reports[configuration] = {
            "records": len(group),
            "raw_official_penman": raw_valid_count,
            "raw_official_penman_rate": raw_valid_count / len(group),
            "prediction_official_penman": extracted_valid_count,
            "prediction_official_penman_rate": extracted_valid_count / len(group),
            "raw_smatchpp_scoreable": raw_smatchpp_scoreable_count,
            "raw_smatchpp_scoreable_rate": raw_smatchpp_scoreable_count / len(group),
            "prediction_smatchpp_scoreable": extracted_smatchpp_scoreable_count,
            "prediction_smatchpp_scoreable_rate": (
                extracted_smatchpp_scoreable_count / len(group)
            ),
            "raw_primary_smatch": raw_corpus,
            "extracted_diagnostic_smatch": extracted_corpus,
            "invalid_policy": (
                "PENMAN-decodable but Smatch++-unscoreable and invalid surfaces "
                "contribute zero matches and zero predicted "
                "triples while retaining all reference triples in the recall "
                "denominator; no record is dropped"
            ),
        }
        integrity_passed = (
            integrity_passed
            and len(group) == manifest["expected_examples"]
            and raw_corpus["all_alignments_optimal"]
            and extracted_corpus["all_alignments_optimal"]
        )

    identities = {
        configuration: {str(row["id"]) for row in by_configuration[configuration]}
        for configuration in configurations
    }
    first_identity = identities[configurations[0]]
    if any(value != first_identity for value in identities.values()):
        parser.error("configuration identity sets are not paired")

    report = {
        "schema_version": 2,
        "audit": "amr-generation-smatchpp",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": {
            "path": str(args.predictions),
            "sha256": sha256_file(args.predictions),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "rows": len(rows),
        },
        "scorer": {
            "name": "smatchpp",
            "version": importlib.metadata.version("smatchpp"),
            "solver": "ilp",
            "max_solver_seconds_per_pair": args.max_solver_seconds,
            "amr_standardizer": "smatchpp.formalism.amr.tools.AMRStandardizer",
            "optimality_required": True,
            "invalid_fallback": None,
            "primary_surface": "raw_prediction",
            "diagnostic_surface": "balanced extracted prediction",
            "penman_version": penman.__version__,
        },
        "groups": group_reports,
        "paired_rows": paired_rows,
        "gate": {
            "passed": integrity_passed,
            "complete_generation_manifest": True,
            "complete_identity_pairing": True,
            "all_alignments_optimal": all(
                group["raw_primary_smatch"]["all_alignments_optimal"]
                and group["extracted_diagnostic_smatch"]["all_alignments_optimal"]
                for group in group_reports.values()
            ),
            "performance_threshold_used": False,
        },
    }
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR score report differs from recomputed evidence")
        print(f"verified AMR score report: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite score report: {args.output}")
        print(f"report: {args.output}")
    for configuration, group in group_reports.items():
        print(
            f"{configuration}: raw PENMAN={group['raw_official_penman']}/"
            f"{group['records']} raw Smatch++={group['raw_primary_smatch']['f1']:.6f}"
        )
    print(f"gate passed: {integrity_passed}")
    return 0 if integrity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
