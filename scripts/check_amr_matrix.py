#!/usr/bin/env python3
"""Immutable integrity gate for AMR U/P pilot, development, and test matrices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import (  # noqa: E402
    complete_generation_manifest,
    sha256_file,
)
from automata_drs.release import (  # noqa: E402
    PENMAN_RELEASE_GUARANTEE,
    has_complete_selective_release_accounting,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--release-audit", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "test"), required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument("--predecessor-gates", type=Path, nargs="*", default=())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/amr_transfer_experiment.json")
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    if args.expected_examples <= 0:
        parser.error("expected-examples must be positive")
    try:
        config = _read_object(args.config)
        manifest_path = Path(str(args.predictions) + ".manifest.json")
        manifest = complete_generation_manifest(args.predictions, manifest_path)
        rows = _read_rows(args.predictions)
        scores = _read_object(args.scores)
        paired = _read_object(args.paired)
        release = _read_object(args.release_audit)
        predecessors = [
            (path, _read_object(path)) for path in args.predecessor_gates
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    source_split = "test" if args.split == "test" else "development"
    expected_data = config["data"][source_split]
    predictions_sha256 = sha256_file(args.predictions)
    expected_rows = args.expected_examples * 2
    by_configuration: dict[str, dict[str, dict[str, Any]]] = {"U": {}, "P": {}}
    row_well_formed = True
    p_accepting_rows = 0
    p_processor_safe_rows = 0
    p_all_masked_rows = 0
    search_evidence_complete = True
    truncations = {"U": 0, "P": 0}
    u_unconstrained = True
    finite_probability = True
    for row in rows:
        configuration = str(row.get("configuration", ""))
        identity = str(row.get("id", ""))
        if (
            configuration not in by_configuration
            or not identity
            or identity in by_configuration.get(configuration, {})
        ):
            row_well_formed = False
            continue
        by_configuration[configuration][identity] = row
        score = row.get("score", {})
        finite_probability = finite_probability and _finite(
            score.get("raw_model_logprob")
        ) and _finite(score.get("raw_model_mean_logprob"))
        generation = row.get("generation", {})
        truncations[configuration] += int(
            bool(generation.get("hit_max_new_tokens"))
        )
        if configuration == "U":
            u_unconstrained = u_unconstrained and row.get("constraint") is None
        else:
            validity = row.get("validity", {})
            accepting = (
                validity.get("raw_local_penman") is True
                and validity.get("raw_official_penman") is True
                and generation.get("ended_with_eos") is True
                and generation.get("hit_max_new_tokens") is False
            )
            p_accepting_rows += int(accepting)
            constraint = row.get("constraint")
            masked = None if not isinstance(constraint, dict) else constraint.get(
                "all_masked_rows"
            )
            allowed_min = None if not isinstance(constraint, dict) else constraint.get(
                "allowed_min"
            )
            evidence_ok = (
                isinstance(masked, int)
                and not isinstance(masked, bool)
                and masked >= 0
                and isinstance(allowed_min, int)
                and not isinstance(allowed_min, bool)
                and allowed_min >= 0
            )
            search_evidence_complete = search_evidence_complete and evidence_ok
            if evidence_ok:
                p_all_masked_rows += masked
                p_processor_safe_rows += int(masked == 0 and allowed_min > 0)
    identity_sets_match = (
        len(by_configuration["U"]) == args.expected_examples
        and set(by_configuration["U"]) == set(by_configuration["P"])
    )
    score_groups = scores.get("groups", {})
    release_summary = release.get("release", {})
    release_accounting_complete = has_complete_selective_release_accounting(
        release_summary,
        attempts=args.expected_examples,
        guarantee=PENMAN_RELEASE_GUARANTEE,
    )
    checks = {
        "predecessors_passed": all(
            gate.get("gate", {}).get("passed") is True
            for _, gate in predecessors
        ),
        "manifest_complete": manifest.get("status") == "complete",
        "manifest_task_and_conditions": (
            manifest.get("task") == "amr"
            and manifest.get("configurations") == ["U", "P"]
            and manifest.get("num_beams") == 1
            and manifest.get("expected_examples") == args.expected_examples
            and manifest.get("expected_rows") == expected_rows
        ),
        "manifest_source": (
            manifest.get("source_sha256") == expected_data["sha256"]
        ),
        "manifest_population_slice": (
            manifest.get("start") == 0
            and (
                manifest.get("limit") == args.expected_examples
                if args.split == "pilot"
                else manifest.get("limit") == -1
            )
        ),
        "row_count": len(rows) == expected_rows,
        "row_identities": row_well_formed and identity_sets_match,
        "u_has_no_processor": u_unconstrained,
        "p_search_evidence_complete": search_evidence_complete,
        "finite_raw_model_probability": finite_probability,
        "score_gate": (
            scores.get("gate", {}).get("passed") is True
            and scores.get("gate", {}).get("performance_threshold_used") is False
            and scores.get("predictions", {}).get("sha256") == predictions_sha256
            and set(score_groups) == {"U", "P"}
            and all(
                score_groups[name].get("records") == args.expected_examples
                for name in ("U", "P")
            )
        ),
        "paired_gate": (
            paired.get("gate", {}).get("passed") is True
            and paired.get("gate", {}).get("performance_threshold_used") is False
            and paired.get("inputs", {}).get("predictions_sha256")
            == predictions_sha256
            and paired.get("inputs", {}).get("scores_sha256")
            == sha256_file(args.scores)
            and paired.get("population", {}).get("records")
            == args.expected_examples
            and paired.get("population", {}).get("paired") is True
            and paired.get("search", {}).get("truncations") == truncations
            and paired.get("search", {}).get("p_all_masked_rows")
            == p_all_masked_rows
            and paired.get("probability", {}).get("paired_records")
            == args.expected_examples
        ),
        "release_gate": (
            release.get("schema_version") == 2
            and release.get("audit") == "penman-certified-release"
            and release.get("gate", {}).get("passed") is True
            and release.get("gate", {}).get("performance_threshold_used") is False
            and release.get("input", {}).get("predictions_sha256")
            == predictions_sha256
            and release.get("population", {}).get("examples")
            == args.expected_examples
            and release.get("population", {}).get("rows")
            == args.expected_examples
            and release_accounting_complete
        ),
    }
    report = {
        "schema_version": 2,
        "audit": "amr-u-p-matrix-gate",
        "split": args.split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "config": str(args.config),
            "config_sha256": sha256_file(args.config),
            "predictions": str(args.predictions),
            "predictions_sha256": predictions_sha256,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "scores": str(args.scores),
            "scores_sha256": sha256_file(args.scores),
            "paired": str(args.paired),
            "paired_sha256": sha256_file(args.paired),
            "release_audit": str(args.release_audit),
            "release_audit_sha256": sha256_file(args.release_audit),
            "predecessor_gates": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path, _ in predecessors
            ],
        },
        "population": {
            "examples": args.expected_examples,
            "rows": len(rows),
            "paired": identity_sets_match,
        },
        "outcomes": {
            "p_accepting_rows": p_accepting_rows,
            "p_nonaccepting_rows": args.expected_examples - p_accepting_rows,
            "p_processor_safe_rows": p_processor_safe_rows,
            "p_processor_unsafe_rows": (
                args.expected_examples - p_processor_safe_rows
            ),
            "p_all_masked_rows": p_all_masked_rows,
            "truncations": truncations,
            "release": {
                "releasable_rows": release_summary.get("releasable_rows"),
                "refused_rows": release_summary.get("refused_rows"),
            },
        },
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "performance_threshold_used": False,
        },
    }
    if args.verify_existing is not None:
        try:
            stored = _read_object(args.verify_existing)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR matrix gate differs from current evidence")
        print(f"verified AMR matrix gate: {args.verify_existing}")
    else:
        assert args.output is not None
        try:
            _write_exclusive(args.output, report)
        except FileExistsError:
            parser.error(f"refusing to overwrite AMR matrix gate: {args.output}")
    print(f"checks: {sum(checks.values())}/{len(checks)}")
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        print("failed checks: " + ", ".join(failed_checks))
    print(f"gate passed: {report['gate']['passed']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
