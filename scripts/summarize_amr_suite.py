#!/usr/bin/env python3
"""Collect the complete AMR transfer experiment into reviewer-facing evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.release import (  # noqa: E402
    PENMAN_RELEASE_GUARANTEE,
    has_complete_selective_release_accounting,
)


def _read(path: Path, evidence: dict[str, str]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    evidence[str(path)] = sha256_file(path)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _without_timestamp(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("created_at_utc", None)
    return result


def _matrix(
    run_root: Path,
    stem: str,
    split: str,
    expected: int,
    config_sha256: str,
    evidence: dict[str, str],
    expected_predecessors: list[Path],
) -> dict[str, Any]:
    predictions = run_root / f"{stem}.jsonl"
    scores_path = run_root / f"{stem}.smatch.json"
    paired_path = run_root / f"{stem}.paired.json"
    release_path = run_root / f"{stem}.p-certified-release.json"
    gate_path = run_root / f"{stem}.gate.json"
    for path in (predictions,):
        _require(path.is_file(), f"missing AMR matrix artifact: {path}")
        evidence[str(path)] = sha256_file(path)
    scores = _read(scores_path, evidence)
    paired = _read(paired_path, evidence)
    release = _read(release_path, evidence)
    gate = _read(gate_path, evidence)
    prediction_sha = evidence[str(predictions)]
    release_summary = release.get("release", {})
    gate_inputs = gate.get("inputs", {})
    _require(
        gate.get("schema_version") == 2
        and gate.get("audit") == "amr-u-p-matrix-gate"
        and gate.get("split") == split
        and gate.get("population", {}).get("examples") == expected
        and gate.get("outcomes", {}).get("release")
        == {
            "releasable_rows": release_summary.get("releasable_rows"),
            "refused_rows": release_summary.get("refused_rows"),
        }
        and gate.get("gate", {}).get("passed") is True
        and gate.get("gate", {}).get("performance_threshold_used") is False,
        f"AMR {split} matrix gate is not schema 2, release-bound, and passing",
    )
    _require(
        gate_inputs.get("config_sha256") == config_sha256
        and gate_inputs.get("predictions_sha256") == prediction_sha
        and gate_inputs.get("scores_sha256") == evidence[str(scores_path)]
        and gate_inputs.get("paired_sha256") == evidence[str(paired_path)]
        and gate_inputs.get("release_audit_sha256") == evidence[str(release_path)],
        f"AMR {split} matrix gate input hashes drifted",
    )
    observed_predecessors = gate_inputs.get("predecessor_gates", [])
    expected_predecessor_rows = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in expected_predecessors
    ]
    _require(
        observed_predecessors == expected_predecessor_rows,
        f"AMR {split} predecessor chain drifted",
    )
    _require(
        scores.get("audit") == "amr-generation-smatchpp"
        and scores.get("gate", {}).get("passed") is True
        and scores.get("gate", {}).get("performance_threshold_used") is False
        and scores.get("predictions", {}).get("sha256") == prediction_sha
        and all(
            scores.get("groups", {}).get(name, {}).get("records") == expected
            for name in ("U", "P")
        ),
        f"AMR {split} score report is incomplete",
    )
    _require(
        paired.get("audit") == "amr-paired-u-p-analysis"
        and paired.get("gate", {}).get("passed") is True
        and paired.get("gate", {}).get("performance_threshold_used") is False
        and paired.get("inputs", {}).get("predictions_sha256") == prediction_sha
        and paired.get("inputs", {}).get("scores_sha256")
        == evidence[str(scores_path)]
        and paired.get("population", {}).get("records") == expected,
        f"AMR {split} paired report is incomplete",
    )
    _require(
        release.get("schema_version") == 2
        and release.get("audit") == "penman-certified-release"
        and release.get("gate", {}).get("passed") is True
        and release.get("gate", {}).get("performance_threshold_used") is False
        and release.get("input", {}).get("predictions_sha256") == prediction_sha
        and release.get("population", {}).get("examples") == expected
        and release.get("population", {}).get("rows") == expected
        and has_complete_selective_release_accounting(
            release_summary,
            attempts=expected,
            guarantee=PENMAN_RELEASE_GUARANTEE,
        ),
        f"AMR {split} certified release report is incomplete",
    )
    return {
        "split": split,
        "examples": expected,
        "gate": str(gate_path),
        "u": scores["groups"]["U"],
        "p": scores["groups"]["P"],
        "paired": paired,
        "release": release_summary,
    }


def collect(config_path: Path, run_root: Path, resource_plan_path: Path) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    config = _read(config_path, evidence)
    resource_plan = _read(resource_plan_path, evidence)
    config_sha = evidence[str(config_path)]
    _require(
        resource_plan.get("audit") == "amr-full-run-resource-plan"
        and resource_plan.get("gate", {}).get("passed") is True
        and resource_plan.get("gate", {}).get("performance_threshold_used") is False
        and resource_plan.get("inputs", {}).get("config_sha256") == config_sha,
        "AMR resource plan does not bind the current config",
    )
    training_gate_path = run_root / "full-training-gate.json"
    training_gate = _read(training_gate_path, evidence)
    _require(
        training_gate.get("audit") == "amr-full-lora-training-gate"
        and training_gate.get("gate", {}).get("passed") is True
        and training_gate.get("gate", {}).get("performance_threshold_used") is False
        and training_gate.get("inputs", {}).get("config_sha256") == config_sha
        and training_gate.get("inputs", {}).get("resource_plan_sha256")
        == evidence[str(resource_plan_path)],
        "AMR full-training gate is not current",
    )

    pilot_records = int(config["development_pilot"]["records"])
    dev_records = int(config["data"]["development"]["records"])
    test_records = int(config["data"]["test"]["records"])
    pilot_stem = f"dev{pilot_records}-u-p-greedy"
    dev_stem = f"dev{dev_records}-u-p-greedy"
    test_stem = f"test{test_records}-u-p-greedy"
    pilot_gate_path = run_root / f"{pilot_stem}.gate.json"
    dev_gate_path = run_root / f"{dev_stem}.gate.json"
    beam_stem = f"dev{dev_records}-u-postp-p-beam4"
    beam_matrix = run_root / f"{beam_stem}.jsonl"
    beam_scores_path = run_root / f"{beam_stem}.smatch.json"
    beam_report_path = run_root / f"{beam_stem}.analysis.json"
    beam_online = run_root / f"dev{dev_records}-p-beam4.jsonl"
    beam_release_path = run_root / f"dev{dev_records}-p-beam4.p-certified-release.json"

    pilot = _matrix(
        run_root, pilot_stem, "pilot", pilot_records, config_sha, evidence,
        [training_gate_path],
    )
    development = _matrix(
        run_root, dev_stem, "dev", dev_records, config_sha, evidence,
        [pilot_gate_path],
    )
    _require(beam_matrix.is_file() and beam_online.is_file(), "missing AMR beam outputs")
    evidence[str(beam_matrix)] = sha256_file(beam_matrix)
    evidence[str(beam_online)] = sha256_file(beam_online)
    beam_scores = _read(beam_scores_path, evidence)
    beam_report = _read(beam_report_path, evidence)
    beam_release = _read(beam_release_path, evidence)
    _require(
        beam_report.get("schema_version") == 2
        and beam_report.get("audit") == "amr-beam-u-postp-p-analysis"
        and beam_report.get("gate", {}).get("passed") is True
        and beam_report.get("gate", {}).get("performance_threshold_used") is False
        and beam_report.get("population", {}).get("records") == dev_records
        and beam_report.get("inputs", {}).get("predictions_sha256")
        == evidence[str(beam_matrix)]
        and beam_report.get("inputs", {}).get("scores_sha256")
        == evidence[str(beam_scores_path)]
        and beam_report.get("inputs", {}).get("predecessor_gate_sha256")
        == evidence[str(dev_gate_path)],
        "AMR beam report does not bind full development",
    )
    _require(
        beam_scores.get("audit") == "amr-generation-smatchpp"
        and beam_scores.get("gate", {}).get("passed") is True
        and beam_scores.get("gate", {}).get("performance_threshold_used") is False
        and beam_scores.get("predictions", {}).get("sha256")
        == evidence[str(beam_matrix)],
        "AMR beam scores are incomplete",
    )
    beam_release_summary = beam_release.get("release", {})
    _require(
        beam_release.get("schema_version") == 2
        and beam_release.get("audit") == "penman-certified-release"
        and beam_release.get("gate", {}).get("passed") is True
        and beam_release.get("gate", {}).get("performance_threshold_used") is False
        and beam_release.get("input", {}).get("predictions_sha256")
        == evidence[str(beam_online)]
        and beam_release.get("population", {}).get("examples") == dev_records
        and beam_release.get("population", {}).get("rows") == dev_records
        and has_complete_selective_release_accounting(
            beam_release_summary,
            attempts=dev_records,
            guarantee=PENMAN_RELEASE_GUARANTEE,
        ),
        "AMR online-P beam release is incomplete",
    )
    heldout = _matrix(
        run_root, test_stem, "test", test_records, config_sha, evidence,
        [dev_gate_path, beam_report_path, beam_release_path],
    )
    return {
        "schema_version": 2,
        "audit": "amr-transfer-suite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "cross-representation constrained generation; structural release "
            "does not imply AMR task-semantic correctness"
        ),
        "certification": {
            "invariant": (
                "every released P graph satisfies both the registered local "
                "PENMAN policy and the independent PENMAN parser"
            ),
            "scope": "released outputs under the registered policy and audited preconditions",
            "orthogonal_to_deterministic_search": True,
            "non_guarantees": [
                "AMR task-semantic correctness",
                "factual correctness",
                "availability of a releasable graph for every decoding attempt",
            ],
        },
        "host": config["host"],
        "evidence": evidence,
        "resource_plan": resource_plan,
        "training": training_gate["observed"],
        "matrices": [pilot, development, heldout],
        "beam": {
            "report_path": str(beam_report_path),
            "groups": beam_report["groups"],
            "comparisons": beam_report["comparisons"],
            "selection": beam_report["selection"],
            "search": beam_report["search"],
            "release": beam_release_summary,
        },
        "gate": {"passed": True, "performance_threshold_used": False},
    }


def _rate(group: dict[str, Any]) -> str:
    return f"{group.get('raw_official_penman', 0)}/{group.get('records', 0)}"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AMR automata-transfer suite",
        "",
        f"Host: `{report['host']['repo_id']}`.",
        "",
        (
            "The certified release turns structural validity from an empirical "
            "rate into a policy-relative invariant over released P outputs. "
            "It does not make search deterministic or certify that a graph "
            "expresses the correct sentence meaning. Release coverage is an "
            "empirical availability result and is reported separately."
        ),
        "",
        "| Split | N | U PENMAN | P PENMAN | U Smatch | P Smatch | Δ P−U [95% CI] | P release coverage | Certified policy validity among released |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for matrix in report["matrices"]:
        paired = matrix["paired"]
        smatch = paired["smatch"]
        attempts = int(matrix["examples"])
        released = int(matrix["release"]["releasable_rows"])
        release_coverage = 100.0 * released / attempts
        lines.append(
            f"| {matrix['split']} | {matrix['examples']} | {_rate(matrix['u'])} | "
            f"{_rate(matrix['p'])} | {smatch['u_corpus_f1']:.4f} | "
            f"{smatch['p_corpus_f1']:.4f} | {smatch['delta_p_minus_u']:+.4f} "
            f"[{smatch['delta_ci_lower']:+.4f}, {smatch['delta_ci_upper']:+.4f}] | "
            f"{released}/{attempts} ({release_coverage:.2f}%) | 100.00% |"
        )
    lines.extend(
        [
            "",
            "## Development beam diagnostic",
            "",
            "| Contrast | Δ Smatch [95% CI] | Repairs/regressions | McNemar/Holm |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in report["beam"]["comparisons"].items():
        lines.append(
            f"| {name} | {comparison['corpus_smatch_delta']:+.4f} "
            f"[{comparison['delta_ci_lower']:+.4f}, {comparison['delta_ci_upper']:+.4f}] | "
            f"{comparison['repairs']}/{comparison['regressions']} | "
            f"{comparison['exact_mcnemar_p']:.4f}/"
            f"{comparison['holm_adjusted_mcnemar_p']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--resource-plan", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/amr_transfer_experiment.json")
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    try:
        report = collect(args.config, args.run_root, args.resource_plan)
        markdown = to_markdown(report)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.verify_existing:
        try:
            stored = json.loads(args.output_json.read_text(encoding="utf-8"))
            stored_markdown = args.output_markdown.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if _without_timestamp(stored) != _without_timestamp(report):
            parser.error("stored AMR suite JSON differs from current evidence")
        if stored_markdown != markdown:
            parser.error("stored AMR suite Markdown differs from current evidence")
        print("verified AMR suite against current evidence")
        return 0
    if args.output_json.exists() or args.output_markdown.exists():
        parser.error("refusing to overwrite AMR suite output")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")
    print(f"AMR suite: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
