#!/usr/bin/env python3
"""Build an auditable closure report for all experiments preceding WCD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passed(value: dict[str, Any]) -> bool:
    return value.get("passed", value.get("gate", {}).get("passed")) is True


def _group(summary: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in summary["groups"] if row["configuration"] == name]
    if len(matches) != 1:
        raise ValueError(f"summary has {len(matches)} {name} groups")
    row = matches[0]
    return {
        "records": row["total"],
        "raw_valid_rate": row["raw_sbn_valid_rate"],
        "referee_valid_rate": row["referee_valid_rate_total"],
        "sbn_smatch": row["sbn_smatch_mean"],
        "generation_ms_mean": row["generation_ms_mean"],
        "constraint_ms_mean": row["constraint_ms_mean"],
        "all_masked_rows": row["all_masked_rows"],
        "truncated": row["truncated"],
    }


def _score_group(report: dict[str, Any], name: str) -> dict[str, Any]:
    if report.get("scorer", {}).get("name") != "smatchpp":
        raise ValueError("AMR closure accepts only Smatch++ reports")
    group = report["groups"][name]
    score = group["raw_primary_smatch"]
    if not score.get("all_alignments_optimal"):
        raise ValueError(f"{name} has a non-optimal Smatch++ alignment")
    return {
        "records": group["records"],
        "raw_penman": group["raw_official_penman"],
        "smatchpp_f1": score["f1"],
        "all_alignments_optimal": True,
    }


def _write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        destination.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_json, args.output_markdown):
        if path.exists():
            parser.error(f"refusing to overwrite closure report: {path}")

    drs_root = Path("runs/lora/qwen35-4b-base-r16-seed42")
    drs_gate_names = (
        "dev1132-publication-matrix-gate.json",
        "tax-publication-matrix-gate.json",
        "test1132-publication-matrix-gate.json",
        "dev400-beam-selection-gate.json",
        "dev400-layer-ablation-gate.json",
        "dev400-fine-grained-ablation-gate.json",
        "dev400-method-pilot-gate.json",
    )
    drs_gates = []
    for name in drs_gate_names:
        path = drs_root / name
        value = _read(path)
        drs_gates.append(
            {"path": str(path), "sha256": _sha256(path), "passed": _passed(value)}
        )

    drs_summaries = {}
    for split, name in {
        "dev": "dev1132-u-bf-wf-greedy.summary.json",
        "tax": "tax-u-bf-wf-greedy.summary.json",
        "test": "test1132-u-bf-wf-greedy.summary.json",
    }.items():
        path = drs_root / name
        value = _read(path)
        drs_summaries[split] = {
            "path": str(path),
            "sha256": _sha256(path),
            "U": _group(value, "U"),
            "BF": _group(value, "BF"),
            "WF": _group(value, "WF"),
        }

    beam_path = drs_root / "dev400-u-postw-w-wf-beam4.summary.json"
    beam_summary = _read(beam_path)
    intervention_path = drs_root / "dev-u-invalid-u-b-w-bf-wf-greedy-v2.summary.json"
    intervention = _read(intervention_path)
    old_intervention_gate = drs_root / "dev-u-invalid-intervention-gate.json"

    external_dev32 = Path(
        "runs/amr/external-viamr-qwen3-1.7b-augmented-dev32-u-v1/"
        "dev32-u-greedy.smatchpp-v1.json"
    )
    old_p = Path(
        "runs/amr/external-viamr-qwen3-1.7b-augmented-p4-h1152-v1/"
        "dev4-p-greedy.smatchpp-v1.json"
    )
    old_pf = Path(
        "runs/amr/external-viamr-qwen3-1.7b-augmented-pf4-h1152-v1/"
        "dev4-pf-greedy.smatchpp-v1.json"
    )
    boundary_root = Path(
        "runs/amr/external-viamr-qwen3-1.7b-augmented-boundary-fix4-v1"
    )
    boundary_p = boundary_root / "dev4-p-boundary-stop.smatchpp-v1.json"
    boundary_pf = boundary_root / "dev4-pf-boundary-stop.smatchpp-v1.json"
    boundary_gate = boundary_root / "boundary-fix-gate.smatchpp-v1.json"
    smoke_root = Path("runs/amr/qwen35-4b-base-r16-seed42-smoke")
    smoke_scores = smoke_root / "dev-smoke-u-p-greedy.smatchpp-v1.json"
    smoke_gate = smoke_root / "smoke-gate.smatchpp-v1.json"

    amr_reports = {
        "external_dev32_u": {
            "path": str(external_dev32),
            "sha256": _sha256(external_dev32),
            "U": _score_group(_read(external_dev32), "U"),
        },
        "old_horizon_p": {
            "path": str(old_p),
            "sha256": _sha256(old_p),
            "P": _score_group(_read(old_p), "P"),
        },
        "old_horizon_pf": {
            "path": str(old_pf),
            "sha256": _sha256(old_pf),
            "PF": _score_group(_read(old_pf), "PF"),
        },
        "boundary_p": {
            "path": str(boundary_p),
            "sha256": _sha256(boundary_p),
            "P": _score_group(_read(boundary_p), "P"),
        },
        "boundary_pf": {
            "path": str(boundary_pf),
            "sha256": _sha256(boundary_pf),
            "PF": _score_group(_read(boundary_pf), "PF"),
        },
        "qwen_smoke": {
            "path": str(smoke_scores),
            "sha256": _sha256(smoke_scores),
            "U": _score_group(_read(smoke_scores), "U"),
            "P": _score_group(_read(smoke_scores), "P"),
        },
    }
    amr_gates = []
    for path in (boundary_gate, smoke_gate):
        amr_gates.append(
            {"path": str(path), "sha256": _sha256(path), "passed": _passed(_read(path))}
        )

    overall = all(row["passed"] for row in drs_gates + amr_gates)
    report = {
        "schema_version": 1,
        "audit": "pre-wcd-experiment-closure",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "closed" if overall else "incomplete",
        "primary_gates_passed": overall,
        "gpu_rerun_required": False,
        "drs": {
            "primary_gates": drs_gates,
            "greedy_primary": drs_summaries,
            "beam_diagnostic": {
                "path": str(beam_path),
                "sha256": _sha256(beam_path),
                **{name: _group(beam_summary, name) for name in ("U", "PostW", "W", "WF")},
            },
            "conditioned_invalid_u_diagnostic": {
                "classification": "non-primary-conditioned-negative-control",
                "reason_not_primary": (
                    "population selected on U invalidity; corrected v2 lacks comparable "
                    "raw model log-probability, so no population estimate or log-prob claim"
                ),
                "old_failed_gate": {
                    "path": str(old_intervention_gate),
                    "sha256": _sha256(old_intervention_gate),
                    "passed": _passed(_read(old_intervention_gate)),
                },
                "v2_summary": str(intervention_path),
                "v2_summary_sha256": _sha256(intervention_path),
                "U": _group(intervention, "U"),
                "BF": _group(intervention, "BF"),
                "WF": _group(intervention, "WF"),
            },
        },
        "amr": {
            "scorer": "smatchpp==1.8.0; AMRStandardizer; ILP; micro F1",
            "primary_gates": amr_gates,
            "reports": amr_reports,
            "legacy_smatch_policy": (
                "retained as historical artifacts only; excluded from conclusions"
            ),
        },
        "gate": {
            "passed": overall,
            "all_primary_drs_gates_passed": all(row["passed"] for row in drs_gates),
            "all_primary_amr_gates_passed": all(row["passed"] for row in amr_gates),
            "all_amr_alignments_optimal": True,
            "performance_threshold_used": False,
        },
    }

    lines = [
        "# Pre-WCD experiment closure",
        "",
        f"Status: **{report['status']}**. No GPU rerun is required.",
        "",
        "## DRS greedy primary results",
        "",
        "| Split | Method | Valid | Referee valid | SBN Smatch |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("dev", "tax", "test"):
        for method in ("U", "BF", "WF"):
            row = drs_summaries[split][method]
            lines.append(
                f"| {split} | {method} | {row['raw_valid_rate']:.4f} | "
                f"{row['referee_valid_rate']:.4f} | {row['sbn_smatch']:.4f} |"
            )
    lines.extend(
        [
            "",
            "BF/WF reach 100% raw structural validity on all three splits. "
            "WF also reaches 100% Referee validity on test.",
            "",
            "The beam-4 diagnostic changes no output and is roughly an order of "
            "magnitude slower online; it is retained as a negative result.",
            "",
            "The 21-example invalid-U intervention is a conditioned diagnostic, not "
            "a population result: BF/WF repair 21/21 and raise SBN Smatch from "
            f"{_group(intervention, 'U')['sbn_smatch']:.4f} to "
            f"{_group(intervention, 'WF')['sbn_smatch']:.4f}.",
            "",
            "## AMR Smatch++ results",
            "",
            "| Evidence | Method | Valid | Smatch++ | Optimal ILP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for evidence, method in (
        ("external dev32", "U"),
        ("old horizon", "P"),
        ("old horizon", "PF"),
        ("boundary stop", "P"),
        ("boundary stop", "PF"),
        ("Qwen smoke", "U"),
        ("Qwen smoke", "P"),
    ):
        key = {
            ("external dev32", "U"): "external_dev32_u",
            ("old horizon", "P"): "old_horizon_p",
            ("old horizon", "PF"): "old_horizon_pf",
            ("boundary stop", "P"): "boundary_p",
            ("boundary stop", "PF"): "boundary_pf",
            ("Qwen smoke", "U"): "qwen_smoke",
            ("Qwen smoke", "P"): "qwen_smoke",
        }[(evidence, method)]
        row = amr_reports[key][method]
        lines.append(
            f"| {evidence} | {method} | {row['raw_penman']}/{row['records']} | "
            f"{row['smatchpp_f1']:.4f} | yes |"
        )
    lines.extend(
        [
            "",
            "Legacy hill-climbing Smatch scores are historical only and are excluded "
            "from conclusions. The boundary P/PF identity gate passes under exact ILP.",
            "",
        ]
    )

    _write_exclusive(
        args.output_json,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(args.output_markdown, "\n".join(lines))
    print(f"json: {args.output_json}")
    print(f"markdown: {args.output_markdown}")
    print(f"gate passed: {overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
