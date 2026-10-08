"""Merge complete frozen results for paper tables without rewriting a gate.

No inference, scoring, output repair, model selection or new confidence interval.
Run on the server where the source-hashed experimental inputs are available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from arr1012_review_mechanism_analysis import validate_aggregate


N = 1132


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("paper data contain a nonfinite or nonnumeric value")
    return value


def count(value):
    if type(value) is not int or not 0 <= value <= N:
        raise ValueError("paper count is outside the full-test denominator")
    return value


def validate_pair(pair):
    for policy in ("native", "compatibility_zeroed"):
        result = pair[policy]
        if result["n"] != N:
            raise ValueError("paired result does not retain the full input denominator")
        counts = result["counts"]
        if set(counts) != {"common_valid", "rescue", "lost", "neither_valid"}:
            raise ValueError("paired validity strata differ")
        if sum(count(v) for v in counts.values()) != N:
            raise ValueError("paired strata do not partition the full test")
        for name in ("delta", "common_valid", "rescue", "lost"):
            value = result[name]
            finite(value["mean_points"])
            lo, hi = value["ci95_points"]
            if finite(lo) > finite(hi):
                raise ValueError("paired confidence interval is inverted")
        total = sum(result[name]["mean_points"] for name in ("common_valid", "rescue", "lost"))
        if not math.isclose(total, result["delta"]["mean_points"], abs_tol=1e-10, rel_tol=0):
            raise ValueError("common-valid/rescue/lost contributions do not close")


def merge_reports(frozen, mechanism, audit):
    validate_aggregate(frozen)
    if mechanism["status"] != "complete_mechanism_evidence" or audit["status"] != "complete_corrected_audit_not_release":
        raise ValueError("paper export requires completed mechanism and corrected audit reports")
    for report in (mechanism, audit):
        if report["frozen_scientific_gate"] != frozen["status"] or report["frozen_gate_errors"] != frozen["errors"]:
            raise ValueError("an analysis changes the frozen scientific gate")
        if set(report["summaries"]) != set(frozen["summaries"]):
            raise ValueError("analysis is missing a registered matrix condition")
    if set(audit["comparisons"]) != set(frozen["comparisons"]) or len(audit["comparisons"]) != 9:
        raise ValueError("not all nine registered paired comparisons are complete")
    if audit["bootstrap"].get("resamples") != 20000 or audit["bootstrap"].get("unit") != "input" or audit["bootstrap"].get("seed") != 20261001:
        raise ValueError("paired-bootstrap protocol differs")
    false_accepts = audit["existing_corrected_false_accepts"] + audit["new_false_accept_observations"]
    if false_accepts < 0 or (false_accepts and frozen["status"] == "passed"):
        raise ValueError("independent checker failure was presented as a gate pass")
    cells = {}
    for key, summary in frozen["summaries"].items():
        checked, measured = audit["summaries"][key], mechanism["summaries"][key]
        if checked["n"] != N or measured["n"] != N:
            raise ValueError("a source report is not a complete full-test cell")
        native = count(summary["raw_valid"])
        compatible = count(checked["fallback_referee_compatible_delivered_valid"])
        score = finite(summary["macro_f1"]) * 100
        if not 0 <= score <= 100 or native != checked["delivered_native_valid"] or compatible > native:
            raise ValueError("native/independent validity counts are inconsistent")
        if not math.isclose(score, finite(checked["native_macro_points"]), abs_tol=1e-10, rel_tol=0):
            raise ValueError("frozen and audited native scores differ")
        zeroed = finite(checked["compatibility_zeroed_macro_points"])
        if not 0 <= zeroed <= score + 1e-10:
            raise ValueError("rejection-zeroing increased the frozen score")
        if key.endswith(":WWCD"):
            paths = summary["certificate_paths"]
            if native != N or summary["accepting_eos"] != N or summary["all_masked"] or summary["prefix_unavailable"]:
                raise ValueError("APD native acceptance/EOS/support invariant failed")
            if sum(paths[name] for name in ("carried_tokens", "cached_tokens", "replanned_tokens")) != paths["rows"]:
                raise ValueError("APD carry/cache/replan accounting does not close")
        cells[key] = {
            "n": N, "native_macro_f1_points": score,
            "native_delivered_valid": native,
            "fallback_referee_compatible_delivered_valid": compatible,
            "compatibility_zeroed_macro_f1_points": zeroed,
            "accepting_eos": count(summary["accepting_eos"]),
            "decoder_failures": count(summary["decoder_failures"]),
            "operations": summary["operations"], "certificate_paths": summary["certificate_paths"],
            "output_length_mean": finite(summary["output_length_mean"]),
            "online_generation_seconds": finite(summary["online_generation_seconds"]),
            "posthoc_seconds": finite(summary["posthoc_seconds"]),
            "mechanism": measured,
        }
    comparisons = {}
    for key, pair in audit["comparisons"].items():
        validate_pair(pair)
        prefix, baseline = key.split(":APD-minus-", 1)
        apd, base = cells[prefix + ":WWCD"], cells[prefix + ":" + baseline]
        for policy, validity in (("native", "native_delivered_valid"),
                                 ("compatibility_zeroed", "fallback_referee_compatible_delivered_valid")):
            strata = pair[policy]["counts"]
            if strata["common_valid"] + strata["rescue"] != apd[validity] or strata["common_valid"] + strata["lost"] != base[validity]:
                raise ValueError("paired strata do not match the compared full-test cells")
        original = frozen["comparisons"][key]
        if original["paired_n"] != N or not math.isclose(pair["native"]["delta"]["mean_points"], finite(original["delta"]["mean"]) * 100, abs_tol=1e-10, rel_tol=0):
            raise ValueError("native paired point estimate differs from the frozen comparison")
        comparisons[key] = {"n": N, "byte_identical": count(original["byte_identical"]), **pair}
    return {
        "status": "complete_paper_data_not_scientific_release",
        "frozen_scientific_gate": frozen["status"], "frozen_gate_errors": frozen["errors"],
        "corrected_independent_agreement_passed": false_accepts == 0,
        "corrected_false_accept_observations": false_accepts,
        "cells": cells, "comparisons": comparisons, "bootstrap": audit["bootstrap"],
        "interpretation": [
            "Native acceptance and independent target compatibility are separate measured quantities.",
            "Compatibility-zeroed scores retain all inputs and original exact alignments; no graph is repaired.",
            "Confidence intervals use paired inputs at a fixed checkpoint, not training-seed uncertainty.",
            "New-proposal denial is synthetic availability stress, not a measured natural timeout.",
            "Fresh proposal requests include denied requests; constructor executions are a separate operation count.",
            "Verified completion lengths are constructive upper bounds, not shortest completion costs.",
            "Cached W320 timing is not measured generation time at shorter output budgets.",
            "An export cannot pass or waive a failed scientific agreement gate.",
        ],
    }


def load_completed(path, manifest=False):
    report = json.loads(path.read_text(encoding="utf-8"))
    if manifest:
        lock = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if lock["status"] != "complete" or lock["output_sha256"] != digest(path):
            raise ValueError("analysis manifest or output hash mismatch")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("paper data must use a new path outside the frozen run")
    aggregate = args.root / "cpu/aggregate-0/review-response-report.json"
    frozen = load_completed(aggregate)
    mechanism = load_completed(args.mechanism, manifest=True)
    audit = load_completed(args.audit, manifest=True)
    result = merge_reports(frozen, mechanism, audit)
    sources = {str(p): digest(p) for p in (aggregate, args.mechanism, args.audit, Path(__file__))}
    for report, field in ((frozen, "evidence"), (mechanism, "evidence"), (audit, "sources")):
        for name, expected in report[field].items():
            if digest(Path(name)) != expected:
                raise ValueError("an experimental or analysis source hash changed")
            if name in sources and sources[name] != expected:
                raise ValueError("source locks disagree across reports")
            sources[name] = expected
    result["sources"] = sources
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump({"status": "complete", "output_sha256": digest(args.output),
                   "scientific_gate": frozen["status"],
                   "corrected_independent_agreement_passed": result["corrected_independent_agreement_passed"]}, stream, indent=2)
    print(json.dumps({"status": result["status"], "conditions": len(result["cells"]),
                      "paired_comparisons": len(result["comparisons"]), "scientific_gate": frozen["status"]}))


if __name__ == "__main__":
    main()
