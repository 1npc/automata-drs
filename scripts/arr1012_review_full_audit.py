"""Corrected independent audit and paired sensitivity of the complete matrix.

No model inference, new alignment, output repair, exclusion, or gate waiver.
Require all 18 full-test cells before writing a new immutable diagnostic report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from arr1012_review_mechanism_analysis import validate_aggregate
from arr1012_review_referee_diagnostics import (
    checked, diagnose, digest, load_module, paired_contributions,
)


def condition_key(row, manifest):
    horizon = row["decode"].get("horizon", manifest.get("max_new_tokens"))
    rate = row.get("review_operations", {}).get("denial_rate", 0)
    if type(horizon) is not int or horizon <= 0 or type(rate) not in (int, float):
        raise ValueError("invalid registered horizon or denial rate")
    return f"H{horizon}:deny{rate:g}:{row['configuration']}"


def score_vector(row, score, raw_accepted, structural_accepted):
    """Preserve the exact original delivered-validity and failure-zero policy."""
    if (row["id"], row["configuration"]) != (score["id"], score["configuration"]):
        raise ValueError("score identity/order changed")
    f1 = score["f1"]
    if score["exact"] is not True or f1 != score["f1_upper"] or not math.isfinite(f1) or not 0 <= f1 <= 1:
        raise ValueError("per-input exact score invalid")
    valid = raw_accepted and row["generation"].get("decoder_failure") is None
    if type(score["raw_W_valid"]) is not bool or score["raw_W_valid"] != valid:
        raise ValueError("delivered validity differs from frozen zero policy")
    if not valid and f1 != 0:
        raise ValueError("invalid or explicit failure received nonzero score")
    compatible = valid and structural_accepted
    return (f1, f1 if compatible else 0, valid, compatible)


def require_complete_cells(cells, frozen):
    if set(cells) != set(frozen["summaries"]):
        raise ValueError("corrected audit lacks a registered matrix condition")
    order = None
    for key, mapping in cells.items():
        ids = list(mapping)
        if len(ids) != 1132 or len(set(ids)) != 1132:
            raise ValueError(f"incomplete or duplicated condition: {key}")
        if order is None:
            order = ids
        elif ids != order:
            raise ValueError("condition identity order differs")
        native = sum(vector[2] for vector in mapping.values())
        if native != frozen["summaries"][key]["raw_valid"]:
            raise ValueError("native validity differs from complete frozen aggregate")
        mean = sum(vector[0] for vector in mapping.values()) / 1132
        if not math.isclose(mean, frozen["summaries"][key]["macro_f1"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("native score differs from complete frozen aggregate")
    return order


def check_surface(batch, upstream, signature, raw):
    accepted, reason = batch.local_valid(raw)
    try:
        projection = batch.sbn_graph_to_clf(batch.parse_raw(raw)).clauses
    except (ValueError, IndexError, KeyError) as exc:
        return accepted, reason, {"accepted": False, "reason": str(exc)}, {
            "accepted": False, "reason": str(exc)}, True
    structural = checked(upstream, projection)
    try:
        upstream.check_clf(list(projection), signature)
        official = {"accepted": True, "reason": None}
    except RuntimeError as exc:
        official = {"accepted": False, "reason": str(exc)}
    return accepted, reason, structural, official, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--auditor", required=True, type=Path)
    parser.add_argument("--existing-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output_manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or output_manifest.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("new diagnostic evidence must be outside the frozen run")
    aggregate_path = args.root / "cpu/aggregate-0/review-response-report.json"
    frozen = json.loads(aggregate_path.read_text(encoding="utf-8"))
    validate_aggregate(frozen)
    sources = {str(aggregate_path): digest(aggregate_path), str(Path(__file__)): digest(Path(__file__)),
               str(args.auditor): digest(args.auditor)}
    for name, expected in frozen["evidence"].items():
        if digest(Path(name)) != expected:
            raise ValueError("frozen aggregate evidence hash changed")
        sources[name] = expected
    upstream_path, signature_path = (args.root / "vendor" / name for name in ("clf_referee.py", "clf_signature.yaml"))
    sys.path.insert(0, str(args.root / "code/src"))
    batch = load_module("corrected_full_matrix_batch", args.auditor)
    upstream = load_module("pinned_full_matrix_referee", upstream_path)
    signature = upstream.get_signature(str(signature_path))
    for path in (upstream_path, signature_path, Path(batch.spec.__file__),
                 args.root / "code/src/automata_drs/sbn/clf.py",
                 Path(__file__).with_name("arr1012_review_mechanism_analysis.py"),
                 Path(__file__).with_name("arr1012_review_referee_diagnostics.py")):
        sources[str(path)] = digest(path)
    existing = json.loads(args.existing_audit.read_text(encoding="utf-8"))
    existing_manifest_path = args.existing_audit.with_suffix(".manifest.json")
    existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
    if existing_manifest["status"] != "complete" or existing_manifest["output_sha256"] != digest(args.existing_audit):
        raise ValueError("corrected original-output audit manifest/hash mismatch")
    if existing_manifest["scientific_agreement_passed"] != (existing["false_accept_structural"] == 0):
        raise ValueError("corrected original-output audit gate changed")
    for key, path in (("auditor_sha256", args.auditor), ("referee_sha256", upstream_path),
                      ("signature_sha256", signature_path)):
        if existing[key] != digest(path):
            raise ValueError("original-output and new-output audit sources differ")
    for name, expected in existing["sources"].items():
        if digest(Path(name)) != expected:
            raise ValueError("original audit input changed")
        sources[name] = expected
    sources[str(args.existing_audit)] = digest(args.existing_audit)
    sources[str(existing_manifest_path)] = digest(existing_manifest_path)
    cells, counts, rejected = defaultdict(dict), defaultdict(Counter), []
    references = {}
    for index in range(20):
        for stem in (("natural68", "natural78", "offline") if index < 8 else ("denial",)):
            predictions = args.root / "matrix" / f"task-{index}" / f"{stem}.jsonl"
            scores_path = args.root / "cpu" / f"score-{index}" / f"{stem}.scores.jsonl"
            gen_manifest = json.loads(Path(str(predictions) + ".manifest.json").read_text(encoding="utf-8"))
            score_manifest = json.loads(Path(str(scores_path) + ".manifest.json").read_text(encoding="utf-8"))
            if gen_manifest["status"] != "complete" or score_manifest["status"] != "complete" or not score_manifest["all_exact"]:
                raise ValueError("incomplete generation or exact ILP scoring")
            if score_manifest["smatchpp"] != "1.8.0":
                raise ValueError("unexpected scorer version")
            rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
            scores = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
            for row, score in zip(rows, scores, strict=True):
                key = condition_key(row, gen_manifest)
                ident = row["id"]
                if ident in cells[key]:
                    raise ValueError("duplicate condition identity")
                if ident in references and references[ident] != row["reference"]:
                    raise ValueError("reference differs across matched conditions")
                references[ident] = row["reference"]
                raw, reason, structural, official, conversion_failed = check_surface(batch, upstream, signature, row["raw_prediction"])
                vector = score_vector(row, score, raw, structural["accepted"])
                cells[key][ident] = vector
                counts[key].update({"raw_native_accept": int(raw), "structural_accept": int(structural["accepted"]),
                                    "official_signature_accept": int(official["accepted"]),
                                    "conversion_failure": int(conversion_failed),
                                    "decoder_failure": int(row["generation"].get("decoder_failure") is not None)})
                if row["configuration"] == "WWCD" and not row["generation"]["ended_with_eos"]:
                    raise ValueError("APD lacks accepting EOS")
                if raw and not structural["accepted"]:
                    case = {"id": ident, "raw": row["raw_prediction"]}
                    if conversion_failed:
                        diagnosis = {"diagnosis": "independent conversion failure", "original_check": structural}
                    else:
                        diagnosis = diagnose(upstream, batch.parse_raw, batch.sbn_graph_to_clf, case)
                    rejected.append({"condition": key, "id": ident, "source": str(predictions),
                                     "raw_sha256": hashlib.sha256(row["raw_prediction"].encode("utf-8")).hexdigest(),
                                     "native_delivered_valid": vector[2], **diagnosis})
    ordered = require_complete_cells(cells, frozen)
    summaries = {}
    for key, mapping in cells.items():
        values = list(mapping.values())
        summaries[key] = {"n": 1132, **dict(counts[key]),
                          "delivered_native_valid": sum(v[2] for v in values),
                          "fallback_referee_compatible_delivered_valid": sum(v[3] for v in values),
                          "native_macro_points": sum(v[0] for v in values) / 1132 * 100,
                          "compatibility_zeroed_macro_points": sum(v[1] for v in values) / 1132 * 100}
    comparisons = {}
    pairs = [(h, 0, baseline) for h in (68, 78) for baseline in ("W", "WF", "WRollback")]
    pairs += [(68, rate, "WF") for rate in (.25, .5, .75)]
    for horizon, rate, baseline in pairs:
        apd_key, base_key = f"H{horizon}:deny{rate:g}:WWCD", f"H{horizon}:deny{rate:g}:{baseline}"
        comparisons[f"H{horizon}:deny{rate:g}:APD-minus-{baseline}"] = paired_contributions(
            [cells[apd_key][ident] for ident in ordered], [cells[base_key][ident] for ident in ordered])
    result = {"status": "complete_corrected_audit_not_release", "sources": sources,
              "frozen_scientific_gate": frozen["status"], "frozen_gate_errors": frozen["errors"],
              "existing_corrected_false_accepts": existing["false_accept_structural"],
              "new_false_accept_observations": len(rejected), "rejected_cases": rejected,
              "summaries": summaries, "comparisons": comparisons,
              "bootstrap": {"resamples": 20000, "unit": "input", "seed": 20261001,
                            "interval": "paired percentile 95%", "scoring_unit": "example-macro F1, 0--100"},
              "interpretation": [
                  "Every original output, exact alignment and all 1132 inputs per condition remain unchanged.",
                  "Only original checker-rejected delivered outputs receive zero in the sensitivity policy.",
                  "Raw parseability and delivered validity differ for explicit WF failure prefixes.",
                  "Inventory-free structural checks and the pinned official signature are distinct audits.",
                  "Predicate-renaming probes diagnose incompatibility; they are not prediction repairs.",
                  "This report cannot pass, overwrite or waive the frozen scientific agreement gate.",
                  "Bootstrap intervals condition on this checkpoint and its outputs, not training-seed uncertainty.",
              ]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    with output_manifest.open("x", encoding="utf-8") as stream:
        json.dump({"status": "complete", "output_sha256": digest(args.output),
                   "frozen_scientific_gate": frozen["status"],
                   "corrected_scientific_agreement_passed": existing["false_accept_structural"] + len(rejected) == 0}, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "frozen_scientific_gate", "new_false_accept_observations")}))


if __name__ == "__main__":
    main()
