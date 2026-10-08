"""Diagnose, without waiving, frozen Referee agreement failures.

The original projections, outputs, scores, and gate remain unchanged. Renaming
an unsupported lexical predicate is a graph-isomorphic *diagnostic control*,
not a repair of a prediction and not evidence that the original CLF is valid.
Run against the experiment's pinned source and Referee, outside its frozen root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shlex
import sys
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    description = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(description)
    sys.modules[name] = module
    description.loader.exec_module(module)
    return module


def checked(upstream, clauses):
    try:
        upstream.check_clf(list(clauses), {})
    except RuntimeError as exc:
        return {"accepted": False, "reason": str(exc)}
    return {"accepted": True, "reason": None}


def lexical_type(upstream, clause):
    try:
        kind, _ = upstream.operator_type(clause[1], (clause[0],) + tuple(clause[2:]), {})
    except RuntimeError:
        return False
    return kind == "LEX"


def diagnose(upstream, parser, project, example):
    raw = example["raw"]
    graph = parser(raw)
    projection = [tuple(clause) for clause in project(graph).clauses]
    original = checked(upstream, projection)
    if original["accepted"]:
        raise RuntimeError("frozen rejection no longer reproduces")
    renamed = list(projection)
    replacements = []
    for index, concept in enumerate(graph.concepts):
        at = 2 * index + 1
        clause = projection[at]
        lemma, pos, sense = concept.label.rsplit(".", 2)
        if clause[1:3] != (lemma, f'"{pos}.{sense}"'):
            raise RuntimeError("unexpected lexical projection layout")
        if lexical_type(upstream, clause):
            continue
        # Keep every variable, box, sense, role, constant and edge unchanged.
        candidate = (clause[0], f"audit_lexeme_{index}", *clause[2:])
        supported = lexical_type(upstream, candidate)
        replacements.append({"concept_index": index, "original_synset": concept.label,
                             "original_lemma": lemma, "probe_supported": supported})
        renamed[at] = candidate
    probe = checked(upstream, renamed)
    category = "lexical-format incompatibility" if replacements and probe["accepted"] else "unresolved rejection"
    return {"id": example["id"], "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "original_check": original, "lexical_replacements": replacements,
            "renamed_predicate_control": probe, "diagnosis": category,
            "original_is_independently_valid": False,
            "probe_is_prediction_or_score_repair": False}


def paired_contributions(apd, baseline):
    """Input-paired sensitivity, using the frozen report's resampling policy."""
    import numpy as np

    a, b = np.asarray(apd, dtype=float), np.asarray(baseline, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 4 or not len(a):
        raise ValueError("expected a nonempty matched (score, adjusted, valid, compatible) grid")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite paired evidence")
    if ((a < 0) | (a > 1)).any() or ((b < 0) | (b > 1)).any():
        raise ValueError("scores and indicators must be in [0, 1]")
    for rows in (a, b):
        if not np.isin(rows[:, 2:], (0, 1)).all():
            raise ValueError("validity indicators must be Boolean")
        if (rows[:, 3] > rows[:, 2]).any():
            raise ValueError("compatibility cannot exceed native validity")
        if ((rows[:, :2] != 0) & (rows[:, 2:] == 0)).any():
            raise ValueError("invalid output must have zero score")
        if ((rows[:, 3] == 1) & (rows[:, 0] != rows[:, 1])).any():
            raise ValueError("compatible output score changed")
    result = {}
    for policy, score_at, valid_at in (("native", 0, 2), ("compatibility_zeroed", 1, 3)):
        av, bv = a[:, valid_at].astype(bool), b[:, valid_at].astype(bool)
        common, rescue, lost = av & bv, av & ~bv, ~av & bv
        delta = a[:, score_at] - b[:, score_at]
        values = np.column_stack((delta, np.where(common, delta, 0),
                                  np.where(rescue, a[:, score_at], 0),
                                  np.where(lost, -b[:, score_at], 0)))
        if not np.allclose(values[:, 0], values[:, 1:].sum(axis=1), atol=1e-12, rtol=0):
            raise ValueError("paired contribution identity failed")
        rng = np.random.default_rng(20261001)
        means = np.concatenate([values[rng.integers(0, len(values), size=(250, len(values)))].mean(axis=1)
                                for _ in range(80)]) * 100
        estimate = values.mean(axis=0) * 100
        lower, upper = np.quantile(means, (0.025, 0.975), axis=0)
        result[policy] = {"n": len(values), "counts": {
            "common_valid": int(common.sum()), "rescue": int(rescue.sum()),
            "lost": int(lost.sum()), "neither_valid": int((~av & ~bv).sum())},
            **{name: {"mean_points": float(estimate[at]),
                      "ci95_points": [float(lower[at]), float(upper[at])]}
               for at, name in enumerate(("delta", "common_valid", "rescue", "lost"))}}
    return result


def score_impact(root: Path, indices, records, hashes, *, paired=False):
    """Zero already-scored checker-rejected surfaces; never repair or rescore."""
    if sorted(indices) != list(range(8)):
        raise RuntimeError("impact analysis requires all eight complete natural shards")
    rejected = {(row["id"], row["raw_sha256"]) for row in records}
    cells, references = {}, {}
    for index in indices:
        for stem in ("natural68", "natural78", "offline"):
            path = root / "matrix" / f"task-{index}" / (stem + ".jsonl")
            scored_path = root / "cpu" / f"score-{index}" / (stem + ".scores.jsonl")
            generation_manifest = Path(str(path) + ".manifest.json")
            score_manifest = Path(str(scored_path) + ".manifest.json")
            generation = json.loads(generation_manifest.read_text(encoding="utf-8"))
            scoring = json.loads(score_manifest.read_text(encoding="utf-8"))
            if generation["status"] != "complete" or scoring["status"] != "complete":
                raise RuntimeError("incomplete generation or scoring")
            if not scoring["all_exact"] or scoring["smatchpp"] != "1.8.0":
                raise RuntimeError("non-exact or incompatible semantic scoring")
            if generation["output_sha256"] != digest(path) or scoring["input_sha256"] != digest(path) or scoring["output_sha256"] != digest(scored_path):
                raise RuntimeError("score provenance hash mismatch")
            for source in (path, scored_path, generation_manifest, score_manifest):
                hashes[str(source)] = digest(source)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            scores = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines()]
            for row, score in zip(rows, scores, strict=True):
                if (row["id"], row["configuration"]) != (score["id"], score["configuration"]):
                    raise RuntimeError("score identity/order mismatch")
                if not score["exact"] or score["f1"] != score["f1_upper"] or not math.isfinite(score["f1"]) or not 0 <= score["f1"] <= 1:
                    raise RuntimeError("invalid per-example exact F1")
                valid = score["raw_W_valid"]
                if not valid and score["f1"] != 0:
                    raise RuntimeError("invalid output was not zero-scored")
                ident = row["id"]
                if ident in references and references[ident] != row["reference"]:
                    raise RuntimeError("reference changed across conditions")
                references[ident] = row["reference"]
                horizon = row["decode"].get("horizon", generation.get("max_new_tokens"))
                key = f"H{horizon}:{row['configuration']}"
                mapping = cells.setdefault(key, {})
                if ident in mapping:
                    raise RuntimeError("duplicate condition identity")
                raw_sha = hashlib.sha256(row["raw_prediction"].encode("utf-8")).hexdigest()
                compatible = valid and (ident, raw_sha) not in rejected
                mapping[ident] = (score["f1"], score["f1"] if compatible else 0, valid, compatible)
    if len(cells) != 12 or len(references) != 1132:
        raise RuntimeError("expected twelve complete natural full-test conditions")
    ids = set(references)
    summary = {}
    for key, rows in cells.items():
        if set(rows) != ids:
            raise RuntimeError("condition identity grid incomplete")
        native = sum(value[0] for value in rows.values()) / 1132 * 100
        compatible = sum(value[1] for value in rows.values()) / 1132 * 100
        summary[key] = {"n": 1132, "native_valid": sum(value[2] for value in rows.values()),
                        "fallback_referee_compatible_valid": sum(value[3] for value in rows.values()),
                        "native_macro_points": native, "compatibility_zeroed_macro_points": compatible,
                        "score_change_points": compatible - native}
    comparisons = {}
    for horizon in (68, 78):
        apd = summary[f"H{horizon}:WWCD"]
        for baseline in ("W", "WF", "WRollback"):
            base = summary[f"H{horizon}:{baseline}"]
            native = apd["native_macro_points"] - base["native_macro_points"]
            adjusted = apd["compatibility_zeroed_macro_points"] - base["compatibility_zeroed_macro_points"]
            comparisons[f"H{horizon}:APD-minus-{baseline}"] = {
                "n": 1132, "native_delta_points": native,
                "compatibility_zeroed_delta_points": adjusted, "delta_change_points": adjusted - native}
            if paired:
                ordered = sorted(ids)
                result = paired_contributions([cells[f"H{horizon}:WWCD"][ident] for ident in ordered],
                                              [cells[f"H{horizon}:{baseline}"][ident] for ident in ordered])
                if not math.isclose(result["native"]["delta"]["mean_points"], native, abs_tol=1e-10) or not math.isclose(result["compatibility_zeroed"]["delta"]["mean_points"], adjusted, abs_tol=1e-10):
                    raise RuntimeError("paired and descriptive estimates disagree")
                comparisons[f"H{horizon}:APD-minus-{baseline}"]["paired_contributions"] = result
    return {"status": "complete_descriptive_impact_calculation", "summaries": summary,
            "comparisons": comparisons, "scoring_unit": "example-macro F1, 0--100 points",
            "bootstrap": {"computed": paired, "resamples": 20000, "unit": "input",
                          "seed": 20261001, "interval": "percentile 95%", "paired_across_methods": True},
            "interpretation": "This sensitivity calculation sets original fallback-Referee-rejected surfaces to zero without changing any output, alignment, denominator or frozen gate. It is not the official-signature metric, a repaired-parser result, or a new decoding experiment. Bootstrap intervals condition on these fixed outputs and this checkpoint; they do not incorporate training-seed uncertainty."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--indices", nargs="+", type=int, default=[7])
    parser.add_argument("--gold", nargs="*", type=Path, default=[])
    parser.add_argument("--score-impact", action="store_true")
    parser.add_argument("--paired-impact", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.paired_impact and not args.score_impact:
        parser.error("--paired-impact requires --score-impact")
    if args.output.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("use a new diagnostic path, outside the frozen run")
    src = args.root / "code" / "src"
    sys.path.insert(0, str(src))
    batch = load_module("frozen_review_batch", args.root / "code" / "scripts" / "arr1012_review_referee.py")
    upstream_path = args.root / "vendor" / "clf_referee.py"
    upstream = load_module("pinned_diagnostic_referee", upstream_path)
    report_hashes = {str(upstream_path): digest(upstream_path),
                     str(src / "automata_drs" / "sbn" / "spec.py"): digest(src / "automata_drs" / "sbn" / "spec.py"),
                     str(src / "automata_drs" / "sbn" / "clf.py"): digest(src / "automata_drs" / "sbn" / "clf.py"),
                     str(Path(__file__)): digest(Path(__file__))}
    records, counts, original_total = [], Counter(), 0
    for index in args.indices:
        path = args.root / "cpu" / f"score-{index}" / "referee.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["referee_sha256"] != digest(upstream_path):
            raise RuntimeError("Referee source hash changed")
        report_hashes[str(path)] = digest(path)
        for source, expected in audit["sources"].items():
            if digest(Path(source)) != expected:
                raise RuntimeError("audited prediction changed")
            report_hashes[source] = expected
        cases = [item for item in audit["examples"] if item["kind"] == "structural"
                 and item["local_accept"] and not item["upstream_accept"]]
        captured = len(cases)
        expected = audit["false_accept_structural"]
        original_total += expected
        if captured != expected:
            raise RuntimeError("capped examples or conversion failures require a full separate re-audit")
        for example in cases:
            record = diagnose(upstream, batch.parse_raw, batch.sbn_graph_to_clf, example)
            record["audit_task"] = index
            records.append(record)
            counts[record["diagnosis"]] += 1
    gold = {}
    for path in args.gold:
        report_hashes[str(path)] = digest(path)
        unsupported, entries = Counter(), 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entries += 1
            text = line.split("\t", 1)[-1]
            lexer = shlex.shlex(text, posix=False)
            lexer.whitespace_split, lexer.commenters = True, ""
            # SBN uses double-quoted constants; apostrophes are ordinary lemma
            # characters (e.g. contracted names), not shell-style quotations.
            lexer.quotes = '"'
            for word in lexer:
                if batch.spec.is_synset(word):
                    lemma, pos, sense = word.rsplit(".", 2)
                    clause = ("b0", lemma, f'"{pos}.{sense}"', "x0")
                    if not lexical_type(upstream, clause):
                        unsupported[word] += 1
        gold[str(path)] = {"input_rows": entries, "unsupported_synset_occurrences": sum(unsupported.values()),
                           "unsupported_labels": dict(sorted(unsupported.items()))}
    result = {"status": "diagnosed_not_exonerated", "original_structural_gate_passed": original_total == 0,
              "original_false_accept_observations": original_total,
              "unique_raw_cases": len({(r["id"], r["raw_sha256"]) for r in records}),
              "diagnosis_counts": dict(counts), "cases": records, "gold_lexical_compatibility": gold,
              "sources": report_hashes,
              "interpretation": "The predicate-renaming control preserves graph topology, references, scopes and senses. It isolates lexical transport incompatibility; it never changes the original rejection, frozen gate, generation or Smatch++ score."}
    if args.score_impact:
        result["score_impact"] = score_impact(args.root, args.indices, records, report_hashes,
                                             paired=args.paired_impact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "original_structural_gate_passed", "original_false_accept_observations", "unique_raw_cases", "diagnosis_counts")}), flush=True)


if __name__ == "__main__":
    main()
