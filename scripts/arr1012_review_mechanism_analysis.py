"""Full-matrix operation/trajectory analysis; never changes predictions or gates.

Requires the frozen complete aggregate. An independently diagnosed audit
disagreement may remain, but missing/invalid scientific artifacts are fatal.
Lengths describe this proposer's verified suffixes, not shortest distances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_ONLY_ERROR = "independent semantic checker disagreement requires diagnosis"

def validate_aggregate(report):
    """An audit disagreement is diagnosable; incomplete evidence is not."""
    if report["status"] not in ("passed", "failed"):
        raise ValueError("unrecognized frozen gate status")
    if (report["status"] == "failed") != bool(report["errors"]):
        raise ValueError("frozen status/error list inconsistent")
    if any(e != AUDIT_ONLY_ERROR for e in report["errors"]):
        raise ValueError("frozen aggregate has non-audit evidence errors")
    if bool(report["errors"]) != bool(report["independent_false_accepts"]):
        raise ValueError("frozen independent-audit count/error mismatch")
    expected = {f"H{h}:deny0:{arm}" for h in (68,78,256,320) for arm in ("W","WRollback")}
    expected |= {f"H{h}:deny0:{arm}" for h in (68,78) for arm in ("WF","WWCD")}
    expected |= {f"H68:deny{rate:g}:{arm}" for rate in (.25,.5,.75) for arm in ("WF","WWCD")}
    if set(report["summaries"]) != expected or any(s["n"] != 1132 for s in report["summaries"].values()):
        raise ValueError("not all 18 registered full-test cells are complete")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(histogram):
    """Linear sample quantiles from integer multiplicities, without expansion."""
    if any(type(count) is not int or count <= 0 for count in histogram.values()):
        raise ValueError("distribution requires positive integer multiplicities")
    if any(not math.isfinite(value) for value in histogram):
        raise ValueError("nonfinite measured quantity")
    n = sum(histogram.values())
    if not n:
        return {"n": 0, "mean": None, "quantiles": None}
    pairs = sorted(histogram.items())

    def order_statistic(index):
        running = 0
        for value, count in pairs:
            running += count
            if index < running:
                return value
        raise AssertionError("quantile outside distribution")

    quantiles = {}
    for q in (0, .25, .5, .75, .9, .95, .99, 1):
        position = q * (n - 1)
        lower = math.floor(position)
        fraction = position - lower
        quantiles[f"p{q*100:g}"] = (
            (1 - fraction) * order_statistic(lower)
            + fraction * order_statistic(math.ceil(position))
        )
    return {"n": n, "mean": sum(v*c for v,c in pairs)/n, "quantiles": quantiles}


def common_prefix(left, right):
    at = 0
    for a, b in zip(left, right):
        if a != b:
            break
        at += 1
    return at


def online_measurements(row, horizon):
    """Validate and summarize complete instrumentation for one WF/APD input."""
    review = row["review_operations"]
    ids = row["generation"]["native_token_ids"]
    ops = review["operations"]
    if review["initial_certificate_verified"] is not True:
        raise ValueError("initial certificate unverified")
    if len(review["initial_certificate_token_ids"]) > horizon:
        raise ValueError("initial certificate exceeds budget")
    calls, status = Counter(), Counter()
    lengths, slack = Counter(), Counter()
    for item in review["completion_histogram"]:
        remaining, length, kind, count = (
            item["remaining"], item["certified_length_upper_bound"],
            item["request_type"], item["count"]
        )
        if kind not in ("initial", "natural", "forced") or type(count) is not int or count <= 0:
            raise ValueError("invalid completion histogram")
        if type(remaining) is not int or not 0 <= remaining <= horizon:
            raise ValueError("invalid constructor budget")
        calls[kind] += count
        status[f"{kind}-{'unknown' if length is None else 'verified'}"] += count
        if length is not None:
            if type(length) is not int or not 0 <= length <= remaining or kind == "forced":
                raise ValueError("invalid certified suffix bound")
            lengths[length] += count
            slack[remaining-length] += count
        elif kind == "initial":
            raise ValueError("registered initialization failed")
    if calls["initial"] != ops.get("initial_proposer_calls", 0):
        raise ValueError("initial calls do not match histogram")
    if calls["natural"] + calls["forced"] != ops.get("fresh_proposer_calls", 0):
        raise ValueError("fresh calls do not match histogram")
    if sum(v for k,v in status.items() if k.endswith("-unknown")) != ops.get("unknown_results", 0):
        raise ValueError("unknown-result accounting mismatch")
    if sum(v for k,v in status.items() if k.endswith("-verified")) != ops.get("verified_witnesses", 0):
        raise ValueError("verified-result accounting mismatch")
    chosen_ids = []
    decisions = Counter()
    length_by_decision = defaultdict(Counter)
    slack_by_decision = defaultdict(Counter)
    for expected, item in enumerate(review["decision_trace"]):
        if item["position"] != expected:
            raise ValueError("decision positions are not consecutive")
        chosen = item["chosen_token_id"]
        if chosen is None:
            if expected != len(ids) or expected != len(review["decision_trace"])-1:
                raise ValueError("failed decision not at first unreleased position")
            decisions["failed"] += 1
            continue
        chosen_ids.append(chosen)
        rank = item["chosen_model_rank"]
        length = item["certified_length_upper_bound"]
        gap = item["slack"]
        if type(rank) is not int or rank < 0:
            raise ValueError("invalid chosen-token rank")
        if (rank == 0) != (chosen == item["model_top1_token_id"]):
            raise ValueError("model-top1/rank inconsistency")
        if length is None or type(length) is not int or length < 0:
            raise ValueError("chosen decision lacks a completion bound")
        if gap != horizon-expected-1-length or gap < 0:
            raise ValueError("chosen certificate/slack budget mismatch")
        branch = item["justification"]
        changed = "changed" if rank else "unchanged"
        decisions["chosen"] += 1
        decisions[changed] += 1
        decisions[f"{branch}:{changed}"] += 1
        length_by_decision[changed][length] += 1
        slack_by_decision[changed][gap] += 1
    if chosen_ids != ids:
        raise ValueError("decision trace does not reconstruct emitted native tokens")
    if decisions["failed"] != int(row["generation"].get("decoder_failure") is not None):
        raise ValueError("failed-decision capture mismatch")
    if row["configuration"] == "WWCD":
        stats = row["constraint"]
        if any(stats.get(k,0) for k in ("all_masked_rows", "prefix_rows_unavailable")):
            raise ValueError("APD lost a releasable prefix")
        for branch in ("carried", "cached", "replanned"):
            if sum(decisions[f"{branch}:{change}"] for change in ("changed","unchanged")) != stats[f"{branch}_tokens"]:
                raise ValueError("APD trace/path accounting mismatch")
        if decisions["chosen"] != stats["rows"]:
            raise ValueError("APD decision/row accounting mismatch")
    return {
        "calls": calls, "status": status, "lengths": lengths, "slack": slack,
        "decisions": decisions, "length_by_decision": length_by_decision,
        "slack_by_decision": slack_by_decision,
    }


def rollback_measurements(row, source_ids, horizon):
    info = row["offline"]
    kept, dropped, closed = (info[k] for k in ("kept_tokens", "dropped_tokens", "completion_tokens"))
    if any(type(v) is not int or v < 0 for v in (kept,dropped,closed)):
        raise ValueError("invalid rollback operation count")
    limited = source_ids[:horizon]
    output = row["generation"]["native_token_ids"]
    if kept + dropped != len(limited) or kept + closed != len(output):
        raise ValueError("rollback length accounting mismatch")
    if output[:kept] != limited[:kept]:
        raise ValueError("rollback did not retain exact source prefix")
    if len(output) > horizon or not row["generation"]["ended_with_eos"]:
        raise ValueError("rollback budget/EOS gate failed")
    if info["fallback"] and (kept or output != row["initial_certificate_token_ids"]):
        raise ValueError("rollback fallback differs from registered certificate")
    return {"kept": kept, "dropped": dropped, "completion": closed,
            "kept_fraction": kept/len(limited) if limited else 0,
            "fresh_calls": info["fresh_calls"], "fallback": int(info["fallback"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("refusing to overwrite mechanism evidence")
    report_path = args.root/"cpu/aggregate-0/review-response-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_aggregate(report)
    evidence = {str(report_path): digest(report_path),
                str(Path(__file__)): digest(Path(__file__))}
    for name, expected in report["evidence"].items():
        path = Path(name)
        if digest(path) != expected:
            raise ValueError(f"aggregate source hash changed: {path}")
        evidence[name] = expected
    source_ids = {}
    ordered_ids = []
    for i in range(8):
        path = args.root/"matrix"/f"task-{i}"/"W320.jsonl"
        evidence[str(path)] = digest(path)
        manifest_path = Path(str(path)+".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "complete" or manifest["output_sha256"] != evidence[str(path)]:
            raise ValueError("cached W source manifest/hash mismatch")
        evidence[str(manifest_path)] = digest(manifest_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in source_ids:
                raise ValueError("duplicate cached W source")
            source_ids[row["id"]] = row["generation"]["native_token_ids"]
            ordered_ids.append(row["id"])
    if len(ordered_ids) != 1132:
        raise ValueError("cached W source is not full test")
    cells = defaultdict(list)
    for i in range(20):
        stems = ("natural68","natural78","offline") if i < 8 else ("denial",)
        for stem in stems:
            path = args.root/"matrix"/f"task-{i}"/f"{stem}.jsonl"
            manifest = json.loads(Path(str(path)+".manifest.json").read_text(encoding="utf-8"))
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                horizon = row["decode"].get("horizon", manifest.get("max_new_tokens"))
                if type(horizon) is not int or horizon <= 0:
                    raise ValueError("cell lacks a registered integer horizon")
                rate = row.get("review_operations",{}).get("denial_rate",0)
                key = f"H{horizon}:deny{rate:g}:{row['configuration']}"
                cells[key].append((row, horizon))
    summaries = {}
    for key, entries in cells.items():
        rows = [row for row, _ in entries]
        if [r["id"] for r in rows] != ordered_ids:
            raise ValueError(f"cell order is not matched to cached W: {key}")
        horizon = entries[0][1]
        if any(h != horizon for _,h in entries):
            raise ValueError("cell horizons differ")
        lengths, slack, calls, status, decisions = (Counter() for _ in range(5))
        length_by, slack_by = defaultdict(Counter), defaultdict(Counter)
        prefix = Counter()
        offline = defaultdict(Counter)
        for row in rows:
            source = source_ids[row["id"]][:horizon]
            prefix[common_prefix(row["generation"]["native_token_ids"], source)] += 1
            if row["configuration"] in ("WF","WWCD"):
                item = online_measurements(row, horizon)
                for target, name in ((lengths,"lengths"),(slack,"slack"),(calls,"calls"),(status,"status"),(decisions,"decisions")):
                    target.update(item[name])
                for name, hist in item["length_by_decision"].items():
                    length_by[name].update(hist)
                for name, hist in item["slack_by_decision"].items():
                    slack_by[name].update(hist)
            if row["configuration"] == "WRollback":
                for name, value in rollback_measurements(row, source_ids[row["id"]], horizon).items():
                    offline[name][value] += 1
        summaries[key] = {
            "n": len(rows), "cached_W_common_prefix_tokens": distribution(prefix),
            "constructor_calls": dict(calls), "constructor_outcomes": dict(status),
            "constructor_returned_suffix_length_upper_bound": distribution(lengths),
            "constructor_suffix_slack": distribution(slack),
            "chosen_decisions": dict(decisions),
            "chosen_suffix_length_upper_bound_by_model_top1_change": {k: distribution(v) for k,v in length_by.items()},
            "chosen_slack_by_model_top1_change": {k: distribution(v) for k,v in slack_by.items()},
            "rollback": {k: distribution(v) for k,v in offline.items()},
        }
    if set(summaries) != set(report["summaries"]):
        raise ValueError("mechanism cells differ from frozen aggregate")
    result = {
        "status": "complete_mechanism_evidence", "frozen_scientific_gate": report["status"],
        "frozen_gate_errors": report["errors"], "evidence": evidence, "summaries": summaries,
        "interpretation": [
            "Completion lengths are proposer-specific verified upper bounds, never minimum c(z).",
            "Model-top1 replacement includes local structural checks, not only bounded-viability changes.",
            "Cached-W common prefixes compare independent runs; they are not causal token-retention effects.",
            "Rollback kept-prefix fractions measure exact original native tokens and include fallback cases.",
            "This report cannot pass, replace or waive the frozen independent-checker agreement gate.",
            "Cached W320 time is not actual generation time at a shorter horizon.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    with args.output.with_suffix(".manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"status":"complete", "output_sha256":digest(args.output),
                   "frozen_scientific_gate":report["status"]}, stream, indent=2)
    print(json.dumps({"status":result["status"], "frozen_scientific_gate":report["status"]}))


if __name__ == "__main__":
    main()
