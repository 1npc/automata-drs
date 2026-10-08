"""Full-test paired review-response analysis; incomplete evidence fails closed."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261001)
    means = []
    for _ in range(80):
        means.extend(values[rng.integers(0, len(values), size=(250,len(values)))].mean(axis=1))
    return [float(v) for v in np.quantile(means,[.025,.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("report already exists")
    errors, artifacts, cells = [], {}, defaultdict(dict)
    recovery = (args.root/"recovery-plan.json").exists()
    if recovery:
        import subprocess
        checked = subprocess.run(["sha256sum", "-c", str(args.root/"PREDECESSOR_SHA256SUMS")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if checked.returncode:
            errors.append("predecessor hash lock failed: " + checked.stderr)
        artifacts[str(args.root/"recovery-plan.json")] = hashlib.sha256((args.root/"recovery-plan.json").read_bytes()).hexdigest()
    for index in range(20):
        g = args.root/"matrix"/f"task-{index}"
        c = args.root/"cpu"/f"score-{index}"
        failed = False
        for folder in (g,c):
            if not (folder/"exit").exists() or (folder/"exit").read_text().strip()!="0":
                errors.append(f"task incomplete/nonzero: {folder}")
                failed = True
        if failed:
            continue
        stems = ["natural68","natural78","offline"] if index<8 else ["denial"]
        for stem in stems:
            path = g/(stem+".jsonl")
            scores_path = c/(stem+".scores.jsonl")
            scores_manifest = json.loads(Path(str(scores_path)+".manifest.json").read_text())
            if not scores_manifest["all_exact"]:
                errors.append("non-exact Smatch++ ILP")
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts[str(scores_path)] = hashlib.sha256(scores_path.read_bytes()).hexdigest()
            manifest = json.loads(Path(str(path)+".manifest.json").read_text())
            if manifest["status"]!="complete":
                errors.append("incomplete generation/offline manifest")
            if manifest.get("output_sha256") != artifacts[str(path)]:
                errors.append("generation hash mismatch " + str(path))
            if scores_manifest.get("input_sha256") != artifacts[str(path)] or scores_manifest.get("output_sha256") != artifacts[str(scores_path)]:
                errors.append("score hash mismatch " + str(path))
            if scores_manifest.get("smatchpp") != "1.8.0":
                errors.append("unexpected scorer version")
            for mpath in (Path(str(path)+".manifest.json"), Path(str(scores_path)+".manifest.json")):
                artifacts[str(mpath)] = hashlib.sha256(mpath.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            scored = [json.loads(line) for line in scores_path.read_text().splitlines()]
            if len(rows)!=len(scored):
                errors.append("score identity grid incomplete")
                continue
            for row,score in zip(rows,scored,strict=True):
                if (row["id"],row["configuration"])!=(score["id"],score["configuration"]):
                    errors.append("score identity order mismatch")
                    continue
                if not score.get("exact") or score["f1"] != score["f1_upper"]:
                    errors.append("per-row ILP exactness failed")
                failure = row["generation"].get("decoder_failure")
                masked = (row.get("constraint") or {}).get("all_masked_rows", 0)
                if recovery and masked:
                    if failure is None or masked != 1 or score["raw_W_valid"] or score["f1"] != 0:
                        errors.append("failure termination/zero policy failed " + row["id"])
                    elif failure["native_prefix_token_ids"] != row["generation"]["native_token_ids"] or failure["position"] != len(row["generation"]["native_token_ids"]):
                        errors.append("failure prefix changed")
                if failure and (not masked or row["generation"]["ended_with_eos"]):
                    errors.append("inconsistent failure marker")
                h = row["decode"].get("horizon",manifest.get("max_new_tokens"))
                rate = row.get("review_operations",{}).get("denial_rate",0)
                key = f'H{h}:deny{rate:g}:{row["configuration"]}'
                if row["id"] in cells[key]:
                    errors.append("duplicate cell identity")
                cells[key][row["id"]] = (row,score)
    summaries = {}
    if len(cells)!=18:
        errors.append(f"expected 18 full-test cells, observed {len(cells)}")
    id_grid = None
    for key,mapping in cells.items():
        ids = sorted(mapping)
        if len(ids)!=1132:
            errors.append(f"cell {key} has {len(ids)}/1132")
        if id_grid is None:
            id_grid = ids
        elif id_grid != ids:
            errors.append("cell identity grids differ")
        rows = [mapping[i][0] for i in ids]
        scores = [mapping[i][1] for i in ids]
        ops = defaultdict(float)
        for row in rows:
            for name,value in row.get("review_operations",{}).get("operations",{}).items():
                ops[name] += value
        lengths = [row["generation"]["output_tokens_including_eos"] for row in rows]
        summaries[key] = {"n": len(rows), "raw_valid": sum(score["raw_W_valid"] for score in scores),
                          "accepting_eos": sum(row["generation"]["ended_with_eos"] for row in rows),
                          "macro_f1": float(np.mean([score["f1"] for score in scores])),
                          "operations": dict(ops), "output_length_mean": float(np.mean(lengths)),
                          "decoder_failures": sum(row["generation"].get("decoder_failure") is not None for row in rows),
                          "reused_rows": sum("review_reuse" in row for row in rows),
                          "certificate_paths": {name: sum((row.get("constraint") or {}).get(name, 0) for row in rows)
                                                for name in ("carried_tokens", "cached_tokens", "replanned_tokens", "rejected_tokens", "rows")},
                          "online_generation_seconds": sum(row.get("timing",{}).get("generation_ms",0) for row in rows)/1000,
                          "posthoc_seconds": sum(row.get("timing",{}).get("posthoc_ms",0) for row in rows)/1000,
                          "all_masked": sum(row.get("constraint",{}).get("all_masked_rows",0) for row in rows),
                          "prefix_unavailable": sum(row.get("constraint",{}).get("prefix_rows_unavailable",0) for row in rows)}
        if key.endswith(":WWCD"):
            if summaries[key]["raw_valid"]!=1132 or summaries[key]["all_masked"] or summaries[key]["prefix_unavailable"]:
                errors.append("APD release/accounting gate failed "+key)
            for row in rows:
                stats = row["constraint"]
                if stats["carried_tokens"]+stats["cached_tokens"]+stats["replanned_tokens"]!=stats["rows"]:
                    errors.append("APD path accounting not closed")
    comparisons = {}
    comparisons_to_run = [(h,0,base) for h in (68,78) for base in ("W","WF","WRollback")]
    comparisons_to_run += [(68,rate,"WF") for rate in (.25,.5,.75)]
    for h,rate,baseline in comparisons_to_run:
            apdkey = f"H{h}:deny{rate:g}:WWCD"
            key = f"H{h}:deny{rate:g}:{baseline}"
            if key not in cells or apdkey not in cells:
                errors.append("missing primary paired cell")
                continue
            ids = sorted(set(cells[key])&set(cells[apdkey]))
            delta, common, rescue, lost = [], [], [], []
            exact = 0
            for ident in ids:
                b,bs = cells[key][ident]; a,ascore = cells[apdkey][ident]
                d = ascore["f1"]-bs["f1"]
                delta.append(d)
                common.append(d if bs["raw_W_valid"] and ascore["raw_W_valid"] else 0)
                rescue.append(ascore["f1"] if not bs["raw_W_valid"] and ascore["raw_W_valid"] else 0)
                lost.append(-bs["f1"] if bs["raw_W_valid"] and not ascore["raw_W_valid"] else 0)
                exact += int(a["raw_prediction"]==b["raw_prediction"])
            if not np.allclose(delta,np.array(common)+rescue+lost):
                errors.append("decomposition identity failed")
            comparisons[f"H{h}:deny{rate:g}:APD-minus-{baseline}"] = {
                "paired_n": len(ids), "byte_identical": exact,
                **{name:{"mean":float(np.mean(values)),"ci95":interval(values)}
                   for name,values in (("delta",delta),("common_valid",common),("rescue",rescue),("lost",lost))}}
    audit = args.root/"cpu"/"audit-existing-0"/"referee.json"
    if not audit.exists():
        errors.append("independent original-output audit missing")
    independent_false_accepts = 0
    for path in [audit]+[args.root/"cpu"/f"score-{i}"/"referee.json" for i in range(20)]:
        if not path.exists():
            errors.append("independent audit incomplete")
            continue
        report = json.loads(path.read_text())
        independent_false_accepts += report["false_accept_structural"]
        artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if independent_false_accepts:
        errors.append("independent semantic checker disagreement requires diagnosis")
    report = {"status": "passed" if not errors else "failed", "errors": errors,
              "evidence": artifacts, "summaries":summaries, "comparisons":comparisons,
              "independent_false_accepts":independent_false_accepts,
              "failure_termination_amendment": recovery,
              "interpretation": "post-review confirmation; synthetic availability, not natural timeout; certified lengths are upper bounds; cached W timing is not bounded decode timing",
              "bootstrap": {"resamples":20000,"unit":"input","seed":20261001}}
    with args.output.open("x",encoding="utf-8") as sink:
        json.dump(report,sink,indent=2)
    print(json.dumps({"status":report["status"],"errors":errors}),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
