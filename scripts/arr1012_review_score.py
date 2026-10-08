"""Raw-W-valid end-to-end Smatch++ scoring, with exactness explicitly gated."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from automata_drs import validate_sbn, check_sbn_state_wellformedness
from automata_supervision.evaluation.smatchpp import SBNPair, SmatchppBackend, evaluate_sbn_pairs
from arr1012_review_runtime import deliverable_raw_valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-rows", type=int, required=True)
    args = p.parse_args()
    manifest_path = Path(str(args.output)+".manifest.json")
    if args.output.exists() or manifest_path.exists():
        p.error("refusing to overwrite scoring evidence")
    if importlib.metadata.version("smatchpp") != "1.8.0":
        raise RuntimeError("Smatch++ 1.8.0 required")
    digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if len(rows) != args.expected_rows:
        raise RuntimeError("incomplete scoring input")
    valid = []
    for row in rows:
        valid.append(deliverable_raw_valid(row))
    pairs = [SBNPair(identifier=row["id"], reference=row["reference"],
                     prediction=row["raw_prediction"] if ok else "") for row,ok in zip(rows,valid)]
    report = evaluate_sbn_pairs(pairs, SmatchppBackend(max_seconds_per_pair=30))
    if hashlib.sha256(args.predictions.read_bytes()).hexdigest()!=digest:
        raise RuntimeError("scoring source changed")
    all_exact = all(score.exact for score in report.pairs)
    with args.output.open("x", encoding="utf-8") as out:
        for row,score,ok in zip(rows,report.pairs,valid,strict=True):
            result = {"id": row["id"], "configuration": row["configuration"], "decode": row["decode"],
                      "raw_W_valid": ok, "decoder_failure": row.get("generation",{}).get("decoder_failure"),
                      "f1": score.f1_lower, "f1_upper": score.f1_upper,
                      "exact": score.exact, "semantic_scores": {"sbn_smatch": score.f1_lower}}
            out.write(json.dumps(result)+"\n")
    with manifest_path.open("x", encoding="utf-8") as out:
        json.dump({"status": "complete", "rows": len(rows), "input_sha256": digest,
                   "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   "protocol": report.protocol, "smatchpp": "1.8.0", "all_exact": all_exact,
                   "raw_validity_zero_policy": True, "decoder_failure_zero_policy": True,
                   "max_seconds_per_pair": 30}, out, indent=2)
    if not all_exact:
        raise RuntimeError("ILP exactness gate failed; bounds and evidence preserved")
    print(f"score complete and exact: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
