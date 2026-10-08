"""Campaign SBN Smatch++ scorer.

Uses the same evaluation function as automata-supervision's
scripts/score_sbn_smatchpp.py (SBNPair, SmatchppBackend, evaluate_sbn_pairs,
30 s per pair) and writes the same per-row score schema. It does not apply that
repository's clean-working-tree provenance gate; the manifest records this so
claim-bearing numbers can be re-scored with the registered script later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from automata_supervision.evaluation.smatchpp import SBNPair, SmatchppBackend, evaluate_sbn_pairs


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--max-seconds-per-pair", type=int, default=30)
    args = parser.parse_args()
    if args.scores.exists():
        parser.error(f"scores exist: {args.scores}")
    digest = sha256(args.input)
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    if len(rows) != args.expected_rows:
        parser.error(f"{len(rows)} rows; expected {args.expected_rows}")
    pairs = [SBNPair(identifier=str(r["id"]), reference=str(r["reference"]), prediction=str(r["prediction"]))
             for r in rows]
    report = evaluate_sbn_pairs(pairs, SmatchppBackend(max_seconds_per_pair=args.max_seconds_per_pair))
    if sha256(args.input) != digest:
        raise RuntimeError("prediction input changed while scoring")
    per_config: dict[str, list[float]] = defaultdict(list)
    valid: dict[str, int] = defaultdict(int)
    with args.scores.open("x", encoding="utf-8") as sink:
        for row, score in zip(rows, report.pairs, strict=True):
            config = str(row.get("configuration", ""))
            per_config[config].append(float(score.f1_lower))
            valid[config] += int(bool(score.structural_valid))
            out = {"id": score.identifier, "configuration": config,
                   "semantic_scores": {"sbn_smatch": score.f1_lower, "sbn_smatch_lower": score.f1_lower,
                                       "sbn_smatch_upper": score.f1_upper, "sbn_smatch_exact": score.exact,
                                       "protocol": report.protocol},
                   "structural_valid": score.structural_valid,
                   "structural_failure": score.structural_failure}
            if "decode" in row:
                out["decode"] = row["decode"]
            sink.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "status": "succeeded",
        "scorer": "campaign score_sbn_campaign.py (same evaluate_sbn_pairs/SmatchppBackend as score_sbn_smatchpp.py)",
        "provenance_gate": "clean-working-tree gate NOT applied; rescore with the registered script for claims",
        "input": str(args.input), "input_sha256": digest, "rows": len(rows),
        "max_seconds_per_pair": args.max_seconds_per_pair, "protocol": report.protocol,
        "macro_f1_by_configuration": {c: sum(v) / len(v) for c, v in per_config.items()},
        "structurally_valid_by_configuration": dict(valid),
        "scores_sha256": sha256(args.scores),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "macro_f1_by_configuration",
                                              "structurally_valid_by_configuration")}, indent=2))


if __name__ == "__main__":
    main()
