"""Seed replication of the training placement (none vs ul) on one dataset.

Reports, per seed: the pre-registered comparison (best-by-dev-loss final adapter,
unconstrained decoding, paired bootstrap), and two exploratory statistics chosen
after seeing seed 42 (last-step snapshot and mean over all snapshots).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import check_sbn_state_wellformedness, validate_sbn  # noqa: E402


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def evaluate(path: Path) -> tuple[float, float, dict[str, float]] | None:
    scores = Path(str(path).replace(".jsonl", ".smatchpp.jsonl"))
    if not path.exists() or not scores.exists():
        return None
    rows = [json.loads(line) for line in path.open() if line.strip()]
    smatch = {json.loads(line)["id"]: json.loads(line)["semantic_scores"]["sbn_smatch"] for line in scores.open()}
    per = {r["id"]: smatch[r["id"]] for r in rows}
    valid = float(np.mean([w_valid(r["raw_prediction"]) for r in rows])) * 100
    return valid, float(np.mean(list(per.values()))) * 100, per


def paired(a: dict[str, float], b: dict[str, float], seed: int = 0) -> dict:
    ids = sorted(a)
    diff = np.array([b[i] - a[i] for i in ids])
    means = diff[np.random.default_rng(seed).integers(0, len(ids), (20000, len(ids)))].mean(axis=1)
    return {"gain": round(float(diff.mean()) * 100, 2),
            "ci95": [round(float(np.percentile(means, q)) * 100, 2) for q in (2.5, 97.5)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", default="en600")
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44"])
    parser.add_argument("--decoders", nargs="+", default=["U", "W"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report: dict = {"dataset": args.dataset, "seeds": {}}
    for seed in args.seeds:
        entry: dict = {}
        arms = {}
        for mode in ("none", "ul"):
            d = args.root / (f"train-{args.dataset}-{mode}" + ("" if seed == "42" else f"-s{seed}"))
            steps = sorted(int(re.search(r"step(\d+)", p).group(1)) for p in glob.glob(f"{d}/dev-step*-U.smatchpp.jsonl"))
            curve = {s: evaluate(d / f"dev-step{s}-U.jsonl") for s in steps}
            finals = {dec: evaluate(d / f"dev-final-{dec}.jsonl") for dec in args.decoders}
            arms[mode] = (curve, finals)
            entry[mode] = {"curve": {s: [round(v[0], 2), round(v[1], 2)] for s, v in curve.items() if v},
                           "final": {dec: ([round(v[0], 2), round(v[1], 2)] if v else None) for dec, v in finals.items()}}
        (c0, f0), (c1, f1) = arms["none"], arms["ul"]
        entry["prereg_final"] = {dec: paired(f0[dec][2], f1[dec][2]) for dec in args.decoders if f0[dec] and f1[dec]}
        common = sorted(set(c0) & set(c1))
        if common:
            entry["exploratory_last_step"] = paired(c0[common[-1]][2], c1[common[-1]][2])
            entry["exploratory_curve_mean_gain"] = round(float(np.mean([c1[s][1] - c0[s][1] for s in common])), 2)
        report["seeds"][seed] = entry
    finals = [e["prereg_final"]["U"]["gain"] for e in report["seeds"].values() if "U" in e.get("prereg_final", {})]
    report["prereg_U_gains"] = finals
    report["prereg_rule_met"] = bool(finals) and len(finals) == len(args.seeds) and all(g >= 0 for g in finals) and float(np.mean(finals)) >= 2
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
