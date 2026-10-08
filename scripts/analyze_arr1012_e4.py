"""E4 analysis: English SBN test under a binding output budget H.

Arms: W (local mask), WF (independent bounded feasibility), WWCD (APD), and
offline arms from the W stream: FreeClose (may exceed H) and Reserve-k.
Reports W-validity (SBN + Referee-aligned well-formedness on raw output),
macro Smatch++, placeholder insertions, and paired bootstrap APD minus each arm.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import check_sbn_state_wellformedness, validate_sbn  # noqa: E402

PLACEHOLDER = re.compile(r"(?:^|\s)\?(?:\s|$)|\.[nvarx]\.00\b")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def analyze(root: Path, resamples: int) -> dict:
    rows = load(root / "test.jsonl") + load(root / "offline.jsonl")
    scores = load(root / "test.smatchpp.jsonl") + load(root / "offline.smatchpp.jsonl")
    smatch = {(s["id"], s["configuration"]): float(s["semantic_scores"]["sbn_smatch"]) for s in scores}
    by = {(r["id"], r["configuration"]): r for r in rows}
    arms = sorted({r["configuration"] for r in rows}, key=lambda a: (a.startswith("Reserve"), len(a), a))
    ids = sorted({r["id"] for r in rows if r["configuration"] == "WWCD"})
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(ids), size=(resamples, len(ids)))
    apd = np.array([smatch[(i, "WWCD")] for i in ids])
    out = {"records": len(ids), "arms": {}}
    for arm in arms:
        valid = np.array([w_valid(by[(i, arm)]["raw_prediction"]) for i in ids])
        score = np.array([smatch[(i, arm)] for i in ids])
        placeholder = sum(1 for i in ids if PLACEHOLDER.search(by[(i, arm)]["prediction"])
                          and not PLACEHOLDER.search(by[(i, arm)]["reference"]))
        diff = apd - score
        means = diff[idx].mean(axis=1)
        entry = {"valid": int(valid.sum()), "valid_rate": round(float(valid.mean()) * 100, 2),
                 "macro_smatch": round(float(score.mean()) * 100, 2),
                 "records_with_placeholder": placeholder,
                 "apd_minus_arm_points": round(float(diff.mean()) * 100, 3),
                 "apd_minus_arm_ci95": [round(float(np.percentile(means, q)) * 100, 3) for q in (2.5, 97.5)]}
        offline = [by[(i, arm)].get("offline") for i in ids]
        if any(offline):
            if arm == "FreeClose":
                entry["over_budget_records"] = sum(1 for o in offline if o and o.get("overrun", 0) > 0)
                entry["max_overrun_tokens"] = max((o.get("overrun", 0) for o in offline if o), default=0)
            else:
                entry["did_not_fit"] = sum(1 for o in offline if o and not o.get("fits", True))
                entry["dropped_model_tokens"] = sum(o.get("dropped", 0) for o in offline if o)
        out["arms"][arm] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--resamples", type=int, default=20000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {root.name: analyze(root, args.resamples) for root in args.roots}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
