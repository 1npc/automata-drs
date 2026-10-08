"""Learning curves for the training placement (none / mask / ul).

For each arm and snapshot: unconstrained (U) validity and macro Smatch++ on dev,
plus net training wall-clock from snapshot.json. Also the final adapter under U
and W. Pre-registered criterion: mask or ul "converges faster" only if it
reaches none's final U validity AND U Smatch++ at <= 70% of the optimizer steps
and its own final U Smatch++ is not lower than none's by more than 0.5 points.
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


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def evaluate(pred: Path) -> dict | None:
    scores = Path(str(pred).replace(".jsonl", ".smatchpp.jsonl"))
    if not pred.exists() or not scores.exists():
        return None
    rows = load(pred)
    smatch = {s["id"]: float(s["semantic_scores"]["sbn_smatch"]) for s in load(scores)}
    valid = np.array([w_valid(r["raw_prediction"]) for r in rows])
    score = np.array([smatch[r["id"]] for r in rows])
    return {"valid_rate": round(float(valid.mean()) * 100, 2), "macro_smatch": round(float(score.mean()) * 100, 2),
            "records": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing train-<dataset>-<mode>/")
    parser.add_argument("--dataset", default="en600")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report: dict = {"dataset": args.dataset, "arms": {}}
    for mode in ("none", "mask", "ul"):
        arm = args.root / f"train-{args.dataset}-{mode}"
        if not arm.exists():
            continue
        curve = []
        for pred in sorted(arm.glob("dev-step*-U.jsonl"), key=lambda p: int(re.search(r"step(\d+)", p.name).group(1))):
            step = int(re.search(r"step(\d+)", pred.name).group(1))
            result = evaluate(pred)
            snap = arm / "lora" / "checkpoints" / f"step-{step}" / "snapshot.json"
            info = json.loads(snap.read_text()) if snap.exists() else {}
            if result:
                curve.append({"step": step, **result,
                              "net_train_seconds": info.get("train_wall_seconds_net"),
                              "automaton_seconds": info.get("automaton_seconds_cumulative"),
                              "diag_seconds": info.get("baseline_diagnostic_seconds_cumulative")})
        report["arms"][mode] = {"curve": curve,
                                "final_U": evaluate(arm / "dev-final-U.jsonl"),
                                "final_W": evaluate(arm / "dev-final-W.jsonl")}
    base = report["arms"].get("none")
    if base and base["curve"]:
        target_valid = base["curve"][-1]["valid_rate"]
        target_smatch = base["curve"][-1]["macro_smatch"]
        total_steps = base["curve"][-1]["step"]
        report["criterion"] = {"target_valid": target_valid, "target_smatch": target_smatch}
        for mode, arm in report["arms"].items():
            hit = next((c for c in arm["curve"] if c["valid_rate"] >= target_valid and c["macro_smatch"] >= target_smatch), None)
            last = arm["curve"][-1]["macro_smatch"] if arm["curve"] else None
            arm["steps_to_none_final"] = hit["step"] if hit else None
            arm["fraction_of_steps"] = round(hit["step"] / total_steps, 3) if hit else None
            arm["faster_by_prereg"] = bool(
                mode != "none" and hit and hit["step"] <= 0.7 * total_steps
                and last is not None and last >= target_smatch - 0.5)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
