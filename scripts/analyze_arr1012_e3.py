"""E3/E5 analysis: online vs. late use of the SBN automaton under scarce supervision.

Inputs per language: dev-online.jsonl (U/W/WF/WWCD) and dev-late.jsonl (L3/L4)
plus their Smatch++ score JSONL files. Outputs the pre-registered Story A
quantities: validity, macro Smatch++, online surplus over the best late arm
(corpus and conditional on U-invalid records) with paired bootstrap intervals,
and the first-dead-prefix split into context-free vs. non-context-free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import check_sbn_state_wellformedness, validate_sbn  # noqa: E402
from automata_drs.sbn.state import SBNConstraintState, SBNPolicy  # noqa: E402

ONLINE, LATE = ("W", "WWCD"), ("L1", "L3", "L4")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def first_death(text: str, policy: SBNPolicy) -> tuple[int, str | None]:
    """Character offset where the policy first rejects, len(text) if only at EOS."""
    state = SBNConstraintState.initial(policy)
    for offset, char in enumerate(text):
        state = state.advance_text(char)
        if state.is_dead:
            return offset, state.dead_reason
    final = state.finish()
    if final.is_dead or not final.is_accepting:
        return len(text), final.dead_reason
    return -1, None


def classify(raw: str) -> dict:
    w_at, w_reason = first_death(raw, SBNPolicy.for_configuration("W"))
    s_at, _ = first_death(raw, SBNPolicy.for_configuration("S"))
    if w_at < 0:  # W accepts the string; fall back to the end-of-sequence Referee check
        w_at, w_reason = len(raw), "referee check at EOS"
    context_free = s_at == w_at
    return {"offset": w_at, "relative_position": w_at / max(1, len(raw)), "reason": w_reason,
            "context_free": context_free}


def bootstrap(diff: np.ndarray, resamples: int, seed: int) -> list[float]:
    if len(diff) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)) * 100, 3), round(float(np.percentile(means, 97.5)) * 100, 3)]


def analyze(root: Path, resamples: int) -> dict:
    parts = [n for n in ("dev-online", "dev-late", "dev-l1") if (root / f"{n}.smatchpp.jsonl").exists()]
    rows = [r for n in parts for r in load(root / f"{n}.jsonl")]
    scores = [r for n in parts for r in load(root / f"{n}.smatchpp.jsonl")]
    smatch = {(s["id"], s["configuration"]): float(s["semantic_scores"]["sbn_smatch"]) for s in scores}
    by = {(r["id"], r["configuration"]): r for r in rows}
    ids = sorted({r["id"] for r in rows if r["configuration"] == "U"})
    arms = [a for a in ["U", "W", "WF", "WWCD", "L1", "L3", "L4"] if (ids[0], a) in by]
    late_arms = [a for a in LATE if a in arms]
    valid = {a: np.array([w_valid(by[(i, a)]["raw_prediction"]) for i in ids]) for a in arms}
    score = {a: np.array([smatch[(i, a)] for i in ids]) for a in arms}
    u_invalid = ~valid["U"]

    best_online = max(ONLINE, key=lambda a: score[a].mean())
    best_late = max(late_arms, key=lambda a: score[a].mean())
    diff = score[best_online] - score[best_late]
    deaths = {i: classify(by[(i, "U")]["raw_prediction"]) for i, bad in zip(ids, u_invalid) if bad}
    non_cf = np.array([i in deaths and not deaths[i]["context_free"] for i in ids])
    cf = np.array([i in deaths and deaths[i]["context_free"] for i in ids])
    total_surplus = diff[u_invalid].sum()

    return {
        "records": len(ids),
        "validity_rate": {a: round(float(valid[a].mean()) * 100, 2) for a in arms},
        "macro_smatch": {a: round(float(score[a].mean()) * 100, 2) for a in arms},
        "u_invalid_rate": round(float(u_invalid.mean()) * 100, 2),
        "best_online": best_online,
        "best_late": best_late,
        "late_arms_available": late_arms,
        "online_surplus_points": round(float(diff.mean()) * 100, 3),
        "online_surplus_ci95": bootstrap(diff, resamples, 1),
        "conditional_surplus_points_on_u_invalid": round(float(diff[u_invalid].mean()) * 100, 3) if u_invalid.any() else None,
        "conditional_surplus_ci95": bootstrap(diff[u_invalid], resamples, 2),
        "late_any_valid_candidate_rate": {
            a: round(float(np.mean([by[(i, a)]["selection"]["any_valid"] for i in ids])) * 100, 2) for a in ("L3", "L4") if a in arms},
        "first_death": {
            "u_invalid_records": int(u_invalid.sum()),
            "non_context_free": int(non_cf.sum()),
            "context_free": int(cf.sum()),
            "share_of_surplus_on_non_cf": round(float(diff[non_cf].sum() / total_surplus), 3) if total_surplus else None,
            "median_relative_position": round(float(np.median([d["relative_position"] for d in deaths.values()])), 3) if deaths else None,
            "top_reasons": sorted(
                {(d["reason"] or "")[:60] for d in deaths.values()},
                key=lambda r: -sum((d["reason"] or "")[:60] == r for d in deaths.values()))[:8],
        },
    }


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
