"""Extra analyses for the placement paper from existing outputs (no new decoding)."""
from __future__ import annotations
import json, re, sys, glob
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from analyze_arr1012_e3 import classify, w_valid  # noqa: E402

R = REPO / "paper/coling2027/tmp/arr1012"
PLACE = re.compile(r"(?:^|\s)\?(?:\s|$)|\.[nvarx]\.00\b")
out: dict = {}

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def scores(p): return {(s["id"], s.get("configuration")): float(s["semantic_scores"]["sbn_smatch"]) for s in load(p)}

# (a) dead mass during training
dm = {}
for d in sorted(glob.glob(str(R / "train-*/lora/automaton_stats.jsonl"))):
    rows = load(d)
    dm[Path(d).parts[-3]] = [(r["step"], r.get("auto_dead_mass_mean")) for r in rows if r.get("auto_dead_mass_mean") is not None]
out["dead_mass"] = dm

# (b) cost, (c) length quartiles, (d) first-death categories, (e) L1 placeholders
def category(reason: str | None) -> str:
    r = (reason or "").lower()
    if "unresolved forward" in r: return "forward reference unresolved"
    if "points before" in r or "range" in r: return "reference out of range"
    if "proposition" in r or "contexts" in r: return "proposition box obligation"
    if "cyclic" in r or "loop" in r: return "graph cycle"
    if "well-formedness" in r or "referee" in r or "accessib" in r or "connected" in r or "subordinat" in r: return "accessibility / Referee"
    return "lexical or syntactic"

for g in ("en600", "en1256", "enfull"):
    d = R / f"e3-{g}-gold-q08"
    rows = load(d / "dev-online.jsonl") + load(d / "dev-late.jsonl") + load(d / "dev-l1.jsonl")
    sc = {}
    for n in ("dev-online", "dev-late", "dev-l1"): sc.update(scores(d / f"{n}.smatchpp.jsonl"))
    by = {(r["id"], r["configuration"]): r for r in rows}
    ids = sorted({r["id"] for r in rows if r["configuration"] == "U"})
    cost = {}
    for c in ("U", "W", "WF", "WWCD"):
        cost[c] = float(np.mean([by[(i, c)]["timing"]["generation_ms"] for i in ids]))
    for c in ("L3", "L4"):
        cost[c] = float(np.mean([by[(i, c)]["selection"]["seconds"] for i in ids])) * 1000
    dead = [i for i in ids if not w_valid(by[(i, "U")]["raw_prediction"])]
    cost["lazy_regenerated_fraction"] = len(dead) / len(ids)
    out.setdefault("cost_ms_per_graph", {})[g] = cost
    # length quartiles by reference length (characters)
    ln = np.array([len(by[(i, "U")]["reference"]) for i in ids]); qs = np.quantile(ln, [0.25, 0.5, 0.75])
    bins = np.digitize(ln, qs)
    best_late = max(("L1", "L3", "L4"), key=lambda a: np.mean([sc[(i, a)] for i in ids]))
    lq = []
    for b in range(4):
        sel = [i for i, k in zip(ids, bins) if k == b]
        lq.append({"quartile": b + 1, "n": len(sel),
                   "u_invalid": round(100 * float(np.mean([not w_valid(by[(i, "U")]["raw_prediction"]) for i in sel])), 1),
                   "apd": round(100 * float(np.mean([sc[(i, "WWCD")] for i in sel])), 2),
                   "best_late": round(100 * float(np.mean([sc[(i, best_late)] for i in sel])), 2)})
    out.setdefault("length_quartiles", {})[g] = {"best_late": best_late, "rows": lq}
    cats: dict = {}
    for i in dead:
        c = classify(by[(i, "U")]["raw_prediction"])
        cats[category(c["reason"])] = cats.get(category(c["reason"]), 0) + 1
    out.setdefault("first_death_categories_dev", {})[g] = cats
    out.setdefault("placeholders_dev", {})[g] = {a: sum(1 for i in ids if PLACE.search(by[(i, a)]["prediction"]) and not PLACE.search(by[(i, a)]["reference"])) for a in ("W", "WWCD", "L1", "L3", "L4")}

# (f) qualitative examples at H=58 (budget): W invalid, APD valid, Reserve-16 with placeholder
E = R / "e4-en-test-h58-q08"
t = {(r["id"], r["configuration"]): r for r in load(E / "test.jsonl") + load(E / "offline.jsonl")}
ex = []
for (i, c), r in t.items():
    if c != "W" or w_valid(r["raw_prediction"]): continue
    res = t[(i, "Reserve-16")]; apd = t[(i, "WWCD")]
    if PLACE.search(res["prediction"]) and not PLACE.search(apd["prediction"]) and len(r["reference"]) < 170:
        ex.append({"id": i, "input": r["input"], "reference": r["reference"], "W": r["raw_prediction"],
                   "Reserve-16": res["prediction"], "APD": apd["prediction"]})
out["examples_h58"] = sorted(ex, key=lambda e: len(e["reference"]))[:4]
(R / "extra-analyses.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if k not in ("dead_mass", "examples_h58")}, indent=1))
