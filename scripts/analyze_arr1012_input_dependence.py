"""Is invalidity a property of the input or of the sample?

Uses the stored candidates of the sample-filter arm (L3, 8 samples at T=1) and
the greedy unconstrained arm (U). For each training size and split it reports
the per-sample invalid rate, the observed share of inputs whose eight samples
are all invalid against the share expected if samples failed independently at
the same marginal rate, the overdispersion of per-input invalid counts, the
dependence on greedy validity and on reference length, and the persistence of
greedy failures across training sizes. CPU only; reads existing outputs.
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

TMP = REPO / "paper/coling2027/tmp"
CELLS = {
    ("dev", "600"): (TMP / "arr1012/e3-en600-gold-q08", "dev-"),
    ("dev", "1256"): (TMP / "arr1012/e3-en1256-gold-q08", "dev-"),
    ("dev", "9057"): (TMP / "arr1012/e3-enfull-gold-q08", "dev-"),
    ("test", "600"): (TMP / "arr1012-test/e3-en600", ""),
    ("test", "1256"): (TMP / "arr1012-test/e3-en1256", ""),
    ("test", "9057"): (TMP / "arr1012-test/e3-enfull", ""),
}


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def key(record_id: str) -> str:
    return record_id.rsplit(":", 1)[-1]


def cell(root: Path, prefix: str) -> dict:
    online = {key(r["id"]): r for r in load(root / f"{prefix}online.jsonl") if r["configuration"] == "U"}
    late = {key(r["id"]): r for r in load(root / f"{prefix}late.jsonl") if r["configuration"] == "L3"}
    ids = sorted(online, key=int)
    u_invalid = np.array([not w_valid(online[i]["raw_prediction"]) for i in ids])
    bad = np.array([sum(not c["w_valid"] for c in late[i]["candidates"]) for i in ids])
    n = np.array([len(late[i]["candidates"]) for i in ids])
    assert (n == 8).all()
    ref_len = np.array([len(online[i]["reference"].split()) for i in ids])
    p = bad.sum() / n.sum()
    all_bad = bad == 8
    var_obs = bad.var()
    var_bin = 8 * p * (1 - p)
    # intra-input correlation implied by the overdispersion of a beta-binomial
    rho = (var_obs / var_bin - 1) / 7 if var_bin > 0 else float("nan")
    quart = np.quantile(ref_len, [0.25, 0.5, 0.75])
    q = np.digitize(ref_len, quart, right=True)
    return {
        "inputs": len(ids),
        "u_invalid_pct": round(100 * u_invalid.mean(), 2),
        "sample_invalid_pct": round(100 * p, 2),
        "all8_invalid_pct": round(100 * all_bad.mean(), 2),
        "all8_expected_if_independent_pct": float(f"{100 * p ** 8:.3g}"),
        "ratio_observed_to_independent": float(f"{all_bad.mean() / p ** 8:.3g}") if p > 0 else None,
        "zero_invalid_pct": round(100 * (bad == 0).mean(), 2),
        "zero_expected_if_independent_pct": round(100 * (1 - p) ** 8, 2),
        "count_histogram": np.bincount(bad, minlength=9).tolist(),
        "overdispersion_var_ratio": round(float(var_obs / var_bin), 2),
        "intra_input_correlation": round(float(rho), 3),
        "all8_invalid_given_u_invalid_pct": round(100 * all_bad[u_invalid].mean(), 1) if u_invalid.any() else None,
        "all8_invalid_given_u_valid_pct": round(100 * all_bad[~u_invalid].mean(), 2),
        "resampling_recovers_u_invalid_pct": round(100 * (~all_bad[u_invalid]).mean(), 1) if u_invalid.any() else None,
        "all8_invalid_by_length_quartile_pct": [round(100 * all_bad[q == k].mean(), 1) for k in range(4)],
        "_u_invalid_ids": [i for i, b in zip(ids, u_invalid) if b],
        "_all8_ids": [i for i, b in zip(ids, all_bad) if b],
        "_ids": ids,
    }


def persistence(a: dict, b: dict, field: str) -> dict:
    ids = a["_ids"]
    sa, sb = set(a[field]), set(b[field])
    pa, pb = len(sa) / len(ids), len(sb) / len(ids)
    both = len(sa & sb) / len(ids)
    return {
        "p_small": round(100 * pa, 2),
        "p_large": round(100 * pb, 2),
        "p_large_given_small_pct": round(100 * both / pa, 1) if pa else None,
        "lift": round(both / (pa * pb), 1) if pa and pb else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=TMP / "arr1012/input-dependence.json")
    args = parser.parse_args()
    cells = {f"{split}-{size}": cell(*paths) for (split, size), paths in CELLS.items()}
    out = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in cells.items()}
    out["persistence"] = {}
    for split in ("dev", "test"):
        for small, large in (("600", "1256"), ("600", "9057"), ("1256", "9057")):
            a, b = cells[f"{split}-{small}"], cells[f"{split}-{large}"]
            out["persistence"][f"{split} greedy U invalid {small}->{large}"] = persistence(a, b, "_u_invalid_ids")
            out["persistence"][f"{split} all-8 invalid {small}->{large}"] = persistence(a, b, "_all8_ids")
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
