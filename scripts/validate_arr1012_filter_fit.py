"""Out-of-sample checks of the beta-binomial filtering-tail model.

(1) Fit on the first four of the eight stored samples per input and predict
the share of inputs whose eight samples are all invalid. (2) Fit on all eight
development samples and predict the test all-eight rate. Also reports the
overlap of all-invalid inputs across training sizes.
"""

from __future__ import annotations

import json
from math import comb, exp, lgamma, log
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TMP = REPO / "paper/coling2027/tmp"
CELLS = {
    ("dev", "600"): (TMP / "arr1012/e3-en600-gold-q08", "dev-"),
    ("dev", "1256"): (TMP / "arr1012/e3-en1256-gold-q08", "dev-"),
    ("dev", "9057"): (TMP / "arr1012/e3-enfull-gold-q08", "dev-"),
    ("test", "600"): (TMP / "arr1012-test/e3-en600", ""),
    ("test", "1256"): (TMP / "arr1012-test/e3-en1256", ""),
    ("test", "9057"): (TMP / "arr1012-test/e3-enfull", ""),
}


def lbeta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def fit(counts: np.ndarray, n: int) -> tuple[float, float]:
    hist = np.bincount(counts, minlength=n + 1)
    best = None
    for p in np.linspace(0.005, 0.995, 199):
        for rho in np.linspace(0.005, 0.995, 199):
            s = 1 / rho - 1
            a, b = p * s, (1 - p) * s
            ll = sum(c * (log(comb(n, k)) + lbeta(a + k, b + n - k) - lbeta(a, b)) for k, c in enumerate(hist) if c)
            if best is None or ll > best[0]:
                best = (ll, a, b)
    return best[1], best[2]


def all_invalid(a: float, b: float, k: int) -> float:
    return exp(lbeta(a + k, b) - lbeta(a, b))


def load(root: Path, prefix: str) -> dict[str, list[bool]]:
    out = {}
    for line in (root / f"{prefix}late.jsonl").open():
        r = json.loads(line)
        if r["configuration"] == "L3":
            out[r["id"].rsplit(":", 1)[-1]] = [not c["w_valid"] for c in r["candidates"]]
    return out


def main() -> None:
    data = {k: load(*v) for k, v in CELLS.items()}
    res = {}
    for (split, size), recs in data.items():
        bad = np.array([recs[i] for i in sorted(recs, key=int)])
        first4 = bad[:, :4].sum(1)
        a4, b4 = fit(first4, 4)
        a8, b8 = fit(bad.sum(1), 8)
        p4 = bad[:, :4].mean()
        res[f"{split}-{size}"] = {
            "observed_all8_pct": round(100 * bad.all(1).mean(), 2),
            "pred_from_first4_betabinomial_pct": round(100 * all_invalid(a4, b4, 8), 2),
            "pred_from_first4_independent_pct": float(f"{100 * p4 ** 8:.3g}"),
            "observed_all4_pct": round(100 * bad[:, :4].all(1).mean(), 2),
            "fit8_all8_pct": round(100 * all_invalid(a8, b8, 8), 2),
            "fit8_all32_pct": round(100 * all_invalid(a8, b8, 32), 2),
            "fit8_all64_pct": round(100 * all_invalid(a8, b8, 64), 2),
            "fit8_beta": round(b8, 2),
            "fit8_rho": round(1 / (a8 + b8 + 1), 3),
            "_ab8": (a8, b8),
        }
    for size in ("600", "1256", "9057"):
        a8, b8 = res[f"dev-{size}"]["_ab8"]
        res[f"test-{size}"]["pred_from_dev_fit_pct"] = round(100 * all_invalid(a8, b8, 8), 2)
    sets = {k: {i for i, v in recs.items() if all(v)} for k, recs in data.items()}
    overlap = {}
    for split in ("dev", "test"):
        for s, l in (("600", "9057"), ("1256", "9057"), ("600", "1256")):
            A, B = sets[(split, s)], sets[(split, l)]
            overlap[f"{split} {s}->{l}"] = {"n_small": len(A), "n_large": len(B), "both": len(A & B)}
    for v in res.values():
        v.pop("_ab8")
    out = {"cells": res, "all8_overlap": overlap}
    (TMP / "arr1012/filter-fit-validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
