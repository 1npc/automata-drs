"""Figure: probability that every one of j samples is invalid, by training size.

Observed points use all j-subsets of the eight stored samples per input
(C(m, j) / C(8, j) for an input with m invalid samples). Dashed lines give the
independent-failure prediction p^j; solid lines a beta-binomial fitted to the
per-input invalid counts by maximum likelihood, extended to 64 samples.
"""

from __future__ import annotations

import json
from math import comb, exp, lgamma, log
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "paper/coling2027/tmp/arr1012/input-dependence.json"
OUT = REPO / "paper/arr1012-drs-placement/figures/filter_failure.pdf"
FIT = REPO / "paper/coling2027/tmp/arr1012/filter-failure-fit.json"


def lbeta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def fit(hist: list[int]) -> tuple[float, float]:
    best = None
    for p in np.linspace(0.005, 0.995, 199):
        for rho in np.linspace(0.005, 0.995, 199):
            s = 1 / rho - 1
            a, b = p * s, (1 - p) * s
            ll = sum(c * (log(comb(8, k)) + lbeta(a + k, b + 8 - k) - lbeta(a, b)) for k, c in enumerate(hist))
            if best is None or ll > best[0]:
                best = (ll, a, b)
    return best[1], best[2]


def main() -> None:
    data = json.loads(DATA.read_text())
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    colors = {"600": "#d95f02", "1256": "#7570b3", "9057": "#1b9e77"}
    fits = {}
    for size, label in (("600", "600"), ("1256", "1,256"), ("9057", "9,057")):
        hist = data[f"dev-{size}"]["count_histogram"]
        n = sum(hist)
        js = np.arange(1, 9)
        obs = [sum(c * comb(m, j) / comb(8, j) for m, c in enumerate(hist)) / n for j in js]
        p = sum(m * c for m, c in enumerate(hist)) / (8 * n)
        a, b = fit(hist)
        ks = np.arange(1, 65)
        bb = [exp(lbeta(a + k, b) - lbeta(a, b)) for k in ks]
        fits[size] = {
            "p": round(p, 4), "alpha": round(a, 3), "beta": round(b, 3),
            "rho": round(1 / (a + b + 1), 3),
            "observed_all_j_invalid_pct": [round(100 * x, 3) for x in obs],
            "fit_all_k_invalid_pct": {str(k): round(100 * bb[k - 1], 3) for k in (8, 16, 32, 64)},
        }
        col = colors[size]
        ax.plot(js, np.array(obs) * 100, "o", color=col, ms=3.5, label=f"{label} graphs")
        ax.plot(ks, np.array(bb) * 100, "-", color=col, lw=1.1)
        ax.plot(ks[:20], (p ** ks[:20]) * 100, "--", color=col, lw=0.9, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 100)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("samples per input $k$")
    ax.set_ylabel("inputs with no valid sample (%)")
    ax.plot([], [], "k-", lw=1.1, label="beta-binomial fit")
    ax.plot([], [], "k--", lw=0.9, label="independent failures")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT)
    FIT.write_text(json.dumps(fits, indent=2))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
