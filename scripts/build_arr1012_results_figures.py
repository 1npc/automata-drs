#!/usr/bin/env python3
"""Publication plots from stored ARR1012 evidence; no generation or scoring.

Parsing-result figures use held-out English test results. Gold-prefix mass is
explicitly a training diagnostic; development learning curves stay in the appendix.
All outputs retain source hashes in build/results-figure-data.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper/arr1012-drs-placement"
DEV = REPO / "paper/coling2027/tmp/arr1012"
TEST = REPO / "paper/coling2027/tmp/arr1012-test"
BUILD = PAPER / "build/results-figures"
SOURCES: dict[str, str] = {}
AUDIT: dict = {"main_split": "test", "diagnostic_split": "dev / gold prefixes"}
GRAY = "#595959"
GREEN = "#147D70"
ORANGE = "#C47428"
BLUE = "#356A9A"
SIZE_COLORS = [BLUE, ORANGE, GREEN]


def read_json(path: Path):
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def scores(path: Path) -> dict[str, float]:
    SOURCES[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [r["id"] for r in rows]
    assert len(rows) == len(set(ids)) == 1132, path
    assert all("/test/" in i for i in ids), path
    assert all(r["semantic_scores"]["sbn_smatch_exact"] for r in rows), path
    return {r["id"]: 100 * r["semantic_scores"]["sbn_smatch"] for r in rows}


def style() -> None:
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.65, "axes.edgecolor": "#444444",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.5, "lines.markersize": 4,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def clean(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E3E3E3", linewidth=0.5)
    ax.xaxis.grid(False)
    ax.minorticks_off()
    ax.tick_params(pad=3)


def save(fig, name: str) -> None:
    fig.canvas.draw()
    # Reject clipping before replacing any previous usable export.
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, box)
        assert box.x1 <= frame.x1 + 1 and box.y1 <= frame.y1 + 1, (name, box)
    for obj in [*fig.texts, *fig.legends]:
        box = obj.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, (name, box)
        assert box.x1 <= frame.x1 + 1 and box.y1 <= frame.y1 + 1, (name, box)
    target = PAPER / "figures" / f"{name}.pdf"
    temporary = target.with_suffix(".tmp.pdf")
    fig.savefig(temporary, metadata={"Creator": "build_arr1012_results_figures.py"})
    temporary.replace(target)
    fig.savefig(target.with_suffix(".svg"))
    fig.savefig(BUILD / f"{name}.png", dpi=240)
    plt.close(fig)


def budget() -> None:
    data = read_json(DEV / "e4-report-all.json")
    horizons = [58, 68, 78, 100]
    arms = ["W", "Reserve-16", "WWCD"]
    labels = [r"$W$", r"Reserve $r=16$", "APD"]
    colors = [GRAY, ORANGE, GREEN]
    markers = ["s", "^", "o"]
    # One-column primary comparison; validity and placeholders remain in tables.
    fig, ax = plt.subplots(figsize=(3.08, 1.86))
    fig.subplots_adjust(left=.17, right=.985, bottom=.27, top=.81)
    metrics = ["valid_rate", "macro_smatch", "records_with_placeholder"]
    plotted = {}
    for arm, label, color, marker in zip(arms, labels, colors, markers):
        rows = [data[f"e4-en-test-h{h}-q08"] for h in horizons]
        assert all(r["records"] == 1132 for r in rows)
        values = []
        for k, metric in enumerate(metrics):
            y = np.array([r["arms"][arm][metric] for r in rows], dtype=float)
            if k == 2:
                y *= 100 / 1132
            if metric == "macro_smatch":
                ax.plot(horizons, y, marker=marker, color=color,
                        linestyle="--" if arm == "Reserve-16" else "-", label=label,
                        zorder=4 if arm == "WWCD" else 3)
            values.append(y.tolist())
        plotted[arm] = values
    clean(ax)
    ax.set(xlim=(55, 103), ylim=(79, 93), xticks=horizons,
           yticks=[80, 84, 88, 92], xlabel=r"Output budget $H$", ylabel="Smatch++")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=3, frameon=False, handlelength=1.4, columnspacing=.85,
               handletextpad=.45)
    AUDIT["budget"] = {"split": "test", "records": 1132,
                       "horizons": horizons, "metrics": metrics, "plotted": plotted,
                       "displayed_metrics": ["macro_smatch"], "width_inches": 3.08}
    save(fig, "budget_curve")


def filtering() -> None:
    data = read_json(DEV / "input-dependence.json")
    uncertainty = read_json(PAPER / "evidence/stored-score-and-filter-uncertainty-v2.json")
    assert uncertainty["status"] == "passed"
    fig = plt.figure(figsize=(3.08, 2.16))
    ax = fig.add_axes([.165, .555, .805, .335])
    tail = fig.add_axes([.165, .165, .805, .205])
    k = np.arange(1, 9)
    plotted = {}
    for j, (size, label, color, marker) in enumerate(zip(
        ["600", "1256", "9057"], ["600", "1,256", "9,057"],
        SIZE_COLORS, ["o", "s", "^"]
    )):
        hist = data[f"test-{size}"]["count_histogram"]
        n = sum(hist)
        assert n == 1132
        observed = np.array([sum(c * comb(m, j) / comb(8, j)
                                 for m, c in enumerate(hist)) / n for j in k])
        p = sum(m * c for m, c in enumerate(hist)) / (8 * n)
        independent = p ** k
        ax.plot(k, 100 * observed, color=color, marker=marker,
                markersize=3.3, linewidth=1.35, label=label, zorder=4)
        entry = uncertainty["filter_uncertainty"][f"test-{size}"]
        assert entry["n"] == n and entry["all8_invalid_count"] == hist[8]
        assert np.isclose(entry["all8_invalid_percent"], 100*observed[-1])
        bounds = np.array(entry["all8_invalid_percent_wilson_ci95"])
        errors = np.array([[100*observed[-1]-bounds[0]], [bounds[1]-100*observed[-1]]])
        ax.errorbar([8], [100*observed[-1]], yerr=errors, fmt="none", color=color,
                    capsize=2, elinewidth=.75, capthick=.65, zorder=3)
        ax.text(8.32, 100*observed[-1], label, color=color,
                va="center", fontsize=9, clip_on=False)
        y = 2-j
        tail.plot([100*independent[-1], 100*observed[-1]], [y, y],
                  color=color, linewidth=.9, linestyle=(0, (2, 2)), alpha=.65)
        tail.errorbar([100*observed[-1]], [y],
                      xerr=errors, fmt="o", color=color, markersize=3.6,
                      capsize=2, elinewidth=.75, capthick=.65, zorder=4)
        tail.scatter([100*independent[-1]], [y], marker="D", s=15,
                     facecolor="white", edgecolor=color, linewidth=.8, zorder=4)
        plotted[size] = {"observed_pct": (100 * observed).tolist(),
                         "independent_pct": (100 * independent).tolist(),
                         "enrichment": (observed / independent).tolist(),
                         "k8_wilson_ci95_percent": bounds.tolist(),
                         "k8_unresolved_inputs": hist[8]}
    ax.set(yscale="log", ylim=(.14, 90), ylabel="Failure (%)")
    ax.set_yticks([.2, 1, 10, 50])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    clean(ax)
    ax.tick_params(labelsize=8.5, length=2.5, pad=2)
    ax.set(xlim=(.8, 10.3), xticks=[1, 2, 4, 6, 8])
    fig.text(.961, .512, r"$k$", ha="right", fontsize=9)
    tail.set(xscale="log", xlim=(8e-7, 30), ylim=(-.55, 2.55),
             yticks=[2, 1, 0], yticklabels=["600", "1,256", "9,057"],
             xticks=[1e-6, 1e-3, 1, 10],
             xticklabels=[r"$10^{-6}$", r"$10^{-3}$", "1", "10"])
    tail.spines[["top", "right", "left"]].set_visible(False)
    tail.tick_params(axis="y", length=0, pad=5, labelsize=8.5)
    tail.tick_params(axis="x", length=2.5, width=.6, pad=2, labelsize=8.5)
    tail.minorticks_off()
    tail.grid(axis="x", color="#E5E7EB", linewidth=.45)
    tail.set_axisbelow(True)
    fig.text(.165, .943, "(a) Candidate coverage", fontsize=9, fontweight="bold")
    fig.text(.165, .425, r"(b) Eight-sample tail vs. $p^8$", fontsize=9, fontweight="bold")
    fig.text(.57, .022, "Failure rate (%)", ha="center", fontsize=9)
    AUDIT["filtering"] = {"split": "test", "records": 1132,
                          "k": k.tolist(), "plotted": plotted,
                          "extrapolation": False, "model_fit": False,
                          "displayed_metrics": ["observed_pct", "independent_k8_pct", "k8_wilson_ci95"],
                          "width_inches": 3.08, "height_inches": 2.16,
                          "independence_reference": "p**8, p from the same eight-sample histogram"}
    save(fig, "filter_failure")


def training_endpoint() -> None:
    fig, ax = plt.subplots(figsize=(3.08, 1.92))
    fig.subplots_adjust(left=.17, right=.985, bottom=.265, top=.81)
    differences = {}
    for j, (size, directories) in enumerate([
        ("600", [TEST / f"train-en600-s{s}" for s in range(42, 47)]),
        ("1,256", [TEST / "train-en1256"]),
        ("9,057", [TEST / "train-enfull"]),
    ]):
        for decoder, marker, shift in [("U", "o", -.15), ("W", "s", .15)]:
            deltas = []
            for directory in directories:
                base = scores(directory / f"none-{decoder}.smatchpp.jsonl")
                ul = scores(directory / f"ul-{decoder}.smatchpp.jsonl")
                assert base.keys() == ul.keys()
                deltas.append(float(np.mean([ul[i] - base[i] for i in base])))
            offsets = np.linspace(-.075, .075, len(deltas)) if len(deltas) > 1 else [0]
            ax.scatter(j + shift + np.array(offsets), deltas,
                       facecolor=SIZE_COLORS[j] if decoder == "U" else "white",
                       edgecolor=SIZE_COLORS[j], marker=marker, s=20,
                       linewidth=.85, zorder=3)
            if len(deltas) > 1:
                mean_shift = -.34 if decoder == "U" else .34
                ax.scatter(j + mean_shift, np.mean(deltas), color="white", edgecolor="#222222",
                           marker="D", s=26, linewidth=.85, zorder=4)
            differences[f"{size}-{decoder}"] = deltas
    clean(ax)
    ax.axhline(0, color="#555555", linewidth=.9, linestyle="--", zorder=2)
    ax.set(xlim=(-.5, 2.5), ylim=(-5.5, 6), yticks=[-4, -2, 0, 2, 4, 6],
           xticks=[0, 1, 2], xticklabels=["600", "1,256", "9,057"],
           xlabel="Training graphs", ylabel=r"$\Delta$ Smatch++")
    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=GRAY,
                         markersize=4, label=r"Unconstrained $U$"),
               plt.Line2D([], [], marker="s", linestyle="none", color=GRAY,
                         markerfacecolor="white", markersize=4, label=r"Local mask $W$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=2, frameon=False, handlelength=.8, handletextpad=.5, columnspacing=1)
    AUDIT["training_endpoint"] = {"split": "test", "records": 1132,
                                  "seed_effects": differences, "width_inches": 3.08,
                                  "mean_600_U": float(np.mean(differences["600-U"])),
                                  "mean_600_W": float(np.mean(differences["600-W"]))}
    save(fig, "training_test_endpoints")


def dead_mass() -> None:
    data = read_json(DEV / "extra-analyses.json")["dead_mass"]
    fig = plt.figure(figsize=(3.08, 2.18))
    ax = fig.add_axes([.175, .235, .425, .515])
    early = fig.add_axes([.74, .235, .24, .515])
    plotted = {}
    for mode, label, color, ls, marker in [("none", "CE", GRAY, "-", "o"),
                                           ("ul", "UL", GREEN, "-", "s"),
                                           ("mask", "Masked", ORANGE, "--", "^")]:
        names = [f"train-en600-{mode}" + ("" if s == 42 else f"-s{s}")
                 for s in (range(42, 47) if mode != "mask" else [42])]
        series = [np.array(data[n], dtype=float) for n in names]
        assert all(np.array_equal(a[:, 0], series[0][:, 0]) for a in series)
        x = 100 * series[0][:, 0] / 380
        ys = np.array([a[:, 1] for a in series]) * 100
        mean = ys.mean(0)
        for axis, keep in ((ax, np.ones(len(x), dtype=bool)), (early, x <= 25)):
            if len(series) > 1:
                for trajectory in ys:
                    axis.plot(x[keep], trajectory[keep], color=color,
                              linewidth=.45, alpha=.2, zorder=1)
                axis.fill_between(x[keep], ys.min(0)[keep], ys.max(0)[keep],
                                  color=color, alpha=.12, linewidth=0, zorder=2)
            axis.plot(x[keep], mean[keep], color=color, linestyle=ls,
                      label=label, linewidth=1.35, marker=marker, markersize=2.7,
                      markevery=3 if axis is ax else 1, markeredgewidth=.55, zorder=4)
        plotted[label] = {"optimizer_steps": series[0][:, 0].tolist(),
                          "training_percent": x.tolist(), "seeds": len(series),
                          "seed_mass_percent": ys.tolist(), "mean_mass_percent": mean.tolist(),
                          "min_mass_percent": ys.min(0).tolist(), "max_mass_percent": ys.max(0).tolist()}
    clean(ax)
    ax.set(yscale="log", ylim=(.02, 22), xlim=(0, 101),
           xticks=[0, 50, 100], ylabel="Rejected mass (%)")
    ax.set_yticks([.03, .1, 1, 10])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.tick_params(labelsize=8.5, pad=2, length=2.5)
    clean(early)
    early.set(xlim=(0, 26), ylim=(0, 17), xticks=[0, 15, 25], yticks=[0, 8, 16])
    early.tick_params(labelsize=8.5, pad=2, length=2.5)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, .985),
               ncol=3, frameon=False, handlelength=1.3, columnspacing=.85,
               handletextpad=.45, fontsize=9)
    titles = [fig.text(.175, .795, "(a) Full training", fontsize=9, fontweight="bold"),
              fig.text(.74, .795, "(b) Early", fontsize=9, fontweight="bold")]
    fig.text(.55, .032, "Optimizer steps (%)", ha="center", fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    title_boxes = [title.get_window_extent(renderer) for title in titles]
    assert all(not legend_box.overlaps(box) for box in title_boxes)
    assert not ax.get_tightbbox(renderer).overlaps(early.get_tightbbox(renderer))
    layout = {"legend": list(legend_box.bounds),
              "titles": [list(box.bounds) for box in title_boxes],
              "panels": [list(axis.get_tightbbox(renderer).bounds) for axis in (ax, early)]}
    AUDIT["dead_mass"] = {"split": "gold training prefixes", "training_graphs": 600,
                          "candidates": "top-32 W-rejected tokens", "plotted": plotted,
                          "band": "min-max across seeds, not a confidence interval",
                          "main_y_scale": "log", "early_y_scale": "linear",
                          "width_inches": 3.08, "height_inches": 2.18,
                          "layout_pixels": layout}
    save(fig, "dead_mass")


def learning_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.08), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.23, top=.78, wspace=.18)
    for ax, dataset, title in zip(axes, ["en600", "en1256", "enfull"],
                                  ["(a) 600", "(b) 1,256", "(c) 9,057"]):
        data = read_json(DEV / f"seeds-{dataset}-report.json")["seeds"]
        for mode, label, color, marker in [("none", "CE", GRAY, "o"),
                                           ("ul", "UL", GREEN, "s")]:
            curves = [seed[mode]["curve"] for seed in data.values()]
            steps = sorted(int(s) for s in curves[0])
            assert all(set(c) == set(curves[0]) for c in curves)
            y = np.array([[c[str(s)][1] for s in steps] for c in curves])
            x = 100 * np.array(steps) / max(steps)
            ax.plot(x, y.mean(0), color=color, marker=marker,
                    linestyle="--" if mode == "ul" else "-", label=label)
            if len(curves) > 1:
                ax.fill_between(x, y.min(0), y.max(0), color=color, alpha=.12, linewidth=0)
        clean(ax)
        ax.set(xlim=(5, 103), ylim=(38, 96), xticks=[10, 35, 50, 75, 100],
               xlabel="Training steps (%)")
        ax.set_title(title, loc="left", pad=4)
    axes[0].set_ylabel("Dev Smatch++")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995),
               ncol=2, frameon=False, handlelength=2)
    save(fig, "learning_curves")


def main() -> None:
    global AUDIT
    functions = {"budget_curve": budget, "filter_failure": filtering,
                 "training_test_endpoints": training_endpoint,
                 "dead_mass": dead_mass, "learning_curves": learning_curves}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=functions,
                        help="Regenerate only the selected figures; preserve all other exports.")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    audit_path = BUILD / "results-figure-data.json"
    if args.only and audit_path.exists():
        AUDIT = json.loads(audit_path.read_text(encoding="utf-8"))
    style()
    selected = args.only or list(functions)
    for name in selected:
        functions[name]()
    AUDIT.setdefault("sources_sha256", {}).update(SOURCES)
    AUDIT["font"] = "Times New Roman (embedded TrueType)"
    (BUILD / "results-figure-data.json").write_text(
        json.dumps(AUDIT, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(PAPER / "figures"), "updated_figures": selected,
                      "audited_sources": len(SOURCES)}, indent=2))


if __name__ == "__main__":
    main()
