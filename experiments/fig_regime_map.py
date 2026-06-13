#!/usr/bin/env python
"""Figure F2 (guide §10.6): the networking-story centerpiece — sensitivity of
the methods to the network parameters (p, W, T_cut) and the proactive-vs-lazy
regime map. Reads results/phase6_main.parquet (the single source of truth);
writes results/fig_regime_map.png at 300 dpi. Regenerable by one command.

Panels:
  (a) J/J_AGG vs p        (b) J/J_AGG vs T_cut
  (c) J/J_AGG vs W        (d) regime map: proactive advantage
                              (J_GreedyJIT - J_GreedyEager)/J_GreedyJIT over
                              p x T_cut  (>0 red: always-on wins; <0 blue:
                              lazy wins = the waste regime)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

METHODS = [("eager", "EAGER (ours)", "C3", "-", "o"),
           ("agg", "AGG", "C0", "--", "s"),
           ("greedy_eager", "GreedyEager (always-on)", "C2", ":", "^"),
           ("greedy_jit", "GreedyJIT (lazy)", "C1", "-.", "v")]


def rel_to_agg(df):
    keys = ["p", "W", "T_cut", "instance", "seed"]
    agg = df[df.method == "agg"].set_index(keys).J.rename("agg_J")
    j = df.join(agg, on=keys)
    j["rel"] = j.J / j["agg_J"]
    return j


def line_panel(ax, j, axis, methods, title):
    for m, label, c, ls, mk in methods:
        sub = j[j.method == m]
        g = sub.groupby(axis).rel.agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"], yerr=g["sem"], label=label, color=c,
                    ls=ls, marker=mk, ms=4, capsize=2, lw=1.5)
    ax.axhline(1.0, color="gray", lw=0.8, alpha=0.6)
    ax.set_xlabel(axis)
    ax.set_ylabel("J / J(AGG)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main(argv=None) -> int:
    df = pd.read_parquet(RESULTS / "phase6_main.parquet")
    j = rel_to_agg(df)
    methods = [m for m in METHODS if m[0] in set(df.method)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    line_panel(axes[0, 0], j, "p", methods, "(a) vs success prob. p")
    line_panel(axes[0, 1], j, "T_cut", methods, "(b) vs cutoff T_cut")
    line_panel(axes[1, 0], j, "W", methods, "(c) vs channels W")
    axes[0, 0].legend(fontsize=8, loc="best")

    # (d) regime map heatmap: proactive advantage over p x T_cut (mean over W)
    keys = ["p", "W", "T_cut", "instance", "seed"]
    piv = df.pivot_table(index=keys, columns="method", values="J")
    piv["adv"] = (piv["greedy_jit"] - piv["greedy_eager"]) / piv["greedy_jit"]
    heat = piv.reset_index().groupby(["p", "T_cut"]).adv.mean().unstack()
    ax = axes[1, 1]
    vmax = float(np.nanmax(np.abs(heat.values)))
    im = ax.imshow(heat.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("T_cut")
    ax.set_ylabel("p")
    ax.set_title("(d) proactive advantage\n(red: always-on wins; blue: waste regime)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="(J_lazy - J_eager)/J_lazy")

    fig.suptitle("Network-parameter sensitivity and the proactive-provisioning "
                 "regime map", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS / "fig_regime_map.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    # sanity-print the numbers the figure must match
    print("\nmean J/J(AGG) by method:")
    print(j.groupby("method").rel.mean().sort_values().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
