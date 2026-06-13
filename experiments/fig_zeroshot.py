#!/usr/bin/env python
"""Figure F3 (guide §10.3): zero-shot generalization of the LOCKED path-B
EAGER policy (trained on synthetic N in [10,30], K in {2,4}) to real
QASMBench circuit families at N up to 98. Reads results/phase6_zeroshot.parquet
(K=4, the trained topology) and results/phase6_zeroshot_k8.parquet (K=8, an
unseen topology); writes results/fig_zeroshot.png at 300 dpi.

Bars: J(EAGER)/J(AGG-reactive) per circuit (mean over seeds), one bar per
regime. <1 (below the dashed line) = EAGER wins zero-shot.
  (a) K=4 (trained topology): EAGER generalizes across real circuit families
      and 3x-larger N, still beating AGG-reactive.
  (b) K=8 (UNSEEN topology): transfer is unreliable (no significant gain;
      degenerate on small-N over-partitioned circuits) — a stated limitation.
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


def ratios(df):
    g = (df.assign(r=df.eager_J / df.reactive_J)
         .groupby(["circuit", "N", "regime"]).r.mean().reset_index())
    g = g.sort_values("N")
    return g


def panel(ax, g, title):
    circuits = list(dict.fromkeys(g.circuit))
    labels = [f"{c}\n(N={int(g[g.circuit==c].N.iloc[0])})" for c in circuits]
    x = np.arange(len(circuits))
    regimes = ["normal", "prov_bound"]
    w = 0.38
    for i, reg in enumerate(regimes):
        vals = [g[(g.circuit == c) & (g.regime == reg)].r.mean()
                for c in circuits]
        ax.bar(x + (i - 0.5) * w, vals, w, label=reg,
               color=["#4C72B0", "#DD8452"][i])
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("J(EAGER) / J(AGG-reactive)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main(argv=None) -> int:
    k4 = ratios(pd.read_parquet(RESULTS / "phase6_zeroshot.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panel(axes[0], k4, "(a) K=4 trained topology: zero-shot to real circuits, "
                       "N up to 98\n(below 1 = EAGER wins; overall 0.90, "
                       "p=4.5e-11)")
    k8_path = RESULTS / "phase6_zeroshot_k8.parquet"
    if k8_path.exists():
        k8 = ratios(pd.read_parquet(k8_path))
        panel(axes[1], k8, "(b) K=8 UNSEEN topology: transfer unreliable "
                           "(no significant gain, p=0.95)\ndegenerate on "
                           "over-partitioned small-N circuits")
        axes[1].set_yscale("log")
    fig.suptitle("Zero-shot generalization of the locked EAGER policy "
                 "(no retraining)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / "fig_zeroshot.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    print("\nK=4 per-circuit J(EAGER)/J(AGG-react):")
    print(k4.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
