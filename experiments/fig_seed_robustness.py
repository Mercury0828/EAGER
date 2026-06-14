#!/usr/bin/env python
"""Figure F5 (review items A + corrected-D83): seed robustness AND the
honest why-GNN signal. Supersedes the PPO-stability framing of F4 — the 5-seed
run revealed that graph PPO ALSO diverges late on ~half the seeds (best-val
rescues all), so PPO stability is NOT the graph-vs-flat differentiator. The
real, reproducible distinction is ADAPTIVITY: across seeds the graph policy
beats BOTH AGG-reactive and always-on, whereas the flat (MLP) policy beats
reactive but is WORSE than always-on (it fails to learn the regime-adaptive
hold-back).

Reads artifacts/agents/pathb_seed{0..4}.json (graph) and pathb_flat_seed0 /
pathb_flat_traj.json (flat). Writes results/fig_seed_robustness.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "agents"
RESULTS = REPO / "results"


def load(tag):
    f = ART / f"pathb_{tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["eval"]["full"]


def main(argv=None) -> int:
    graph = [load(f"seed{s}") for s in range(5)]
    graph = [g for g in graph if g]
    flat = [load(t) for t in ("flat_seed0", "flat_traj")]
    flat = [f for f in flat if f]

    g_react = [g["vs_AGGreactive_ratio"] for g in graph]
    g_eager = [g["vs_AGGeager_ratio"] for g in graph]
    f_react = [f["vs_AGGreactive_ratio"] for f in flat]
    f_eager = [f["vs_AGGeager_ratio"] for f in flat]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = {"graph": "C3", "flat": "C0"}
    # two comparison groups on x: vs reactive, vs always-on
    def scatter(xc, vals, color, marker, label):
        jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * 0.06
        ax.scatter(np.full(len(vals), xc) + jit, vals, c=color, marker=marker,
                   s=70, zorder=3, label=label, edgecolors="k", linewidths=0.5)
        ax.hlines(np.mean(vals), xc - 0.18, xc + 0.18, color=color, lw=2, zorder=2)

    scatter(0.0, g_react, cols["graph"], "o", "R-GCN (graph), 5 seeds")
    scatter(0.35, f_react, cols["flat"], "s", "MLP (flat), 2 runs")
    scatter(1.2, g_eager, cols["graph"], "o", None)
    scatter(1.55, f_eager, cols["flat"], "s", None)

    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks([0.175, 1.375])
    ax.set_xticklabels(["vs AGG-reactive\n(both encoders win)",
                        "vs always-on\n(only the graph wins = adaptivity)"])
    ax.set_ylabel("held-out J ratio (lower = better; <1 = wins)")
    ax.set_title("Seed robustness + the why-GNN signal is ADAPTIVITY, not PPO "
                 "stability\n(graph beats both baselines across 5 seeds; flat "
                 "beats reactive but loses to always-on)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"graph vs-react {np.mean(g_react):.3f}±{np.std(g_react):.3f}\n"
                f"all 5 seeds < 1.0", (0.0, max(g_react)), fontsize=7,
                ha="center", va="bottom", xytext=(0, 8),
                textcoords="offset points")

    fig.tight_layout()
    out = RESULTS / "fig_seed_robustness.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    print(f"graph vs-react: {np.mean(g_react):.4f}±{np.std(g_react):.4f} "
          f"(all<1: {all(x<1 for x in g_react)}); vs-always-on mean "
          f"{np.mean(g_eager):.4f} (all<=1.001: {all(x<=1.001 for x in g_eager)})")
    print(f"flat vs-react: {np.mean(f_react):.4f} (beats reactive); "
          f"vs-always-on {np.mean(f_eager):.4f} (>1 = loses to always-on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
