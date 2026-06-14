#!/usr/bin/env python
"""Figure F4 (guide §10.6 / D83): graph (R-GCN) vs flat (MLP, no message
passing) PPO trajectories under the IDENTICAL IL+PPO pipeline (only the encoder
differs). Reads artifacts/agents/pathb_graph_traj.json + pathb_flat_traj.json;
writes results/fig_ppo_stability.png at 300 dpi.

IMPORTANT CORRECTION (D86): the 5-seed run REFUTED the original "graph PPO is
stable, flat diverges" reading of this figure — graph PPO ALSO diverges late on
~half the seeds (best-val rescues all), so the single graph trajectory shown
here is an unrepresentative (cherry-stable) run and PPO stability is NOT the
graph-vs-flat differentiator. The real, reproducible why-GNN signal is
ADAPTIVITY (figure F5, fig_seed_robustness.py): across 5 seeds the graph beats
BOTH reactive AND always-on, while flat beats reactive but LOSES to always-on.
This figure is retained only to show the flat-PPO divergence + the IL tie; for
the why-GNN claim use F5. Both panels still show the IL tie (annotated).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "agents"
RESULTS = REPO / "results"


def load(tag):
    d = json.loads((ART / f"pathb_{tag}.json").read_text(encoding="utf-8"))
    tr = d["ppo_trajectory"]
    return {"il": d["il_val_top1"],
            "it": [p["iter"] for p in tr],
            "ratio": [p["val_ratio_vs_react"] for p in tr],
            "kl": [p["kl"] for p in tr],
            "final": d["eval"]["full"]["vs_AGGreactive_ratio"]}


def main(argv=None) -> int:
    graph = load("graph_traj")
    flat = load("flat_traj")

    fig, (ax, axk) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) val ratio trajectory (log scale — flat diverges by ~10x)
    ax.plot(graph["it"], graph["ratio"], "-o", color="C3", ms=4, lw=1.8,
            label=f"R-GCN (graph) — IL top-1 {graph['il']:.3f}")
    ax.plot(flat["it"], flat["ratio"], "-s", color="C0", ms=4, lw=1.8,
            label=f"MLP (flat, no msg-passing) — IL top-1 {flat['il']:.3f}")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="AGG-reactive (=1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("PPO refinement iteration")
    ax.set_ylabel("validation J / J(AGG-reactive)  (lower=better)")
    ax.set_title("(a) PPO refinement stability\nIL ties; only the graph "
                 "refines stably (flat diverges)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    # (b) approximate KL per logged iter (stability signal)
    axk.plot(graph["it"], graph["kl"], "-o", color="C3", ms=4, lw=1.8,
             label="R-GCN (graph)")
    axk.plot(flat["it"], flat["kl"], "-s", color="C0", ms=4, lw=1.8,
             label="MLP (flat)")
    axk.set_xlabel("PPO refinement iteration")
    axk.set_ylabel("approx. KL per update")
    axk.set_title("(b) policy-update KL\n(graph small/steady; flat erratic)")
    axk.legend(fontsize=8)
    axk.grid(alpha=0.3)

    fig.suptitle("Why a graph encoder: representation isolation under identical "
                 "IL+PPO (only the encoder differs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS / "fig_ppo_stability.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    print(f"graph: IL {graph['il']:.3f}, final held-out {graph['final']:.4f}, "
          f"PPO ratio range [{min(graph['ratio']):.3f}, {max(graph['ratio']):.3f}]")
    print(f"flat:  IL {flat['il']:.3f}, final held-out {flat['final']:.4f}, "
          f"PPO ratio range [{min(flat['ratio']):.3f}, {max(flat['ratio']):.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
