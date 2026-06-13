#!/usr/bin/env python
"""Phase 6 main results table (guide §10.6 T3): the unified method ranking
including the path-B EAGER (AGG placement + aggregation + LEARNED proactive
provisioning), CRN-paired on the regime grid panel.

Static heuristic rows are read from the already-computed
results/phase6_regime_grid.parquet; this script adds the EAGER rows by
running the learned path-B policy on the SAME (instance, config, seed)
triples (AGG-transformed, AGG-placement pre-applied) and writes the merged
results/phase6_main.parquet + the ranking summary.

Usage (from the repo root):
    python experiments/phase6_main_table.py --eager-ckpt artifacts/agents/eager_final.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from eager.baselines.agg import transform_instance
from eager.baselines.greedy_jit import compute_placement
from eager.config import load_hardware_config
from eager.env import EagerEnv
from eager.env.actions import Map
from eager.baselines.greedy_jit import map_emission_order
from eager.model.policy import EagerPolicy, act_greedy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase6_regime_grid import P_GRID, W_GRID, TCUT_GRID, build_panel, hardware

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def eager_on_agg_J(policy, inst, hw, placement, agg_inst, seed, device):
    env = EagerEnv(hw, agg_inst)
    env.reset(seed)
    for q in map_emission_order(agg_inst):
        if q in env._unmapped:
            env.step(Map(q, placement[q]))
    done = False
    while not done:
        _, _, done, info = env.step(act_greedy(policy, env, device))
    return info["metrics"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eager-ckpt", default="artifacts/agents/eager_final.pt")
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--qpus", type=int, default=4)
    args = ap.parse_args(argv)

    grid = pd.read_parquet(RESULTS / "phase6_regime_grid.parquet")
    grid = grid[grid.method != "eager"]          # drop the old clone-GreedyJIT eager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = EagerPolicy()
    policy.load_state_dict(torch.load(args.eager_ckpt, map_location="cpu",
                                      weights_only=False)["state_dict"])
    policy.to(device)
    print(f"path-B EAGER loaded on {device}")

    panel = build_panel(args.instances)
    pre = {}
    for inst in panel:
        ref = hardware(args.qpus, inst.num_qubits, 0.12, 2, 20)
        pl = compute_placement(inst, ref, seed=0)
        agg_inst, _ = transform_instance(inst, pl)
        pre[inst.name] = (pl, agg_inst)

    rows = []
    t0 = time.perf_counter()
    cfgs = [(p, w, tc) for p in P_GRID for w in W_GRID for tc in TCUT_GRID]
    for ci, (p, w, tc) in enumerate(cfgs, 1):
        for inst in panel:
            h = hardware(args.qpus, inst.num_qubits, p, w, tc)
            pl, agg_inst = pre[inst.name]
            for e in range(args.seeds):
                m = eager_on_agg_J(policy, inst, h, pl, agg_inst, e, device)
                rows.append({"method": "eager", "p": p, "W": w, "T_cut": tc,
                             "instance": inst.name, "N": inst.num_qubits,
                             "M": inst.num_gates, "seed": e, "J": m["J"],
                             "T": m["T"], "C_comm": m["C_comm"],
                             "C_waste": m["C_waste"],
                             "truncated": m["truncated"]})
        print(f"[{ci}/{len(cfgs)}] p={p} W={w} T_cut={tc} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    merged = pd.concat([grid, pd.DataFrame(rows)], ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    merged.to_parquet(RESULTS / "phase6_main.parquet", index=False)

    print("\n=== method ranking (mean J over the regime grid, lower=better) ===")
    print(merged.groupby("method").J.mean().sort_values().round(2).to_string())

    # CRN-paired EAGER vs AGG (and vs each method) on matched (instance,cfg,seed)
    print("\n=== EAGER vs each method, CRN-paired (Wilcoxon, EAGER<other) ===")
    keys = ["p", "W", "T_cut", "instance", "seed"]
    em = merged[merged.method == "eager"].set_index(keys).J
    for other in ["agg", "mhsa_ls", "greedy_eager", "greedy_jit", "random_prog"]:
        om = merged[merged.method == other].set_index(keys).J
        j = em.to_frame("e").join(om.to_frame("o"), how="inner").dropna()
        a, b = j.e.values, j.o.values
        pv = stats.wilcoxon(a, b, alternative="less").pvalue if not np.allclose(a, b) else 1.0
        print(f"  EAGER vs {other:<13} ratio={a.mean()/b.mean():.4f} "
              f"won={(a<b).sum()}/{len(a)} p={pv:.2e}")
    print(f"\nresults -> {RESULTS / 'phase6_main.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
