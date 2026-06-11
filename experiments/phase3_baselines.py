#!/usr/bin/env python
"""Phase 3 acceptance runs (guide §11):

A. MHSA placement vs the §9.1 partitioner: remote-gate count (interaction-
   graph cut) on a 20-instance panel (14 frozen panel circuits + 6 seeded
   synthetics). Acceptance: MHSA <= partitioner on >= 70%.
B. AGG vs GreedyJIT, CRN-paired: consumed pairs (and J) per instance, burst
   statistics. Acceptance: strict consumed-pair reduction on every burst-
   carrying instance; burst-free instances must be exactly unchanged.

Writes results/phase3_mhsa.parquet, results/phase3_agg.parquet and updates
results/index.json. Exit 0 only if both acceptance blocks hold.

Usage (from the repo root):
    python experiments/phase3_baselines.py [--seeds 3] [--mhsa-budget 20000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from eager.baselines.agg import make_agg_method
from eager.baselines.greedy_jit import GreedyJITPolicy, compute_placement
from eager.baselines.mhsa import mhsa_placement
from eager.baselines.partition import cut_weight, interaction_graph
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import SynthParams, load_circuit_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_instance

REPO = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO / "configs" / "circuits" / "qasmbench"
RESULTS = REPO / "results"

SYNTH_PANEL = [(10, 3, 11), (20, 3, 12), (30, 3, 13),
               (40, 2, 14), (50, 2, 15), (60, 1, 16)]


def panel_instances():
    insts = [build_instance(load_circuit_config(y))
             for y in sorted(PANEL_DIR.glob("*.yaml"))]
    insts += [generate_instance(SynthParams(n, n * d, None), seed=s)
              for (n, d, s) in SYNTH_PANEL]
    return insts


def part_a_mhsa(budget: int) -> tuple[pd.DataFrame, bool]:
    rows = []
    for inst in panel_instances():
        hw = default_panel_hardware(inst.num_qubits)
        w = interaction_graph(inst)
        t0 = time.perf_counter()
        cut_part = cut_weight(
            compute_placement(inst, hw, seed=0), w)
        t1 = time.perf_counter()
        cut_m = cut_weight(mhsa_placement(inst, hw, seed=0, budget=budget), w)
        t2 = time.perf_counter()
        rows.append({"instance": inst.name, "N": inst.num_qubits,
                     "M": inst.num_gates, "cut_partitioner": cut_part,
                     "cut_mhsa": cut_m, "mhsa_le": cut_m <= cut_part,
                     "t_part_s": round(t1 - t0, 3),
                     "t_mhsa_s": round(t2 - t1, 3)})
        print(f"[MHSA] {inst.name:<22} cut_part={cut_part:>5} "
              f"cut_mhsa={cut_m:>5} {'<=' if cut_m <= cut_part else '> '}",
              flush=True)
    df = pd.DataFrame(rows)
    wins = int(df["mhsa_le"].sum())
    need = -(-len(df) * 7 // 10)        # ceil(0.7 * n)
    ok = wins >= need
    print(f"\n[MHSA] remote-gate count <= partitioner on {wins}/{len(df)} "
          f"instances (need >= {need}, budget={budget}): "
          f"{'PASS' if ok else 'FAIL'}")
    return df, ok


def part_b_agg(seeds: int) -> tuple[pd.DataFrame, bool]:
    rows = []
    ok = True
    for ypath in sorted(PANEL_DIR.glob("*.yaml")):
        inst = build_instance(load_circuit_config(ypath))
        hw = default_panel_hardware(inst.num_qubits)
        transformed, agg_policy, placement, stats = make_agg_method(inst, hw)
        cons_g, cons_a, j_g, j_a = [], [], [], []
        for e in range(seeds):
            env = EagerEnv(hw, inst)
            info, _, _ = run_episode(env, GreedyJITPolicy(placement_seed=0), e)
            m = info["metrics"]
            cons_g.append(m["pairs"]["consumed"])
            j_g.append(m["J"])
            env = EagerEnv(hw, transformed)
            info, _, _ = run_episode(env, agg_policy, e)
            m = info["metrics"]
            cons_a.append(m["pairs"]["consumed"])
            j_a.append(m["J"])
        has_bursts = stats["gates_aggregated"] > 0
        if has_bursts:
            inst_ok = all(a < g for a, g in zip(cons_a, cons_g))
        else:
            inst_ok = cons_a == cons_g
        ok &= inst_ok
        rows.append({"instance": ypath.stem, "N": inst.num_qubits,
                     "M": inst.num_gates, "n_bursts": stats["n_bursts"],
                     "gates_aggregated": stats["gates_aggregated"],
                     "consumed_greedy_mean": sum(cons_g) / seeds,
                     "consumed_agg_mean": sum(cons_a) / seeds,
                     "J_greedy_mean": sum(j_g) / seeds,
                     "J_agg_mean": sum(j_a) / seeds,
                     "ok": inst_ok})
        print(f"[AGG] {ypath.stem:<16} bursts={stats['n_bursts']:>3} "
              f"agg_gates={stats['gates_aggregated']:>4} "
              f"pairs {sum(cons_g)/seeds:>7.1f} -> {sum(cons_a)/seeds:>7.1f}  "
              f"J {sum(j_g)/seeds:>9.1f} -> {sum(j_a)/seeds:>9.1f}"
              f"{'' if inst_ok else '   <-- FAIL'}", flush=True)
    df = pd.DataFrame(rows)
    n_burst_carrying = int((df["gates_aggregated"] > 0).sum())
    print(f"\n[AGG] strict pair reduction on all {n_burst_carrying} "
          f"burst-carrying instances; burst-free unchanged: "
          f"{'PASS' if ok else 'FAIL'}")
    return df, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--mhsa-budget", type=int, default=20_000)
    args = parser.parse_args(argv)

    df_a, ok_a = part_a_mhsa(args.mhsa_budget)
    df_b, ok_b = part_b_agg(args.seeds)

    RESULTS.mkdir(exist_ok=True)
    df_a.to_parquet(RESULTS / "phase3_mhsa.parquet", index=False)
    df_b.to_parquet(RESULTS / "phase3_agg.parquet", index=False)
    index_path = RESULTS / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    index.update({
        "phase3_mhsa": {
            "path": "phase3_mhsa.parquet",
            "panel": "14 frozen panel circuits + 6 seeded synthetics",
            "mhsa_budget": args.mhsa_budget, "placement_seed": 0,
        },
        "phase3_agg": {
            "path": "phase3_agg.parquet",
            "seeds": list(range(args.seeds)), "crn_paired": True,
            "placement": "shared with greedy_jit (seed 0)",
        },
    })
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    print(f"\nphase 3 acceptance: MHSA={'PASS' if ok_a else 'FAIL'} "
          f"AGG={'PASS' if ok_b else 'FAIL'}")
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    sys.exit(main())
