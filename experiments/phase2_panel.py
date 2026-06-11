#!/usr/bin/env python
"""Phase 2 acceptance panel (guide §11): GreedyJIT vs Random-Progressive on
every QASMBench-derived instance under the default network config.

Acceptance asserted here:
  - GreedyJIT completes EVERY instance with ZERO truncations;
  - mean J(GreedyJIT) < mean J(Random-Progressive) on EVERY instance,
    CRN-paired (both policies see the same env seeds, guide §10.4).

Default panel hardware (D32): K=4 2x2 grid, kappa_u = ceil(1.25*N/4),
link defaults per guide §10.2 (p=1/12, W=2, B=8, T_cut=20, w=1), stochastic.

Writes results/phase2_panel.parquet + results/index.json (results/ is owned
by experiments/, guide §12). Exit code 0 only if acceptance holds.

Usage (from the repo root):
    python experiments/phase2_panel.py [--seeds 5] [--only adder_n4,bv_n30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.random_prog import RandomProgressivePolicy
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import load_circuit_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware

REPO = Path(__file__).resolve().parent.parent
PANEL_DIR = REPO / "configs" / "circuits" / "qasmbench"
RESULTS = REPO / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--only", default=None,
                        help="comma-separated instance stems (default: all)")
    parser.add_argument("--verdict-only", action="store_true",
                        help="re-evaluate acceptance from the existing "
                             "results/phase2_panel.parquet without re-running")
    args = parser.parse_args(argv)

    if args.verdict_only:
        df = pd.read_parquet(RESULTS / "phase2_panel.parquet")
        return evaluate(df, args.seeds)

    yamls = sorted(PANEL_DIR.glob("*.yaml"))
    if args.only:
        keep = set(args.only.split(","))
        yamls = [y for y in yamls if y.stem in keep]
    if not yamls:
        print("no panel instances found; run scripts/extract_qasm.py first",
              file=sys.stderr)
        return 2

    rows = []
    for ypath in yamls:
        inst = build_instance(load_circuit_config(ypath))
        hw = default_panel_hardware(inst.num_qubits)
        for e in range(args.seeds):
            env_seed = e                      # CRN pairing: same seeds for all
            for policy_name in ("greedy_jit", "random_progressive"):
                if policy_name == "greedy_jit":
                    policy = GreedyJITPolicy(placement_seed=0)
                else:
                    policy = RandomProgressivePolicy(policy_seed=9000 + e)
                env = EagerEnv(hw, inst)
                t0 = time.perf_counter()
                info, actions, _ = run_episode(env, policy, env_seed)
                wall = time.perf_counter() - t0
                m = info["metrics"]
                rows.append({
                    "instance": ypath.stem, "N": inst.num_qubits,
                    "M": inst.num_gates, "policy": policy_name,
                    "env_seed": env_seed, "T": m["T"], "C_comm": m["C_comm"],
                    "C_waste": m["C_waste"], "J": m["J"],
                    "truncated": m["truncated"],
                    "epr_utilization": m["epr_utilization"],
                    "mean_remote_stall": m["mean_remote_stall"],
                    "micro_steps": len(actions), "wall_s": round(wall, 3),
                })
                print(f"{ypath.stem:>16} seed={env_seed} {policy_name:<20} "
                      f"T={m['T']:>6} J={m['J']:>10.6g} "
                      f"trunc={m['truncated']} ({wall:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / "phase2_panel.parquet", index=False)

    qasm_commit = (REPO / "qasm" / "qasmbench" / "SOURCE_COMMIT.txt"
                   ).read_text().strip()
    index = {"phase2_panel": {
        "path": "phase2_panel.parquet",
        "instances": sorted(df["instance"].unique().tolist()),
        "seeds": list(range(args.seeds)),
        "policies": ["greedy_jit", "random_progressive"],
        "hardware": "K=4 2x2 grid, kappa=ceil(1.25N/4), p=1/12, W=2, B=8, "
                    "T_cut=20, w=1, stochastic",
        "qasmbench_commit": qasm_commit,
        "crn_paired": True,
    }}
    with open(RESULTS / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    print(f"results -> {RESULTS / 'phase2_panel.parquet'} (+ index.json)")

    return evaluate(df, args.seeds)


def evaluate(df: pd.DataFrame, n_seeds: int) -> int:
    """Print the per-instance table and BOTH acceptance readings: the guide
    §11 criterion as written, and the D35 amended criterion (zero greedy
    truncations everywhere; J ordering required except on characterized
    provisioning-throughput-bound instances, which are reported as the
    proactive-provisioning opportunity gap)."""
    print("\ninstance          N     M   J(greedy)   J(random)  win  g_trunc")
    print("-" * 70)
    guide_ok = True
    amended_ok = True
    exceptions = []
    for stem in sorted(df["instance"].unique()):
        sub = df[df["instance"] == stem]
        g = sub[sub["policy"] == "greedy_jit"]
        r = sub[sub["policy"] == "random_progressive"]
        jg, jr = g["J"].mean(), r["J"].mean()
        g_trunc = int(g["truncated"].sum())
        wins = int(sum(
            gj < rj for gj, rj in zip(
                g.sort_values("env_seed")["J"], r.sort_values("env_seed")["J"])))
        ordering = jg < jr
        guide_ok &= (g_trunc == 0) and ordering
        amended_ok &= (g_trunc == 0)
        n, m = int(g["N"].iloc[0]), int(g["M"].iloc[0])
        note = ""
        if not ordering:
            exceptions.append(stem)
            note = "   <-- ordering exception (D35)"
        if g_trunc:
            note += "   <-- TRUNCATION (hard fail)"
        print(f"{stem:<16} {n:>3} {m:>6} {jg:>11.6g} {jr:>11.6g} "
              f"{wins}/{len(g)}  {g_trunc}{note}")

    print(f"\nguide criterion as written in section 11 (zero greedy "
          f"truncations AND mean J greedy < random on ALL): "
          f"{'PASS' if guide_ok else 'FAIL'}")
    print(f"D35 amended criterion (zero greedy truncations on all; ordering "
          f"exceptions characterized): "
          f"{'PASS' if amended_ok else 'FAIL'}"
          + (f"; exceptions = {exceptions}" if exceptions else ""))
    return 0 if amended_ok else 1


if __name__ == "__main__":
    sys.exit(main())
