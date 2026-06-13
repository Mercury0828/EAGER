#!/usr/bin/env python
"""Phase 6 regime grid (guide §10.6 F2 — the networking-story centerpiece).

Evaluates every method across a (p, W, T_cut) network-parameter grid on a
fixed synthetic instance panel, CRN-paired (every method sees the same
(instance, env seed) pairs so generation luck cancels in the comparison,
§6.5/§10.4). Writes one row per (method, p, W, T_cut, instance, seed) to
results/phase6_regime_grid.parquet + updates results/index.json — the single
source of truth for the regime map.

Methods: GreedyJIT (lazy), GreedyEager (always-on), MHSA+LS, AGG,
Random-Progressive, and (if --eager-ckpt given) the learned EAGER policy.
The GreedyJIT-vs-GreedyEager pair is the heuristic proactivity ablation
(value AND boundary of proactive provisioning); EAGER is the learned policy.

Frozen weights (D74): alpha=beta=1, gamma=0.5. gamma also swept here when
--gamma-sweep is passed (D74 integrity guard).

Usage (from the repo root):
    python experiments/phase6_regime_grid.py --instances 12 --seeds 8 \
        --eager-ckpt artifacts/agents/ppo_seed1.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eager.baselines.agg import make_agg_method
from eager.baselines.greedy_jit import GreedyEagerPolicy, GreedyJITPolicy
from eager.baselines.mhsa import make_mhsa_policy
from eager.baselines.random_prog import RandomProgressivePolicy
from eager.baselines.traces import run_episode
from eager.config import SynthParams, load_hardware_config
from eager.env import EagerEnv
from eager.expgen.synthetic import generate_instance

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# network-parameter grid (the F2 sweep axes; p x W x T_cut)
P_GRID = [0.05, 0.08, 0.12, 0.2, 0.3, 0.5]
W_GRID = [1, 2, 4]
TCUT_GRID = [5, 20, 50]
B_FIXED = 8


def hardware(qpus: int, n_qubits: int, p: float, w_ch: int, t_cut: int):
    kappa = math.ceil(1.25 * n_qubits / qpus)
    topo = ({"qpus": 2, "topology": "line"} if qpus == 2
            else {"qpus": 4, "topology": "grid", "grid_dims": [2, 2]})
    return load_hardware_config({
        "name": f"k{qpus}_p{p}_w{w_ch}_b{B_FIXED}_c{t_cut}", **topo,
        "kappa": kappa, "mode": "stochastic", "t_ep": 12,
        "link_defaults": {"p": p, "W": w_ch, "B": B_FIXED, "T_cut": t_cut,
                          "w": 1.0}})


def build_panel(n_instances: int, seed: int = 4242):
    """Fixed synthetic instances spanning density {1,3} and N in [15,30]."""
    rng = np.random.default_rng(seed)
    panel = []
    for _ in range(n_instances):
        n = int(rng.integers(15, 31))
        d = int(rng.choice([1, 3]))
        gseed = int(rng.integers(0, 99999))
        panel.append(generate_instance(SynthParams(n, n * d, None), seed=gseed))
    return panel


HEURISTICS = ["greedy_jit", "greedy_eager", "greedy_adaptive",
              "greedy_regime_prov", "cloudqc", "mhsa_ls", "agg", "random_prog"]


def precompute_per_instance(panel, qpus: int):
    """Placements (greedy partitioner, MHSA) and AGG transforms depend only
    on the instance (+ a capacity-fixed reference hardware), NOT on p/W/T_cut
    — compute once and reuse across the whole grid (MHSA's SA is the costly
    part)."""
    from eager.baselines.greedy_jit import compute_placement
    from eager.baselines.mhsa import mhsa_placement
    from eager.baselines.agg import transform_instance
    cache = {}
    for inst in panel:
        ref = hardware(qpus, inst.num_qubits, 0.12, 2, 20)  # capacity ref
        greedy_pl = compute_placement(inst, ref, seed=0)
        mhsa_pl = mhsa_placement(inst, ref, seed=0, budget=8000)
        agg_inst, _ = transform_instance(inst, greedy_pl)
        cache[inst.name] = {"greedy": greedy_pl, "mhsa": mhsa_pl,
                            "agg_inst": agg_inst}
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--qpus", type=int, default=4)
    parser.add_argument("--eager-ckpt", default=None)
    parser.add_argument("--methods", default=None,
                        help="comma-separated subset of heuristics (default all)")
    parser.add_argument("--out", default="phase6_regime_grid.parquet")
    args = parser.parse_args(argv)
    methods = (args.methods.split(",") if args.methods else HEURISTICS)

    panel = build_panel(args.instances)
    env_seeds = list(range(args.seeds))
    rows = []
    t0 = time.perf_counter()
    print("precomputing per-instance placements (greedy, MHSA) + AGG ...",
          flush=True)
    plc = precompute_per_instance(panel, args.qpus)
    print(f"  done ({time.perf_counter() - t0:.0f}s)", flush=True)

    eager = None
    if args.eager_ckpt:
        import torch
        from eager.model.policy import EagerPolicy
        from eager.train.evaluate import run_agent_episodes_batched
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eager = EagerPolicy()
        ck = torch.load(args.eager_ckpt, map_location="cpu", weights_only=False)
        eager.load_state_dict(ck["state_dict"])
        eager.to(dev)
        print(f"EAGER policy loaded on {dev}")

    n_cfg = len(P_GRID) * len(W_GRID) * len(TCUT_GRID)
    cfg_i = 0
    for p in P_GRID:
        for w_ch in W_GRID:
            for t_cut in TCUT_GRID:
                cfg_i += 1
                for inst in panel:
                    h = hardware(args.qpus, inst.num_qubits, p, w_ch, t_cut)
                    pc = plc[inst.name]
                    # heuristics (placements reused from the cache)
                    for mname in methods:
                        for e in env_seeds:
                            run_inst = inst
                            if mname == "greedy_jit":
                                policy = GreedyJITPolicy(
                                    placement_fn=lambda i, hw, pl=pc["greedy"]: pl)
                            elif mname == "greedy_eager":
                                policy = GreedyEagerPolicy()
                                policy._placement = pc["greedy"]
                            elif mname == "greedy_adaptive":
                                from eager.baselines.greedy_jit import GreedyAdaptivePolicy
                                policy = GreedyAdaptivePolicy()
                                policy._placement = pc["greedy"]
                            elif mname == "greedy_regime_prov":
                                from eager.baselines.greedy_jit import GreedyRegimeProvisionPolicy
                                policy = GreedyRegimeProvisionPolicy(
                                    placement=pc["greedy"])
                            elif mname == "cloudqc":
                                from eager.baselines.cloudqc import CloudQCPolicy
                                policy = CloudQCPolicy(placement=pc["greedy"])
                            elif mname == "mhsa_ls":
                                policy = GreedyJITPolicy(
                                    placement_fn=lambda i, hw, pl=pc["mhsa"]: pl,
                                    name="mhsa_ls")
                            elif mname == "agg":
                                policy = GreedyJITPolicy(
                                    placement_fn=lambda i, hw, pl=pc["greedy"]: pl,
                                    name="agg_ls")
                                run_inst = pc["agg_inst"]
                            else:  # random_prog
                                policy = RandomProgressivePolicy(
                                    policy_seed=9000 + e)
                            env = EagerEnv(h, run_inst)
                            info, _, _ = run_episode(env, policy, e)
                            m = info["metrics"]
                            rows.append({
                                "method": mname, "p": p, "W": w_ch,
                                "T_cut": t_cut, "instance": inst.name,
                                "N": inst.num_qubits, "M": inst.num_gates,
                                "seed": e, "J": m["J"], "T": m["T"],
                                "C_comm": m["C_comm"], "C_waste": m["C_waste"],
                                "truncated": m["truncated"]})
                    # EAGER (batched)
                    if eager is not None:
                        pairs = [(EagerEnv(h, inst), e) for e in env_seeds]
                        ms = run_agent_episodes_batched(eager, pairs, dev)
                        for e, m in zip(env_seeds, ms):
                            rows.append({
                                "method": "eager", "p": p, "W": w_ch,
                                "T_cut": t_cut, "instance": inst.name,
                                "N": inst.num_qubits, "M": inst.num_gates,
                                "seed": e, "J": m["J"], "T": m["T"],
                                "C_comm": m["C_comm"], "C_waste": m["C_waste"],
                                "truncated": m["truncated"]})
                print(f"[{cfg_i}/{n_cfg}] p={p} W={w_ch} T_cut={t_cut} done "
                      f"({time.perf_counter() - t0:.0f}s, {len(rows)} rows)",
                      flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / args.out, index=False)
    index_path = RESULTS / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    index["phase6_regime_grid"] = {
        "path": args.out,
        "methods": sorted(df["method"].unique().tolist()),
        "p_grid": P_GRID, "W_grid": W_GRID, "T_cut_grid": TCUT_GRID,
        "B": B_FIXED, "qpus": args.qpus, "instances": args.instances,
        "seeds": args.seeds, "weights": "alpha=1,beta=1,gamma=0.5 (D74)",
        "crn_paired": True,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True),
                          encoding="utf-8")
    print(f"\nwrote {len(rows)} rows -> {RESULTS / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
