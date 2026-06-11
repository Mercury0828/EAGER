#!/usr/bin/env python
"""Phase 4 gap harness (guide §9.6): exact MILP vs GreedyJIT on
deterministic-mode instances within the §9.6 envelope (N <= 12, M <= 30,
K in {2,3}); reports J*, status, MIP gap, runtime, GreedyJIT's J and its
optimality gap (J_greedy - J*)/J*. Every solution is replay-verified in the
env before being recorded.

The full T4 table (1h/instance limit, EAGER rows) belongs to Phase 6; this
harness is the reusable runner plus the Phase 4 evidence run.

Usage (from the repo root):
    python experiments/phase4_gurobi_gap.py [--time-limit 600]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import SynthParams, load_circuit_config, load_hardware_config
from eager.env import EagerEnv, EnvParams
from eager.exact.milp import replay_exact, solve_exact
from eager.expgen.synthetic import generate_instance

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

SYNTH = [(8, 16, 21), (10, 20, 22), (12, 30, 23)]      # (N, M, seed)


def det_hw(qpus: int, n_qubits: int):
    kappa = math.ceil(1.25 * n_qubits / qpus)
    return load_hardware_config({
        "name": f"det_k{qpus}_line_kap{kappa}", "qpus": qpus,
        "topology": "line", "kappa": kappa,
        "mode": "deterministic", "t_ep": 12,
        "link_defaults": {"p": 1.0, "W": 2, "B": 8, "T_cut": None, "w": 1.0}})


def cases():
    hw_golden = load_hardware_config(
        REPO / "configs" / "hardware" / "golden_k2_det.yaml")
    for stem in ("golden_micro_1", "golden_micro_2"):
        inst = build_instance(load_circuit_config(
            REPO / "configs" / "circuits" / f"{stem}.yaml"))
        yield stem, hw_golden, inst
    for (n, m, s) in SYNTH:
        inst = generate_instance(SynthParams(n, m, None), seed=s)
        for k in (2, 3):
            yield f"synth_n{n}_m{m}_k{k}", det_hw(k, n), inst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=600.0)
    args = parser.parse_args(argv)

    params = EnvParams()
    rows = []
    for name, hw, inst in cases():
        env = EagerEnv(hw, inst, params)
        info, _, _ = run_episode(env, GreedyJITPolicy(placement_seed=0), 0)
        mg = info["metrics"]
        assert not mg["truncated"], name
        res = solve_exact(hw, inst, params, time_limit=args.time_limit)
        replay_exact(res, hw, inst, params)         # raises on any mismatch
        gap = (mg["J"] - res.j_star) / res.j_star
        rows.append({
            "instance": name, "N": inst.num_qubits, "M": inst.num_gates,
            "K": hw.num_qpus, "horizon": res.horizon,
            "J_star": res.j_star, "T_star": res.t_makespan,
            "C_comm_star": res.c_comm, "status": res.status,
            "mip_gap": res.mip_gap, "runtime_s": round(res.runtime_s, 2),
            "J_greedy": mg["J"], "greedy_gap": round(gap, 4),
        })
        print(f"{name:<18} N={inst.num_qubits:>3} M={inst.num_gates:>3} "
              f"K={hw.num_qpus} H={res.horizon:>4}  J*={res.j_star:>8.6g} "
              f"({res.status}, mip_gap={res.mip_gap:.2e}, "
              f"{res.runtime_s:6.1f}s)  J_greedy={mg['J']:>8.6g}  "
              f"gap={gap:6.1%}  replay=OK", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / "phase4_gap.parquet", index=False)
    index_path = RESULTS / "index.json"
    index = {}
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
    index.update({"phase4_gap": {
        "path": "phase4_gap.parquet",
        "time_limit_s": args.time_limit,
        "envelope": "golden micros (t_ep=2) + synthetics N<=12 M<=30 "
                    "K in {2,3} (t_ep=12, kappa=ceil(1.25N/K))",
        "replay_verified": True,
    }})
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)

    all_opt = all(r["status"] == "OPTIMAL" for r in rows)
    ok = all(r["J_star"] <= r["J_greedy"] + 1e-9 for r in rows)
    print(f"\nall solves optimal: {all_opt}; J* <= J_greedy everywhere: {ok}")
    print(f"results -> {RESULTS / 'phase4_gap.parquet'} (+ index.json)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
