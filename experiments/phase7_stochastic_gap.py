#!/usr/bin/env python
"""Phase 7 STOCHASTIC optimal-gap (T4 stochastic extension, D84).

The deterministic MILP (Phase 4) cannot measure proactive provisioning — at
p=1 provisioning is trivial (D82). This computes the CLAIRVOYANT stochastic
optimum (eager.exact.stochastic_opt): for each CRN seed the env is
deterministic, so the min J on the seeded env (exhaustive B&B) is the
perfect-information optimum; averaged over seeds it lower-bounds the expected
cost of ANY non-anticipative policy. On tiny forced-remote instances with
hideable latency it shows the provisioning strategies' gap to optimum:
reactive (GreedyJIT) leaves a gap, adaptive/proactive (the strategy EAGER
imitates) ~= optimum.

Methods: clairvoyant optimum; GreedyJIT (reactive); GreedyEager (always-on);
GreedyRegimeProvision (regime-adaptive = EAGER's expert); EAGER (path-B —
OOD here, N far below its [10,30] training band; reported with that caveat).
CRN-paired (every method + the optimum see the same seed). Writes
results/phase7_stochastic_gap.parquet + index.json.

Usage (from the repo root):
    python experiments/phase7_stochastic_gap.py --seeds 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eager.baselines.agg import transform_instance
from eager.baselines.greedy_jit import (
    GreedyEagerPolicy, GreedyJITPolicy, GreedyRegimeProvisionPolicy,
    compute_placement,
)
from eager.baselines.traces import run_episode
from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import EagerEnv
from eager.env.env import EnvParams
from eager.exact.stochastic_opt import NodeCapExceeded, clairvoyant_optimum
from eager.train.pathb import PathBCase

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

# tiny forced-remote instances; kappa tight so the cross-QPU gate(s) are remote
# and earlier local work gives latency for proactive provisioning to hide.
INSTANCES = [
    ("q2m1", 2, 1, ((0, 1),)),                      # minimal: no hideable lat.
    ("q3m2", 3, 2, ((0, 1), (1, 2))),               # 1 remote, small hideable
    ("q4m3", 4, 2, ((0, 1), (2, 3), (1, 2))),       # 1 remote, hideable
]
NODE_CAP = 3_000_000


def hw(kappa, p=0.5, w=1, b=2, tcut=20, tep=2):
    return load_hardware_config({
        "name": f"k2_kap{kappa}_p{p}", "qpus": 2, "topology": "line",
        "kappa": kappa, "mode": "stochastic", "t_ep": tep,
        "link_defaults": {"p": p, "W": w, "B": b, "T_cut": tcut, "w": 1.0}})


def heuristic_J(policy, hardware, inst, seed):
    env = EagerEnv(hardware, inst, EnvParams())
    return run_episode(env, policy, seed)[0]["metrics"]["J"]


def eager_J(eager, hardware, inst, seed, device):
    from train_pathb import run_pathb_agent
    placement = tuple(compute_placement(inst, hardware, seed=0))
    agg_inst, _ = transform_instance(inst, list(placement))
    case = PathBCase(hardware=hardware, agg_instance=agg_inst,
                     placement=placement, label=f"{inst.name}")
    return run_pathb_agent(eager, case, seed, device)["J"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--eager-ckpt", default="artifacts/agents/eager_final.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "scripts"))
    import torch
    from eager.model.policy import EagerPolicy
    device = torch.device(args.device)
    eager = EagerPolicy()
    eager.load_state_dict(torch.load(args.eager_ckpt, map_location="cpu",
                                     weights_only=False)["state_dict"])
    eager.to(device).eval()

    rows = []
    t0 = time.perf_counter()
    for name, n, kappa, gates in INSTANCES:
        inst = instance_from_gates(name, n, gates)
        h = hw(kappa)
        skipped = 0
        for seed in range(args.seeds):
            jit = heuristic_J(GreedyJITPolicy(placement_seed=0), h, inst, seed)
            eag = heuristic_J(GreedyEagerPolicy(), h, inst, seed)
            reg = heuristic_J(
                GreedyRegimeProvisionPolicy(
                    placement=list(compute_placement(inst, h, seed=0))),
                h, inst, seed)
            ea = eager_J(eager, h, inst, seed, device)
            inc = min(jit, eag, reg, ea)
            try:
                opt = clairvoyant_optimum(h, inst, seed, EnvParams(),
                                          incumbent=inc + 1e-9,
                                          node_cap=NODE_CAP)
            except NodeCapExceeded:
                skipped += 1
                print(f"  {name} seed={seed} SKIPPED (node cap)", flush=True)
                continue
            rows.append({"instance": name, "N": n, "M": len(gates),
                         "seed": seed, "opt_J": opt["J"], "opt_nodes": opt["nodes"],
                         "jit_J": jit, "eager_always_J": eag,
                         "regime_J": reg, "EAGER_J": ea})
        n_ok = sum(1 for r in rows if r["instance"] == name)
        print(f"[{name}] solved {n_ok}/{args.seeds} (skipped {skipped}) "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / "phase7_stochastic_gap.parquet", index=False)

    print("\n=== gap to clairvoyant stochastic optimum (lower = closer) ===")
    summary = {}
    for name in df.instance.unique():
        d = df[df.instance == name]
        row = {"n": int(len(d)), "opt": float(d.opt_J.mean())}
        for col, lab in [("jit_J", "GreedyJIT(reactive)"),
                         ("eager_always_J", "GreedyEager(always-on)"),
                         ("regime_J", "RegimeProvision(adaptive)"),
                         ("EAGER_J", "EAGER(path-B,OOD)")]:
            gap = (d[col].mean() - d.opt_J.mean()) / d.opt_J.mean()
            row[lab] = {"meanJ": float(d[col].mean()), "gap": float(gap)}
        summary[name] = row
        print(f"\n[{name}] n={row['n']} optimum meanJ={row['opt']:.3f}")
        for lab in ("GreedyJIT(reactive)", "GreedyEager(always-on)",
                    "RegimeProvision(adaptive)", "EAGER(path-B,OOD)"):
            print(f"    {lab:<28} meanJ={row[lab]['meanJ']:7.3f}  "
                  f"gap={row[lab]['gap']:+7.1%}")

    index_path = RESULTS / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) \
        if index_path.exists() else {}
    index["phase7_stochastic_gap"] = {
        "path": "phase7_stochastic_gap.parquet",
        "instances": [i[0] for i in INSTANCES], "seeds": args.seeds,
        "anchor": "clairvoyant perfect-information optimum (lower bound on any "
                  "non-anticipative policy; B&B per CRN seed)",
        "regime": "p=0.5 W=1 B=2 t_ep=2 T_cut=20", "summary": summary,
        "crn_paired": True}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True),
                          encoding="utf-8")
    print(f"\nwrote {len(rows)} rows -> {RESULTS/'phase7_stochastic_gap.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
