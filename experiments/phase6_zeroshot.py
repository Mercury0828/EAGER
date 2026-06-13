#!/usr/bin/env python
"""Phase 6 zero-shot transfer (guide §10.3): the generalization claim — the
LOCKED path-B EAGER policy (trained on synthetic N in [10,30], K in {2,4})
is evaluated WITHOUT any retraining on:
  (1) real QASMBench circuit structures (vs synthetic random DAGs),
  (2) larger N (up to N=120, far beyond the training band),
  (3) an unseen topology / qubit count (K=8, 2x4 grid — beyond K in {2,4}).
A graph policy can even REPRESENT these (varying N/K/topology); a flat-state
DQN cannot (its state/action dims are config-locked, §9.4) — so this
experiment is structurally impossible for the DDQN baseline, which is itself
part of the "why GNN" message.

Same provisioning-only setup as the main result: AGG placement + aggregation
(matched static base), EAGER learns only proactive provisioning, compared to
AGG-reactive (= EAGER-NoProactive) and AGG-eager (always-on), CRN-paired so
generation luck cancels. Reports J(EAGER)/J(AGG-reactive) per circuit x K and
aggregate Wilcoxon. Writes results/phase6_zeroshot.parquet + index.json.

Usage (from the repo root):
    python experiments/phase6_zeroshot.py --eager-ckpt artifacts/agents/eager_final.pt
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
import torch
from scipy import stats

from eager.baselines.agg import transform_instance
from eager.baselines.greedy_jit import GreedyJITPolicy, compute_placement
from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.model.policy import EagerPolicy
from eager.train.pathb import PathBCase

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO / "scripts"))
from train_pathb import _eager, run_pathb_agent, run_pathb_heuristic  # noqa: E402

# real circuit structures spanning N well past the [10,30] training band,
# diverse families (arith / ml / entangling / oracle / variational / physics),
# tractable gate counts (the M~3-4k qft/multiplier are excluded for runtime)
QASM = ["adder_n28", "dnn_n51", "cat_n65", "bv_n70", "ghz_n78",
        "ghz_fanout_n78", "qugan_n71", "ising_n98"]
# two representative network regimes (normal + provisioning-bound)
REGIMES = [("normal", 0.12, 2, 20), ("prov_bound", 0.08, 1, 50)]


def hardware(qpus: int, n_qubits: int, p: float, w_ch: int, t_cut: int):
    kappa = math.ceil(1.25 * n_qubits / qpus)
    if qpus == 2:
        topo = {"qpus": 2, "topology": "line"}
    elif qpus == 4:
        topo = {"qpus": 4, "topology": "grid", "grid_dims": [2, 2]}
    elif qpus == 8:
        topo = {"qpus": 8, "topology": "grid", "grid_dims": [2, 4]}
    else:
        raise ValueError(f"unsupported qpus={qpus}")
    return load_hardware_config({
        "name": f"k{qpus}_p{p}_w{w_ch}_c{t_cut}", **topo, "kappa": kappa,
        "mode": "stochastic", "t_ep": 12,
        "link_defaults": {"p": p, "W": w_ch, "B": 8, "T_cut": t_cut, "w": 1.0}})


def make_case(inst, qpus, p, w_ch, t_cut) -> PathBCase:
    ref = hardware(qpus, inst.num_qubits, 0.12, 2, 20)        # capacity ref
    placement = tuple(compute_placement(inst, ref, seed=0))
    agg_inst, _ = transform_instance(inst, list(placement))
    hw = hardware(qpus, inst.num_qubits, p, w_ch, t_cut)
    return PathBCase(hardware=hw, agg_instance=agg_inst, placement=placement,
                     label=f"{inst.name}@{hw.name}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eager-ckpt", default="artifacts/agents/eager_final.pt")
    ap.add_argument("--qpus", default="4,8",
                    help="comma-separated QPU counts (4 = seen, 8 = unseen)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--circuits", default=None,
                    help="comma-separated subset of QASM circuits (default all)")
    ap.add_argument("--out", default="phase6_zeroshot.parquet")
    ap.add_argument("--index-key", default="phase6_zeroshot")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    qpu_list = [int(x) for x in args.qpus.split(",")]
    print(f"device: {device}; QPU counts: {qpu_list}")

    eager = EagerPolicy()
    ck = torch.load(args.eager_ckpt, map_location="cpu", weights_only=False)
    eager.load_state_dict(ck["state_dict"])
    eager.to(device).eval()
    print(f"EAGER loaded from {args.eager_ckpt}")

    circuits = args.circuits.split(",") if args.circuits else QASM
    rows = []
    t0 = time.perf_counter()
    for cname in circuits:
        path = REPO / "configs" / "circuits" / "qasmbench" / f"{cname}.yaml"
        try:
            inst = build_instance(load_circuit_config(path), seed=0)
        except Exception as exc:                              # noqa: BLE001
            print(f"  skip {cname}: {exc}")
            continue
        for qpus in qpu_list:
            for rlabel, p, w_ch, t_cut in REGIMES:
                case = make_case(inst, qpus, p, w_ch, t_cut)
                for e in range(args.seeds):
                    ma = run_pathb_agent(eager, case, e, device)
                    mr = run_pathb_heuristic(
                        lambda c: GreedyJITPolicy(
                            placement_fn=lambda i, h, pl=list(c.placement): pl),
                        case, e)
                    me = run_pathb_heuristic(lambda c: _eager(c), case, e)
                    rows.append({
                        "circuit": cname, "N": inst.num_qubits, "qpus": qpus,
                        "regime": rlabel, "seed": e,
                        "eager_J": ma["J"], "reactive_J": mr["J"],
                        "eager_full_J": me["J"],
                        "eager_trunc": int(ma["truncated"]),
                        "reactive_trunc": int(mr["truncated"]),
                        "seen_K": qpus in (2, 4)})
                seen = "SEEN" if qpus in (2, 4) else "UNSEEN-K"
                sub = [r for r in rows if r["circuit"] == cname
                       and r["qpus"] == qpus and r["regime"] == rlabel]
                rr = np.mean([r["eager_J"] for r in sub]) / np.mean(
                    [r["reactive_J"] for r in sub])
                print(f"  {cname:<16} N={inst.num_qubits:>3} K={qpus} "
                      f"[{seen:<8}] {rlabel:<10} EAGER/AGG-react={rr:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_parquet(RESULTS / args.out, index=False)

    # aggregate report (CRN-paired Wilcoxon over all rows, and by seen/unseen K)
    print("\n=== zero-shot transfer summary (EAGER vs AGG-reactive) ===")
    summary = {}
    for label, mask in [("ALL", df.index == df.index),
                        ("seen-K (4)", df.seen_K),
                        ("unseen-K (8)", ~df.seen_K)]:
        d = df[mask]
        if len(d) == 0:
            continue
        ratio = d.eager_J.mean() / d.reactive_J.mean()
        won = int((d.eager_J.values < d.reactive_J.values).sum())
        p = (stats.wilcoxon(d.eager_J, d.reactive_J, alternative="less").pvalue
             if not np.allclose(d.eager_J, d.reactive_J) else 1.0)
        summary[label] = {"n": int(len(d)), "ratio": float(ratio),
                          "won": won, "p": float(p)}
        print(f"  [{label:<14}] n={len(d):>4} EAGER/AGG-react={ratio:.4f} "
              f"won={won}/{len(d)} p={p:.2e}")

    index_path = RESULTS / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) \
        if index_path.exists() else {}
    index[args.index_key] = {
        "path": args.out,
        "circuits": QASM, "qpus": qpu_list, "regimes": [r[0] for r in REGIMES],
        "seeds": args.seeds, "ckpt": args.eager_ckpt,
        "summary": summary, "crn_paired": True,
        "note": "locked path-B EAGER, no retraining; K=8 is an unseen topology"}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True),
                          encoding="utf-8")
    print(f"\nwrote {len(rows)} rows -> {RESULTS / args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
