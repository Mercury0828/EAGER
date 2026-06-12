#!/usr/bin/env python
"""Regime-stratified CRN-paired evaluation of an agent checkpoint vs
GreedyJIT on the D61 held-out protocol (20 cases x 20 env seeds = 400
pairs), reporting the full-distribution verdict AND the per-regime strata:

  provisioning-bound: p = 0.2 OR W = 1 (the 4/6 of the hardware grid where
  proactive provisioning has measured headroom, D61/D64)
  comfortable:        p < 0.2 AND W = 2

Used for the Phase 5 acceptance evidence and (if the full-distribution gate
stays out of reach) the D38-style regime-characterized escalation.

Usage (from the repo root):
    python scripts/eval_agent_regimes.py --ckpt artifacts/agents/ppo_seed1.pt
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from scipy import stats

from eager.env import EagerEnv
from eager.model.policy import EagerPolicy
from eager.train.distribution import held_out_cases
from eager.train.evaluate import run_agent_episodes_batched, run_greedy_episode


def is_provisioning_bound(case) -> bool:
    lc = case.hardware.links[0]
    return lc.p >= 0.2 or lc.W == 1


def stratum_stats(pairs: list[tuple[float, float]]) -> dict:
    ja = np.array([a for a, _ in pairs])
    jg = np.array([g for _, g in pairs])
    diff = ja - jg
    if len(ja) == 0 or np.allclose(diff, 0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(ja, jg, alternative="less").pvalue)
    return {
        "n_pairs": len(ja),
        "mean_J_agent": float(ja.mean()) if len(ja) else None,
        "mean_J_greedy": float(jg.mean()) if len(ja) else None,
        "ratio": float(ja.mean() / jg.mean()) if len(ja) else None,
        "wins": int((ja < jg).sum()),
        "ties": int((ja == jg).sum()),
        "wilcoxon_p_less": p_value,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--env-seeds", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)

    policy = EagerPolicy()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(ck["state_dict"])
    policy.to(device)

    cases = held_out_cases(args.cases)
    seeds = list(range(args.env_seeds))
    pairs_env = [(EagerEnv(c.hardware, c.instance), e)
                 for c in cases for e in seeds]
    agent_metrics = run_agent_episodes_batched(policy, pairs_env, device)

    strata = {"full": [], "provisioning_bound": [], "comfortable": []}
    truncs = 0
    idx = 0
    for c in cases:
        bound = is_provisioning_bound(c)
        for e in seeds:
            ma = agent_metrics[idx]
            idx += 1
            env = EagerEnv(c.hardware, c.instance)
            mg = run_greedy_episode(env, e)
            truncs += int(ma["truncated"])
            pair = (ma["J"], mg["J"])
            strata["full"].append(pair)
            strata["provisioning_bound" if bound else "comfortable"].append(pair)

    report = {"ckpt": args.ckpt, "agent_truncations": truncs,
              "strata": {k: stratum_stats(v) for k, v in strata.items()}}
    for name, s in report["strata"].items():
        print(f"{name:<20} n={s['n_pairs']:>3} ratio={s['ratio'] and round(s['ratio'], 4)} "
              f"wins={s['wins']}/{s['n_pairs']} ties={s['ties']} "
              f"p={s['wilcoxon_p_less']:.3e}")
    print(f"agent truncations: {truncs}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
