#!/usr/bin/env python
"""Path B (D76): train a provisioning-only EAGER on AGG placement.

IL-clones GreedyRegimeProvision (eager/reactive regime switch) on
pre-mapped AGG instances across the full regime grid, then evaluates the
learned policy vs AGG-reactive (and AGG-eager) CRN-paired on the held-out
set, stratified by regime. The win over AGG-reactive is attributable purely
to learned proactive provisioning (placement+aggregation matched to AGG).

Usage (from the repo root):
    python scripts/train_pathb.py --transitions 200000 --seed 0 --max-epochs 30
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from eager.baselines.greedy_jit import (
    GreedyEagerPolicy,
    GreedyJITPolicy,
    GreedyRegimeProvisionPolicy,
)
from eager.baselines.traces import run_episode
from eager.env import EagerEnv
from eager.model.policy import EagerPolicy, act_greedy, build_action_set
from eager.model.graph import build_graph
from eager.train.il import Transition, split_episodes, train_il
from eager.train.pathb import (
    held_out_pathb_cases,
    premapped_env,
    sample_pathb_case,
)

ART = Path("artifacts") / "agents"


def collect_pathb_dataset(min_transitions: int, seed: int, log_every=100):
    rng = np.random.default_rng(seed)
    episodes, n_tr, ep = [], 0, 0
    t0 = time.perf_counter()
    while n_tr < min_transitions:
        case = sample_pathb_case(rng)
        env = premapped_env(case, seed=ep)            # placement pre-applied
        expert = GreedyRegimeProvisionPolicy(placement=list(case.placement))
        episode, done = [], False
        while not done:
            snap = build_graph(env)
            aset = build_action_set(env, snap)
            action = expert(env)
            pos = aset.actions.index(action)
            episode.append(Transition(snap=snap, aset=aset, expert_pos=pos))
            _, _, done, info = env.step(action)
        assert not info["metrics"]["truncated"], case.label
        episodes.append(episode)
        n_tr += len(episode)
        ep += 1
        if ep % log_every == 0:
            print(f"  {ep} episodes / {n_tr} transitions "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return episodes, {"episodes": ep, "transitions": n_tr,
                      "collect_seconds": round(time.perf_counter() - t0, 1)}


def run_pathb_agent(policy, case, seed, device, max_micro=2_000_000):
    env = premapped_env(case, seed)
    done, steps = False, 0
    while not done:
        action = act_greedy(policy, env, device)
        _, _, done, info = env.step(action)
        steps += 1
        if steps > max_micro:
            raise RuntimeError("guard tripped")
    return info["metrics"]


def run_pathb_heuristic(factory, case, seed):
    env = premapped_env(case, seed)
    policy = factory(case)
    done = False
    while not done:
        _, _, done, info = env.step(policy(env))
    return info["metrics"]


def paired(a, b):
    a, b = np.array(a), np.array(b)
    p = (stats.wilcoxon(a, b, alternative="less").pvalue
         if not np.allclose(a, b) else 1.0)
    return a.mean() / b.mean(), int((a < b).sum()), p


def evaluate(policy, cases, env_seeds, device, log=print):
    strata = collections.defaultdict(lambda: collections.defaultdict(list))
    for case in cases:
        waste = GreedyRegimeProvisionPolicy.is_waste_regime(case.hardware)
        reg = "waste" if waste else "normal"
        for e in env_seeds:
            ma = run_pathb_agent(policy, case, e, device)
            mr = run_pathb_heuristic(
                lambda c: GreedyJITPolicy(
                    placement_fn=lambda i, h, p=list(c.placement): p), case, e)
            me = run_pathb_heuristic(
                lambda c: _eager(c), case, e)
            for grp in ("full", reg):
                strata[grp]["agent"].append(ma["J"])
                strata[grp]["reactive"].append(mr["J"])
                strata[grp]["eager"].append(me["J"])
            strata["full"]["trunc"].append(int(ma["truncated"]))
    out = {}
    for grp, d in strata.items():
        r_react = paired(d["agent"], d["reactive"])
        r_eager = paired(d["agent"], d["eager"])
        out[grp] = {"n": len(d["agent"]),
                    "vs_AGGreactive_ratio": r_react[0], "vs_react_p": r_react[2],
                    "vs_react_wins": r_react[1],
                    "vs_AGGeager_ratio": r_eager[0], "vs_eager_p": r_eager[2],
                    "truncations": int(np.sum(d.get("trunc", [0])))}
        log(f"  [{grp:<7}] n={out[grp]['n']:>4} vs AGG-reactive "
            f"ratio={r_react[0]:.4f} won={r_react[1]}/{out[grp]['n']} "
            f"p={r_react[2]:.2e} | vs AGG-eager ratio={r_eager[0]:.4f} "
            f"p={r_eager[2]:.2e}")
    return out


def _eager(case):
    po = GreedyEagerPolicy()
    po._placement = list(case.placement)
    return po


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transitions", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--eval-cases", type=int, default=20)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    print(f"device: {device}")

    print("collecting path-B expert dataset (RegimeProvision on AGG) ...")
    episodes, stats_ds = collect_pathb_dataset(args.transitions, args.seed)
    print(f"dataset: {stats_ds}")
    train_data, val_data = split_episodes(episodes, 0.1, args.seed + 1)
    print(f"split: {len(train_data)} train / {len(val_data)} val")

    policy = EagerPolicy()
    result = train_il(policy, train_data, val_data, device,
                      max_epochs=args.max_epochs, seed=args.seed,
                      batch_size=args.batch_size, patience=args.patience)
    print(f"IL best val top-1: {result['best_val_top1']:.4f}")

    print("held-out eval (EAGER-on-AGG vs AGG-reactive / AGG-eager) ...")
    cases = held_out_pathb_cases(args.eval_cases)
    ev = evaluate(policy, cases, list(range(args.eval_seeds)), device)

    ART.mkdir(parents=True, exist_ok=True)
    ckpt = ART / f"pathb_seed{args.seed}.pt"
    torch.save({"state_dict": policy.state_dict()}, ckpt)
    with open(ART / f"pathb_seed{args.seed}.json", "w", encoding="utf-8") as fh:
        json.dump({"dataset": stats_ds, "il_val_top1": result["best_val_top1"],
                   "eval": ev}, fh, indent=2)
    print(f"checkpoint -> {ckpt}")
    full = ev["full"]
    ok = full["vs_AGGreactive_ratio"] < 1.0 and full["vs_react_p"] < 0.05
    print(f"VERDICT: EAGER-on-AGG vs AGG-reactive full ratio "
          f"{full['vs_AGGreactive_ratio']:.4f} p={full['vs_react_p']:.2e} "
          f"-> {'BEATS AGG' if ok else 'not significant'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
