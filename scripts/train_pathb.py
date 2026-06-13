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


def pathb_ppo_refine(policy, device, seed, iters, eval_cb, log=print):
    """Provisioning-only PPO refinement (D78): rollouts on pre-mapped AGG
    envs; episode-constant CRN-paired advantage (J_RegimeProvision - J_agent)
    on the SAME (case, env seed) so generation luck cancels and the gradient
    rewards beating the regime-adaptive expert (esp. learning to hold back in
    the waste regime, which the IL clone does imperfectly). Reuses the D58/D63
    machinery. Value-warmup first; best checkpoint by validation."""
    import torch as T
    from eager.model.encoder import BatchedGraphs
    from eager.train.ppo import PPOConfig, RunningStd, ppo_update

    cfg = PPOConfig(paired_advantage=True, clip=0.15, lr=8e-5, lr_min=3e-5,
                    ent_start=0.005, ent_end=0.001, value_warmup_iters=3,
                    total_iters=iters, episodes_per_iter=32)
    T.manual_seed(seed)
    gen = T.Generator(device=device); gen.manual_seed(seed)
    gen_cpu = T.Generator(); gen_cpu.manual_seed(seed + 1)
    rng = np.random.default_rng(seed * 7 + 1)
    opt = T.optim.Adam(policy.parameters(), lr=cfg.lr)
    diff_std = RunningStd()
    greedy_cache: dict = {}

    def collect(value_only=False):
        n = cfg.episodes_per_iter
        cases = [sample_pathb_case(rng) for _ in range(n)]
        seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n)]
        envs = [premapped_env(c, s) for c, s in zip(cases, seeds)]
        steps = [[] for _ in range(n)]
        js = [None] * n
        active = list(range(n))
        policy.eval()
        with T.no_grad():
            while active:
                snaps = [build_graph(envs[i]) for i in active]
                asets = [build_action_set(envs[i], s) for i, s in zip(active, snaps)]
                out = policy(BatchedGraphs(snaps, device), asets)
                pos = out.sample(generator=gen)
                logp = out.log_prob_of(pos).cpu()
                nxt = []
                for j, i in enumerate(active):
                    p = int(pos[j])
                    steps[i].append((snaps[j], asets[j], p, float(logp[j])))
                    _, _, done, info = envs[i].step(asets[j].actions[p])
                    if done:
                        js[i] = info["metrics"]
                    else:
                        nxt.append(i)
                active = nxt
        policy.train()
        flat = {"snaps": [], "asets": [], "pos": [], "logp": [], "adv": []}
        for i in range(n):
            key = (cases[i].label, seeds[i])
            if key not in greedy_cache:
                env = premapped_env(cases[i], seeds[i])
                exp = GreedyRegimeProvisionPolicy(placement=list(cases[i].placement))
                done = False
                while not done:
                    _, _, done, gi = env.step(exp(env))
                greedy_cache[key] = gi["metrics"]["J"]
            diff = greedy_cache[key] - js[i]["J"]
            diff_std.update(diff)
            adv = diff / diff_std.std()
            for (snap, aset, p, lp) in steps[i]:
                flat["snaps"].append(snap); flat["asets"].append(aset)
                flat["pos"].append(p); flat["logp"].append(lp); flat["adv"].append(adv)
        flat["pos"] = T.tensor(flat["pos"]); flat["logp"] = T.tensor(flat["logp"])
        flat["adv"] = T.tensor(flat["adv"], dtype=T.float32)
        flat["ret"] = flat["adv"].clone()
        return flat

    # value warmup (policy frozen)
    frozen = [p for n_, p in policy.named_parameters() if not n_.startswith("v_mlp")]
    for p in frozen:
        p.requires_grad_(False)
    wopt = T.optim.Adam(policy.v_mlp.parameters(), lr=1e-3)
    for _ in range(cfg.value_warmup_iters):
        ppo_update(policy, wopt, collect(), cfg, device, 0.0, gen_cpu, value_only=True)
    for p in frozen:
        p.requires_grad_(True)

    sched = T.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=cfg.lr_min)
    best = {"ratio": float("inf"), "state": None}
    for it in range(iters):
        frac = it / max(1, iters - 1)
        ent = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
        stats = ppo_update(policy, opt, collect(), cfg, device, ent, gen_cpu)
        sched.step()
        if (it + 1) % 10 == 0:
            r = eval_cb()
            if r < best["ratio"]:
                best.update(ratio=r, state={k: v.detach().cpu().clone()
                                            for k, v in policy.state_dict().items()})
            log(f"  ppo it {it+1}: kl={stats['approx_kl']:.4f} val_ratio_vs_react={r:.4f}")
    if best["state"] is not None:
        policy.load_state_dict(best["state"])
    return best["ratio"]


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
    ap.add_argument("--ppo-iters", type=int, default=0,
                    help="provisioning-only PPO refinement iters after IL (D78)")
    ap.add_argument("--init-ckpt", default=None,
                    help="skip IL, load this checkpoint and PPO-refine")
    ap.add_argument("--flat-encoder", action="store_true",
                    help="ablation: MLP encoder (no message passing) instead "
                         "of R-GCN — the clean graph-vs-flat isolation (D83)")
    ap.add_argument("--tag", default=None,
                    help="checkpoint/json filename tag (default seed-based)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    tag = args.tag or f"seed{args.seed}"
    print(f"device: {device}; encoder: "
          f"{'MLP-flat (ablation)' if args.flat_encoder else 'R-GCN'}; tag={tag}")

    if args.flat_encoder:
        from eager.model.encoder import MLPEncoder
        policy = EagerPolicy(encoder=MLPEncoder())
    else:
        policy = EagerPolicy()
    if args.init_ckpt:
        policy.load_state_dict(torch.load(args.init_ckpt, map_location="cpu",
                                          weights_only=False)["state_dict"])
        policy.to(device)
        stats_ds = {"note": f"loaded {args.init_ckpt}"}
        result = {"best_val_top1": float("nan")}
        print(f"loaded {args.init_ckpt}; skipping IL")
    else:
        print("collecting path-B expert dataset (RegimeProvision on AGG) ...")
        episodes, stats_ds = collect_pathb_dataset(args.transitions, args.seed)
        print(f"dataset: {stats_ds}")
        train_data, val_data = split_episodes(episodes, 0.1, args.seed + 1)
        print(f"split: {len(train_data)} train / {len(val_data)} val")
        result = train_il(policy, train_data, val_data, device,
                          max_epochs=args.max_epochs, seed=args.seed,
                          batch_size=args.batch_size, patience=args.patience)
        print(f"IL best val top-1: {result['best_val_top1']:.4f}")

    cases = held_out_pathb_cases(args.eval_cases)
    eval_seeds = list(range(args.eval_seeds))

    if args.ppo_iters > 0:
        print(f"provisioning-only PPO refinement ({args.ppo_iters} iters) ...")
        val_cases = held_out_pathb_cases(args.eval_cases, seed=778)

        def val_ratio():
            ja, jr = [], []
            for c in val_cases:
                for e in range(4):
                    ja.append(run_pathb_agent(policy, c, e, device)["J"])
                    jr.append(run_pathb_heuristic(
                        lambda cc: GreedyJITPolicy(
                            placement_fn=lambda i, h, p=list(cc.placement): p),
                        c, e)["J"])
            return float(np.mean(ja) / np.mean(jr))
        pathb_ppo_refine(policy, device, args.seed, args.ppo_iters, val_ratio)

    print("held-out eval (EAGER-on-AGG vs AGG-reactive / AGG-eager) ...")
    ev = evaluate(policy, cases, eval_seeds, device)

    ART.mkdir(parents=True, exist_ok=True)
    ckpt = ART / f"pathb_{tag}.pt"
    torch.save({"state_dict": policy.state_dict()}, ckpt)
    with open(ART / f"pathb_{tag}.json", "w", encoding="utf-8") as fh:
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
