#!/usr/bin/env python
"""DDQN-flat learning baseline on the SAME path-B provisioning task as EAGER
(guide §9.4): premapped AGG envs, provisioning-only, full regime grid. This
isolates the architecture — EAGER's R-GCN graph encoder + attention decoder
vs a flat fixed-size state + MLP Double-DQN — on an identical learning
problem, answering 'is the graph encoder needed?'.

The flat state (FlatFeaturizer) is fixed-size for fixed K and link count
(it uses top-k ready-gate features, not all gates), so it spans the path-B
instance-size band; only the Schedule action index varies, handled by a
max-size (M_max) padded action layout with masking (guide §9.4 "flattened,
max-size masks"). Trained with the same env-step budget as EAGER's path-B
training (fairness).

Usage (from the repo root):
    python scripts/train_pathb_ddqn.py --env-steps 400000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from eager.baselines.ddqn_flat import (
    DDQNConfig,
    FlatFeaturizer,
    QNetwork,
    ReplayBuffer,
)
from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.env.actions import ADVANCE, GenEPR, Schedule
from eager.train.pathb import (
    held_out_pathb_cases,
    premapped_env,
    sample_pathb_case,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_pathb import run_pathb_heuristic, _eager  # noqa: E402

ART = Path("artifacts") / "agents"
N_MAX, K_FIX, M_MAX, L_MAX = 30, 4, 90, 4
NEG = torch.finfo(torch.float32).min


def padded_index(action, n, m):
    """Map an env action to the fixed padded action layout
    [Map(N_MAX*K) | Schedule(M_MAX) | GenEPR(L_MAX) | ADVANCE]."""
    if isinstance(action, Schedule):
        return N_MAX * K_FIX + action.gate
    if isinstance(action, GenEPR):
        return N_MAX * K_FIX + M_MAX + action.link
    return N_MAX * K_FIX + M_MAX + L_MAX          # ADVANCE


PADDED_SIZE = N_MAX * K_FIX + M_MAX + L_MAX + 1


def padded_mask(env):
    mask = np.zeros(PADDED_SIZE, dtype=bool)
    for a in env.valid_actions():
        mask[padded_index(a, env.instance.num_qubits,
                          env.instance.num_gates)] = True
    return mask


def padded_action(idx, env):
    if idx == PADDED_SIZE - 1:
        return ADVANCE
    if idx >= N_MAX * K_FIX + M_MAX:
        return GenEPR(idx - (N_MAX * K_FIX + M_MAX))
    return Schedule(idx - N_MAX * K_FIX)


def featurize(env, featurizer):
    return featurizer(env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-cases", type=int, default=24)
    ap.add_argument("--eval-seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = DDQNConfig()

    # DDQN-flat's state dim is K-specific (3K + 4*n_links + ...), so a fixed
    # Q-net REQUIRES a fixed QPU count — we train/eval it on the K=4 grid
    # stratum only. EAGER, by contrast, handles K in {2,4} (and unseen
    # topologies, §10.3) with one policy: that config-lock IS the "why GNN"
    # message. (guide §9.4 "trained per configuration".)
    def sample_k4(r):
        return sample_pathb_case(r, qpus_choices=(4,))

    # reference featurizer dim (fixed for K=4, L<=4): build on a sample env
    ref_case = sample_k4(rng)
    ref_env = premapped_env(ref_case, 0)
    featurizer = FlatFeaturizer(ref_env)
    state_dim = featurizer.dim
    print(f"flat state dim {state_dim}, padded action size {PADDED_SIZE}")

    online = QNetwork(state_dim, PADDED_SIZE).to(device)
    target = QNetwork(state_dim, PADDED_SIZE).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(cfg.buffer_capacity, state_dim, PADDED_SIZE)

    steps = 0
    ep = 0
    losses = []
    t0 = time.perf_counter()
    while steps < args.env_steps:
        case = sample_k4(rng)
        env = premapped_env(case, int(rng.integers(0, 1_000_000)))
        done = False
        while not done and steps < args.env_steps:
            s = featurize(env, featurizer)
            mask = padded_mask(env)
            eps = max(cfg.eps_end, cfg.eps_start + (cfg.eps_end - cfg.eps_start)
                      * steps / cfg.eps_decay_steps)
            if rng.random() < eps:
                idx = int(rng.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q = online(torch.from_numpy(s).unsqueeze(0).to(device))[0]
                    q = q.masked_fill(~torch.from_numpy(mask).to(device), NEG)
                    idx = int(torch.argmax(q).item())
            action = padded_action(idx, env)
            _, r, done, info = env.step(action)
            s2 = featurize(env, featurizer)
            mask2 = padded_mask(env) if not done else np.zeros(PADDED_SIZE, bool)
            buf.push(s, idx, r, s2, done, mask2)
            steps += 1
            if buf.size >= max(cfg.train_start, cfg.batch_size):
                S, A, R, S2, D, M2 = buf.sample(cfg.batch_size, rng)
                S, A, R, S2, D, M2 = (x.to(device) for x in (S, A, R, S2, D, M2))
                qsa = online(S).gather(1, A.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q2 = online(S2).masked_fill(~M2, NEG)
                    a2 = torch.argmax(q2, 1, keepdim=True)
                    qt = target(S2).gather(1, a2).squeeze(1)
                    y = R + cfg.gamma * (1 - D) * qt
                loss = torch.nn.functional.smooth_l1_loss(qsa, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(float(loss.item()))
                if steps % cfg.target_sync_every == 0:
                    target.load_state_dict(online.state_dict())
        ep += 1
        if ep % 100 == 0:
            print(f"  ep {ep} steps {steps}/{args.env_steps} "
                  f"loss {np.mean(losses[-200:]) if losses else 0:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # eval: DDQN-flat (greedy decode) vs AGG-reactive / AGG-eager
    print("eval DDQN-flat vs AGG-reactive / AGG-eager ...")
    online.eval()

    def ddqn_J(case, seed):
        env = premapped_env(case, seed)
        done = False
        while not done:
            s = featurize(env, featurizer)
            mask = padded_mask(env)
            with torch.no_grad():
                q = online(torch.from_numpy(s).unsqueeze(0).to(device))[0]
                q = q.masked_fill(~torch.from_numpy(mask).to(device), NEG)
                idx = int(torch.argmax(q).item())
            _, _, done, info = env.step(padded_action(idx, env))
        return info["metrics"]["J"]

    # K=4 held-out stratum (filter the standard held-out set to qpus=4 so
    # DDQN-flat's fixed state dim applies; EAGER is evaluated on the same)
    cases = [c for c in held_out_pathb_cases(args.eval_cases * 3)
             if c.hardware.num_qpus == 4][:args.eval_cases]
    print(f"K=4 held-out eval cases: {len(cases)}")
    jd, jr, je = [], [], []
    for c in cases:
        for e in range(args.eval_seeds):
            jd.append(ddqn_J(c, e))
            jr.append(run_pathb_heuristic(
                lambda cc: GreedyJITPolicy(
                    placement_fn=lambda i, h, p=list(cc.placement): p), c, e))
            je.append(run_pathb_heuristic(lambda cc: _eager(cc), c, e))
    jr = [m["J"] for m in jr]; je = [m["J"] for m in je]
    jd, jr, je = np.array(jd), np.array(jr), np.array(je)
    pr = stats.wilcoxon(jd, jr, alternative="less").pvalue if not np.allclose(jd, jr) else 1.0
    print(f"DDQN-flat vs AGG-reactive: ratio={jd.mean()/jr.mean():.4f} "
          f"won={(jd<jr).sum()}/{len(jd)} p={pr:.2e}")
    print(f"DDQN-flat vs AGG-eager:    ratio={jd.mean()/je.mean():.4f}")
    print(f"DDQN-flat mean J {jd.mean():.2f}  (AGG-reactive {jr.mean():.2f}, "
          f"AGG-eager {je.mean():.2f})")

    ART.mkdir(parents=True, exist_ok=True)
    torch.save({"online": online.state_dict()}, ART / f"ddqn_pathb_seed{args.seed}.pt")
    with open(ART / f"ddqn_pathb_seed{args.seed}.json", "w", encoding="utf-8") as fh:
        json.dump({"env_steps": args.env_steps,
                   "ddqn_mean_J": float(jd.mean()),
                   "agg_reactive_mean_J": float(jr.mean()),
                   "agg_eager_mean_J": float(je.mean()),
                   "vs_reactive_ratio": float(jd.mean()/jr.mean()),
                   "vs_reactive_p": float(pr)}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
