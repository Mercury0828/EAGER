"""CRN-paired evaluation (guide §10.4): the agent (greedy decoding) and
GreedyJIT run the SAME (case, env seed) pairs — the counter-based CRN
engine guarantees identical generation luck wherever tasking coincides —
and the comparison is a paired Wilcoxon signed-rank over per-pair J."""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from ..baselines.greedy_jit import GreedyJITPolicy
from ..env.env import EagerEnv
from ..model.policy import EagerPolicy, act_greedy
from .distribution import Case


def run_agent_episode(policy: EagerPolicy, env: EagerEnv, env_seed: int,
                      device, max_micro: int = 2_000_000) -> dict:
    policy.eval()
    env.reset(env_seed)
    done = False
    steps = 0
    while not done:
        action = act_greedy(policy, env, device)
        _, _, done, info = env.step(action)
        steps += 1
        if steps > max_micro:
            raise RuntimeError("micro-step guard tripped in agent episode")
    return info["metrics"]


def run_greedy_episode(env: EagerEnv, env_seed: int) -> dict:
    env.reset(env_seed)
    policy = GreedyJITPolicy(placement_seed=0)
    done = False
    while not done:
        _, _, done, info = env.step(policy(env))
    return info["metrics"]


def run_agent_episodes_batched(policy: EagerPolicy, pairs, device,
                               max_micro: int = 2_000_000) -> list[dict]:
    """Greedy-decode many episodes concurrently (stragglers shrink the
    batch); pairs = [(env, env_seed), ...]."""
    import torch as _torch
    from ..model.encoder import BatchedGraphs
    from ..model.graph import build_graph
    from ..model.policy import build_action_set

    for env, seed in pairs:
        env.reset(seed)
    n = len(pairs)
    metrics: list[dict | None] = [None] * n
    active = list(range(n))
    steps = 0
    policy.eval()
    with _torch.no_grad():
        while active:
            snaps = [build_graph(pairs[i][0]) for i in active]
            asets = [build_action_set(pairs[i][0], s)
                     for i, s in zip(active, snaps)]
            out = policy(BatchedGraphs(snaps, device), asets)
            pos = out.greedy()
            nxt = []
            for j, i in enumerate(active):
                action = asets[j].actions[int(pos[j])]
                _, _, done, info = pairs[i][0].step(action)
                if done:
                    metrics[i] = info["metrics"]
                else:
                    nxt.append(i)
            active = nxt
            steps += 1
            if steps > max_micro:
                raise RuntimeError("batched eval micro-step guard tripped")
    return metrics


def paired_eval(policy: EagerPolicy, cases: list[Case], env_seeds: list[int],
                device, log=None) -> dict:
    """Returns per-pair J arrays + summary + paired Wilcoxon (agent < greedy).
    Agent episodes run batched (greedy decode); GreedyJIT runs serially."""
    pairs = [(EagerEnv(case.hardware, case.instance), e)
             for case in cases for e in env_seeds]
    agent_metrics = run_agent_episodes_batched(policy, pairs, device)
    j_agent, j_greedy, trunc_agent = [], [], 0
    idx = 0
    for case in cases:
        for e in env_seeds:
            ma = agent_metrics[idx]
            idx += 1
            env = EagerEnv(case.hardware, case.instance)
            mg = run_greedy_episode(env, e)
            j_agent.append(ma["J"])
            j_greedy.append(mg["J"])
            trunc_agent += int(ma["truncated"])
            if log:
                log(f"    {case.label} seed={e}: agent J={ma['J']:.1f} "
                    f"greedy J={mg['J']:.1f}")
    ja, jg = np.array(j_agent), np.array(j_greedy)
    diff = ja - jg
    if np.allclose(diff, 0):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(ja, jg, alternative="less").pvalue)
    return {
        "n_pairs": len(ja),
        "mean_J_agent": float(ja.mean()),
        "mean_J_greedy": float(jg.mean()),
        "ratio": float(ja.mean() / jg.mean()),
        "pairs_won": int((ja < jg).sum()),
        "pairs_tied": int((ja == jg).sum()),
        "wilcoxon_p_less": p_value,
        "agent_truncations": trunc_agent,
        "j_agent": ja.tolist(),
        "j_greedy": jg.tolist(),
    }
