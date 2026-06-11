"""PPO fine-tuning from the IL initialization (guide §8.2): custom
CleanRL-style single-file loop adapted to graph batches and variable masked
action sets. GAE lambda=0.95, gamma=0.995, clip 0.2, value coef 0.5, entropy
coef 0.01 -> 0.001 (linear), lr 3e-4 cosine decay, 4 epochs/iter, minibatch
1024 transitions, 16 parallel CPU envs x 512-step rollouts, advantage
normalization, grad-norm clip 0.5, KL early stop (target 0.02).

Reward normalization (documented per the guide): per-env discounted return
accumulator R_t = gamma*R_{t-1} + r_t feeds a running variance; rewards are
scaled r / sqrt(var + 1e-8) with NO mean subtraction (CleanRL
NormalizeReward scheme), so reward signs are preserved [D51]. Episode ends
(incl. T_budget truncation, whose penalty is already in the reward) are
treated as terminal for bootstrapping.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..env.env import EagerEnv
from ..model.encoder import BatchedGraphs
from ..model.graph import build_graph
from ..model.policy import EagerPolicy, build_action_set
from .distribution import sample_case


@dataclass
class PPOConfig:
    n_envs: int = 16
    rollout_steps: int = 512
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    ent_start: float = 0.01
    ent_end: float = 0.001
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch: int = 1024
    grad_clip: float = 0.5
    target_kl: float = 0.02
    total_iters: int = 150
    stage: str = "A"


class RunningReturnStd:
    def __init__(self, gamma: float, n_envs: int):
        self.gamma = gamma
        self.acc = np.zeros(n_envs)
        self.count = 1e-4
        self.mean = 0.0
        self.m2 = 1.0

    def normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.acc = self.acc * self.gamma + rewards
        for x in self.acc:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (x - self.mean)
        self.acc[dones] = 0.0
        var = self.m2 / max(1.0, self.count - 1)
        return rewards / math.sqrt(var + 1e-8)


class VecEnvs:
    """n synchronous envs over the stage distribution; auto-resample the
    case and advance the env seed on every episode end."""

    def __init__(self, n_envs: int, case_seed: int, env_seed_base: int,
                 stage: str = "A"):
        self.rng = np.random.default_rng(case_seed)
        self.stage = stage
        self.env_seed_base = env_seed_base
        self.episode_counter = [0] * n_envs
        self.envs: list[EagerEnv] = []
        self.episode_js: list[float] = []
        self.episode_truncs = 0
        self.episodes_done = 0
        for i in range(n_envs):
            self.envs.append(self._fresh_env(i))

    def _fresh_env(self, i: int) -> EagerEnv:
        case = sample_case(self.rng, stage=self.stage)
        env = EagerEnv(case.hardware, case.instance)
        env.reset(self.env_seed_base + 1000 * self.episode_counter[i] + i)
        return env

    def step(self, actions) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros(len(self.envs))
        dones = np.zeros(len(self.envs), dtype=bool)
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            _, r, done, info = env.step(a)
            rewards[i] = r
            dones[i] = done
            if done:
                m = info["metrics"]
                self.episode_js.append(m["J"])
                self.episode_truncs += int(m["truncated"])
                self.episodes_done += 1
                self.episode_counter[i] += 1
                self.envs[i] = self._fresh_env(i)
        return rewards, dones


def collect_rollout(policy: EagerPolicy, vec: VecEnvs, cfg: PPOConfig,
                    device, gen: torch.Generator, ret_norm: RunningReturnStd):
    n = cfg.n_envs
    snaps_t, asets_t, pos_t = [], [], []
    logp = torch.zeros(cfg.rollout_steps, n)
    values = torch.zeros(cfg.rollout_steps, n)
    rewards = np.zeros((cfg.rollout_steps, n))
    dones = np.zeros((cfg.rollout_steps, n), dtype=bool)
    policy.eval()
    with torch.no_grad():
        for t in range(cfg.rollout_steps):
            snaps = [build_graph(e) for e in vec.envs]
            asets = [build_action_set(e, s) for e, s in zip(vec.envs, snaps)]
            batch = BatchedGraphs(snaps, device)
            out = policy(batch, asets)
            positions = out.sample(generator=gen)
            logp[t] = out.log_prob_of(positions).cpu()
            values[t] = out.value.cpu()
            actions = [asets[i].actions[int(positions[i])] for i in range(n)]
            r, d = vec.step(actions)
            rewards[t] = ret_norm.normalize(r, d)
            dones[t] = d
            snaps_t.append(snaps)
            asets_t.append(asets)
            pos_t.append(positions.cpu())
        # bootstrap values of the post-rollout states
        snaps = [build_graph(e) for e in vec.envs]
        asets = [build_action_set(e, s) for e, s in zip(vec.envs, snaps)]
        out = policy(BatchedGraphs(snaps, device), asets)
        last_value = out.value.cpu()
    policy.train()

    adv = torch.zeros(cfg.rollout_steps, n)
    last_gae = torch.zeros(n)
    for t in reversed(range(cfg.rollout_steps)):
        not_done = torch.from_numpy(~dones[t]).float()
        next_v = last_value if t == cfg.rollout_steps - 1 else values[t + 1]
        delta = (torch.from_numpy(rewards[t]).float()
                 + cfg.gamma * next_v * not_done - values[t])
        last_gae = delta + cfg.gamma * cfg.gae_lambda * not_done * last_gae
        adv[t] = last_gae
    returns = adv + values

    flat = {
        "snaps": [s for row in snaps_t for s in row],
        "asets": [a for row in asets_t for a in row],
        "pos": torch.cat([p for p in pos_t]),
        "logp": logp.flatten(),
        "adv": adv.flatten(),
        "ret": returns.flatten(),
        "value": values.flatten(),
    }
    return flat


def ppo_update(policy: EagerPolicy, opt, flat: dict, cfg: PPOConfig, device,
               ent_coef: float, gen: torch.Generator) -> dict:
    n_total = len(flat["snaps"])
    adv = flat["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": [],
             "clipfrac": [], "early_stop": False}
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n_total, generator=gen)
        for i in range(0, n_total, cfg.minibatch):
            idx = perm[i:i + cfg.minibatch]
            snaps = [flat["snaps"][j] for j in idx.tolist()]
            asets = [flat["asets"][j] for j in idx.tolist()]
            batch = BatchedGraphs(snaps, device)
            out = policy(batch, asets)
            pos = flat["pos"][idx].to(device)
            new_logp = out.log_prob_of(pos)
            old_logp = flat["logp"][idx].to(device)
            ratio = (new_logp - old_logp).exp()
            mb_adv = adv[idx].to(device)
            l1 = -mb_adv * ratio
            l2 = -mb_adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
            pi_loss = torch.max(l1, l2).mean()
            v_loss = ((out.value - flat["ret"][idx].to(device)) ** 2).mean()
            ent = out.entropy().mean()
            loss = pi_loss + cfg.value_coef * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()
            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean().item()
                clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
            stats["pi_loss"].append(pi_loss.item())
            stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(ent.item())
            stats["approx_kl"].append(approx_kl)
            stats["clipfrac"].append(clipfrac)
            if approx_kl > cfg.target_kl:
                stats["early_stop"] = True
                return {k: (np.mean(v) if isinstance(v, list) and v else v)
                        for k, v in stats.items()}
    return {k: (np.mean(v) if isinstance(v, list) and v else v)
            for k, v in stats.items()}


def train_ppo(policy: EagerPolicy, cfg: PPOConfig, device, seed: int,
              log=print, on_eval=None, eval_every: int = 10) -> dict:
    """on_eval(iter) -> optional dict; if it returns {"stop": True} the
    driver stops early (used for beat-greedy acceptance checks)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    vec = VecEnvs(cfg.n_envs, case_seed=seed * 7 + 3,
                  env_seed_base=seed * 100_000, stage=cfg.stage)
    ret_norm = RunningReturnStd(cfg.gamma, cfg.n_envs)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.total_iters)
    history = []
    t0 = time.perf_counter()
    for it in range(cfg.total_iters):
        frac = it / max(1, cfg.total_iters - 1)
        ent_coef = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
        flat = collect_rollout(policy, vec, cfg, device, gen, ret_norm)
        stats = ppo_update(policy, opt, flat, cfg, device, ent_coef, gen)
        sched.step()
        recent = vec.episode_js[-32:]
        row = {"iter": it, "mean_recent_J": float(np.mean(recent)) if recent
               else None, "episodes": vec.episodes_done,
               "truncs": vec.episode_truncs, **stats}
        history.append(row)
        log(f"  it {it:3d}: J~{row['mean_recent_J'] and round(row['mean_recent_J'], 1)} "
            f"eps={vec.episodes_done} kl={stats['approx_kl']:.4f} "
            f"ent={stats['entropy']:.3f} ({time.perf_counter() - t0:.0f}s)")
        if on_eval is not None and (it + 1) % eval_every == 0:
            verdict = on_eval(it)
            if verdict and verdict.get("stop"):
                log(f"  acceptance reached at iter {it}; stopping seed early")
                break
    return {"history": history, "episodes": vec.episodes_done,
            "truncations": vec.episode_truncs,
            "wall_s": round(time.perf_counter() - t0, 1)}
