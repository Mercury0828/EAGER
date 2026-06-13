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
    """Fine-tuning regime (D58): the §8.2 defaults (lr 3e-4, clip 0.2)
    re-learn from scratch and destroy the IL initialization before slowly
    recovering (measured: held-out ratio 1.04 -> 1.2-1.9 in the first 30
    iters, back to ~1.00 by iter 120, never significant); §15's sanctioned
    fallbacks apply: smaller clip, gentler lr, low entropy, plus a VALUE
    WARMUP phase (policy frozen) so the freshly initialized value head
    cannot feed noise advantages to the policy gradient."""

    n_envs: int = 16
    rollout_steps: int = 512
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.15
    value_coef: float = 0.5
    ent_start: float = 0.01       # exploration must SURVIVE (D60): the
    ent_end: float = 0.003        # frozen v2/v3 regime had policy entropy
    lr: float = 1e-4              # ~0.05 and KL ~0.002 — no discovery
    lr_min: float = 3e-5
    update_epochs: int = 4
    minibatch: int = 1024
    grad_clip: float = 0.5
    target_kl: float = 0.02
    total_iters: int = 300
    value_warmup_iters: int = 3
    stage: str = "A"
    # ---- targeted-exploration self-imitation stream (D60) ----
    sil_episodes_per_iter: int = 16
    sil_epsilon: float = 0.15     # P(force a GenEPR when ADVANCE is argmax)
    sil_buffer_cap: int = 60_000
    sil_minibatch: int = 512
    sil_steps_per_iter: int = 4
    sil_coef: float = 1.0
    sil_gen_weight: float = 1.0   # extra BC weight on GenEPR steps of
                                  # winning episodes (the decisive, rare
                                  # class; ~5-10% of a winner's states)
    sil_win_margin: float = 0.05  # clone only episodes beating greedy by
                                  # this RELATIVE margin: at p=0.08 the
                                  # per-episode luck noise admits abundant
                                  # spurious sub-margin "wins" whose cloning
                                  # teaches gambling (D62)
    anchor_coef: float = 0.0      # CE toward the FROZEN IL policy's argmax
                                  # on rollout states: counters gambling
                                  # drift in regimes where SIL has no win
                                  # evidence (D62); 0 disables
    comfortable_rl_off: bool = False  # v9 (D67): zero the policy-gradient
                                      # advantage on comfortable-regime
                                      # episodes entirely — comfortable
                                      # behavior is shaped ONLY by the
                                      # anchor; RL/SIL act only where
                                      # headroom is proven
    comfortable_greedy_anchor: bool = False  # v10 (D70): anchor comfortable-
                                      # regime states toward the (conditional)
                                      # GREEDY action, not IL — IL itself is
                                      # ~1.04 vs greedy, so an IL anchor caps
                                      # comfortable at ~1.04; greedy is the
                                      # near-optimal target where proactivity
                                      # only wastes pairs
    paired_advantage: bool = False  # v6 (D63): episode-constant advantage
    episodes_per_iter: int = 32     # = (J_greedy - J_agent) on the SAME
                                    # (case, env seed) via CRN pairing --
                                    # generation luck cancels exactly in the
                                    # difference; rollouts become whole
                                    # episodes instead of fixed-step windows
    regime_stage1_iters: int = 0    # v7 (D64): iters trained on the
                                    # provisioning-bound regime ONLY with the
                                    # anchor OFF (skill specialization),
                                    # before broadening to the full
                                    # distribution with the anchor ON


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


def collect_sil_winners(policy: EagerPolicy, cfg: PPOConfig, device,
                        rng: np.random.Generator, greedy_cache: dict,
                        stage: str, regime: str = "full") -> tuple[list, dict]:
    """Targeted-exploration self-imitation stream (D60): run episodes with
    the policy's GREEDY action EXCEPT that, with probability sil_epsilon at
    states where ADVANCE is the argmax and a GenEPR is valid, a demanded
    (else any valid) GenEPR is forced instead — exactly the proactive-
    provisioning margin that IL drove to ~zero probability and that global
    entropy bonuses cannot reach. Episodes that BEAT the CRN-paired
    GreedyJIT J on the same (case, env seed) return their full
    (state, action) sequences for behavioral cloning."""
    from ..baselines.greedy_jit import GreedyJITPolicy
    from ..env.actions import Advance, GenEPR
    from ..model.encoder import BatchedGraphs

    n = cfg.sil_episodes_per_iter
    cases = [sample_case(rng, stage=stage, regime=regime) for _ in range(n)]
    seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n)]
    envs = [EagerEnv(c.hardware, c.instance) for c in cases]
    for env, s in zip(envs, seeds):
        env.reset(s)
    episodes: list[list] = [[] for _ in range(n)]
    js = [None] * n
    active = list(range(n))
    policy.eval()
    with torch.no_grad():
        while active:
            snaps = [build_graph(envs[i]) for i in active]
            asets = [build_action_set(envs[i], s)
                     for i, s in zip(active, snaps)]
            out = policy(BatchedGraphs(snaps, device), asets)
            pos = out.greedy()
            nxt = []
            for j, i in enumerate(active):
                aset = asets[j]
                p = int(pos[j])
                action = aset.actions[p]
                if (isinstance(action, Advance)
                        and rng.random() < cfg.sil_epsilon):
                    demand, _ = envs[i].deficit_demand()
                    gens = [(k, a) for k, a in enumerate(aset.actions)
                            if isinstance(a, GenEPR)]
                    if gens:
                        demanded = [(k, a) for k, a in gens
                                    if demand[a.link] > 0]
                        pool = demanded if demanded else gens
                        p, action = pool[int(rng.integers(len(pool)))]
                episodes[i].append((snaps[j], aset, p))
                _, _, done, info = envs[i].step(action)
                if done:
                    js[i] = info["metrics"]
                else:
                    nxt.append(i)
            active = nxt
    policy.train()

    winners = []
    n_won = 0
    for i in range(n):
        key = (cases[i].label, seeds[i])
        if key not in greedy_cache:
            env = EagerEnv(cases[i].hardware, cases[i].instance)
            env.reset(seeds[i])
            gp = GreedyJITPolicy(placement_seed=0)
            done = False
            while not done:
                _, _, done, ginfo = env.step(gp(env))
            greedy_cache[key] = ginfo["metrics"]["J"]
        bar = greedy_cache[key] * (1.0 - cfg.sil_win_margin)
        if (not js[i]["truncated"]) and js[i]["J"] <= bar:
            winners.extend(episodes[i])
            n_won += 1
    return winners, {"episodes": n, "won": n_won, "new_states": len(winners)}


class RunningStd:
    def __init__(self):
        self.count, self.mean, self.m2 = 1e-4, 0.0, 1.0

    def update(self, x: float) -> None:
        self.count += 1
        d = x - self.mean
        self.mean += d / self.count
        self.m2 += d * (x - self.mean)

    def std(self) -> float:
        return math.sqrt(self.m2 / max(1.0, self.count - 1) + 1e-8)


def collect_paired_episodes(policy: EagerPolicy, cfg: PPOConfig, device,
                            gen: torch.Generator, rng: np.random.Generator,
                            greedy_cache: dict, diff_std: RunningStd,
                            stage: str, regime: str = "full"):
    """v6 (D63): sample whole episodes (policy sampling) on fresh cases and
    assign every step the EPISODE-CONSTANT advantage
    (J_greedy - J_agent) / running_std, where J_greedy is the CRN-paired
    GreedyJIT objective on the SAME (case, env seed) — an unbiased control
    variate through which generation luck cancels exactly (§6.5)."""
    from ..baselines.greedy_jit import ConditionalGreedyJIT, GreedyJITPolicy
    from ..model.encoder import BatchedGraphs

    n = cfg.episodes_per_iter
    cases = [sample_case(rng, stage=stage, regime=regime) for _ in range(n)]
    seeds = [int(rng.integers(0, 1_000_000)) for _ in range(n)]
    envs = [EagerEnv(c.hardware, c.instance) for c in cases]
    for env, s in zip(envs, seeds):
        env.reset(s)
    # per-episode comfortable tag + (optional) live greedy oracle for the
    # comfortable-regime greedy anchor (D70)
    comfy = []
    for c in cases:
        lc = c.hardware.links[0]
        comfy.append(lc.p < 0.2 and lc.W >= 2)
    oracle = [ConditionalGreedyJIT(0) if (cfg.comfortable_greedy_anchor
                                          and comfy[i]) else None
              for i in range(n)]
    steps: list[list] = [[] for _ in range(n)]   # (snap, aset, pos, logp, gpos)
    js = [None] * n
    active = list(range(n))
    policy.eval()
    with torch.no_grad():
        while active:
            snaps = [build_graph(envs[i]) for i in active]
            asets = [build_action_set(envs[i], s)
                     for i, s in zip(active, snaps)]
            out = policy(BatchedGraphs(snaps, device), asets)
            pos = out.sample(generator=gen)
            logp = out.log_prob_of(pos).cpu()
            nxt = []
            for j, i in enumerate(active):
                p = int(pos[j])
                gpos = -1
                if oracle[i] is not None:
                    ga = oracle[i](envs[i])         # greedy action at THIS state
                    gpos = asets[j].actions.index(ga)
                steps[i].append((snaps[j], asets[j], p, float(logp[j]), gpos))
                _, _, done, info = envs[i].step(asets[j].actions[p])
                if done:
                    js[i] = info["metrics"]
                else:
                    nxt.append(i)
            active = nxt
    policy.train()

    flat = {"snaps": [], "asets": [], "pos": [], "logp": [], "adv": [],
            "comfortable": [], "greedy_pos": []}
    ep_js, n_trunc = [], 0
    for i in range(n):
        key = (cases[i].label, seeds[i])
        if key not in greedy_cache:
            env = EagerEnv(cases[i].hardware, cases[i].instance)
            env.reset(seeds[i])
            gp = GreedyJITPolicy(placement_seed=0)
            done = False
            while not done:
                _, _, done, ginfo = env.step(gp(env))
            greedy_cache[key] = ginfo["metrics"]["J"]
        diff = greedy_cache[key] - js[i]["J"]        # >0: agent better
        diff_std.update(diff)
        adv = diff / diff_std.std()
        ep_js.append(js[i]["J"])
        n_trunc += int(js[i]["truncated"])
        for (snap, aset, p, lp, gpos) in steps[i]:
            flat["snaps"].append(snap)
            flat["asets"].append(aset)
            flat["pos"].append(p)
            flat["logp"].append(lp)
            flat["adv"].append(adv)
            flat["comfortable"].append(comfy[i])
            flat["greedy_pos"].append(gpos)
    flat["pos"] = torch.tensor(flat["pos"])
    flat["logp"] = torch.tensor(flat["logp"])
    flat["adv"] = torch.tensor(flat["adv"], dtype=torch.float32)
    flat["comfortable"] = np.array(flat["comfortable"], dtype=bool)
    flat["greedy_pos"] = np.array(flat["greedy_pos"], dtype=np.int64)
    if cfg.comfortable_rl_off:
        flat["adv"][torch.from_numpy(flat["comfortable"])] = 0.0
    flat["ret"] = flat["adv"].clone()                # value target: paired adv
    return flat, {"episodes": n, "mean_J": float(np.mean(ep_js)),
                  "truncs": n_trunc}


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
               ent_coef: float, gen_cpu: torch.Generator,
               value_only: bool = False) -> dict:
    n_total = len(flat["snaps"])
    adv = flat["adv"]
    if cfg.paired_advantage:
        # paired advantages have a MEANINGFUL zero (= ties greedy under the
        # same luck); scale-normalize only, so masked zeros stay zero (D67)
        adv = adv / (adv.std() + 1e-8)
    else:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": [],
             "clipfrac": [], "early_stop": False}
    for _ in range(cfg.update_epochs):
        perm = torch.randperm(n_total, generator=gen_cpu)
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
            if value_only:
                # warmup: fit the value head only; the policy gradient stays
                # OFF until advantages mean something (D58)
                loss = cfg.value_coef * v_loss
            else:
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
            if not value_only and approx_kl > cfg.target_kl:
                stats["early_stop"] = True
                return {k: (np.mean(v) if isinstance(v, list) and v else v)
                        for k, v in stats.items()}
    return {k: (np.mean(v) if isinstance(v, list) and v else v)
            for k, v in stats.items()}


def train_ppo(policy: EagerPolicy, cfg: PPOConfig, device, seed: int,
              log=print, on_eval=None, eval_every: int = 10,
              anchor_policy: EagerPolicy | None = None) -> dict:
    """on_eval(iter) -> optional dict; if it returns {"stop": True} the
    driver stops early (used for beat-greedy acceptance checks)."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)       # device gen: action sampling
    gen.manual_seed(seed)
    gen_cpu = torch.Generator()                # cpu gen: minibatch permutation
    gen_cpu.manual_seed(seed + 1)
    vec = VecEnvs(cfg.n_envs, case_seed=seed * 7 + 3,
                  env_seed_base=seed * 100_000, stage=cfg.stage)
    ret_norm = RunningReturnStd(cfg.gamma, cfg.n_envs)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.total_iters, eta_min=cfg.lr_min)
    history = []
    t0 = time.perf_counter()

    # ---- value warmup: fit V on frozen features, policy untouched (D58) ----
    if cfg.value_warmup_iters > 0:
        frozen = [p for n, p in policy.named_parameters()
                  if not n.startswith("v_mlp")]
        for p in frozen:
            p.requires_grad_(False)
        warm_opt = torch.optim.Adam(policy.v_mlp.parameters(), lr=1e-3)
        for wit in range(cfg.value_warmup_iters):
            flat = collect_rollout(policy, vec, cfg, device, gen, ret_norm)
            stats = ppo_update(policy, warm_opt, flat, cfg, device, 0.0,
                               gen_cpu, value_only=True)
            log(f"  warmup {wit}: v_loss={stats['v_loss']:.4f}")
        for p in frozen:
            p.requires_grad_(True)

    sil_buffer: list = []
    sil_rng = np.random.default_rng(seed * 13 + 5)
    greedy_cache: dict = {}
    diff_std = RunningStd()
    for it in range(cfg.total_iters):
        frac = it / max(1, cfg.total_iters - 1)
        ent_coef = cfg.ent_start + frac * (cfg.ent_end - cfg.ent_start)
        in_stage1 = it < cfg.regime_stage1_iters
        regime = "provisioning" if in_stage1 else "full"
        if cfg.paired_advantage:
            flat, ep_stats = collect_paired_episodes(
                policy, cfg, device, gen, sil_rng, greedy_cache, diff_std,
                cfg.stage, regime=regime)
            vec.episode_js.extend([ep_stats["mean_J"]])
            vec.episodes_done += ep_stats["episodes"]
            vec.episode_truncs += ep_stats["truncs"]
        else:
            flat = collect_rollout(policy, vec, cfg, device, gen, ret_norm)
        stats = ppo_update(policy, opt, flat, cfg, device, ent_coef, gen_cpu)
        sched.step()

        # IL-anchor toward the frozen IL policy, REGIME-CONDITIONAL (D65):
        # applied ONLY to comfortable-regime states (p<0.2 and W>=2), where
        # the measured agent deviations are pure loss (ratio 1.08, 0 wins
        # signal) — the provisioning-bound regime, where the agent wins with
        # p=1e-10 significance, is left free. OFF during regime stage 1.
        comf_pool = (np.flatnonzero(flat["comfortable"])
                     if cfg.paired_advantage and "comfortable" in flat
                     else np.arange(len(flat["snaps"])))
        use_greedy_anchor = (cfg.comfortable_greedy_anchor
                             and "greedy_pos" in flat)
        if use_greedy_anchor:
            # restrict the pool to comfortable states that carry a recorded
            # greedy target (D70)
            comf_pool = np.flatnonzero(
                flat["comfortable"] & (flat["greedy_pos"] >= 0))
        if ((anchor_policy is not None or use_greedy_anchor)
                and cfg.anchor_coef > 0 and not in_stage1 and len(comf_pool) > 0):
            a_idx = comf_pool[sil_rng.integers(
                len(comf_pool), size=min(cfg.sil_minibatch, len(comf_pool)))]
            chunk_s = [flat["snaps"][int(k)] for k in a_idx]
            chunk_a = [flat["asets"][int(k)] for k in a_idx]
            batch_a = BatchedGraphs(chunk_s, device)
            if use_greedy_anchor:
                ref_pos = torch.tensor(
                    [int(flat["greedy_pos"][int(k)]) for k in a_idx],
                    device=device)
            else:
                with torch.no_grad():
                    ref = anchor_policy(batch_a, chunk_a)
                    ref_pos = ref.greedy()
            out_a = policy(batch_a, chunk_a)
            anchor_loss = cfg.anchor_coef * (
                -out_a.log_prob_of(ref_pos).mean())
            opt.zero_grad()
            anchor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()

        # self-imitation stream (D60)
        sil_stats = {"episodes": 0, "won": 0, "new_states": 0}
        sil_loss = float("nan")
        if cfg.sil_episodes_per_iter > 0:
            winners, sil_stats = collect_sil_winners(
                policy, cfg, device, sil_rng, greedy_cache, cfg.stage,
                regime=regime)
            sil_buffer.extend(winners)
            if len(sil_buffer) > cfg.sil_buffer_cap:
                del sil_buffer[: len(sil_buffer) - cfg.sil_buffer_cap]
            if len(sil_buffer) >= cfg.sil_minibatch:
                from ..env.actions import GenEPR as _GenEPR
                for _ in range(cfg.sil_steps_per_iter):
                    idx = sil_rng.integers(len(sil_buffer),
                                           size=cfg.sil_minibatch)
                    chunk = [sil_buffer[int(k)] for k in idx]
                    batch = BatchedGraphs([c[0] for c in chunk], device)
                    out = policy(batch, [c[1] for c in chunk])
                    targets = torch.tensor([c[2] for c in chunk],
                                           device=device)
                    nll = -out.log_prob_of(targets)
                    if cfg.sil_gen_weight != 1.0:
                        w = torch.tensor(
                            [cfg.sil_gen_weight if isinstance(
                                c[1].actions[c[2]], _GenEPR) else 1.0
                             for c in chunk], device=device)
                        nll = w * nll / w.mean()
                    loss = cfg.sil_coef * nll.mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),
                                                   cfg.grad_clip)
                    opt.step()
                sil_loss = float(loss.item())

        recent = vec.episode_js[-32:]
        row = {"iter": it, "mean_recent_J": float(np.mean(recent)) if recent
               else None, "episodes": vec.episodes_done,
               "truncs": vec.episode_truncs, "sil_won": sil_stats["won"],
               "sil_buffer": len(sil_buffer), **stats}
        history.append(row)
        log(f"  it {it:3d}: J~{row['mean_recent_J'] and round(row['mean_recent_J'], 1)} "
            f"eps={vec.episodes_done} kl={stats['approx_kl']:.4f} "
            f"ent={stats['entropy']:.3f} sil={sil_stats['won']}/"
            f"{sil_stats['episodes']} buf={len(sil_buffer)} "
            f"({time.perf_counter() - t0:.0f}s)")
        if on_eval is not None and (it + 1) % eval_every == 0:
            verdict = on_eval(it)
            if verdict and verdict.get("stop"):
                log(f"  acceptance reached at iter {it}; stopping seed early")
                break
    return {"history": history, "episodes": vec.episodes_done,
            "truncations": vec.episode_truncs,
            "wall_s": round(time.perf_counter() - t0, 1)}
