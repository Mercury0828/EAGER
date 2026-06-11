"""DDQN-flat — ICC'25-style learning baseline (guide §9.4).

Double DQN + target network + uniform replay over a FLAT fixed-size state
(per-QPU loads, per-link [stored, busy, free, p], top-k ready-gate features,
globals; zero-padded) with the SAME action space as every other method (the
D15 enumeration, max-size boolean masks). Trained PER CONFIGURATION (sizes
are derived from the bound env config); training itself runs in Phase 6 with
the same env-step budget as EAGER's PPO phase — this module ships the
implementation and a smoke-trainable loop.

Its expected failure to generalize/scale is part of the paper's message; the
implementation is still kept honest (masked argmax everywhere, Double-DQN
target, gradient clipping) so the comparison is fair at small scale.

RNG separation: torch/agent randomness is seeded independently; env
stochasticity stays in the CRN engine (guide §12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..env.actions import Action
from ..env.env import EagerEnv
from ..env.state import UNSCHEDULED

TOP_K_READY = 8


class FlatFeaturizer:
    """Fixed-size float32 state vector for one env configuration."""

    def __init__(self, env: EagerEnv):
        self.k = env.hardware.num_qpus
        self.n_links = env.hardware.num_links
        self.top_k = TOP_K_READY
        self.m = env.instance.num_gates
        self.n = env.instance.num_qubits
        self.depth = max(1, env.instance.depth)
        self.t_budget = env.t_budget
        self.kappa = env.hardware.kappa
        self.link_caps = [(lc.W, lc.B, lc.p) for lc in env.hardware.links]
        self.total_b = max(1, sum(lc.B for lc in env.hardware.links))
        self.max_route = max(
            (len(r) for r in env.routing.link_routes.values()), default=1)
        self.dim = 3 * self.k + 4 * self.n_links + 4 * self.top_k + 4

    def __call__(self, env: EagerEnv) -> np.ndarray:
        x = np.zeros(self.dim, dtype=np.float32)
        i = 0
        mapped_per = [0] * self.k
        for u in env.qubit_qpu:
            if u is not None:
                mapped_per[u] += 1
        ready = env.ready_gates()
        ready_local_per = [0] * self.k
        for g in ready:
            gr = env.gates[g]
            if gr.remote is False:
                a, _ = env.instance.gates[g]
                ready_local_per[env.qubit_qpu[a]] += 1
        for u in range(self.k):
            x[i] = env.kappa_res[u] / self.kappa[u]
            x[i + 1] = mapped_per[u] / self.kappa[u]
            x[i + 2] = ready_local_per[u] / max(1, len(ready)) if ready else 0.0
            i += 3
        for l, ls in enumerate(env.links):
            w_cap, b_cap, p = self.link_caps[l]
            x[i] = ls.stored / b_cap
            x[i + 1] = ls.busy_channels / w_cap
            x[i + 2] = ls.free_channels / w_cap
            x[i + 3] = p
            i += 4
        crit = env.instance.criticality
        top = sorted(ready, key=lambda g: (-crit[g], g))[: self.top_k]
        for g in top:
            gr = env.gates[g]
            a, b = env.instance.gates[g]
            n_mapped = (env.qubit_qpu[a] is not None) + (env.qubit_qpu[b] is not None)
            x[i] = crit[g] / self.depth
            x[i + 1] = n_mapped / 2.0
            x[i + 2] = 1.0 if gr.remote else 0.0
            x[i + 3] = (len(gr.route) / self.max_route) if gr.route else 0.0
            i += 4
        i = 3 * self.k + 4 * self.n_links + 4 * self.top_k
        x[i] = env.t / self.t_budget
        x[i + 1] = env.done_count / self.m
        x[i + 2] = (self.n - len(env._unmapped)) / self.n
        x[i + 3] = sum(ls.stored for ls in env.links) / self.total_b
        return x


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, num_actions: int):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.mask2 = np.zeros((capacity, num_actions), dtype=bool)
        self.size = 0
        self._next = 0

    def push(self, s, a, r, s2, done, mask2) -> None:
        j = self._next
        self.s[j], self.a[j], self.r[j] = s, a, r
        self.s2[j], self.done[j], self.mask2[j] = s2, float(done), mask2
        self._next = (j + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch)
        return (torch.from_numpy(self.s[idx]),
                torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]),
                torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.done[idx]),
                torch.from_numpy(self.mask2[idx]))


@dataclass(frozen=True)
class DDQNConfig:
    gamma: float = 0.995
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    buffer_capacity: int = 50_000
    batch_size: int = 128
    target_sync_every: int = 500
    train_start: int = 500
    grad_clip: float = 1.0


NEG_INF = torch.finfo(torch.float32).min


class DDQNFlatAgent:
    def __init__(self, env: EagerEnv, config: DDQNConfig | None = None,
                 seed: int = 0):
        self.config = config or DDQNConfig()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.featurizer = FlatFeaturizer(env)
        self.num_actions = env.action_space.size
        self.online = QNetwork(self.featurizer.dim, self.num_actions)
        self.target = QNetwork(self.featurizer.dim, self.num_actions)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity,
                                   self.featurizer.dim, self.num_actions)
        self.steps = 0

    def epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self.steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def select_action(self, env: EagerEnv, eps: float | None = None
                      ) -> tuple[Action, int, np.ndarray]:
        mask = env.valid_action_mask()
        state = self.featurizer(env)
        if (eps if eps is not None else self.epsilon()) > self.rng.random():
            valid = np.flatnonzero(mask)
            idx = int(valid[self.rng.integers(len(valid))])
        else:
            with torch.no_grad():
                q = self.online(torch.from_numpy(state).unsqueeze(0))[0]
            q = q.masked_fill(~torch.from_numpy(mask), NEG_INF)
            idx = int(torch.argmax(q).item())
        return env.action_space.action_at(idx), idx, state

    def update(self) -> float | None:
        c = self.config
        if self.buffer.size < max(c.train_start, c.batch_size):
            return None
        s, a, r, s2, done, mask2 = self.buffer.sample(c.batch_size, self.rng)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2_online = self.online(s2).masked_fill(~mask2, NEG_INF)
            a_star = torch.argmax(q2_online, dim=1, keepdim=True)
            q2_target = self.target(s2).gather(1, a_star).squeeze(1)
            y = r + c.gamma * (1.0 - done) * q2_target
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()
        if self.steps % c.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path) -> None:
        torch.save({"online": self.online.state_dict(),
                    "config": self.config.__dict__}, path)

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])


def train_ddqn(env: EagerEnv, total_env_steps: int,
               config: DDQNConfig | None = None, seed: int = 0,
               env_seed0: int = 0) -> dict:
    """Per-configuration training loop (budgeted in env steps; Phase 6 sets
    the budget to match EAGER's PPO phase for fairness)."""
    agent = DDQNFlatAgent(env, config, seed=seed)
    history = {"episode_J": [], "episode_T": [], "losses": [], "episodes": 0}
    episode = 0
    while agent.steps < total_env_steps:
        obs = env.reset(env_seed0 + episode)
        done = False
        while not done and agent.steps < total_env_steps:
            action, idx, state = agent.select_action(env)
            _, reward, done, info = env.step(action)
            mask2 = env.valid_action_mask() if not done else np.zeros(
                agent.num_actions, dtype=bool)
            state2 = agent.featurizer(env)
            agent.buffer.push(state, idx, reward, state2, done, mask2)
            agent.steps += 1
            loss = agent.update()
            if loss is not None:
                history["losses"].append(loss)
        if done:
            m = info["metrics"]
            history["episode_J"].append(m["J"])
            history["episode_T"].append(m["T"])
        episode += 1
    history["episodes"] = episode
    history["agent"] = agent
    return history
