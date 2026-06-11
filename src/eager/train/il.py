"""Imitation learning (guide §8.1): behavioral cloning of GreedyJIT.

The expert is GreedyJIT executed live over the env API; at every micro-step
we capture (graph snapshot, valid action set, the expert action's POSITION
inside the D15-ordered valid set). Loss = cross-entropy over the valid set
(the decoder's segment softmax); target >= 90% val top-1 before PPO.

The dataset is split 90/10 BY EPISODE (transition-level splits would leak
near-duplicate consecutive states across the split; D50).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from ..baselines.greedy_jit import GreedyJITPolicy
from ..env.env import EagerEnv
from ..model.encoder import BatchedGraphs
from ..model.graph import GraphSnapshot, build_graph
from ..model.policy import ActionSet, EagerPolicy, build_action_set
from .distribution import sample_case


@dataclass
class Transition:
    snap: GraphSnapshot
    aset: ActionSet
    expert_pos: int


def collect_expert_dataset(min_transitions: int = 50_000, seed: int = 0,
                           episodes_seed0: int = 0, stage: str = "A",
                           log_every: int = 50) -> tuple[list[list[Transition]], dict]:
    """Run GreedyJIT episodes over the stage distribution until the
    transition budget is met; returns episodes (list of transition lists)."""
    rng = np.random.default_rng(seed)
    episodes: list[list[Transition]] = []
    n_tr = 0
    sizes = []
    t0 = time.perf_counter()
    ep = 0
    while n_tr < min_transitions:
        case = sample_case(rng, stage=stage)
        env = EagerEnv(case.hardware, case.instance)
        policy = GreedyJITPolicy(placement_seed=0)
        env.reset(episodes_seed0 + ep)
        done = False
        episode: list[Transition] = []
        while not done:
            snap = build_graph(env)
            aset = build_action_set(env, snap)
            action = policy(env)
            pos = aset.actions.index(action)
            episode.append(Transition(snap=snap, aset=aset, expert_pos=pos))
            _, _, done, info = env.step(action)
        assert not info["metrics"]["truncated"], case.label
        episodes.append(episode)
        n_tr += len(episode)
        sizes.append(case.instance.num_qubits)
        ep += 1
        if ep % log_every == 0:
            print(f"  collected {ep} episodes / {n_tr} transitions "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    stats = {"episodes": ep, "transitions": n_tr,
              "n_qubits_min": int(min(sizes)), "n_qubits_max": int(max(sizes)),
              "n_qubits_mean": float(np.mean(sizes)),
              "collect_seconds": round(time.perf_counter() - t0, 1)}
    return episodes, stats


def split_episodes(episodes: list[list[Transition]], val_frac: float = 0.1,
                   seed: int = 1) -> tuple[list[Transition], list[Transition]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_frac))
    val_eps = set(order[:n_val].tolist())
    train, val = [], []
    for i, ep in enumerate(episodes):
        (val if i in val_eps else train).extend(ep)
    return train, val


def _batch_forward(policy: EagerPolicy, batch_tr: list[Transition], device):
    batch = BatchedGraphs([t.snap for t in batch_tr], device)
    out = policy(batch, [t.aset for t in batch_tr])
    targets = torch.tensor([t.expert_pos for t in batch_tr], device=device)
    return out, targets


def expert_action_code(t: Transition) -> int:
    return int(t.aset.spec[t.expert_pos, 0])


def type_weights(data: list[Transition],
                 boost: dict[int, float] | None = None) -> dict[int, float]:
    """Inverse-frequency weights over expert action TYPES (Map/Schedule/
    GenEPR/ADVANCE), optionally boosted per type, normalized to mean 1 over
    the dataset (D53): the easy, dominant ADVANCE class must not drown out
    the consequential placement and scheduling decisions."""
    counts: dict[int, int] = {}
    for t in data:
        c = expert_action_code(t)
        counts[c] = counts.get(c, 0) + 1
    n_types = len(counts)
    raw = {c: len(data) / (n_types * cnt) for c, cnt in counts.items()}
    for c, b in (boost or {}).items():
        if c in raw:
            raw[c] *= b
    mean_w = sum(raw[expert_action_code(t)] for t in data) / len(data)
    return {c: w / mean_w for c, w in raw.items()}


def evaluate_top1(policy: EagerPolicy, data: list[Transition], device,
                  batch_size: int = 512) -> float:
    policy.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            out, targets = _batch_forward(policy, chunk, device)
            correct += int((out.greedy() == targets).sum().item())
    policy.train()
    return correct / max(1, len(data))


def evaluate_breakdown(policy: EagerPolicy, data: list[Transition], device,
                       batch_size: int = 512) -> dict[str, dict]:
    """Per-expert-action-type top-1 accuracy."""
    names = {0: "map", 1: "schedule", 2: "gen_epr", 3: "advance"}
    hit: dict[int, int] = {c: 0 for c in names}
    tot: dict[int, int] = {c: 0 for c in names}
    policy.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i:i + batch_size]
            out, targets = _batch_forward(policy, chunk, device)
            pred = out.greedy()
            for j, t in enumerate(chunk):
                c = expert_action_code(t)
                tot[c] += 1
                hit[c] += int(pred[j].item() == targets[j].item())
    policy.train()
    return {names[c]: {"n": tot[c],
                       "top1": (hit[c] / tot[c]) if tot[c] else None}
            for c in names}


def train_il(policy: EagerPolicy, train_data: list[Transition],
             val_data: list[Transition], device, max_epochs: int = 20,
             batch_size: int = 256, lr: float = 3e-4, patience: int = 3,
             seed: int = 0, weighted: bool = True,
             boost: dict[int, float] | None = None, log=print) -> dict:
    torch.manual_seed(seed)
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    rng = np.random.default_rng(seed)
    weights = type_weights(train_data, boost=boost) if weighted else None
    best = {"val_top1": -1.0, "epoch": -1, "state": None}
    history = []
    stall = 0
    for epoch in range(max_epochs):
        order = rng.permutation(len(train_data))
        losses = []
        for i in range(0, len(order), batch_size):
            chunk = [train_data[j] for j in order[i:i + batch_size]]
            out, targets = _batch_forward(policy, chunk, device)
            nll = -out.log_prob_of(targets)
            if weights is not None:
                w = torch.tensor([weights[expert_action_code(t)]
                                  for t in chunk], device=device)
                loss = (w * nll).mean()
            else:
                loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()
        val_top1 = evaluate_top1(policy, val_data, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_top1": val_top1})
        log(f"  IL epoch {epoch:2d}: loss={np.mean(losses):.4f} "
            f"val_top1={val_top1:.4f}")
        if val_top1 > best["val_top1"] + 1e-4:
            best = {"val_top1": val_top1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in policy.state_dict().items()}}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                log(f"  early stop at epoch {epoch} "
                    f"(best val_top1={best['val_top1']:.4f} "
                    f"@ epoch {best['epoch']})")
                break
    policy.load_state_dict(best["state"])
    return {"best_val_top1": best["val_top1"], "best_epoch": best["epoch"],
            "history": history}
