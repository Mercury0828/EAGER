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
    positives: np.ndarray | None = None   # cut-equivalent-optimal positions
                                          # (Map states, D57); None -> {expert_pos}


def _hop_matrix(hardware):
    from ..env.routing import build_routing
    rt = build_routing(hardware)
    k = hardware.num_qpus
    hop = [[0] * k for _ in range(k)]
    for u in range(k):
        for v in range(k):
            if u != v:
                hop[u][v] = len(rt.route(u, v))
    return hop


def _completion_comm_cost(instance, hardware, weights, hop,
                          pinned: dict[int, int], seed: int = 0) -> float:
    from ..baselines.partition import balanced_partition
    plan = balanced_partition(instance.num_qubits, list(hardware.kappa),
                              weights, seed=seed, preassigned=pinned)
    cost = 0.0
    for (a, b), w in weights.items():
        cost += w * hop[plan[a]][plan[b]]
    return cost


def map_positive_positions(env: EagerEnv, aset: ActionSet, expert_action,
                           weights, hop) -> np.ndarray:
    """All valid Map(q, u) positions (same qubit as the expert's choice)
    whose optimal-completion COMM COST (route hops weighted by gate counts)
    ties the minimum — the J-equivalent placement choices (D57). The expert's
    own position is always included."""
    from ..env.actions import Map as MapAction
    q = expert_action.qubit
    pinned_base = {i: u for i, u in enumerate(env.qubit_qpu) if u is not None}
    costs: dict[int, float] = {}
    for pos, a in enumerate(aset.actions):
        if isinstance(a, MapAction) and a.qubit == q:
            costs[pos] = _completion_comm_cost(
                env.instance, env.hardware, weights, hop,
                {**pinned_base, q: a.qpu})
    best = min(costs.values())
    pos = np.array(sorted(p for p, c in costs.items()
                          if c <= best + 1e-9), dtype=np.int64)
    expert_pos = aset.actions.index(expert_action)
    if expert_pos not in pos:
        pos = np.append(pos, expert_pos)
    return pos


def collect_expert_dataset(min_transitions: int = 50_000, seed: int = 0,
                           episodes_seed0: int = 0, stage: str = "A",
                           log_every: int = 50, expert_factory=None
                           ) -> tuple[list[list[Transition]], dict]:
    """Run EXPERT episodes over the stage distribution until the transition
    budget is met. ``expert_factory() -> policy`` defaults to GreedyJIT;
    pass ``make_mhsa_policy`` for the stronger MHSA-placement teacher (D71)."""
    from ..baselines.partition import interaction_graph
    from ..env.actions import Map as MapAction
    if expert_factory is None:
        expert_factory = lambda: GreedyJITPolicy(placement_seed=0)
    rng = np.random.default_rng(seed)
    episodes: list[list[Transition]] = []
    n_tr = 0
    sizes = []
    t0 = time.perf_counter()
    ep = 0
    while n_tr < min_transitions:
        case = sample_case(rng, stage=stage)
        env = EagerEnv(case.hardware, case.instance)
        policy = expert_factory()
        env.reset(episodes_seed0 + ep)
        weights = interaction_graph(case.instance)
        hop = _hop_matrix(case.hardware)
        done = False
        episode: list[Transition] = []
        while not done:
            snap = build_graph(env)
            aset = build_action_set(env, snap)
            action = policy(env)
            pos = aset.actions.index(action)
            positives = None
            if isinstance(action, MapAction):
                positives = map_positive_positions(env, aset, action,
                                                   weights, hop)
            episode.append(Transition(snap=snap, aset=aset, expert_pos=pos,
                                      positives=positives))
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


def collect_dagger_states(policy, n_transitions: int, device,
                          case_seed: int = 100, env_seed_base: int = 500_000,
                          n_envs: int = 16, stage: str = "A",
                          log_every: int = 20_000) -> list[Transition]:
    """DAgger round (D55): roll the CURRENT agent (greedy decode — the
    deployment policy) over the training distribution and label every
    visited state with the CONDITIONAL expert's action (a completion
    partition pinned to the agent's partial mapping; recovery supervision)."""
    import torch as _torch
    from ..baselines.greedy_jit import ConditionalGreedyJIT
    from ..model.encoder import BatchedGraphs

    rng = np.random.default_rng(case_seed)
    envs: list = [None] * n_envs
    experts: list = [None] * n_envs
    counters = [0] * n_envs

    def fresh(i: int) -> None:
        case = sample_case(rng, stage=stage)
        env = EagerEnv(case.hardware, case.instance)
        env.reset(env_seed_base + 1000 * counters[i] + i)
        envs[i] = env
        experts[i] = ConditionalGreedyJIT(placement_seed=0)

    for i in range(n_envs):
        fresh(i)
    from ..baselines.partition import interaction_graph
    from ..env.actions import Map as MapAction
    out: list[Transition] = []
    policy.eval()
    t0 = time.perf_counter()
    with _torch.no_grad():
        while len(out) < n_transitions:
            snaps = [build_graph(e) for e in envs]
            asets = [build_action_set(e, s) for e, s in zip(envs, snaps)]
            pol_out = policy(BatchedGraphs(snaps, device), asets)
            positions = pol_out.greedy()
            for i, env in enumerate(envs):
                expert_action = experts[i](env)
                pos = asets[i].actions.index(expert_action)
                positives = None
                if isinstance(expert_action, MapAction):
                    positives = map_positive_positions(
                        env, asets[i], expert_action,
                        interaction_graph(env.instance),
                        _hop_matrix(env.hardware))
                out.append(Transition(snap=snaps[i], aset=asets[i],
                                      expert_pos=pos, positives=positives))
                agent_action = asets[i].actions[int(positions[i])]
                _, _, done, _ = env.step(agent_action)
                if done:
                    counters[i] += 1
                    fresh(i)
            if log_every and len(out) % log_every < n_envs:
                print(f"  dagger: {len(out)} labeled states "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    policy.train()
    return out[:n_transitions]


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


def _multi_positive_nll(out, chunk: list[Transition], device) -> torch.Tensor:
    """-log sum_{p in positives} prob(p): every J-equivalent placement choice
    counts as correct (D57); non-map transitions have the singleton set."""
    from ..model.policy import segment_logsumexp
    logp = out.log_softmax()
    flat_idx, seg_ids = [], []
    for i, t in enumerate(chunk):
        pos = (t.positives if t.positives is not None
               else np.array([t.expert_pos]))
        flat_idx.extend((out.ptr[i] + pos).tolist())
        seg_ids.extend([i] * len(pos))
    vals = logp[torch.tensor(flat_idx, device=device)]
    seg = torch.tensor(seg_ids, device=device)
    return -segment_logsumexp(vals, seg, len(chunk))


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
    """Per-expert-action-type STRICT top-1, plus class-correct accuracy for
    maps (prediction inside the J-equivalent positives set, D57)."""
    names = {0: "map", 1: "schedule", 2: "gen_epr", 3: "advance"}
    hit: dict[int, int] = {c: 0 for c in names}
    tot: dict[int, int] = {c: 0 for c in names}
    class_hit = 0
    class_tot = 0
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
                if c == 0 and t.positives is not None:
                    class_tot += 1
                    class_hit += int(pred[j].item() in t.positives)
    policy.train()
    result = {names[c]: {"n": tot[c],
                         "top1": (hit[c] / tot[c]) if tot[c] else None}
              for c in names}
    result["map_class_correct"] = {
        "n": class_tot, "acc": (class_hit / class_tot) if class_tot else None}
    return result


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
            # equivalence-aware main term + small strict-CE auxiliary that
            # keeps the convention tie-break (and the strict top-1 gate)
            nll = (_multi_positive_nll(out, chunk, device)
                   + 0.25 * (-out.log_prob_of(targets)))
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
