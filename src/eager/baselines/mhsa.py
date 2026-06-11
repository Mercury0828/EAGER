"""MHSA — multistage hybrid simulated annealing placement (guide §9.2;
INFOCOM'23-style, Mao/Liu/Yang "Qubit Allocation for Distributed Quantum
Computing": a local-search heuristic hybridized with simulated annealing).

Reimplemented for *placement only* and paired with the §9.1 list scheduler +
JIT provisioning (via GreedyJITPolicy's placement_fn), so MHSA-vs-GreedyJIT
comparisons isolate placement quality. Objective: remote-gate count = the
gate-weighted cut of the qubit interaction graph, under per-QPU capacities.

Structure (fidelity notes in docs/BASELINE_FIDELITY.md):
  greedy initialization -> stages of [SA exploration at a per-stage
  temperature schedule over capacity-feasible single moves and pairwise
  swaps] -> [local-search descent polishing] -> next stage at lower
  temperature. Fixed proposal budget, reported with results.
"""

from __future__ import annotations

import math

import numpy as np

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from .partition import interaction_graph

DEFAULT_BUDGET = 20_000
DEFAULT_STAGES = 4


def _build_adj(num_items: int, weights: dict[tuple[int, int], int]):
    adj: list[dict[int, int]] = [dict() for _ in range(num_items)]
    for (a, b), w in weights.items():
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w
    return adj


def mhsa_placement(instance: CircuitInstance, hardware: HardwareConfig,
                   seed: int = 0, budget: int = DEFAULT_BUDGET,
                   stages: int = DEFAULT_STAGES) -> list[int]:
    n = instance.num_qubits
    caps = list(hardware.kappa)
    if n > sum(caps):
        raise ValueError(f"cannot place {n} qubits into capacities {caps}")
    k = len(caps)
    weights = interaction_graph(instance)
    adj = _build_adj(n, weights)
    rng = np.random.default_rng(seed)

    def aff(q: int, part: int, assign) -> int:
        return sum(w for nbr, w in adj[q].items() if assign[nbr] == part)

    # ---- greedy initialization (highest-interaction qubits first) ----------
    total_w = [sum(adj[q].values()) for q in range(n)]
    order = sorted(range(n), key=lambda q: (-total_w[q], q))
    assign = [-1] * n
    load = [0] * k
    for q in order:
        cands = [u for u in range(k) if load[u] < caps[u]]
        best = max(cands, key=lambda u: (aff(q, u, assign), -u))
        assign[q] = best
        load[best] += 1

    def cut_cost(a) -> int:
        return sum(w for (x, y), w in weights.items() if a[x] != a[y])

    def move_delta(q: int, dst: int) -> int:
        return aff(q, assign[q], assign) - aff(q, dst, assign)

    def swap_delta(qa: int, qb: int) -> int:
        pa, pb = assign[qa], assign[qb]
        d_a = aff(qa, pb, assign) - aff(qa, pa, assign)
        d_b = aff(qb, pa, assign) - aff(qb, pb, assign)
        return -(d_a + d_b - 2 * adj[qa].get(qb, 0))

    cur = cut_cost(assign)
    best_assign, best_cost = list(assign), cur

    # initial temperature from sampled proposal magnitudes
    samples = []
    for _ in range(64):
        qa, qb = rng.integers(n), rng.integers(n)
        if assign[qa] != assign[qb]:
            samples.append(abs(swap_delta(int(qa), int(qb))))
    t0 = max(1.0, float(np.mean(samples)) if samples else 1.0)

    iters_per_stage = max(1, budget // max(1, stages))
    for stage in range(stages):
        temp = t0 * (0.5 ** stage)
        cool = (0.01) ** (1.0 / iters_per_stage)      # within-stage geometric
        for _ in range(iters_per_stage):
            temp *= cool
            if rng.random() < 0.5:                    # single move
                q = int(rng.integers(n))
                dst = int(rng.integers(k))
                if dst == assign[q] or load[dst] >= caps[dst]:
                    continue
                delta = move_delta(q, dst)
                if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                    load[assign[q]] -= 1
                    load[dst] += 1
                    assign[q] = dst
                    cur += delta
            else:                                     # pairwise swap
                qa, qb = int(rng.integers(n)), int(rng.integers(n))
                if assign[qa] == assign[qb]:
                    continue
                delta = swap_delta(qa, qb)
                if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                    assign[qa], assign[qb] = assign[qb], assign[qa]
                    cur += delta
            if cur < best_cost:
                best_cost, best_assign = cur, list(assign)

        # ---- hybrid local-search descent between stages --------------------
        improved = True
        while improved:
            improved = False
            for q in range(n):
                for dst in range(k):
                    if dst != assign[q] and load[dst] < caps[dst]:
                        delta = move_delta(q, dst)
                        if delta < 0:
                            load[assign[q]] -= 1
                            load[dst] += 1
                            assign[q] = dst
                            cur += delta
                            improved = True
            for qa in range(n):
                for qb in range(qa + 1, n):
                    if assign[qa] != assign[qb]:
                        delta = swap_delta(qa, qb)
                        if delta < 0:
                            assign[qa], assign[qb] = assign[qb], assign[qa]
                            cur += delta
                            improved = True
        if cur < best_cost:
            best_cost, best_assign = cur, list(assign)

    return best_assign


def make_mhsa_policy(seed: int = 0, budget: int = DEFAULT_BUDGET):
    """MHSA placement + the shared §9.1 list scheduler / JIT provisioning."""
    from .greedy_jit import GreedyJITPolicy
    return GreedyJITPolicy(
        placement_fn=lambda inst, hw: mhsa_placement(inst, hw, seed=seed,
                                                     budget=budget),
        name="mhsa_ls")
