"""Capacity-constrained balanced k-way partitioning of the qubit interaction
graph (guide §9.1 step 1).

METIS/KaHyPar have no installable Windows wheels in this environment (pymetis
needs a C++ toolchain; see DESIGN_DECISIONS D29), so this is a deterministic
pure-Python stand-in with the same contract: minimize the weighted edge cut
of the interaction graph subject to per-part capacities. Algorithm: weighted
greedy growth (highest-interaction qubits first, placed where their already-
placed neighbors live) followed by Fiduccia–Mattheyses-style single-move
refinement passes. The hypergraph-partitioning lineage and this substitution
are disclosed in docs/BASELINE_FIDELITY.md.
"""

from __future__ import annotations

import numpy as np

from ..circuit import CircuitInstance


def interaction_graph(instance: CircuitInstance) -> dict[tuple[int, int], int]:
    """Edge weights: number of two-qubit gates between each qubit pair."""
    weights: dict[tuple[int, int], int] = {}
    for a, b in instance.gates:
        key = (a, b) if a < b else (b, a)
        weights[key] = weights.get(key, 0) + 1
    return weights


def cut_weight(assign: list[int], weights: dict[tuple[int, int], int]) -> int:
    return sum(w for (a, b), w in weights.items() if assign[a] != assign[b])


def balanced_partition(num_items: int, caps: list[int],
                       weights: dict[tuple[int, int], int],
                       seed: int = 0, max_passes: int = 10) -> list[int]:
    """Assign items 0..num_items-1 to parts with |part u| <= caps[u],
    minimizing the weighted cut. Deterministic given seed."""
    k = len(caps)
    if num_items > sum(caps):
        raise ValueError(f"cannot place {num_items} items into capacities "
                         f"{caps} (total {sum(caps)})")
    rng = np.random.default_rng(seed)

    adj: list[dict[int, int]] = [dict() for _ in range(num_items)]
    for (a, b), w in weights.items():
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w

    total_w = [sum(adj[q].values()) for q in range(num_items)]
    order = sorted(range(num_items), key=lambda q: (-total_w[q], q))

    assign = [-1] * num_items
    load = [0] * k

    def part_affinity(q: int, u: int) -> int:
        return sum(w for nbr, w in adj[q].items() if assign[nbr] == u)

    for q in order:
        candidates = [u for u in range(k) if load[u] < caps[u]]
        # max affinity to already-placed neighbors; tie -> most residual
        # capacity; tie -> lowest part id (all deterministic)
        best = max(candidates,
                   key=lambda u: (part_affinity(q, u), caps[u] - load[u], -u))
        assign[q] = best
        load[best] += 1

    # Refinement: FM-style single moves (need residual capacity) plus
    # Kernighan-Lin pairwise swaps (work at tight capacity), seeded order.
    def single_move_pass() -> bool:
        improved = False
        visit = list(range(num_items))
        rng.shuffle(visit)
        for q in visit:
            cur = assign[q]
            internal = part_affinity(q, cur)
            best_gain, best_u = 0, cur
            for u in range(k):
                if u == cur or load[u] >= caps[u]:
                    continue
                gain = part_affinity(q, u) - internal
                if gain > best_gain or (gain == best_gain and gain > 0 and u < best_u):
                    best_gain, best_u = gain, u
            if best_gain > 0:
                load[cur] -= 1
                load[best_u] += 1
                assign[q] = best_u
                improved = True
        return improved

    def swap_pass() -> bool:
        # KL gain for swapping a<->b across parts:
        #   D_a + D_b - 2*w(a,b),  D_x = aff(other part) - aff(own part)
        improved = False
        for a in range(num_items):
            for b in range(a + 1, num_items):
                pa, pb = assign[a], assign[b]
                if pa == pb:
                    continue
                d_a = part_affinity(a, pb) - part_affinity(a, pa)
                d_b = part_affinity(b, pa) - part_affinity(b, pb)
                gain = d_a + d_b - 2 * adj[a].get(b, 0)
                if gain > 0:
                    assign[a], assign[b] = pb, pa
                    improved = True
        return improved

    for _ in range(max_passes):
        if not (single_move_pass() | swap_pass()):
            break
    return assign
