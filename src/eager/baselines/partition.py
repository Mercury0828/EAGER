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


def placement_order(num_items: int,
                    weights: dict[tuple[int, int], int]) -> list[int]:
    """The greedy-growth visit order (highest total interaction weight first,
    tie: lowest id) — shared by the partitioner AND by the experts' Map
    emission order (D56): mapping along this order makes every Map decision a
    LOCAL affinity readout (place q with its already-placed neighbors), which
    message passing can actually see, instead of a stepwise readout of a
    global partition solution."""
    totals = [0] * num_items
    for (a, b), w in weights.items():
        totals[a] += w
        totals[b] += w
    return sorted(range(num_items), key=lambda q: (-totals[q], q))


def balanced_partition(num_items: int, caps: list[int],
                       weights: dict[tuple[int, int], int],
                       seed: int = 0, max_passes: int = 10,
                       preassigned: dict[int, int] | None = None) -> list[int]:
    """Assign items 0..num_items-1 to parts with |part u| <= caps[u],
    minimizing the weighted cut. Deterministic given seed.

    ``preassigned`` pins items to parts (completion mode, D55): pinned items
    seed the loads and never move during refinement — used by the
    conditional expert that must complete an arbitrary partial mapping."""
    k = len(caps)
    if num_items > sum(caps):
        raise ValueError(f"cannot place {num_items} items into capacities "
                         f"{caps} (total {sum(caps)})")
    preassigned = preassigned or {}
    for q, u in preassigned.items():
        if not 0 <= u < k:
            raise ValueError(f"preassigned item {q} -> invalid part {u}")
    rng = np.random.default_rng(seed)

    adj: list[dict[int, int]] = [dict() for _ in range(num_items)]
    for (a, b), w in weights.items():
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w

    order = [q for q in placement_order(num_items, weights)
             if q not in preassigned]

    assign = [-1] * num_items
    load = [0] * k
    for q, u in preassigned.items():
        assign[q] = u
        load[u] += 1
    if any(load[u] > caps[u] for u in range(k)):
        raise ValueError(f"preassignment violates capacities {caps}")

    def part_affinity(q: int, u: int) -> int:
        return sum(w for nbr, w in adj[q].items() if assign[nbr] == u)

    for q in order:
        candidates = [u for u in range(k) if load[u] < caps[u]]
        # max affinity to already-placed neighbors; tie -> LOWEST part id
        # with room (sequential fill). The tie-break is cut-neutral but not
        # burst-neutral: most-residual round-robins zero-affinity qubits
        # across parts, shredding the consecutive remote runs that
        # Autocomm-style aggregation consumes; sequential fill matches the
        # contiguous-register-split mapping convention of the DQC
        # literature (D41).
        best = max(candidates, key=lambda u: (part_affinity(q, u), -u))
        assign[q] = best
        load[best] += 1

    # Refinement: FM-style single moves (need residual capacity) plus
    # Kernighan-Lin pairwise swaps (work at tight capacity), seeded order.
    free = [q for q in range(num_items) if q not in preassigned]

    def single_move_pass() -> bool:
        improved = False
        visit = list(free)
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
        for ia, a in enumerate(free):
            for b in free[ia + 1:]:
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
