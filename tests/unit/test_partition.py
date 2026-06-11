"""Phase 2: capacity-constrained balanced partitioner (METIS-style contract,
pure-Python implementation per D29)."""

import numpy as np
import pytest

from eager.baselines.partition import balanced_partition, cut_weight, interaction_graph
from eager.circuit import instance_from_gates


def loads(assign, k):
    out = [0] * k
    for u in assign:
        out[u] += 1
    return out


def test_interaction_graph_counts():
    inst = instance_from_gates("g", 3, ((0, 1), (1, 2), (0, 1)))
    assert interaction_graph(inst) == {(0, 1): 2, (1, 2): 1}


def test_capacity_respected_and_all_assigned():
    rng = np.random.default_rng(0)
    weights = {(int(a), int(b)): 1
               for a, b in (sorted(rng.choice(12, 2, replace=False))
                            for _ in range(40))}
    caps = [4, 4, 4, 4]
    assign = balanced_partition(12, caps, weights, seed=1)
    assert len(assign) == 12 and all(0 <= u < 4 for u in assign)
    assert all(l <= c for l, c in zip(loads(assign, 4), caps))


def test_deterministic_per_seed():
    weights = {(a, b): (a + b) % 3 + 1 for a in range(10) for b in range(a + 1, 10)}
    a1 = balanced_partition(10, [5, 5], weights, seed=7)
    a2 = balanced_partition(10, [5, 5], weights, seed=7)
    assert a1 == a2


def test_chain_optimal_cut_with_slack():
    # path 0-1-2-3-4-5: optimal 2-way cut = 1 (a single chain edge)
    weights = {(i, i + 1): 1 for i in range(5)}
    assign = balanced_partition(6, [4, 4], weights, seed=0)
    assert cut_weight(assign, weights) == 1


def test_complete_graph_balanced_cut():
    # K4 complete, unit weights, caps [2,2]: every balanced split cuts 4
    weights = {(a, b): 1 for a in range(4) for b in range(a + 1, 4)}
    assign = balanced_partition(4, [2, 2], weights, seed=0)
    assert loads(assign, 2) == [2, 2]
    assert cut_weight(assign, weights) == 4


def test_two_clusters_separated():
    # two unit-weight triangles joined by one weak edge -> cut = 1
    weights = {(0, 1): 3, (1, 2): 3, (0, 2): 3,
               (3, 4): 3, (4, 5): 3, (3, 5): 3, (2, 3): 1}
    assign = balanced_partition(6, [3, 3], weights, seed=0)
    assert cut_weight(assign, weights) == 1
    assert assign[0] == assign[1] == assign[2]
    assert assign[3] == assign[4] == assign[5]


def test_refinement_beats_sequential_fill():
    rng = np.random.default_rng(3)
    n, k = 24, 4
    weights = {}
    for _ in range(120):
        a, b = sorted(int(x) for x in rng.choice(n, 2, replace=False))
        weights[(a, b)] = weights.get((a, b), 0) + 1
    caps = [8] * k                      # 1.33x slack like the panel sizing
    assign = balanced_partition(n, caps, weights, seed=0)
    sequential = [min(q // 6, k - 1) for q in range(n)]
    assert cut_weight(assign, weights) <= cut_weight(sequential, weights)


def test_infeasible_caps_rejected():
    with pytest.raises(ValueError, match="capacities"):
        balanced_partition(5, [2, 2], {(0, 1): 1}, seed=0)
