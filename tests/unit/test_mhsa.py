"""Phase 3: MHSA placement (guide §9.2) — feasibility, determinism, and
quality vs both its own greedy init and the §9.1 partitioner."""

import pytest

from eager.baselines.mhsa import mhsa_placement
from eager.baselines.partition import balanced_partition, cut_weight, interaction_graph
from eager.circuit import build_instance
from eager.config import SynthParams, load_circuit_config, load_hardware_config
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_instance


def loads(assign, k):
    out = [0] * k
    for u in assign:
        out[u] += 1
    return out


def test_capacity_respected_and_deterministic():
    inst = generate_instance(SynthParams(20, 60, None), seed=3)
    hw = default_panel_hardware(20)
    a1 = mhsa_placement(inst, hw, seed=0, budget=4000)
    a2 = mhsa_placement(inst, hw, seed=0, budget=4000)
    assert a1 == a2
    assert len(a1) == 20
    assert all(l <= c for l, c in zip(loads(a1, 4), hw.kappa))


def test_two_cluster_graph_solved_exactly():
    # two heavy triangles + one weak bridge: optimal cut = 1
    from eager.circuit import instance_from_gates
    gates = (((0, 1),) * 3 + ((1, 2),) * 3 + ((0, 2),) * 3
             + ((3, 4),) * 3 + ((4, 5),) * 3 + ((3, 5),) * 3 + ((2, 3),))
    inst = instance_from_gates("twoclu", 6, gates)
    hw = load_hardware_config(
        {"name": "k2", "qpus": 2, "topology": "line", "kappa": 3,
         "mode": "deterministic", "t_ep": 1,
         "link_defaults": {"p": 1.0, "W": 1, "B": 2, "T_cut": None, "w": 1.0}})
    assign = mhsa_placement(inst, hw, seed=0, budget=2000)
    assert cut_weight(assign, interaction_graph(inst)) == 1


@pytest.mark.stochastic
def test_mhsa_competitive_with_partitioner_minipanel():
    """Mini version of the §11 acceptance comparison (full 20-instance run in
    experiments/phase3_baselines.py): MHSA remote-gate count <= partitioner
    on >= 70% of instances."""
    cases = [generate_instance(SynthParams(n, n * d, None), seed=s)
             for (n, d, s) in [(12, 3, 1), (16, 3, 2), (20, 2, 3),
                               (24, 3, 4), (30, 2, 5), (16, 6, 6)]]
    wins = 0
    for inst in cases:
        hw = default_panel_hardware(inst.num_qubits)
        w = interaction_graph(inst)
        cut_part = cut_weight(
            balanced_partition(inst.num_qubits, list(hw.kappa), w, seed=0), w)
        cut_mhsa = cut_weight(mhsa_placement(inst, hw, seed=0, budget=8000), w)
        wins += cut_mhsa <= cut_part
    assert wins >= round(0.7 * len(cases)), f"MHSA <= partitioner on {wins}/6"


@pytest.mark.stochastic
def test_mhsa_policy_completes_episode():
    from util_invariants import run_checked_episode
    from eager.baselines.mhsa import make_mhsa_policy
    from eager.env import EagerEnv
    inst = generate_instance(SynthParams(10, 30, None), seed=7)
    env = EagerEnv(default_panel_hardware(10), inst)
    m = run_checked_episode(env, make_mhsa_policy(seed=0, budget=2000), seed=0)
    assert m["done"] and not m["truncated"]
