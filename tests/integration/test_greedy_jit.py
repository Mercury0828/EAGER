"""Phase 2: GreedyJIT expert — placement quality on the golden micro, exact
hand-checked episode, and invariant-checked runs across modes."""

import pytest
from util_invariants import run_checked_episode

from eager.baselines.greedy_jit import GreedyJITPolicy, compute_placement
from eager.baselines.partition import cut_weight, interaction_graph
from eager.circuit import build_instance, instance_from_gates
from eager.config import SynthParams, load_circuit_config, load_hardware_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_instance


def test_placement_micro1_min_cut(hardware_dir, circuits_dir):
    """golden_micro_1 interaction graph: w(q0,q1)=2, w(q1,q2)=1 on kappa=[2,2]
    -> the only min-cut placement puts {q0,q1} together (cut=1)."""
    hw = load_hardware_config(hardware_dir / "golden_k2_det.yaml")
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    placement = compute_placement(inst, hw, seed=0)
    assert placement[0] == placement[1] != placement[2]
    assert cut_weight(placement, interaction_graph(inst)) == 1


def test_greedy_jit_micro1_exact_episode(hardware_dir, circuits_dir):
    """Hand-checked GreedyJIT timeline on golden micro 1 (det, t_ep=2):

      slot 0: maps {q0,q1}->u0, q2->u1; Schedule(g0) [local]; g1 not ready
              -> no deficit -> ADVANCE. resolve: g0 done, g1 ready@1.
      slot 1: g1 remote blocked (no pair) -> deficit on l0 -> GenEPR; ADVANCE.
      slot 2: pair lands at the END of slot 2 (tasked slot 1, t_ep=2)
              -> still blocked, deficit covered by busy channel -> ADVANCE.
      slot 3: Schedule(g1) consumes (-1); ADVANCE. slot 4: ADVANCE (g1 runs).
      slot 5: Schedule(g2) [local]; ADVANCE -> done.

      T=6, C_comm=1, C_waste=0 -> J=7; the JIT latency penalty vs. the
      proactive golden schedule (J=6) is exactly one slot of generation
      latency it could not hide.
    """
    hw = load_hardware_config(hardware_dir / "golden_k2_det.yaml")
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    env = EagerEnv(hw, inst)
    m = run_checked_episode(env, GreedyJITPolicy(placement_seed=0), seed=0)
    assert m["T"] == 6 and m["C_comm"] == 1.0 and m["C_waste"] == 0.0
    assert m["J"] == 7.0 and not m["truncated"]


@pytest.mark.stochastic
def test_greedy_jit_invariants_stochastic_k4():
    hw = default_panel_hardware(num_qubits=12)
    inst = generate_instance(SynthParams(12, 36, None), seed=4)
    env = EagerEnv(hw, inst)
    for seed in (0, 1):
        m = run_checked_episode(env, GreedyJITPolicy(placement_seed=0), seed=seed)
        assert m["done"] and not m["truncated"]
        p = m["pairs"]
        assert p["generated"] == p["consumed"] + p["expired"] + p["stored"]


@pytest.mark.stochastic
def test_greedy_jit_multi_hop_line():
    hw = load_hardware_config(
        {"name": "k3s", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": "stochastic",
         "link_defaults": {"p": 0.3, "W": 1, "B": 2, "T_cut": 6, "w": 1.0}})
    inst = instance_from_gates(
        "hop", 5, ((0, 1), (2, 3), (1, 2), (3, 4), (0, 4)))
    env = EagerEnv(hw, inst)
    m = run_checked_episode(env, GreedyJITPolicy(placement_seed=0), seed=3)
    assert m["done"] and not m["truncated"]


def test_placement_reused_across_episodes():
    hw = default_panel_hardware(num_qubits=8)
    inst = generate_instance(SynthParams(8, 16, None), seed=2)
    env = EagerEnv(hw, inst)
    policy = GreedyJITPolicy(placement_seed=0)
    run_checked_episode(env, policy, seed=0)
    first = list(policy._placement)
    run_checked_episode(env, policy, seed=1)
    assert policy._placement == first, "static placement cached per config"
