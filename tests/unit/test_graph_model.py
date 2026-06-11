"""Phase 5: state-graph builder exactness (guide §6.2) and the
encoder/decoder stack (guide §7) — segment-softmax correctness, batching
consistency (batched logits == single-graph logits), value head shapes."""

import numpy as np
import pytest
import torch

from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import ADVANCE, EagerEnv, GenEPR, Map, Schedule
from eager.model.encoder import BatchedGraphs
from eager.model.graph import NUM_RELATIONS, build_graph
from eager.model.policy import EagerPolicy, build_action_set

DEVICE = torch.device("cpu")


def golden_env(hardware_dir, circuits_dir):
    hw = load_hardware_config(
        {"name": "g", "qpus": 2, "topology": "line", "kappa": [2, 2],
         "mode": "stochastic",
         "link_defaults": {"p": 0.5, "W": 2, "B": 4, "T_cut": 8, "w": 1.0}})
    inst = build_instance(
        load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    return EagerEnv(hw, inst)


def test_graph_builder_exactness(hardware_dir, circuits_dir):
    env = golden_env(hardware_dir, circuits_dir)
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 0))
    env.step(Map(2, 1))
    env.step(Schedule(0))                       # g0 local, leaves the graph
    env.step(GenEPR(0))
    env.step(ADVANCE)                           # g0 done; pair lands (p=.5? seed 0)

    snap = build_graph(env)
    # gate nodes: only unscheduled gates (g1, g2)
    assert snap.gate_ids.tolist() == [1, 2]
    g1 = snap.gate_row[1]
    # g1=(1,2) is remote (q1@u0, q2@u1): one-hot remote, ready (g0 done)
    assert snap.x_gate[g1, 2] == 1.0 and snap.x_gate[g1, 4] == 1.0
    g2 = snap.gate_row[2]
    # g2 has 1 unfinished pred (g1) -> 0.5; not ready
    assert snap.x_gate[g2, 5] == pytest.approx(0.5)
    assert snap.x_gate[g2, 4] == 0.0
    # qubits all mapped
    assert all(snap.x_qubit[q, 0] == 1.0 for q in range(3))
    # QPU 0 hosts q0,q1 with kappa 2 -> kappa_res 0, mapped 1.0
    assert snap.x_qpu[0, 0] == 0.0 and snap.x_qpu[0, 1] == 1.0
    # link features: p, channels, buffer occupancy
    assert snap.x_link[0, 0] == pytest.approx(0.5)
    ls = env.links[0]
    assert snap.x_link[0, 2] == pytest.approx(ls.stored / 4)
    # age-bucket mass equals stored/B
    assert snap.x_link[0, 4:8].sum() == pytest.approx(ls.stored / 4)
    # pending demand: g1 ready+remote through link 0, 1 ready gate total
    assert snap.x_link[0, 8] == pytest.approx(1.0)
    # relations present and well-formed
    assert snap.edge_type.max() < NUM_RELATIONS
    assert snap.edge_index.max() < snap.num_nodes
    # routed-through edges exist for the remote gate
    assert (snap.edge_type == 7).sum() == 1 and (snap.edge_type == 8).sum() == 1
    # globals
    assert snap.globals[2] == pytest.approx(1.0)        # all qubits mapped
    assert snap.globals[1] == pytest.approx(1 / 3)      # 1 of 3 gates done


def test_policy_distribution_and_batching(hardware_dir, circuits_dir):
    torch.manual_seed(0)
    env1 = golden_env(hardware_dir, circuits_dir)
    env1.reset(0)
    env2 = golden_env(hardware_dir, circuits_dir)
    env2.reset(1)
    env2.step(Map(0, 0))
    env2.step(Map(1, 1))

    s1, s2 = build_graph(env1), build_graph(env2)
    a1, a2 = build_action_set(env1, s1), build_action_set(env2, s2)
    policy = EagerPolicy(hidden=32)
    policy.eval()

    with torch.no_grad():
        out_b = policy(BatchedGraphs([s1, s2], DEVICE), [a1, a2])
        out_1 = policy(BatchedGraphs([s1], DEVICE), [a1])

    # probabilities sum to 1 per graph; entropy nonnegative
    logp = out_b.log_softmax()
    p_sum0 = logp[: len(a1.actions)].exp().sum().item()
    p_sum1 = logp[len(a1.actions):].exp().sum().item()
    assert p_sum0 == pytest.approx(1.0, abs=1e-5)
    assert p_sum1 == pytest.approx(1.0, abs=1e-5)
    assert (out_b.entropy() >= -1e-6).all()

    # batched logits for graph 1 == single-graph logits (flattening is sound)
    assert torch.allclose(out_b.logits[: len(a1.actions)], out_1.logits[0:len(a1.actions)],
                          atol=1e-4)
    assert torch.allclose(out_b.value[0], out_1.value[0], atol=1e-4)

    # greedy/sampled positions are valid indices
    g = out_b.greedy()
    assert 0 <= int(g[0]) < len(a1.actions)
    assert 0 <= int(g[1]) < len(a2.actions)
    gen = torch.Generator().manual_seed(3)
    s = out_b.sample(generator=gen)
    assert 0 <= int(s[0]) < len(a1.actions)
    assert 0 <= int(s[1]) < len(a2.actions)

    # log_prob_of agrees with manual log-softmax lookup
    pos = torch.tensor([1, 0])
    lp = out_b.log_prob_of(pos)
    assert lp[0].item() == pytest.approx(logp[1].item(), abs=1e-5)
    assert lp[1].item() == pytest.approx(
        logp[len(a1.actions)].item(), abs=1e-5)


def test_action_set_matches_env_enumeration(hardware_dir, circuits_dir):
    env = golden_env(hardware_dir, circuits_dir)
    env.reset(0)
    snap = build_graph(env)
    aset = build_action_set(env, snap)
    assert aset.actions == env.valid_actions()
    assert aset.actions[-1] == ADVANCE
    assert aset.spec.shape == (len(aset.actions), 3)


def test_curriculum_unlock():
    from eager.train.curriculum import Curriculum
    cur = Curriculum()
    assert cur.record_eval(10, 9) == "A"        # loss resets the streak
    for _ in range(2):
        assert cur.record_eval(8, 9) == "A"
    assert cur.record_eval(8, 9) == "B"         # third consecutive win
