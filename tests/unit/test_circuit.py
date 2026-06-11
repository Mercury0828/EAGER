"""Phase 0: circuit instance building — per-qubit serialization DAG,
criticality, and the synthetic generator."""

import pytest

from eager.circuit import build_instance, instance_from_gates
from eager.config import ConfigError, SynthParams, load_circuit_config
from eager.expgen.synthetic import generate_instance


def test_golden_micro_1_dag(circuits_dir):
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    # g0=(0,1), g1=(1,2), g2=(0,1):
    #   g1 depends on g0 (last toucher of q1); g2 on g1 (q1) and g0 (q0)
    assert inst.preds == ((), (0,), (0, 1))
    assert inst.succs == ((1, 2), (2,), ())
    assert inst.criticality == (3, 2, 1)
    assert inst.depth == 3


def test_golden_micro_2_dag(circuits_dir):
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_2.yaml"))
    # g0=(0,2), g1=(1,3) independent; g2=(0,1) <- {g0,g1}; g3=(2,3) <- {g0,g1}
    assert inst.preds == ((), (), (0, 1), (0, 1))
    assert inst.succs == ((2, 3), (2, 3), (), ())
    assert inst.criticality == (2, 2, 1, 1)
    assert inst.depth == 2


def test_per_qubit_serialization_property():
    """For every qubit, consecutive gates touching it must be DAG-linked."""
    inst = generate_instance(SynthParams(num_qubits=12, num_gates=60, seed=3), seed=3)
    touchers = {q: [] for q in range(inst.num_qubits)}
    for g, (a, b) in enumerate(inst.gates):
        touchers[a].append(g)
        touchers[b].append(g)
    for q, gs in touchers.items():
        for earlier, later in zip(gs, gs[1:]):
            assert earlier in inst.preds[later], (
                f"qubit {q}: gate {later} must depend on its predecessor "
                f"{earlier} (per-qubit serialization)")


def test_generator_seeded_reproducibility():
    p = SynthParams(num_qubits=15, num_gates=45, seed=None)
    a = generate_instance(p, seed=11)
    b = generate_instance(p, seed=11)
    c = generate_instance(p, seed=12)
    assert a.gates == b.gates
    assert a.gates != c.gates
    assert all(x != y for (x, y) in a.gates)


def test_build_instance_seed_handling():
    cfg = load_circuit_config({"name": "s", "kind": "synthetic",
                               "params": {"num_qubits": 8, "density": 1}})
    with pytest.raises(ConfigError, match="seed"):
        build_instance(cfg)
    inst = build_instance(cfg, seed=5)
    assert inst.num_gates == 8


def test_instance_from_gates_rejects_bad_gate():
    with pytest.raises(ConfigError, match="valid"):
        instance_from_gates("bad", 2, ((0, 0),))


def test_dag_edges_and_depth_chain():
    inst = instance_from_gates("chain", 2, ((0, 1), (0, 1), (0, 1)))
    assert inst.dag_edges == ((0, 1), (1, 2))
    assert inst.criticality == (3, 2, 1)
