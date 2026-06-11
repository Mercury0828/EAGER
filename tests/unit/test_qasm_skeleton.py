"""Phase 2: OPENQASM 2q-skeleton extractor — crafted-program exactness, the
ccx/custom-gate expansions, and consistency between qasm/ sources and the
committed extracted YAMLs."""

import pytest

from eager.circuit import build_instance
from eager.config import ConfigError, load_circuit_config
from eager.expgen.qasm_skeleton import instance_from_qasm, parse_qasm_skeleton
from eager.expgen.synthetic import generate_layered_random_instance

CRAFTED = """
OPENQASM 2.0;
include "qelib1.inc";
gate foo a,b { cx a,b; h a; cx b,a; }
qreg q[3];
qreg anc[1];
creg c[3];
h q[0];
cx q[0],q[1];
ccx q[0],q[1],q[2];
foo q[2],anc[0];
swap q[1],q[2];
barrier q;
measure q[0] -> c[0];
"""


def test_crafted_program_exact_pairs():
    n, gates = parse_qasm_skeleton(CRAFTED, "crafted")
    assert n == 4                       # q[3] flattened 0..2, anc[0] -> 3
    assert gates == (
        (0, 1),                          # cx q0,q1
        (1, 2), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1),   # ccx -> 6 CNOTs
        (2, 3), (3, 2),                  # foo body: cx a,b ; cx b,a (h dropped)
        (1, 2),                          # swap = one 2q instruction (D31)
    )


def test_unsupported_constructs_raise():
    with pytest.raises(ConfigError, match="unsupported gate"):
        parse_qasm_skeleton("qreg q[2]; mystery q[0],q[1];", "x")
    with pytest.raises(ConfigError, match="whole-register"):
        parse_qasm_skeleton("qreg q[2]; cx q,q;", "x")
    with pytest.raises(ConfigError, match="conditionals"):
        parse_qasm_skeleton("qreg q[2]; creg c[1]; if(c==1) cx q[0],q[1];", "x")
    with pytest.raises(ConfigError, match="no two-qubit gates"):
        parse_qasm_skeleton("qreg q[2]; h q[0];", "x")
    with pytest.raises(ConfigError, match="twice"):
        parse_qasm_skeleton("qreg q[2]; cx q[1],q[1];", "x")


def test_cswap_expansion_count():
    n, gates = parse_qasm_skeleton("qreg q[3]; cswap q[0],q[1],q[2];", "x")
    assert n == 3 and len(gates) == 8   # cx + 6 (ccx) + cx


def test_shipped_qasm_matches_committed_yaml(repo_root):
    """Every committed extracted YAML must equal a fresh extraction of its
    qasm source (guards drift between qasm/ and configs/)."""
    qasm_dir = repo_root / "qasm" / "qasmbench"
    out_dir = repo_root / "configs" / "circuits" / "qasmbench"
    qasm_files = sorted(qasm_dir.glob("*.qasm"))
    assert qasm_files, "qasm/qasmbench is empty"
    for qasm_path in qasm_files:
        fresh = instance_from_qasm(qasm_path)
        committed = build_instance(
            load_circuit_config(out_dir / f"{qasm_path.stem}.yaml"))
        assert committed.num_qubits == fresh.num_qubits, qasm_path.name
        assert committed.gates == fresh.gates, qasm_path.name


def test_supremacy_instance_matches_committed_yaml(repo_root):
    fresh = generate_layered_random_instance(num_qubits=120, num_layers=10,
                                             seed=2027)
    committed = build_instance(load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / "supremacy_n120.yaml"))
    assert committed.gates == fresh.gates
    assert fresh.num_gates == 600 and fresh.depth == 10
    # perfect matching per layer: every qubit appears exactly once per layer
    for layer in range(10):
        seen = set()
        for a, b in fresh.gates[layer * 60:(layer + 1) * 60]:
            assert a not in seen and b not in seen
            seen.update((a, b))
        assert len(seen) == 120
