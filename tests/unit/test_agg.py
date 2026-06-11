"""Phase 3: AGG burst detection and cat-comm transform (guide §9.3, D40) —
crafted-instance exactness, transform identities, and zero-burst structure
of chain circuits."""

from eager.baselines.agg import detect_bursts, transform_instance
from eager.baselines.greedy_jit import compute_placement
from eager.circuit import build_instance, instance_from_gates
from eager.config import load_circuit_config
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_fanout_instance


def test_burst_detection_crafted():
    """q0 on QPU0; q1..q4 split across QPU1/QPU2:
       gates: (0,1)(0,2) -> remote run to QPU1 sharing q0, len 2  [burst]
              (0,3)      -> remote to QPU2, breaks the run (len 1, no burst)
              (0,4)(0,1) -> wait: q4 on QPU2 -> run continues? q1 on QPU1
       layout: placement = [0, 1, 1, 2, 2]
       chain of q0: (0,1)Q1 (0,2)Q1 (0,3)Q2 (0,4)Q2 (0,1)Q1
       runs: [(0,1),(0,2)] to Q1; [(0,3),(0,4)] to Q2; [(0,1)] len1.
    """
    inst = instance_from_gates(
        "crafted", 5, ((0, 1), (0, 2), (0, 3), (0, 4), (0, 1)))
    placement = [0, 1, 1, 2, 2]
    bursts = detect_bursts(inst, placement)
    assert [(b.shared, b.anchor, b.target_qpu, b.gates) for b in bursts] == [
        (0, 1, 1, (0, 1)),
        (0, 3, 2, (2, 3)),
    ]


def test_burst_breaks_on_local_gate_and_anchor_repeat():
    # local gate interrupts: (0,1)R (1,2)local-on... make q0's chain explicit
    inst = instance_from_gates("brk", 4, ((0, 1), (0, 2), (0, 1), (0, 2)))
    placement = [0, 1, 1, 1]
    # chain of q0: (0,1)(0,2)(0,1)(0,2) all remote to QPU1 sharing q0;
    # anchor=1; gate2 has other=1 == anchor -> run breaks, new run starts
    bursts = detect_bursts(inst, placement)
    assert [(b.anchor, b.gates) for b in bursts] == [(1, (0, 1)), (1, (2, 3))]


def test_transform_pairs_and_depth_identity():
    inst = instance_from_gates(
        "t", 5, ((0, 1), (0, 2), (0, 3), (0, 4), (0, 1)))
    placement = [0, 1, 1, 2, 2]
    transformed, stats = transform_instance(inst, placement)
    assert stats["n_bursts"] == 2 and stats["gates_aggregated"] == 2
    # heads keep (q, x1); tails become (x1, xi) local on the target QPU
    assert transformed.gates == ((0, 1), (1, 2), (0, 3), (3, 4), (0, 1))
    # remote-gate count under the placement: 5 -> 3
    def remote_count(i):
        return sum(placement[a] != placement[b] for a, b in i.gates)
    assert remote_count(inst) == 5 and remote_count(transformed) == 3
    assert transformed.num_qubits == inst.num_qubits
    assert transformed.num_gates == inst.num_gates
    # within a burst the tail stays serial (anchored chain), so aggregation
    # never deepens the DAG; ACROSS bursts the shared qubit is released
    # early (one-to-many cat copies, see BASELINE_FIDELITY), so global depth
    # may shrink: here 5 -> 3.
    assert transformed.depth <= inst.depth
    assert transformed.depth == 3


def test_chain_ghz_is_burst_free_under_min_cut(repo_root):
    """QASMBench chain-form ghz: min-cut placement leaves only isolated
    remote gates -> zero bursts (the structural D40 result)."""
    inst = build_instance(load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / "ghz_n78.yaml"))
    hw = default_panel_hardware(inst.num_qubits)
    placement = compute_placement(inst, hw, seed=0)
    assert detect_bursts(inst, placement) == []


def test_fanout_ghz_carries_bursts(repo_root):
    inst = build_instance(load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / "ghz_fanout_n78.yaml"))
    hw = default_panel_hardware(inst.num_qubits)
    placement = compute_placement(inst, hw, seed=0)
    bursts = detect_bursts(inst, placement)
    assert bursts, "fan-out GHZ must expose aggregatable bursts"
    transformed, stats = transform_instance(inst, placement)
    assert stats["gates_aggregated"] > 0

    def remote_count(i):
        return sum(placement[a] != placement[b] for a, b in i.gates)
    assert remote_count(transformed) < remote_count(inst)


def test_transform_identity_when_no_bursts():
    inst = instance_from_gates("chain", 4, ((0, 1), (1, 2), (2, 3)))
    placement = [0, 0, 1, 1]
    transformed, stats = transform_instance(inst, placement)
    assert stats["n_bursts"] == 0
    assert transformed.gates == inst.gates
