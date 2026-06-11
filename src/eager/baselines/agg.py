"""AGG — Autocomm-style burst-communication aggregation (guide §9.3;
MICRO'22 top-venue anchor; public artifact checked and followed, see
docs/BASELINE_FIDELITY.md).

Pipeline (controlled comparison: same placement + same scheduler as §9.1):

1. Place qubits with the SAME partitioner as GreedyJIT.
2. Detect bursts a la the artifact's ``consecutive_merge``: maximal runs of
   remote gates between the same QPU pair sharing one operand qubit,
   consecutive in the shared qubit's serialization chain (our skeletons have
   no 1q interleavers; commutation-based block extension has no counterpart
   in our serialization-frozen DAG — disclosed deviation).
3. Cat-comm transform: a burst (q,x1)(q,x2)...(q,xk) with q on u and all
   x_i on v becomes
       (q,x1)            remote head: cat-entangle + first gate
                         (d_rem slots, ONE pair per route link)
       (x1,x2)..(x1,xk)  local tail on v, anchored at x1 (the cat copy's
                         interaction site), serialized through x1
   Durations: d_rem + (k-1)*d_loc = k+1 slots vs 2k unaggregated; pairs:
   1*route vs k*route. Depth of the burst region is unchanged (k).
4. Execute the transformed instance with GreedyJITPolicy under the SAME
   placement (qubit ids unchanged by the transform).

Known modeling concessions (each in BASELINE_FIDELITY): the shared qubit is
released after the head completes rather than after disentangle (affects T
only, never pairs); bursts break when the anchor x1 reappears as the other
operand (the rewrite (x1,x1) would be invalid); direction of CNOTs is
abstract in 2q skeletons, so sharing is side-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..circuit import CircuitInstance, instance_from_gates
from ..config import HardwareConfig
from .greedy_jit import GreedyJITPolicy, compute_placement


@dataclass(frozen=True)
class Burst:
    shared: int                 # chain qubit q (stays on the source QPU)
    anchor: int                 # x1: other operand of the head gate
    target_qpu: int             # QPU hosting every other operand
    gates: tuple[int, ...]      # gate ids, len >= 2, in chain order


def detect_bursts(instance: CircuitInstance, placement: list[int]
                  ) -> list[Burst]:
    """Candidate runs per qubit chain, then greedy non-overlapping selection
    (longest first, then earliest head)."""
    chains: list[list[int]] = [[] for _ in range(instance.num_qubits)]
    for g, (a, b) in enumerate(instance.gates):
        chains[a].append(g)
        chains[b].append(g)

    candidates: list[Burst] = []
    for q in range(instance.num_qubits):
        run: list[int] = []
        run_v = -1
        anchor = -1

        def flush():
            if len(run) >= 2:
                candidates.append(Burst(shared=q, anchor=anchor,
                                        target_qpu=run_v, gates=tuple(run)))

        for g in chains[q]:
            a, b = instance.gates[g]
            other = b if a == q else a
            if placement[a] == placement[b]:        # local gate breaks the run
                flush()
                run, run_v, anchor = [], -1, -1
                continue
            v = placement[other]
            if not run:
                run, run_v, anchor = [g], v, other
            elif v == run_v and other != anchor:
                run.append(g)
            else:
                # different target QPU, or anchor reappears as the other
                # operand (rewrite (x1,x1) would be invalid): start a new run
                flush()
                run, run_v, anchor = [g], v, other
        flush()

    candidates.sort(key=lambda bu: (-len(bu.gates), bu.gates[0]))
    used: set[int] = set()
    selected: list[Burst] = []
    for bu in candidates:
        if not any(g in used for g in bu.gates):
            selected.append(bu)
            used.update(bu.gates)
    selected.sort(key=lambda bu: bu.gates[0])
    return selected


def transform_instance(instance: CircuitInstance, placement: list[int]
                       ) -> tuple[CircuitInstance, dict]:
    """Rewrite burst tails as local gates anchored at the head's partner."""
    bursts = detect_bursts(instance, placement)
    rewrite: dict[int, tuple[int, int]] = {}
    for bu in bursts:
        for g in bu.gates[1:]:
            a, b = instance.gates[g]
            other = b if a == bu.shared else a
            rewrite[g] = (bu.anchor, other)

    gates = [rewrite.get(g, instance.gates[g])
             for g in range(instance.num_gates)]
    transformed = instance_from_gates(f"{instance.name}+agg",
                                      instance.num_qubits, tuple(gates))
    stats = {
        "n_bursts": len(bursts),
        "gates_aggregated": sum(len(bu.gates) - 1 for bu in bursts),
        "burst_lengths": sorted((len(bu.gates) for bu in bursts),
                                reverse=True),
    }
    return transformed, stats


def make_agg_method(instance: CircuitInstance, hardware: HardwareConfig,
                    placement_seed: int = 0):
    """Return (transformed_instance, policy, placement, stats). Build the env
    on the TRANSFORMED instance; the policy reuses the §9.1 scheduler with
    the SAME placement (qubit ids are unchanged by the transform)."""
    placement = compute_placement(instance, hardware, seed=placement_seed)
    transformed, stats = transform_instance(instance, placement)
    policy = GreedyJITPolicy(placement_fn=lambda _inst, _hw: placement,
                             name="agg_ls")
    return transformed, policy, placement, stats
