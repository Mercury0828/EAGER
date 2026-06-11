"""Synthetic random circuit generator (guide §10.1).

Draws M two-qubit gates, each over a uniformly random unordered pair of
distinct qubits. The gate list order defines the circuit's temporal order;
per-qubit serialization (enforced in :func:`eager.circuit.instance_from_gates`)
derives the dependency DAG, so the per-qubit total order required by the guide
holds by construction.

The generator RNG (numpy PCG64) is strictly separate from the environment's
CRN engine (guide §12: env RNG separate from any other RNG).
"""

from __future__ import annotations

import numpy as np

from ..circuit import CircuitInstance, instance_from_gates
from ..config import SynthParams


def generate_gates(num_qubits: int, num_gates: int,
                   rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    gates = []
    for _ in range(num_gates):
        a, b = rng.choice(num_qubits, size=2, replace=False)
        gates.append((int(a), int(b)))
    return tuple(gates)


def generate_instance(params: SynthParams, seed: int,
                      name: str | None = None) -> CircuitInstance:
    rng = np.random.default_rng(seed)
    gates = generate_gates(params.num_qubits, params.num_gates, rng)
    label = name or f"synthetic_n{params.num_qubits}_m{params.num_gates}"
    return instance_from_gates(f"{label}_s{seed}", params.num_qubits, gates)
