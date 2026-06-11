"""GreedyJIT — the self-contained expert + heuristic baseline (guide §9.1).

A policy over the env micro-action API (hard requirement: its traces live in
the agent's action vocabulary, §8.1):

1. **Placement**: balanced capacity-constrained partition of the qubit
   interaction graph (partition.py; METIS-style contract, see
   BASELINE_FIDELITY.md). Static per (instance, hardware, placement_seed);
   emitted as Map micro-actions, qubits in ascending order.
2. **Scheduling**: list scheduling — among ready gates in DESCENDING
   criticality (tie: ascending gate id): local gates schedule immediately;
   remote gates schedule iff every route link holds a stored pair.
3. **Provisioning (JIT)**: for the remaining blocked remote gates, issue
   GenEPR per link deficit (capped by free channels and buffer headroom),
   links ordered by the max criticality of the gates they block. Then
   ADVANCE.

Deterministic given (instance, hardware, placement_seed) and the env state.
"""

from __future__ import annotations

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from ..env.actions import ADVANCE, Action, GenEPR, Map, Schedule
from ..env.env import EagerEnv
from .partition import balanced_partition, interaction_graph


def compute_placement(instance: CircuitInstance, hardware: HardwareConfig,
                      seed: int = 0) -> list[int]:
    return balanced_partition(
        num_items=instance.num_qubits,
        caps=list(hardware.kappa),
        weights=interaction_graph(instance),
        seed=seed,
    )


class GreedyJITPolicy:
    """Callable policy: action = policy(env). The placement is computed once
    per policy instance (static placement, A2) and reused across episodes of
    the same env config."""

    name = "greedy_jit"

    def __init__(self, placement_seed: int = 0):
        self.placement_seed = placement_seed
        self._placement: list[int] | None = None

    def __call__(self, env: EagerEnv) -> Action:
        if self._placement is None:
            self._placement = compute_placement(
                env.instance, env.hardware, self.placement_seed)

        # 1. placement
        for q in sorted(env._unmapped):
            return Map(q, self._placement[q])

        # 2. list scheduling by descending criticality
        crit = env.instance.criticality
        for g in sorted(env.ready_gates(), key=lambda g: (-crit[g], g)):
            if env.is_valid(Schedule(g)):
                return Schedule(g)

        # 3. JIT provisioning: saturate free channels / buffer headroom on
        # links with blocked demand (literal §9.1; D33), most critical first
        demand, max_crit = env.deficit_demand()
        for l in sorted((l for l in range(env.hardware.num_links)
                         if demand[l] > 0),
                        key=lambda l: (-(max_crit[l] or 0), l)):
            if env.is_valid(GenEPR(l)):
                return GenEPR(l)

        return ADVANCE
