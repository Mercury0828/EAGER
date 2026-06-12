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
from .partition import balanced_partition, interaction_graph, placement_order


def map_emission_order(instance: CircuitInstance) -> list[int]:
    """Experts emit Map actions in the partitioner's own greedy order (D56);
    cached on the instance (static)."""
    cached = getattr(instance, "_eager_map_order", None)
    if cached is None:
        cached = placement_order(instance.num_qubits,
                                 interaction_graph(instance))
        object.__setattr__(instance, "_eager_map_order", cached)
    return cached


def compute_placement(instance: CircuitInstance, hardware: HardwareConfig,
                      seed: int = 0) -> list[int]:
    return balanced_partition(
        num_items=instance.num_qubits,
        caps=list(hardware.kappa),
        weights=interaction_graph(instance),
        seed=seed,
    )


def greedy_schedule_or_gen(env: EagerEnv) -> Action | None:
    """The §9.1 scheduling + JIT-provisioning micro-step shared by the
    static expert and the conditional (DAgger-labeling) expert: schedule the
    most critical valid ready gate, else saturate deficit links, else None."""
    crit = env.instance.criticality
    for g in sorted(env.ready_gates(), key=lambda g: (-crit[g], g)):
        if env.is_valid(Schedule(g)):
            return Schedule(g)
    demand, max_crit = env.deficit_demand()
    for l in sorted((l for l in range(env.hardware.num_links)
                     if demand[l] > 0),
                    key=lambda l: (-(max_crit[l] or 0), l)):
        if env.is_valid(GenEPR(l)):
            return GenEPR(l)
    return None


class ConditionalGreedyJIT:
    """State-conditioned expert for DAgger labeling (D55): identical
    scheduling/provisioning to GreedyJIT, but Map decisions come from a
    COMPLETION partition — the current (possibly off-expert) partial mapping
    is pinned and the remaining qubits are partitioned around it — so the
    labeled action is always valid and demonstrates recovery from agent
    mistakes."""

    name = "conditional_greedy_jit"

    def __init__(self, placement_seed: int = 0):
        self.placement_seed = placement_seed
        self._completion: dict | None = None     # {premise: tuple, plan: list}

    def __call__(self, env: EagerEnv) -> Action:
        if env._unmapped:
            premise = tuple(-1 if u is None else u for u in env.qubit_qpu)
            if self._completion is None or self._completion["premise"] != premise:
                pinned = {q: u for q, u in enumerate(env.qubit_qpu)
                          if u is not None}
                plan = balanced_partition(
                    env.instance.num_qubits, list(env.hardware.kappa),
                    interaction_graph(env.instance),
                    seed=self.placement_seed, preassigned=pinned)
                self._completion = {"premise": premise, "plan": plan}
            q = next(q for q in map_emission_order(env.instance)
                     if q in env._unmapped)
            return Map(q, self._completion["plan"][q])
        action = greedy_schedule_or_gen(env)
        return action if action is not None else ADVANCE


class GreedyJITPolicy:
    """Callable policy: action = policy(env). The placement is computed once
    per policy instance (static placement, A2) and reused across episodes of
    the same env config.

    ``placement_fn(instance, hardware) -> list[int]`` swaps the placement
    stage only (used by MHSA+LS and AGG, §9.2/§9.3), so every placement
    baseline shares the identical list-scheduling + JIT-provisioning loop and
    comparisons isolate placement quality."""

    def __init__(self, placement_seed: int = 0, placement_fn=None,
                 name: str = "greedy_jit"):
        self.placement_seed = placement_seed
        self.placement_fn = placement_fn
        self.name = name
        self._placement: list[int] | None = None

    def __call__(self, env: EagerEnv) -> Action:
        if self._placement is None:
            if self.placement_fn is not None:
                self._placement = list(self.placement_fn(env.instance,
                                                         env.hardware))
            else:
                self._placement = compute_placement(
                    env.instance, env.hardware, self.placement_seed)

        # 1. placement, emitted in the partitioner's greedy order (D56;
        # placement and J are order-invariant — all maps land in slot 0)
        for q in map_emission_order(env.instance):
            if q in env._unmapped:
                return Map(q, self._placement[q])

        # 2./3. shared list scheduling + saturating JIT provisioning (D33)
        action = greedy_schedule_or_gen(env)
        return action if action is not None else ADVANCE
