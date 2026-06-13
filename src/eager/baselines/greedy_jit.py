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


class GreedyAdaptivePolicy:
    """GreedyAdaptive (D76): GreedyJIT placement + list scheduling +
    BOUNDED-LOOKAHEAD demand-aware provisioning that resolves the eager-vs-
    lazy tradeoff the regime map (D75) exposed. For each link it provisions
    proactively for remote gates whose number of unfinished predecessors is
    <= K (gates that will become ready 'soon'), capping stored+in-flight at
    that imminent-demand count so it does not over-generate pairs that would
    expire before use.

    K adapts to the regime: a pair generated now arrives in ~1/p slots and
    survives T_cut slots, so it is useful only for gates that become ready
    within ~(1/p + T_cut) slots; in DAG terms K ~= (1/p + T_cut)/d_rem. Low
    p (slow generation) => large K => behaves eager; high p + tight T_cut
    (fast generation, short lifetime) => small K => behaves lazy and avoids
    the waste regime. K=0 reduces to JIT; K=inf to always-on."""

    def __init__(self, placement_seed: int = 0, lookahead: int | None = None,
                 name: str = "greedy_adaptive"):
        self.placement_seed = placement_seed
        self.lookahead = lookahead          # None => regime-adaptive
        self.name = name
        self._placement: list[int] | None = None
        self._K: int | None = None

    def _adaptive_K(self, env: EagerEnv) -> int:
        if self.lookahead is not None:
            return self.lookahead
        p = env.hardware.links[0].p
        t_cut = env.hardware.links[0].T_cut
        horizon = (1.0 / p) + (t_cut if t_cut is not None else 50)
        return max(0, int(round(horizon / max(1, env.params.d_rem))))

    def __call__(self, env: EagerEnv) -> Action:
        if self._placement is None:
            self._placement = compute_placement(
                env.instance, env.hardware, self.placement_seed)
        for q in map_emission_order(env.instance):
            if q in env._unmapped:
                return Map(q, self._placement[q])

        crit = env.instance.criticality
        for g in sorted(env.ready_gates(), key=lambda g: (-crit[g], g)):
            if env.is_valid(Schedule(g)):
                return Schedule(g)

        # bounded-lookahead demand: imminent remote gates per link
        if self._K is None:
            self._K = self._adaptive_K(env)
        from ..env.state import UNSCHEDULED
        nl = env.hardware.num_links
        imminent = [0] * nl
        for gid, gr in enumerate(env.gates):
            if (gr.state == UNSCHEDULED and gr.remote
                    and gr.n_unfinished_preds <= self._K):
                for l in gr.route:
                    imminent[l] += 1
        # provision links whose imminent demand exceeds current supply,
        # most-critical-blocked first (reuse deficit_demand for ordering)
        _, max_crit = env.deficit_demand()
        order = sorted(range(nl),
                       key=lambda l: (-(max_crit[l] or 0), l))
        for l in order:
            ls = env.links[l]
            supply = ls.stored + ls.busy_channels
            if imminent[l] > supply and env.is_valid(GenEPR(l)):
                return GenEPR(l)
        return ADVANCE


class GreedyRegimeProvisionPolicy:
    """Regime-adaptive provisioning (D76): the strong provisioning expert
    that approximates the path-A oracle on a fixed (e.g. AGG) placement.
    Behaves EAGER (always-on) except in the WASTE regime — tight cutoff with
    fast aggregate generation (T_cut small AND p*W large) — where always-on
    over-generates and pairs expire, so it switches to bounded-lookahead
    K=1 (lazy-ish). This is the IL target for the provisioning-only EAGER
    (path B) and a baseline in its own right; placement is supplied
    externally (AGG/MHSA) since this policy learns/decides only provisioning.

    Regime detection uses the link parameters (p, W, T_cut) that are also
    state features, so the learned agent can reproduce the switch from
    observation."""

    def __init__(self, placement_seed: int = 0, placement: list[int] | None = None,
                 name: str = "greedy_regime_prov"):
        self.placement_seed = placement_seed
        self.name = name
        self._placement = placement
        self._delegate = None

    @staticmethod
    def is_waste_regime(hardware) -> bool:
        lc = hardware.links[0]
        tight = lc.T_cut is not None and lc.T_cut <= 5
        fast = lc.p * lc.W >= 0.6
        return tight and fast

    def __call__(self, env: EagerEnv) -> Action:
        if self._delegate is None:
            if self.is_waste_regime(env.hardware):
                # reactive JIT is near-oracle in the waste regime (always-on
                # wastes; bounded-lookahead over-corrects near the boundary)
                pl = self._placement
                self._delegate = GreedyJITPolicy(
                    placement_fn=(lambda i, h, p=pl: p) if pl is not None
                    else None, placement_seed=self.placement_seed)
            else:
                self._delegate = GreedyEagerPolicy(self.placement_seed)
                if self._placement is not None:
                    self._delegate._placement = list(self._placement)
        return self._delegate(env)


class GreedyEagerPolicy:
    """GreedyEager (guide D38, §9.7-adjacent): GreedyJIT placement + list
    scheduling, but MAXIMALLY PROACTIVE provisioning — saturate every free
    generation channel (subject to buffer-overflow safety) on EVERY link that
    routes at least one still-unscheduled remote gate, every slot, instead of
    JIT-on-deficit. This is the honest "always-on provisioning" control that
    answers the reviewer question 'is the learned proactivity needed, or would
    trivially always generating suffice?'. EAGER must beat THIS, not just the
    reactive GreedyJIT, to justify learning the provisioning policy (D75)."""

    def __init__(self, placement_seed: int = 0, name: str = "greedy_eager"):
        self.placement_seed = placement_seed
        self.name = name
        self._placement: list[int] | None = None
        self._routed_links: set[int] | None = None

    def __call__(self, env: EagerEnv) -> Action:
        if self._placement is None:
            self._placement = compute_placement(
                env.instance, env.hardware, self.placement_seed)

        for q in map_emission_order(env.instance):
            if q in env._unmapped:
                return Map(q, self._placement[q])

        # schedule the most critical valid ready gate (same as GreedyJIT)
        crit = env.instance.criticality
        for g in sorted(env.ready_gates(), key=lambda g: (-crit[g], g)):
            if env.is_valid(Schedule(g)):
                return Schedule(g)

        # maximally proactive: saturate channels on every link still serving
        # an unscheduled remote gate (always-on, not deficit-gated)
        from ..env.state import UNSCHEDULED
        needed: set[int] = set()
        for gid, gr in enumerate(env.gates):
            if gr.state == UNSCHEDULED and gr.remote:
                needed.update(gr.route)
        for l in sorted(needed):
            if env.is_valid(GenEPR(l)):
                return GenEPR(l)

        return ADVANCE
