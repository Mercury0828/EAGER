"""CloudQC baseline (guide §9.8 stretch; Yu et al., "CloudQC: A Network-aware
Framework for Multi-tenant Distributed Quantum Computing", arXiv:2504.20389 /
IEEE 2025). Closest published setting to ours (placement + network scheduling
under probabilistic EPR generation). Adapted to our single-circuit,
homogeneous-QPU, micro-action env; every deviation is recorded in
docs/BASELINE_FIDELITY.md and disclosed as "CloudQC-style".

CloudQC has two components:

- **Placement**: filtering-and-scoring — graph partitioning with a tuned
  imbalance factor + community detection + a heuristic mapping, ranked by
  S = alpha/T + beta/C (runtime, communication). In our setting (one
  circuit, K homogeneous QPUs, fixed routing) this reduces to a balanced
  capacity-constrained min-cut of the interaction graph — the same family as
  the other placement baselines, so the comparison isolates CloudQC's
  distinctive SCHEDULER (D79).

- **Network scheduler (the distinctive part)**: priority = longest-path
  depth in the remote DAG (= our criticality); the scheduler "prioritizes
  important gates by allocating REDUNDANT network resources to mitigate
  backlogs". We implement this as criticality-prioritized PROACTIVE
  provisioning with redundancy: each slot, after scheduling the most-critical
  ready gates, provision EPR pairs on the links serving the most-critical
  UNSCHEDULED remote gates, filling buffer headroom (redundant pairs) on
  those critical links first, so high-criticality remote gates never stall on
  generation. This is distinct from GreedyJIT (reactive, no redundancy),
  GreedyEager (always-on, criticality-blind), and the regime-adaptive
  policies — it is criticality-weighted eager-with-redundancy.
"""

from __future__ import annotations

from ..config import HardwareConfig
from ..env.actions import ADVANCE, Action, GenEPR, Map, Schedule
from ..env.env import EagerEnv
from ..env.state import UNSCHEDULED
from .greedy_jit import compute_placement, map_emission_order


class CloudQCPolicy:
    """CloudQC-style: balanced min-cut placement + criticality-prioritized
    redundant proactive provisioning."""

    name = "cloudqc"

    def __init__(self, placement_seed: int = 0, placement=None):
        self.placement_seed = placement_seed
        self._placement = list(placement) if placement is not None else None

    def __call__(self, env: EagerEnv) -> Action:
        if self._placement is None:
            self._placement = compute_placement(
                env.instance, env.hardware, self.placement_seed)
        for q in map_emission_order(env.instance):
            if q in env._unmapped:
                return Map(q, self._placement[q])

        crit = env.instance.criticality
        # 1. schedule the most critical valid ready gate
        for g in sorted(env.ready_gates(), key=lambda g: (-crit[g], g)):
            if env.is_valid(Schedule(g)):
                return Schedule(g)

        # 2. criticality-prioritized redundant provisioning: for each link,
        # the max criticality of an UNSCHEDULED remote gate it serves; fill
        # buffer headroom on the most-critical links first (redundancy)
        nl = env.hardware.num_links
        link_crit: list[int] = [-1] * nl
        for gid, gr in enumerate(env.gates):
            if gr.state == UNSCHEDULED and gr.remote:
                for l in gr.route:
                    if crit[gid] > link_crit[l]:
                        link_crit[l] = crit[gid]
        order = sorted((l for l in range(nl) if link_crit[l] >= 0),
                       key=lambda l: (-link_crit[l], l))
        for l in order:
            if env.is_valid(GenEPR(l)):
                return GenEPR(l)
        return ADVANCE
