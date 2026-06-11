# BASELINE_FIDELITY — published-method adaptation disclosures

> Stub (Phase 0). Filled in Phase 3 when published-style baselines are
> implemented (guide §9). Every adaptation from a published method is recorded
> here and disclosed in the paper as "X-style, adapted to our stochastic
> environment".

Template per baseline:

## <Baseline name> (<venue anchor>)
- **Source method**: citation + which mechanism is reimplemented.
- **Public artifact checked**: yes/no; result of the artifact search.
- **Faithful parts**: what follows the paper exactly.
- **Adaptations**: every deviation forced by our environment (stochastic
  generation, cutoff, buffers, micro-action API), each with rationale.
- **Budget/fairness**: tuning budget, seeds, env-step budget parity notes.

## GreedyJIT (self-contained expert + heuristic baseline, guide §9.1)

- **Source method**: this project's own §9.1 specification (not a published
  method); placement step follows the hypergraph-partitioning lineage
  (Andrés-Martínez & Heunen) via a METIS-style balanced k-way partition of
  the qubit interaction graph.
- **Public artifact checked**: n/a (self-specified). For the partitioner,
  pymetis/KaHyPar were attempted; neither installs on this Windows
  environment (no prebuilt wheels, C++ build required).
- **Faithful parts**: interaction-graph placement under per-QPU capacities;
  per-slot list scheduling in descending criticality; remote gates schedule
  only when every route link holds a stored pair; deficit registration and
  JIT provisioning up to free channels / buffer headroom, prioritized by the
  most critical blocked gate; emits micro-action traces in the agent's
  action vocabulary (§8.1 requirement).
- **Adaptations**: (1) partitioner is a pure-Python greedy + FM + KL-swap
  refinement honoring the same contract (D29) — to be revisited with a real
  METIS before the Phase 3 MHSA comparison; (2) "up to free channels /
  buffer headroom" implemented as saturation of the §6.3 validity bounds on
  links with blocked demand (literal reading, D33).
- **Budget/fairness**: deterministic given placement seed (guide §10.4:
  3 placement seeds at evaluation phases); no tuning knobs.

## Random-Progressive (lower bound, guide §9.5)

- **Source method**: guide §9.5 verbatim — uniform over valid non-ADVANCE
  actions; ADVANCE only when nothing else is valid.
- **Faithful parts**: exactly the above; policy RNG (PCG64) separate from
  the env CRN, fresh per episode, seeded.
- **Disclosure (D35)**: because GenEPR is valid whenever a channel and
  buffer headroom are free, ADVANCE-only-when-forced makes this policy an
  accidental ALWAYS-ON provisioner (it executes essentially every valid
  action every slot; its only true randomness is placement). At default
  p=1/12 it beats reactive JIT on provisioning-throughput-bound serialized
  circuits — characterized in PHASE_STATUS Phase 2 and escalated; the paper
  must present it accordingly.

*(published-style baselines MHSA+LS / AGG / DDQN-flat arrive in Phase 3)*
