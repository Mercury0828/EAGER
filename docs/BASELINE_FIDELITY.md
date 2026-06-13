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

## MHSA+LS (home-venue anchor: Mao, Liu & Yang, "Qubit Allocation for
## Distributed Quantum Computing", IEEE INFOCOM 2023)

- **Source method**: the paper's multistage hybrid simulated annealing (MHSA)
  algorithm for the qubit-allocation problem (QA-DQC) — a local-search
  heuristic hybridized with simulated annealing, minimizing remote-gate
  count under per-QPU capacities.
- **Public artifact checked**: none found for the INFOCOM'23 paper; the full
  text is IEEE-paywalled in this environment. Implemented from the paper's
  published abstract/structure plus the guide §9.2 specification; fidelity
  level disclosed as "MHSA-style" in the paper.
- **Faithful parts**: objective (remote-gate count = gate-weighted
  interaction-graph cut) under capacity constraints; hybrid structure =
  greedy initialization + SA + local-search descent; multistage = 4 stages,
  each an SA sweep (capacity-feasible single moves + pairwise swaps,
  within-stage geometric cooling from a sampled initial temperature, halved
  between stages) followed by a first-improvement local-search descent.
- **Adaptations**: stage count (4), proposal budget (20,000, fixed and
  reported with results), cooling constants — paper values unavailable,
  chosen once and frozen; placement is then paired with the §9.1 list
  scheduler + JIT provisioning (via GreedyJITPolicy.placement_fn) so
  MHSA-vs-GreedyJIT isolates placement quality, per the guide.
- **Budget/fairness**: deterministic given seed; same evaluation seeds as
  every other method (CRN); budget reported in results/index.json.

## AGG (top-venue anchor: Wu et al., "AutoComm", MICRO 2022)

- **Source method**: AutoComm's communication aggregation — extract burst
  communication (consecutive remote gates between the same node pair sharing
  a source qubit) and serve each burst with ONE cat-comm channel.
- **Public artifact checked**: YES — github.com/anbangw/AutoComm (reference
  implementation; inspected 2026-06-11). Burst detection follows its
  `consecutive_merge` stage: maximal runs of remote gates between the same
  QPU pair sharing one operand, consecutive in the shared qubit's gate
  sequence. Their latency table (EP=12 vs CX=1) independently matches this
  project's t_ep ~= 12 x CX calibration (guide §4.5).
- **Faithful parts**: consecutive-run burst detection on the shared qubit;
  cat-comm cost shape: a k-gate burst costs ONE pair (per route link) +
  d_rem + (k-1)*d_loc slots (their EP + local gates), vs k pairs and
  k*d_rem unaggregated; controlled comparison on the SAME placement and the
  SAME §9.1 scheduler as GreedyJIT.
- **Adaptations / deviations (each forced by our model, D40)**:
  (1) no commutation-based block extension (their `linear_merge_iter`):
  our DAG is serialization-frozen by definition (§4.1/D30), so only
  chain-consecutive runs aggregate — this UNDERSTATES AutoComm's gains;
  (2) no TP-comm branch (A2/A6: cat-comm only in v1);
  (3) CNOT direction is abstract in 2q skeletons, so burst sharing is
  side-agnostic (their cat/tp tagging keys on control/target sides);
  (4) bursts break when the head's partner reappears as the other operand
  (the local rewrite (x1,x1) would be degenerate);
  (5) the shared qubit is released after the burst head completes rather
  than after disentangle, and bursts to different targets may overlap
  (one-to-many cat copies — present in AutoComm's one-to-many burst forms);
  affects T only, never pair counts;
  (6) implementation = placement-aware instance transform (burst tails
  rewritten as local gates anchored at the head's partner), executed by the
  unmodified env — no per-method cost code (guide §12).
- **Structural note**: QASMBench's chain-form ghz/cat are burst-FREE in this
  model under min-cut placement (isolated remote gates); AutoComm's
  ghz-class gains presume fan-out/commutation forms. A constructed fan-out
  GHZ instance (ghz_fanout_n78) carries the §11 acceptance signal instead.

## DDQN-flat (learning anchor, ICC'25-style)

- **Source method**: ICC'25-style flat-state Double DQN compiler baseline
  (guide §9.4): Double DQN + target network + uniform replay over a flat
  fixed-size state with the same action space, max-size masks, trained per
  configuration.
- **Public artifact checked**: not applicable at this phase (training runs
  in Phase 6 with EAGER-PPO-matched env-step budgets; this phase ships the
  implementation).
- **Faithful parts**: flat state = per-QPU loads + per-link
  [stored, busy, free, p] + top-k=8 ready-gate features + globals,
  zero-padded; SAME D15 action enumeration with boolean masks (masked
  argmax everywhere, including the Double-DQN target's argmax); epsilon-
  greedy; Huber loss; gradient clipping; periodic target sync.
- **Adaptations**: feature normalizations and network width (256x256 MLP)
  chosen once; per-config sizing derived from the bound env.
- **Budget/fairness**: Phase 6 trains with the same env-step budget as
  EAGER's PPO phase; agent RNG (torch/numpy) seeded separately from the env
  CRN.
- **Phase 6 OUTCOME (D81) — recorded failure, NOT shipped as a headline
  competitor**: trained on the path-B provisioning-only task (premapped AGG,
  K=4 stratum; fixed-K because the flat state dim is K-specific — itself the
  config-lock point), 600k env steps, fixed max-size padded action layout
  (N_max=30, M_max=90, L=4, masked argmax). The learned policy is DEGENERATE:
  it over-emits GenEPR (hundreds-to-thousands per episode) and under-schedules
  gates (e.g. 10/45 gates scheduled), so circuits do not complete and episodes
  truncate — mean J 1650 vs AGG-reactive 71.8 (23x worse) and worse than even
  Random-Progressive (~2x). This is a degenerate-policy training failure on
  the long micro-action horizon (terminal completion signal washed out by
  gamma over ~460 steps), NOT a competently-trained-but-weaker baseline.
  Per the research-integrity rule (failed experiments are recorded, not
  shipped), it is NOT presented as a fair learning competitor in the main
  table. The "why a graph encoder" claim rests instead on (i) representation —
  a flat state is config-locked (cannot represent the varying N/K/topology
  EAGER handles), and (ii) the zero-shot generalization (D80, F3) that a flat
  model is structurally unable to perform. A clean representation-isolating
  ablation (flat-state PPO: same algorithm and budget as EAGER, MLP encoder
  instead of the R-GCN) is the honest future-work isolation; the DQN result
  conflates algorithm (DQN vs PPO) with representation and is logged as such.

## CloudQC (stretch / closest published setting, guide §9.8)

- **Source method**: Yu et al., "CloudQC: A Network-aware Framework for
  Multi-tenant Distributed Quantum Computing" (arXiv:2504.20389, IEEE 2025) —
  the closest published setting (joint placement + network scheduling under
  probabilistic EPR generation).
- **Public artifact checked**: none found; method extracted from the paper's
  abstract/method summary (the PDF is binary-encoded; used the structured
  review of the method).
- **Faithful parts**: (1) placement = balanced graph partition with tuned
  imbalance + community detection + heuristic mapping ranked by S=alpha/T +
  beta/C; (2) network scheduler = priority by longest-path depth in the
  remote DAG (= our criticality), and "allocating REDUNDANT network resources
  to important gates to mitigate backlogs" -> criticality-prioritized
  PROACTIVE provisioning that fills buffer headroom on the links serving the
  most critical unscheduled remote gates first.
- **Adaptations (D79)**: our setting is single-circuit, homogeneous QPUs,
  fixed shortest-path routing, so CloudQC's multi-tenant placement (community
  detection over a QPU pool + the S-score over remaining capacity) reduces to
  a balanced capacity-constrained min-cut — the same placement family as the
  other baselines, so the comparison isolates CloudQC's distinctive
  criticality-prioritized redundant SCHEDULER. The exact redundancy quantity
  is unspecified in the source; we fill buffer headroom on critical links
  (bounded by the §6.3 overflow-safe rule). Labeled "CloudQC-style".
- **Behavior note**: with ample channels its redundant proactive provisioning
  approaches always-on; under channel contention (W=1) the criticality
  ordering is its distinguishing feature; like all proactive heuristics it
  wastes in the high-p / tight-T_cut regime (the D75 regime map).

## Provisioning-spectrum controls (not published; D75/D76)

Principled provisioning-strategy baselines spanning the eager<->lazy spectrum,
introduced to characterize the proactive-provisioning tradeoff (the thesis)
and as honest controls for the learned policy:
- **GreedyEager** (D75): GreedyJIT placement/scheduling + always-on
  generation on every link serving an unscheduled remote gate (maximally
  proactive — the "is learning needed over trivially always-generating"
  control).
- **GreedyAdaptive(K)** (D76): bounded-lookahead provisioning (provision for
  remote gates within K unfinished predecessors); K=0 -> JIT, K->inf -> eager.
- **GreedyRegimeProvision** (D76): eager normally, reactive in the waste
  regime (the regime-adaptive switch; ~oracle on a fixed placement; the IL
  target for the path-B learned policy, D77).

## EAGER (the learned policy, path B, D77/D78)

- **Composition**: AGG placement + AGG aggregation (the strong static base)
  + a LEARNED proactive-provisioning policy (R-GCN encoder + attention
  decoder, IL from GreedyRegimeProvision + provisioning-only PPO). The agent
  does NOT learn placement (D72/D76 showed it cannot match MHSA/AGG); it
  learns only provisioning, so any win over AGG is attributable purely to the
  learned provisioning (placement+aggregation matched).
- **Result**: vs AGG-reactive +5.1% (p=1.9e-22); beats always-on in BOTH
  regimes (normal +0.9%, waste +5.5%); residual: pure-reactive is best in the
  extreme waste regime (reported limitation, D78b).
