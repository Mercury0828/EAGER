# DESIGN_DECISIONS — EAGER (eager-dqc)

Append-only decision log. Format: `D## | date | decision | rationale |
alternatives rejected`. Locked decisions are not reopened without an explicit
override entry; forced deviations get a NEW numbered entry, never an edit.
This file is INTERNAL ONLY (excluded from any public artifact release).

D1–D9 mirror the initial decision log in `docs/guide.md` §14 (canonical).

---

D1 | 2026-06-10 | Multi-hop remote gates via fixed shortest-path swapping with q=1; entanglement *generation* is the only stochastic element in v1. | Networking flavor without routing-decision complexity (guide §14). | Adaptive/learned routing; stochastic swapping.

D2 | 2026-06-10 | Static qubit placement; no TP-Comm/migration in v1. | Scope control; limitation paragraph + future work (guide §14, A2). | Mid-circuit migration actions.

D3 | 2026-06-10 | Objective weights α=1.0, β=1.0, γ=0.5 provisional; one calibration pilot in Phase 6, then frozen. | Terms should be same order of magnitude on medium instances (guide §5.1). | Per-instance adaptive weights; tuning weights per experiment.

D4 | 2026-06-10 | Baseline suite per guide §9 (top-venue anchor MICRO'22 aggregation style; home-venue anchor INFOCOM'23 SA placement; learning anchor ICC'25-style flat-state DDQN; expert GreedyJIT; Random-Progressive lower bound; Gurobi exact upper bound). CloudQC demoted to related-work/stretch. | Authority layering (guide §9). | CloudQC as mandatory baseline.

D5 | 2026-06-10 | Codename EAGER; repo `eager-dqc`. The retired prior framework codename (named only in guide §14/D5) must not appear anywhere in this repo. | Lineage constraint (guide §2.3); fresh self-contained formulation. | Reusing prior naming.

D6 | 2026-06-10 | Cat-Comm only; one end-to-end pair (= one stored pair consumed per route link) per remote gate. | Assumption A6 (guide §4.3). | TP-Comm; per-gate multi-pair purification.

D7 | 2026-06-10 | GenEPR(l) = persistent generate-until-success tasking of one free channel; no cancel action in v1. | Smaller action space; matches channel hardware semantics (guide §4.4). | Cancellable tasks; per-slot one-shot attempts.

D8 | 2026-06-10 | Counter-based CRN (Philox/stable hash) for ALL stochastic draws: `outcome(seed, l, c, t)`; mandatory property tests. | Valid common-random-number paired comparisons across policies (guide §6.5). | Stateful sequential RNG (order-dependent, breaks CRN pairing).

D9 | 2026-06-10 | T_budget = 20·(M+N)+200 slots; truncation penalty P_trunc = α·10·(#unfinished gates). | Bounded episodes; a sane policy never truncates (guide §6.1). | Unbounded episodes; per-step timeout heuristics.

---

Session decisions (Phase 0/1A/1B bootstrap):

D10 | 2026-06-10 | Repo rooted at the existing working directory (project dir name `EAGER`); package name `eager-dqc`, import package `eager`, layout exactly per guide §12. Canonical guide moved from repo root to `docs/guide.md`. | Guide §12 prescribes the inner layout, not the outer dir name; the user-placed guide belongs at `docs/guide.md`. | Renaming the working directory (pointless churn).

D11 | 2026-06-10 | Python 3.12.10 venv at `.venv` (standard CPython); Phase 0/1 deps only: numpy, networkx, pyyaml, pandas, pyarrow (+pytest dev extra). torch/torch_geometric/gurobipy deferred to their phases. | Guide §12 requires 3.11+ and user-local env; 3.12 is the system's standard CPython. Minimal deps per session scope. | conda env (a 3.11 miniconda exists but standard venv is leaner); installing GPU/MILP deps now.

D12 | 2026-06-10 | Config schema concretes: `mode: stochastic|deterministic` and `t_ep` live in the hardware config; `T_cut: null` means no cutoff (∞); per-link overrides keyed `"u-v"`; unknown keys are hard errors; link ids = index into the lexicographically sorted edge list; `kappa` accepts a scalar (uniform) or per-QPU list; `ring` requires K≥3 (K=2 ring would duplicate the line edge). | Deterministic mode is a hardware-level semantics switch (guide §5.2); typo-proof validation with helpful errors (Phase 0 scope). | Separate experiment-config file for mode (overkill at this phase); silent ignoring of unknown keys.

D13 | 2026-06-10 | Timing conventions (implements guide §6.1 exactly): slots are 0-indexed; a channel tasked during slot t makes its first draw at the resolve of slot t (deterministic mode: pair lands at end of slot t+t_ep−1); a pair generated at the resolve of slot t is consumable during slots t+1..t+T_cut and expires (waste, charged once, attached to that ADVANCE's reward) at the resolve of slot t+T_cut; ages increment at resolve step (3), so stored pairs always show age ≥ 1 at micro-action time; a gate scheduled in slot t with duration d completes at the resolve of slot t+d−1; its successors become ready with ready_slot = t+d; source gates have ready_slot = 0; makespan T = number of resolved slots when the last gate completes (= ADVANCE count, so Σ ADVANCE rewards = −α·T); truncation when post-increment t > T_budget; remote-gate stall = schedule_slot − ready_slot. | All downstream hand-derivations, golden tests, and metrics need one frozen convention; this is the direct reading of §6.1's resolve order (generation → gate progress → aging → t+1). | Aging before generation (would let pairs expire one slot early); 1-indexed slots.

D14 | 2026-06-10 | Env API: configs are bound at construction (`EagerEnv(hardware, circuit, params)`), `reset(seed)` (re)starts an episode; `step()` on an invalid action raises ValueError with the violated rule; `reset` rejects instances with N > Σκ (unmappable). | Immutable config per env instance is the standard contract; hard errors surface agent bugs that penalize-and-continue would hide. | `reset(config, seed)` re-binding configs per call; silent penalty for invalid actions.

D15 | 2026-06-10 | Fixed action-space enumeration for masks: [Map(q,u) q-major] ++ [Schedule(g)] ++ [GenEPR(l)] ++ [ADVANCE last]; mask is a bool vector of length N·K+M+L+1. | Agents (Phase 5) and DDQN-flat need a stable index contract from day one. | Dict-only action sets (no stable indexing for replay buffers).

D16 | 2026-06-10 | Pair consumption is FIFO oldest-first per link. | Maximizes utilization under cutoff (oldest pairs die first); deterministic and hand-computable. | Youngest-first (provably wasteful under T_cut); random choice (breaks determinism).

D17 | 2026-06-10 | One canonical route per unordered QPU pair {u,v}: computed from the lower-indexed endpoint with lexicographically-smallest shortest node sequence (greedy min-index next hop); both gate-operand orders use the same route. | Pair consumption must be well-defined regardless of operand order (guide §4.3 fixed routing, lexicographic tie-break). | Direction-dependent routes (would make C_comm depend on operand order).

D18 | 2026-06-10 | CRN engine = numpy Philox, per-draw stateless construction: key = (seed mod 2^64, upper seed bits XOR a fixed stream constant), counter = (t, channel, link, 0); `uniform(l,c,t)` is the first float64 of that stream; success iff uniform < p_l. | Same (seed,l,c,t) → same draw regardless of query order or policy, by construction (guide §6.5, D8); numpy Philox is platform-stable. | xxhash→[0,1) (extra dependency); stateful Generator shared across draws (order-dependent).

D19 | 2026-06-10 | Trajectory hash = chained SHA-256 over canonical JSON of (action repr, integer-only obs snapshot, reward as float.hex(), done) per step, seeded with the reset obs. | Cross-process determinism acceptance needs a stable, platform-reproducible fingerprint; float.hex() avoids repr ambiguity. | Python `hash()` (salted per process); pickling (not canonical).

D20 | 2026-06-10 | Tiny scripted demo policies live in `eager/utils/scripted_policies.py` (first-fit mapper + JIT-style generator + eager-generation variant). They are test/demo helpers shared by tests and `scripts/run_episode.py`, explicitly NOT the Phase-2 GreedyJIT baseline (no METIS partitioning, no criticality-ordered list scheduling). | Phase 1 acceptance requires scripted policies; duplicating them in tests and scripts invites drift. Phase gating concerns baselines/agents, not trivial helpers. | Inlining per-test policies (drift); implementing GreedyJIT early (phase violation).

D21 | 2026-06-10 | auto_jit (guide §9.7) implemented at env level, default OFF: when ON, applying ADVANCE first auto-issues GenEPR for per-link deficits of ready-but-pair-blocked remote gates (deficit = blocked demand − stored − busy, capped by free channels and buffer headroom), links ordered by max criticality of the gates they block (then link id), then the slot resolves normally. | §9.1(3) JIT semantics at the moment the agent yields the slot; deficit cap prevents over-tasking; criticality priority per guide. | Auto-provisioning at gate-ready time inside resolve (would generate within the same slot the demand appears, too strong); uncapped tasking.

D22 | 2026-06-10 | Synthetic instance generator ships in Phase 0 (config system completeness: `kind: synthetic` must be loadable/validatable); per-qubit serialization is enforced structurally by deriving the DAG from the gate-list order via last-toucher edges. | Smoke + config validation need buildable instances; the guide's serialization requirement holds by construction rather than by post-hoc check. | Stub generator until Phase 1A; explicit DAG edges in YAML (redundant, error-prone).

D23 | 2026-06-10 | `python -m eager.smoke` parses and summarizes configs only (no env dependency). | Phase 0 acceptance is "prints a parsed config"; keeps the entry point stable before the env exists. | Smoke running an episode (couples Phase 0 to 1A).

D24 | 2026-06-10 | CLAUDE.md and all repo files carry no owner names, usernames, or external personal URLs; the startup skill package is referenced generically where needed. | Guide §2.4 double-blind hygiene: no author-identifying strings anywhere in the repo. | Embedding the owner/skill-repo identity in CLAUDE.md (template default, rejected for this repo).

D25 | 2026-06-10 | Criticality (features + provisioning priority) = longest path to sink in gate counts over the static DAG, computed once per instance. | Duration-aware criticality depends on the mapping (remote=2 slots), which is a decision variable; the static quantity is well-defined pre-mapping (guide §6.2 normalizes it anyway). | Duration-weighted criticality recomputed per mapping (circular, costlier).
