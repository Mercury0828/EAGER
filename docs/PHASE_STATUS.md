# PHASE_STATUS — EAGER (eager-dqc)

> Live state for artifact-based resume. Updated after every major step.
> Acceptance evidence below is REAL pasted command output, never summarized
> from memory.

## Current state

- **Current phase**: Phase 7 COMPLETE (2026-06-13). Phases 0/1A/1B/2/3/4/5/6/7
  all complete. Architecture pivot to path B (D76) executed and locked; two
  owner-requested enhancements (flat-PPO isolation D83, stochastic-optimum T4
  D84) delivered; T5 ablation table scripted.
- **Last completed step**: Phase 7 close — flat-PPO representation isolation
  (D83: same IL+PPO+budget, MLP encoder vs R-GCN; IL ties at 0.972 but flat
  PPO diverges and cannot beat always-on, graph wins 0.9495 vs 0.9591 — a fair
  SHIPPABLE baseline that supersedes the degenerate DDQN); stochastic-optimum
  T4 (D84: clairvoyant perfect-information B&B; reactive +25.9% off the
  stochastic optimum, EAGER reaches it); T5 ablation table scripted; all
  protocols green (164 passed, 10x ALL STABLE, clean-state byte-identical).
- **Exact next step**: paper writing (prose, polishing); optional remaining
  figures (flat-vs-graph PPO-stability F4). Core experimental program (Phases
  0-7) is complete. Requires owner direction.
- **Blockers**: none. Standing debts: D29 (real METIS — partitioner is still
  pure-Python greedy+FM, acceptable since AGG/MHSA placements carry the main
  result); reported LIMITATIONS (all honestly logged, none hidden):
  (i) EAGER 7.7% worse than pure-reactive in the EXTREME waste regime (D78b),
  (ii) zero-shot topology transfer to unseen K=8 unreliable (D80),
  (iii) flat-state DQN (D81) conflates algo+representation — superseded by the
  clean flat-PPO isolation (D83); (iv) a latent env edge-case is reachable only
  by unpruned exhaustive micro-action search (stochastic_opt seeds an incumbent
  to avoid it; no validated result is affected, D84).

## Session authorization

This bootstrap session is authorized for Phase 0, Phase 1A, Phase 1B only
(guide §11). Phase 2+ (baselines/agents) must NOT be started.

---

## Phase 0 — Scaffold

Status: in progress. Acceptance: fresh clone + fresh env + install + pytest
green + `python -m eager.smoke` works.

### Working-tree evidence (2026-06-10)

`.venv\Scripts\python.exe -m pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
rootdir: E:\Project-git\EAGER
configfile: pyproject.toml
testpaths: tests
collected 23 items

tests\unit\test_circuit.py .......                                       [ 30%]
tests\unit\test_config.py ................                               [100%]

============================= 23 passed in 0.87s ==============================
```

`.venv\Scripts\python.exe -m eager.smoke`:

```
hardware 'k2_line': K=2 topology=line mode=stochastic t_ep=12
  kappa=[12, 12] (total 24)
  link 0: (0,1) p=0.0833333 W=2 B=8 T_cut=20 w=1
circuit 'golden_micro_1': N=3 qubits, M=3 two-qubit gates, depth=3
  g0: (0,1) preds=[] crit=3
  g1: (1,2) preds=[0] crit=2
  g2: (0,1) preds=[0, 1] crit=1
derived: T_budget=20*(M+N)+200=320 slots (guide D9); total capacity=24 for N=3 qubits
smoke OK
```

### Fresh-clone acceptance (2026-06-10)

`git clone <repo> %TEMP%\eager_fresh_p0` + fresh venv (Python 3.12.10) +
`pip install -e ".[dev]"` + pytest + smoke:

```
=== pytest (fresh clone) ===
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\<user>\AppData\Local\Temp\eager_fresh_p0
configfile: pyproject.toml
testpaths: tests
collected 23 items

tests\unit\test_circuit.py .......                                       [ 30%]
tests\unit\test_config.py ................                               [100%]

============================= 23 passed in 1.04s ==============================
=== smoke (fresh clone) ===
hardware 'k2_line': K=2 topology=line mode=stochastic t_ep=12
  kappa=[12, 12] (total 24)
  link 0: (0,1) p=0.0833333 W=2 B=8 T_cut=20 w=1
circuit 'golden_micro_1': N=3 qubits, M=3 two-qubit gates, depth=3
  g0: (0,1) preds=[] crit=3
  g1: (1,2) preds=[0] crit=2
  g2: (0,1) preds=[0, 1] crit=1
derived: T_budget=20*(M+N)+200=320 slots (guide D9); total capacity=24 for N=3 qubits
smoke OK
```

**Phase 0 acceptance: PASS** (fresh clone + fresh env + install + pytest green
+ smoke works). Tagged `phase-0-done`.

## Phase 1A — Deterministic simulator core

Status: COMPLETE (2026-06-10). Tagged `phase-1a-done`.

Scope delivered: env API (`reset(seed)`/`step`/`valid_actions`/
`valid_action_mask`/`info` with §10.5 metrics), Map/Schedule/GenEPR/ADVANCE
with exact §6.3 validity, §6.1 micro-action loop + resolve order, fixed
shortest-path routing with lexicographic tie-break (precomputed), multi-link
pair consumption at Schedule time, §6.4 reward (potential-based ADVANCE term,
NO valid-action bonus), single metrics module, synthetic generator with
per-qubit serialization. Stochastic mode is guarded with NotImplementedError
until Phase 1B; aging/cutoff machinery and the env-level auto_jit hook exist
but their acceptance tests land in Phase 1B. Timing conventions frozen in
DESIGN_DECISIONS D13.

### Acceptance evidence (2026-06-10)

Invariant tests (DAG precedence, capacity, per-slot pair conservation
`generated == consumed + stored`, buffer safety, reward accounting), TWO
golden micro-instances with hand derivations in comments
(tests/integration/test_golden_micro.py), and cross-process determinism:

```
tests/integration/test_golden_micro.py::test_golden_micro_1 PASSED       [  5%]
tests/integration/test_golden_micro.py::test_golden_micro_2 PASSED       [ 11%]
tests/integration/test_determinism_process.py::test_two_process_invocations_identical_deterministic[configs/circuits/golden_micro_1.yaml] PASSED [ 17%]
tests/integration/test_determinism_process.py::test_two_process_invocations_identical_deterministic[configs/circuits/golden_micro_2.yaml] PASSED [ 23%]
tests/integration/test_invariants.py::test_jit_policy_invariants[k2_line] PASSED [ 29%]
tests/integration/test_invariants.py::test_jit_policy_invariants[k3_line] PASSED [ 35%]
tests/integration/test_invariants.py::test_jit_policy_invariants[k4_grid] PASSED [ 41%]
tests/integration/test_invariants.py::test_random_policy_invariants[0-k2_line] PASSED [ 47%]
tests/integration/test_invariants.py::test_random_policy_invariants[0-k3_line] PASSED [ 52%]
tests/integration/test_invariants.py::test_random_policy_invariants[0-k4_grid] PASSED [ 58%]
tests/integration/test_invariants.py::test_random_policy_invariants[1-k2_line] PASSED [ 64%]
tests/integration/test_invariants.py::test_random_policy_invariants[1-k3_line] PASSED [ 70%]
tests/integration/test_invariants.py::test_random_policy_invariants[1-k4_grid] PASSED [ 76%]
tests/integration/test_invariants.py::test_random_policy_invariants[2-k2_line] PASSED [ 82%]
tests/integration/test_invariants.py::test_random_policy_invariants[2-k3_line] PASSED [ 88%]
tests/integration/test_invariants.py::test_random_policy_invariants[2-k4_grid] PASSED [ 94%]
tests/integration/test_invariants.py::test_golden_configs_with_jit_policy PASSED [100%]

============================= 17 passed in 0.75s ==============================
```

Full suite: `62 passed in 1.98s` (unit + integration).

Golden hand-derivations (full derivations as comments in the test file):
- micro 1: T=5, C_comm=1.0, C_waste=0.0, J=6.0, reward_sum=-6.0,
  pairs {generated:1, consumed:1, expired:0, stored:0}, utilization 1.0,
  mean remote stall 1.0 — matched exactly.
- micro 2: T=5, C_comm=2.0, C_waste=0.0, J=7.0, reward_sum=-7.0,
  pairs {generated:2, consumed:2, expired:0, stored:0}, utilization 1.0,
  mean remote stall 2.0 — matched exactly.

`scripts/run_episode.py` demo (scripted reactive-JIT policy, same instance):

```
episode hardware=golden_k2_det circuit=golden_micro_1 seed=123 policy=jit mode=deterministic auto_jit=False
T=6 C_comm=1 C_waste=0 J=7 truncated=False
pairs generated=1 consumed=1 expired=0 stored=0
epr_utilization=1 mean_remote_stall=2
reward_sum=-7 micro_steps=13
trajectory_sha256=707d47b587b7f1a33749b064b5758e59b0a5274a0f046eb9e79c33e397cf268a
```

Observation worth keeping: the reactive JIT demo policy yields J=7 on golden
micro 1; the proactive scripted sequence in the golden test achieves J=6
(provisioning starts during the slot the local predecessor runs, hiding
generation latency). The proactive-vs-JIT gap the paper targets is already
visible at micro scale.

## Phase 1B — Stochastic layer

Status: COMPLETE (2026-06-11). Tagged `phase-1b-done`.

Scope delivered: counter-based CRN engine (`eager/env/crn.py`, numpy Philox;
`uniform(l,c,t)` pure in (seed,l,c,t)); Bernoulli generate-until-success
channels (D7); buffer aging + T_cut expiry -> waste accounting (charged once,
on the resolving ADVANCE); buffer-overflow-safe GenEPR masking incl. in-flight
pairs (1A code, 1B-tested); env-level `auto_jit` flag (guide §9.7, default
OFF) with negative control; deterministic mode retained as a config switch and
anchored by the stochastic-p=1 == deterministic-t_ep=1 trajectory-equivalence
test.

### Acceptance evidence (2026-06-11)

Full suite after 1B: `91 passed in 8.09s`.

#### Flaky-bug protocol: stochastic suite repeated 10x (per-test pass counts)

`python scripts/run_repeat_suite.py --runs 10 --marker stochastic`:

```
run  1/10: 29/29 passed
run  2/10: 29/29 passed
run  3/10: 29/29 passed
run  4/10: 29/29 passed
run  5/10: 29/29 passed
run  6/10: 29/29 passed
run  7/10: 29/29 passed
run  8/10: 29/29 passed
run  9/10: 29/29 passed
run 10/10: 29/29 passed

test                                                                                                  pass_count
----------------------------------------------------------------------------------------------------------------
tests.integration.test_auto_jit::test_auto_jit_completes_map_schedule_only_policy                     10/10
tests.integration.test_auto_jit::test_auto_jit_respects_channel_and_buffer_limits                     10/10
tests.integration.test_auto_jit::test_without_auto_jit_same_policy_truncates                          10/10
tests.integration.test_crn_policies::test_different_seed_changes_luck                                 10/10
tests.integration.test_crn_policies::test_identical_draws_at_identical_coordinates                    10/10
tests.integration.test_crn_policies::test_same_policy_same_seed_identical_log                         10/10
tests.integration.test_determinism_process::test_two_process_invocations_identical_stochastic         10/10
tests.integration.test_expiry::test_conservation_with_interleaved_expiry_and_consumption              10/10
tests.integration.test_expiry::test_expiry_golden_derivation                                          10/10
tests.integration.test_expiry::test_pair_consumable_on_last_window_slot                               10/10
tests.integration.test_expiry::test_pair_gone_one_slot_after_window                                   10/10
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p083_cut20]      10/10
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p30_cut2_tight]  10/10
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p50_cut1_w1b1]   10/10
tests.integration.test_invariants::test_stochastic_random_policy_invariants[0]                        10/10
tests.integration.test_invariants::test_stochastic_random_policy_invariants[1]                        10/10
tests.statistical.test_crn_frequency::test_frequency_across_links_and_channels                        10/10
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.05]                        10/10
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.08333333333333333]        10/10
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.3]                         10/10
tests.unit.test_crn::test_coordinate_separation                                                       10/10
tests.unit.test_crn::test_different_seeds_differ_somewhere                                            10/10
tests.unit.test_crn::test_input_validation                                                            10/10
tests.unit.test_crn::test_large_seed_supported                                                        10/10
tests.unit.test_crn::test_query_order_independence                                                    10/10
tests.unit.test_crn::test_same_seed_same_draws_across_engines                                         10/10
tests.unit.test_crn::test_uniform_range_and_threshold_semantics                                       10/10
tests.unit.test_det_equiv::test_deterministic_mode_still_available_with_cutoff_inf                    10/10
tests.unit.test_det_equiv::test_stochastic_p1_equals_deterministic_tep1                               10/10

verdict: ALL STABLE (29 tests x 10 runs)
```

Notes on the acceptance map:
- CRN property tests: `tests/unit/test_crn.py` (order independence, engine
  agreement, seed separation) + `tests/integration/test_crn_policies.py`
  (two DIFFERENT scripted policies, same seed -> identical draws at identical
  (l,c,t); link-0 draw logs identical; extra activity on another link does
  not perturb shared coordinates).
- Statistical test: `tests/statistical/test_crn_frequency.py`, >=1e5 draws
  per p in {0.05, 1/12, 0.3}, |p_hat - p| within the 99% normal CI.
- Conservation including expiry: asserted after EVERY micro-action by
  `tests/util_invariants.py` across stochastic cases with tight cutoffs
  (T_cut in {1, 2, 20}), plus interleaved consume/expire test.
- Aging/expiry golden test: `tests/integration/test_expiry.py::
  test_expiry_golden_derivation` — pair generated at slot 0, T_cut=3,
  discarded exactly at the resolve of slot 3, charged exactly once
  (ADVANCE reward -1.5 = -alpha - gamma*w), J = 5.5 matched exactly; window
  edges: consumable at slot t+T_cut, gone at t+T_cut+1.
- auto_jit smoke: Map+Schedule-only policy completes a remote-gate instance
  with auto_jit=ON, zero truncation; negative control truncates with
  auto_jit=OFF and generated==0 (D26).

#### Clean-state verification (fresh clone + fresh venv)

`python scripts/clean_state_verify.py` (episode script BEFORE pytest ->
pytest -> episode script again; file-tree snapshot diff):

```
clean-state workdir: C:\Users\<user>\AppData\Local\Temp\eager_clean_enoteo8w
installing into fresh venv ...

=== episode runs BEFORE pytest ===
episode hardware=k2_line circuit=golden_micro_1 seed=0 policy=jit mode=stochastic auto_jit=False
T=3 C_comm=0 C_waste=0 J=3 truncated=False
pairs generated=0 consumed=0 expired=0 stored=0
epr_utilization=None mean_remote_stall=None
reward_sum=-3 micro_steps=9
trajectory_sha256=4e0bf86db49661a1cb91232f70b3fcebf5dd5caa8fd086f4cbe4d03417aee718
episode hardware=golden_k2_det circuit=golden_micro_2 seed=1 policy=jit mode=deterministic auto_jit=False
T=5 C_comm=2 C_waste=0 J=7 truncated=False
pairs generated=2 consumed=2 expired=0 stored=0
epr_utilization=1 mean_remote_stall=2
reward_sum=-7 micro_steps=15
trajectory_sha256=611fda0b4ad8699108d82c763e29fbca2f103bc284937209adf7ea7903281008

=== pytest (fresh clone) ===
........................................................................ [ 79%]
...................                                                      [100%]
91 passed in 9.81s

=== episode runs AFTER pytest ===
episode hardware=k2_line circuit=golden_micro_1 seed=0 policy=jit mode=stochastic auto_jit=False
T=3 C_comm=0 C_waste=0 J=3 truncated=False
pairs generated=0 consumed=0 expired=0 stored=0
epr_utilization=None mean_remote_stall=None
reward_sum=-3 micro_steps=9
trajectory_sha256=4e0bf86db49661a1cb91232f70b3fcebf5dd5caa8fd086f4cbe4d03417aee718
episode hardware=golden_k2_det circuit=golden_micro_2 seed=1 policy=jit mode=deterministic auto_jit=False
T=5 C_comm=2 C_waste=0 J=7 truncated=False
pairs generated=2 consumed=2 expired=0 stored=0
epr_utilization=1 mean_remote_stall=2
reward_sum=-7 micro_steps=15
trajectory_sha256=611fda0b4ad8699108d82c763e29fbca2f103bc284937209adf7ea7903281008

=== clean-state verdict ===
pytest green:               PASS
no test pollution:          PASS
episode outputs identical:  PASS
OVERALL: PASS
```

(Note on the first episode: with k2_line's ample kappa=12 the first-fit demo
policy co-locates all three qubits on QPU 0, so golden_micro_1 runs all-local
with zero pair traffic — correct behavior; placement pressure requires kappa
scarcity, which the experiment configs of later phases impose.)

---

## Phase 2 — GreedyJIT + Random-Progressive + trace recorder

Status: COMPLETE (2026-06-11), tagged `phase-2-done`, with ONE escalation
(D35/D37) awaiting an owner ruling on acceptance wording. Authorized by the
owner on 2026-06-11 ("开始phase2").

Scope delivered: QASMBench 2q-skeleton pipeline (pinned commit, extractor
with ccx/cswap expansion + custom-gate inlining + hard errors on unknowns,
frozen explicit YAMLs + drift-guard test, generated supremacy_n120) [D28,
D30, D31]; pure-Python capacity-constrained balanced partitioner (pymetis
unbuildable on Windows) [D29]; GreedyJIT expert per §9.1 as an env-API
micro-action policy with saturating JIT provisioning [D33]; Random-
Progressive per §9.5; trace record/replay in the agent's action vocabulary
[D36]; env hot-path optimization with consistency guards [D34]; panel
runner writing results/phase2_panel.parquet + results/index.json.

### Acceptance panel (5 CRN-paired seeds per instance, default config D32)

`python experiments/phase2_panel.py --seeds 5` (full per-episode log in the
run output; table re-derived from the parquet via `--verdict-only`):

```
instance          N     M   J(greedy)   J(random)  win  g_trunc
----------------------------------------------------------------------
adder_n28         28    195       120.3       752.3 5/5  0
adder_n4           4     10        27.6        41.3 4/5  0
bv_n30            30     18        82.4        79.6 2/5  0   <-- ordering exception (D35)
bv_n70            70     36       120.8       166.6 5/5  0
cat_n65           65     64        91.5         237 5/5  0
dnn_n51           51    319         844      1298.3 5/5  0
ghz_n78           78     77       112.5       312.9 5/5  0
ising_n98         98    194        46.5       708.3 5/5  0
multiplier_n45    45   2574      4305.5     11261.5 5/5  0
qaoa_n6            6     54       253.9       246.7 3/5  0   <-- ordering exception (D35)
qft_n63           63   3906     17645.2     14117.9 0/5  0   <-- ordering exception (D35)
qugan_n71         71    483      1003.6      1838.2 5/5  0
supremacy_n120   120    600      1348.1        2106 5/5  0

guide criterion as written in section 11 (zero greedy truncations AND mean J greedy < random on ALL): FAIL
D35 amended criterion (zero greedy truncations on all; ordering exceptions characterized): PASS; exceptions = ['bv_n30', 'qaoa_n6', 'qft_n63']
```

### ESCALATION — D35/D37 — **RESOLVED 2026-06-11 by owner ruling (D38)**

> Resolution: amended criterion ADOPTED; both baselines stay as defined;
> p=1/12 stays; Random presented as accidental always-on provisioner;
> GreedyEager variant planned for Phase 6. Under the D41 placement
> tie-break change (Phase 3), bv_n30 LEFT the exception set — caught by the
> strict-xfail sentinel; the regenerated panel below is authoritative.

Original escalation text (kept for the record):

The §11 Phase 2 criterion "J strictly < Random-Progressive on all" is
structurally unsatisfiable at the default p=1/12 on provisioning-throughput-
bound serialized circuits, because §9.5's "ADVANCE only when nothing else is
valid" makes Random-Progressive execute essentially every valid action every
slot — i.e., random placement + schedule-ASAP + ALWAYS-ON generation on all
links: an accidental maximally-proactive provisioner whose only true
randomness is placement. Reactive JIT pays ~1/(2p) slots of generation
latency per serialized remote gate. This is the paper's §1 trade-off
appearing inside the baseline pair (per guide §15, the regime map is itself
content), and it is precisely the gap the EAGER agent is meant to close.

Regime evidence (3 CRN-paired seeds; greedy seed-0 J decomposition shown):

```
bv_n30 p=0.0833: J_greedy=   91.33 J_random=   81.83 random wins  (greedy seed0 T=103 C_comm=12.0 C_waste=6.0)
bv_n30 p=0.2000: J_greedy=   58.00 J_random=   60.50 GREEDY WINS  (greedy seed0 T=52 C_comm=12.0 C_waste=0.0)
bv_n30 p=0.3000: J_greedy=   52.00 J_random=   62.00 GREEDY WINS  (greedy seed0 T=44 C_comm=12.0 C_waste=0.0)
bv_n30 p=0.5000: J_greedy=   47.33 J_random=   63.83 GREEDY WINS  (greedy seed0 T=39 C_comm=12.0 C_waste=0.0)
qaoa_n6 p=0.0833: J_greedy=  249.50 J_random=  220.33 random wins  (greedy seed0 T=212 C_comm=48.0 C_waste=2.0)
qaoa_n6 p=0.2000: J_greedy=  164.00 J_random=  155.17 random wins  (greedy seed0 T=121 C_comm=48.0 C_waste=0.0)
qaoa_n6 p=0.3000: J_greedy=  141.00 J_random=  137.33 random wins  (greedy seed0 T=97 C_comm=48.0 C_waste=0.0)
qaoa_n6 p=0.5000: J_greedy=  126.00 J_random=  137.50 GREEDY WINS  (greedy seed0 T=72 C_comm=48.0 C_waste=0.0)
```

Repo state pending the ruling: the two test-subset exceptions are
strict-xfail tests (they FAIL loudly if the regime ever flips) plus
mechanism-guard tests asserting greedy wins at higher p; the panel reports
BOTH criteria. **Proposed amended criterion**: "zero GreedyJIT truncations on
all instances; J(GreedyJIT) < J(Random-Progressive) except on characterized
provisioning-throughput-bound instances, where the gap is the documented
proactive-provisioning opportunity." Paper implication: Random-Progressive
must be presented as an accidental always-on provisioner (or §9.5 revised)
in the Phase 6 evaluation.

### Trace record/replay evidence (script level)

```
episode seed=0: T=183 J=233.5 truncated=False steps=465 replay=OK
episode seed=1: T=191 J=242.5 truncated=False steps=474 replay=OK
episode seed=2: T=192 J=241.5 truncated=False steps=471 replay=OK
wrote 3 traces -> artifacts\traces\adder_n28_greedy.jsonl
trace 0 (env_seed=0, policy=greedy_jit): replay OK
trace 1 (env_seed=1, policy=greedy_jit): replay OK
trace 2 (env_seed=2, policy=greedy_jit): replay OK
replayed 3 traces, 0 mismatches
```

Suite: `122 passed, 2 xfailed in 4.95s` (xfails = the strict D35 regime
exceptions). Trace semantics also covered by tests/integration/test_traces.py
(tamper detection, wrong-binding rejection, BC action-vocabulary check).

### 10x stochastic repeat (Phase 2)

`python scripts/run_repeat_suite.py --runs 10 --marker stochastic`
(strict xfails count as their EXPECTED outcome; stability = same outcome
10/10):

```
run  1/10: 46/46 as expected
run  2/10: 46/46 as expected
run  3/10: 46/46 as expected
run  4/10: 46/46 as expected
run  5/10: 46/46 as expected
run  6/10: 46/46 as expected
run  7/10: 46/46 as expected
run  8/10: 46/46 as expected
run  9/10: 46/46 as expected
run 10/10: 46/46 as expected

test                                                                                                  expected_outcome_count
----------------------------------------------------------------------------------------------------------------------------
tests.integration.test_auto_jit::test_auto_jit_completes_map_schedule_only_policy                     10/10 PASS
tests.integration.test_auto_jit::test_auto_jit_respects_channel_and_buffer_limits                     10/10 PASS
tests.integration.test_auto_jit::test_without_auto_jit_same_policy_truncates                          10/10 PASS
tests.integration.test_crn_policies::test_different_seed_changes_luck                                 10/10 PASS
tests.integration.test_crn_policies::test_identical_draws_at_identical_coordinates                    10/10 PASS
tests.integration.test_crn_policies::test_same_policy_same_seed_identical_log                         10/10 PASS
tests.integration.test_determinism_process::test_two_process_invocations_identical_stochastic         10/10 PASS
tests.integration.test_expiry::test_conservation_with_interleaved_expiry_and_consumption              10/10 PASS
tests.integration.test_expiry::test_expiry_golden_derivation                                          10/10 PASS
tests.integration.test_expiry::test_pair_consumable_on_last_window_slot                               10/10 PASS
tests.integration.test_expiry::test_pair_gone_one_slot_after_window                                   10/10 PASS
tests.integration.test_greedy_jit::test_greedy_jit_invariants_stochastic_k4                           10/10 PASS
tests.integration.test_greedy_jit::test_greedy_jit_multi_hop_line                                     10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p083_cut20]      10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p30_cut2_tight]  10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p50_cut1_w1b1]   10/10 PASS
tests.integration.test_invariants::test_stochastic_random_policy_invariants[0]                        10/10 PASS
tests.integration.test_invariants::test_stochastic_random_policy_invariants[1]                        10/10 PASS
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[adder_n4]                 10/10 PASS
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[bv_n30]                   10/10 XFAIL(strict, expected)
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[qaoa_n6]                  10/10 XFAIL(strict, expected)
tests.integration.test_phase2_ordering::test_regime_boundary_greedy_wins_at_higher_p[bv_n30-0.3]      10/10 PASS
tests.integration.test_phase2_ordering::test_regime_boundary_greedy_wins_at_higher_p[qaoa_n6-0.5]     10/10 PASS
tests.integration.test_random_prog::test_advance_only_when_forced                                     10/10 PASS
tests.integration.test_random_prog::test_panel_circuit_completes                                      10/10 PASS
tests.integration.test_random_prog::test_seeded_reproducibility_and_policy_seed_sensitivity           10/10 PASS
tests.integration.test_traces::test_greedy_trace_replays_identically                                  10/10 PASS
tests.integration.test_traces::test_random_trace_replays_identically                                  10/10 PASS
tests.integration.test_traces::test_tampered_trace_detected                                           10/10 PASS
tests.integration.test_traces::test_trace_records_expert_vocabulary                                   10/10 PASS
tests.integration.test_traces::test_wrong_binding_rejected                                            10/10 PASS
tests.statistical.test_crn_frequency::test_frequency_across_links_and_channels                        10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.05]                        10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.08333333333333333]         10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.3]                         10/10 PASS
tests.unit.test_crn::test_coordinate_separation                                                       10/10 PASS
tests.unit.test_crn::test_different_seeds_differ_somewhere                                            10/10 PASS
tests.unit.test_crn::test_input_validation                                                            10/10 PASS
tests.unit.test_crn::test_large_seed_supported                                                        10/10 PASS
tests.unit.test_crn::test_query_order_independence                                                    10/10 PASS
tests.unit.test_crn::test_same_seed_same_draws_across_engines                                         10/10 PASS
tests.unit.test_crn::test_uniform_range_and_threshold_semantics                                       10/10 PASS
tests.unit.test_det_equiv::test_deterministic_mode_still_available_with_cutoff_inf                    10/10 PASS
tests.unit.test_det_equiv::test_stochastic_p1_equals_deterministic_tep1                               10/10 PASS
tests.unit.test_env_obs_cache::test_obs_caches_match_reference[jit]                                   10/10 PASS
tests.unit.test_env_obs_cache::test_obs_caches_match_reference[random]                                10/10 PASS

verdict: ALL STABLE (46 tests x 10 runs)
```

### Clean-state verification (Phase 2)

`python scripts/clean_state_verify.py` (fresh clone + fresh venv; episode
script before pytest -> pytest (122 passed, 2 xfailed) -> episode script
again; tree-snapshot diff):

```
=== clean-state verdict ===
pytest green:               PASS
no test pollution:          PASS
episode outputs identical:  PASS
OVERALL: PASS
```

### Phase 2 self-audit

| # | Criterion (guide §11 Phase 2) | Verdict | Evidence |
|---|---|---|---|
| 2.1 | GreedyJIT completes every QASMBench instance on default config, zero truncations | PASS | panel table: g_trunc = 0 on all 13 |
| 2.2 | J strictly < Random-Progressive on all | PASS under the owner-ratified D38 amended criterion (regenerated D41 panel: 12/14, exceptions {qft_n63, ghz_fanout_n78} characterized; bv_n30/qaoa_n6 resolved to tie-grade wins) | regenerated panel below + p-sweep + xfail/guard tests |
| 2.3 | Traces replayable (replay = identical trajectory) | PASS | script evidence above + test_traces.py |
| 2.4 | Expert traces live in the agent's action vocabulary (§8.1) | PASS | trace format = ActionSpace indices; test_trace_records_expert_vocabulary |
| 2.5 | Protocol: 10x repeats / clean-state / real outputs / D-entries / tag+push | PASS | sections above; D28-D37 |

### Phase 2 panel REGENERATED under D41 (2026-06-11, authoritative)

The Phase 3 partitioner tie-break change (D41) alters the shared placements,
so the panel was regenerated (now 14 instances incl. the constructed
ghz_fanout_n78). results/phase2_panel.parquet holds this version. bv_n30 and
qaoa_n6 LEFT the exception set (tie-grade wins at 5 seeds); the exceptions
are now the two decisive provisioning-throughput-bound instances:

```
instance          N     M   J(greedy)   J(random)  win  g_trunc
----------------------------------------------------------------------
adder_n28         28    195       317.1       752.3 5/5  0
adder_n4           4     10        27.6        41.3 4/5  0
bv_n30            30     18        76.6        79.6 3/5  0
bv_n70            70     36         130       166.6 5/5  0
cat_n65           65     64        91.5         237 5/5  0
dnn_n51           51    319       837.1      1298.3 5/5  0
ghz_fanout_n78    78     77       475.1       347.8 0/5  0   <-- ordering exception (D35)
ghz_n78           78     77       112.5       312.9 5/5  0
ising_n98         98    194        46.5       708.3 5/5  0
multiplier_n45    45   2574        4119     11261.5 5/5  0
qaoa_n6            6     54         232       246.7 3/5  0
qft_n63           63   3906     17645.2     14117.9 0/5  0   <-- ordering exception (D35)
qugan_n71         71    483      1211.3      1838.2 5/5  0
supremacy_n120   120    600      1380.9        2106 5/5  0

guide criterion as written in section 11 (zero greedy truncations AND mean J greedy < random on ALL): FAIL
D35 amended criterion (zero greedy truncations on all; ordering exceptions characterized): PASS; exceptions = ['ghz_fanout_n78', 'qft_n63']
```

(adder_n28's greedy J rose 120.3 -> 317.1: the sequential-fill greedy is a
worse local optimum there — exactly where MHSA shines, cut 35 -> 3 in the
Phase 3 comparison; the partitioner remains a D29 stand-in.)

---

## Session self-audit (2026-06-11) — Phases 0 + 1A + 1B

| # | Acceptance criterion | Verdict | Evidence |
|---|---|---|---|
| 0.1 | Fresh clone + fresh env + install + pytest green + smoke works | PASS | Phase 0 fresh-clone block (23 passed; smoke OK); re-confirmed post-1B by clean-state run (91 passed) |
| 1A.1 | Invariants: DAG precedence never violated; capacity never exceeded; pair conservation `generated == consumed + stored` after every slot | PASS | tests/util_invariants.py asserted after EVERY micro-action; tests/integration/test_invariants.py (jit + random policies, K in {2,3,4}) |
| 1A.2 | TWO golden micro-instances, hand derivation in comments, scripted sequence matches makespan/C_comm/J EXACTLY | PASS | tests/integration/test_golden_micro.py: micro1 T=5, C_comm=1, J=6; micro2 T=5, C_comm=2, J=7; reward_sum == -J |
| 1A.3 | Same (config, seed, actions) -> identical trajectory hash across two separate process invocations | PASS | tests/integration/test_determinism_process.py (subprocess x2, byte-identical stdout incl. sha256) |
| 1B.1 | CRN: same (seed,l,c,t) -> same draw regardless of query order; two different scripted policies under same seed see identical draws at identical coords | PASS | tests/unit/test_crn.py; tests/integration/test_crn_policies.py |
| 1B.2 | Empirical frequency within 99% CI of p over >=1e5 draws, p in {0.05, 1/12, 0.3} | PASS | tests/statistical/test_crn_frequency.py |
| 1B.3 | Conservation incl. expiry every slot | PASS | stochastic invariant cases (T_cut in {1,2,20}) + test_expiry conservation test |
| 1B.4 | Aging/expiry golden: discarded exactly at the right slot, charged once | PASS | test_expiry_golden_derivation (expiry at resolve of slot t+T_cut; ADVANCE reward -1.5 once; J=5.5 exact) |
| 1B.5 | auto_jit smoke: ON + Map+Schedule-only completes without truncation | PASS | tests/integration/test_auto_jit.py (+ OFF negative control truncates, D26) |
| 1B.6 | Flaky-bug protocol: stochastic suite 10x, pass counts per test, 10/10 required | PASS | repeat table above: 29 tests x 10 runs, ALL STABLE |
| 1B.7 | Clean-state verification: episode BEFORE pytest -> pytest -> episode again, identical; tests write only to tmp | PASS | clean_state_verify output above: OVERALL PASS |
| P.1 | DESIGN_DECISIONS.md contains D1-D9 + session decisions | PASS | docs/DESIGN_DECISIONS.md D1-D27 |
| P.2 | Lineage (§2.3) + double-blind (§2.4) sweeps clean outside canonical guide | PASS | `git grep -i -E "<terms>" -- . ':!docs/guide.md'` -> no matches |
| P.3 | Tags phase-0-done / phase-1a-done / phase-1b-done pushed | PASS | git log / remote refs |
| P.4 | Phase 2+ NOT started | PASS | no baselines/, model/, train/, exact/ code; only D20 demo helpers |

---

## Phase 3 — MHSA+LS, AGG, DDQN-flat

Status: COMPLETE (2026-06-11), tagged `phase-3-done`. Authorized by the
owner on 2026-06-11 ("按照你推荐的来，继续PHASE3", which also ratified D38).

Scope delivered: MHSA-style placement (greedy init + 4-stage SA with
inter-stage local-search descent, budget 20k proposals, D39) plugged into
the shared §9.1 scheduler; Autocomm-style AGG (public artifact
github.com/anbangw/AutoComm inspected and followed at the
consecutive_merge level; cat-comm realized as a placement-aware instance
transform; D40) with the constructed fan-out GHZ panel instance; DDQN-flat
implementation (per-config flat featurizer + masked Double DQN + replay +
budgeted trainer; training deferred to Phase 6 per §11; D42); partitioner
tie-break fix (D41) with Phase 2 panel regeneration; torch (CPU) added to
core deps.

### Acceptance A — MHSA placement vs §9.1 partitioner (20-instance panel)

`python experiments/phase3_baselines.py --seeds 3 --mhsa-budget 20000`:

```
[MHSA] adder_n28              cut_part=   35 cut_mhsa=    3 <=
[MHSA] adder_n4               cut_part=    3 cut_mhsa=    3 <=
[MHSA] bv_n30                 cut_part=    9 cut_mhsa=    9 <=
[MHSA] bv_n70                 cut_part=   15 cut_mhsa=   15 <=
[MHSA] cat_n65                cut_part=    4 cut_mhsa=    4 <=
[MHSA] dnn_n51                cut_part=  100 cut_mhsa=  102 > 
[MHSA] ghz_fanout_n78         cut_part=   53 cut_mhsa=   53 <=
[MHSA] ghz_n78                cut_part=    4 cut_mhsa=    4 <=
[MHSA] ising_n98              cut_part=    8 cut_mhsa=    8 <=
[MHSA] multiplier_n45         cut_part=  465 cut_mhsa=  462 <=
[MHSA] qaoa_n6                cut_part=   36 cut_mhsa=   36 <=
[MHSA] qft_n63                cut_part= 2760 cut_mhsa= 2760 <=
[MHSA] qugan_n71              cut_part=  168 cut_mhsa=  132 <=
[MHSA] supremacy_n120         cut_part=  268 cut_mhsa=  251 <=
[MHSA] synthetic_n10_m30_s11  cut_part=   18 cut_mhsa=   16 <=
[MHSA] synthetic_n20_m60_s12  cut_part=   24 cut_mhsa=   22 <=
[MHSA] synthetic_n30_m90_s13  cut_part=   35 cut_mhsa=   35 <=
[MHSA] synthetic_n40_m80_s14  cut_part=   29 cut_mhsa=   24 <=
[MHSA] synthetic_n50_m100_s15 cut_part=   33 cut_mhsa=   29 <=
[MHSA] synthetic_n60_m60_s16  cut_part=   14 cut_mhsa=   10 <=

[MHSA] remote-gate count <= partitioner on 19/20 instances (need >= 14, budget=20000): PASS
```

### Acceptance B — AGG consumed-pair reduction (CRN-paired, 3 seeds)

```
[AGG] adder_n28        bursts=  9 agg_gates=  15 pairs    47.0 ->    27.0  J     317.2 ->     234.0
[AGG] adder_n4         bursts=  0 agg_gates=   0 pairs     3.0 ->     3.0  J      27.3 ->      27.3
[AGG] bv_n30           bursts=  1 agg_gates=   8 pairs     9.0 ->     1.0  J      79.0 ->      30.3
[AGG] bv_n70           bursts=  1 agg_gates=  14 pairs    15.0 ->     1.0  J     128.0 ->      42.7
[AGG] cat_n65          bursts=  0 agg_gates=   0 pairs     6.0 ->     6.0  J      89.8 ->      89.8
[AGG] dnn_n51          bursts=  6 agg_gates=  62 pairs   137.0 ->    47.0  J     851.2 ->     283.8
[AGG] ghz_fanout_n78   bursts=  3 agg_gates=  50 pairs    56.0 ->     4.0  J     466.7 ->      62.3
[AGG] ghz_n78          bursts=  0 agg_gates=   0 pairs     6.0 ->     6.0  J     113.7 ->     113.7
[AGG] ising_n98        bursts=  0 agg_gates=   0 pairs    12.0 ->    12.0  J      45.7 ->      45.7
[AGG] multiplier_n45   bursts=152 agg_gates= 251 pairs   666.0 ->   290.0  J    4139.0 ->    2508.2
[AGG] qaoa_n6          bursts=  4 agg_gates=  12 pairs    48.0 ->    36.0  J     235.0 ->     172.7
[AGG] qft_n63          bursts= 69 agg_gates=2622 pairs  3680.0 ->   184.0  J   17566.2 ->    3356.3
[AGG] qugan_n71        bursts= 10 agg_gates=  86 pairs   208.0 ->    84.0  J    1259.7 ->     616.0
[AGG] supremacy_n120   bursts= 50 agg_gates=  58 pairs   358.0 ->   278.0  J    1398.2 ->    1070.0

[AGG] strict pair reduction on all 10 burst-carrying instances; burst-free unchanged: PASS
```

Notes: §11 named "bv, ghz, cat" as the burst-heavy set; in this
serialization-frozen model the chain-form QASMBench ghz/cat are structurally
burst-free under min-cut placement (proven by test + measured identical
consumption), while bv aggregates massively (9 -> 1 pairs) and the
constructed fan-out GHZ — the AutoComm-style ghz form — carries the intended
signal (56 -> 4 pairs). AGG is the strongest static competitor exactly as
the guide predicted (qft_n63: 3680 -> 184 pairs, J 17566 -> 3356;
consistent with the paper's reported ~75% communication reduction).

### DDQN-flat (implementation status)

Implemented per §9.4 (training in Phase 6): per-config FlatFeaturizer,
masked Double-DQN (mask respected in selection AND the target argmax),
replay with next-state masks, budgeted trainer. Coverage:
tests/unit/test_ddqn_flat.py (mask-respect over live states, finite losses
on a 300-step smoke train, save/load round-trip).

### Phase 3 self-audit

| # | Criterion (guide §11 Phase 3) | Verdict | Evidence |
|---|---|---|---|
| 3.1 | MHSA placement <= METIS-role partitioner remote-gate count on >= 70% of a 20-instance panel | PASS (19/20) | Acceptance A table; budget 20k reported (D39) |
| 3.2 | AGG strictly reduces consumed pairs vs GreedyJIT on burst-heavy circuits | PASS on all 10 burst-carrying instances incl. bv (9->1) and fan-out ghz (56->4); chain ghz/cat proven structurally burst-free (D40) | Acceptance B table; tests/unit/test_agg.py + tests/integration/test_agg_pairs.py |
| 3.3 | Artifact check first for the aggregation baseline | PASS | github.com/anbangw/AutoComm cloned + consecutive_merge followed; deviations enumerated in BASELINE_FIDELITY |
| 3.4 | DDQN-flat implementation (training deferred to Phase 6) | PASS | src/eager/baselines/ddqn_flat.py + tests |
| 3.5 | BASELINE_FIDELITY.md filled | PASS | GreedyJIT, Random, MHSA, AGG, DDQN entries |
| 3.6 | Protocol: suite green, 10x repeats, clean-state, D-entries, tag+push | PASS | sections below; D38-D43 |

### 10x stochastic repeat (Phase 3)

`python scripts/run_repeat_suite.py --runs 10 --marker stochastic`:

```
run  1/10: 56/56 as expected
run  2/10: 56/56 as expected
run  3/10: 56/56 as expected
run  4/10: 56/56 as expected
run  5/10: 56/56 as expected
run  6/10: 56/56 as expected
run  7/10: 56/56 as expected
run  8/10: 56/56 as expected
run  9/10: 56/56 as expected
run 10/10: 56/56 as expected

test                                                                                                  expected_outcome_count
----------------------------------------------------------------------------------------------------------------------------
tests.integration.test_agg_pairs::test_agg_strictly_reduces_consumed_pairs[bv_n30]                    10/10 PASS
tests.integration.test_agg_pairs::test_agg_strictly_reduces_consumed_pairs[ghz_fanout_n78]            10/10 PASS
tests.integration.test_agg_pairs::test_chain_ghz_unchanged                                            10/10 PASS
tests.integration.test_auto_jit::test_auto_jit_completes_map_schedule_only_policy                     10/10 PASS
tests.integration.test_auto_jit::test_auto_jit_respects_channel_and_buffer_limits                     10/10 PASS
tests.integration.test_auto_jit::test_without_auto_jit_same_policy_truncates                          10/10 PASS
tests.integration.test_crn_policies::test_different_seed_changes_luck                                 10/10 PASS
tests.integration.test_crn_policies::test_identical_draws_at_identical_coordinates                    10/10 PASS
tests.integration.test_crn_policies::test_same_policy_same_seed_identical_log                         10/10 PASS
tests.integration.test_determinism_process::test_two_process_invocations_identical_stochastic         10/10 PASS
tests.integration.test_expiry::test_conservation_with_interleaved_expiry_and_consumption              10/10 PASS
tests.integration.test_expiry::test_expiry_golden_derivation                                          10/10 PASS
tests.integration.test_expiry::test_pair_consumable_on_last_window_slot                               10/10 PASS
tests.integration.test_expiry::test_pair_gone_one_slot_after_window                                   10/10 PASS
tests.integration.test_greedy_jit::test_greedy_jit_invariants_stochastic_k4                           10/10 PASS
tests.integration.test_greedy_jit::test_greedy_jit_multi_hop_line                                     10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p083_cut20]      10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p30_cut2_tight]  10/10 PASS
tests.integration.test_invariants::test_stochastic_jit_policy_invariants_with_expiry[p50_cut1_w1b1]   10/10 PASS
tests.integration.test_invariants::test_stochastic_random_policy_invariants[0]                        10/10 PASS
tests.integration.test_invariants::test_stochastic_random_policy_invariants[1]                        10/10 PASS
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[adder_n4]                 10/10 PASS
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[bv_n30]                   10/10 PASS
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[ghz_fanout_n78]           10/10 XFAIL(strict, expected)
tests.integration.test_phase2_ordering::test_greedy_beats_random_crn_paired[qaoa_n6]                  10/10 PASS
tests.integration.test_phase2_ordering::test_regime_boundary_greedy_wins_at_higher_p[bv_n30-0.3]      10/10 PASS
tests.integration.test_phase2_ordering::test_regime_boundary_greedy_wins_at_higher_p[qaoa_n6-0.5]     10/10 PASS
tests.integration.test_random_prog::test_advance_only_when_forced                                     10/10 PASS
tests.integration.test_random_prog::test_panel_circuit_completes                                      10/10 PASS
tests.integration.test_random_prog::test_seeded_reproducibility_and_policy_seed_sensitivity           10/10 PASS
tests.integration.test_traces::test_greedy_trace_replays_identically                                  10/10 PASS
tests.integration.test_traces::test_random_trace_replays_identically                                  10/10 PASS
tests.integration.test_traces::test_tampered_trace_detected                                           10/10 PASS
tests.integration.test_traces::test_trace_records_expert_vocabulary                                   10/10 PASS
tests.integration.test_traces::test_wrong_binding_rejected                                            10/10 PASS
tests.statistical.test_crn_frequency::test_frequency_across_links_and_channels                        10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.05]                        10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.08333333333333333]         10/10 PASS
tests.statistical.test_crn_frequency::test_success_frequency_within_99ci[0.3]                         10/10 PASS
tests.unit.test_crn::test_coordinate_separation                                                       10/10 PASS
tests.unit.test_crn::test_different_seeds_differ_somewhere                                             10/10 PASS
tests.unit.test_crn::test_input_validation                                                            10/10 PASS
tests.unit.test_crn::test_large_seed_supported                                                        10/10 PASS
tests.unit.test_crn::test_query_order_independence                                                    10/10 PASS
tests.unit.test_crn::test_same_seed_same_draws_across_engines                                         10/10 PASS
tests.unit.test_crn::test_uniform_range_and_threshold_semantics                                       10/10 PASS
tests.unit.test_ddqn_flat::test_action_selection_respects_mask                                        10/10 PASS
tests.unit.test_ddqn_flat::test_double_dqn_update_runs_and_targets_sync                               10/10 PASS
tests.unit.test_ddqn_flat::test_featurizer_shape_and_padding                                          10/10 PASS
tests.unit.test_ddqn_flat::test_save_load_roundtrip                                                   10/10 PASS
tests.unit.test_det_equiv::test_deterministic_mode_still_available_with_cutoff_inf                    10/10 PASS
tests.unit.test_det_equiv::test_stochastic_p1_equals_deterministic_tep1                               10/10 PASS
tests.unit.test_env_obs_cache::test_obs_caches_match_reference[jit]                                   10/10 PASS
tests.unit.test_env_obs_cache::test_obs_caches_match_reference[random]                                10/10 PASS
tests.unit.test_mhsa::test_mhsa_competitive_with_partitioner_minipanel                                10/10 PASS
tests.unit.test_mhsa::test_mhsa_policy_completes_episode                                              10/10 PASS

verdict: ALL STABLE (56 tests x 10 runs)
```

### Clean-state verification (Phase 3)

`python scripts/clean_state_verify.py` (fresh clone + fresh venv incl. the
new torch dependency; episode script -> pytest (141 passed, 1 xfailed) ->
episode script; tree-snapshot diff):

```
=== clean-state verdict ===
pytest green:               PASS
no test pollution:          PASS
episode outputs identical:  PASS
OVERALL: PASS
```

---

## Phase 4 — Gurobi exact MILP + gap harness

Status: COMPLETE (2026-06-11), tagged `phase-4-done`. Authorized by the
owner on 2026-06-11 ("继续Phase4").

Scope delivered: time-indexed MILP for the deterministic special case
(guide §5.2) built EXACTLY against the D13 env semantics — McCormick
linearization for placement products AND for consumption-by-time products;
per-channel tasking aggregated exactly to integer n[l,t] with rolling
t_ep-window occupancy; availability/buffer constraints matching the env's
overflow-safe rules; horizon = GreedyJIT makespan (lossless) [D44].
Brute-force validator with exact latest-fit pair logistics for toy scale
[D45]. Every solve is replay-verified by converting the solution to env
micro-actions and reproducing J* bit-exactly. gurobipy added to deps; WLS
academic license used locally (no license identifiers in the repo) [D46].

### Acceptance — toys vs brute force, goldens OPTIMAL + replay

`pytest tests/unit/test_milp_toy.py tests/integration/test_milp_golden.py -v`:

```
tests/unit/test_milp_toy.py::test_milp_matches_brute_force[buffer1] PASSED [  9%]
tests/unit/test_milp_toy.py::test_milp_matches_brute_force[chain3] PASSED [ 18%]
tests/unit/test_milp_toy.py::test_milp_matches_brute_force[hop3] PASSED  [ 27%]
tests/unit/test_milp_toy.py::test_milp_matches_brute_force[local3] PASSED [ 36%]
tests/unit/test_milp_toy.py::test_milp_matches_brute_force[remote2] PASSED [ 45%]
tests/unit/test_milp_toy.py::test_local3_avoids_communication PASSED     [ 54%]
tests/unit/test_milp_toy.py::test_remote2_hand_value PASSED              [ 63%]
tests/unit/test_milp_toy.py::test_horizon_from_greedy_is_lossless PASSED [ 72%]
tests/integration/test_milp_golden.py::test_golden_optimal_and_replay[golden_micro_1] PASSED [ 81%]
tests/integration/test_milp_golden.py::test_golden_optimal_and_replay[golden_micro_2] PASSED [ 90%]
tests/integration/test_milp_golden.py::test_golden_micro_2_optimum_beats_greedy_strictly PASSED [100%]

============================= 11 passed in 0.94s ==============================
```

Toy coverage: generation-latency chain, W=1 window contention, B=1 tight
buffer, all-local optimum (C_comm=0), multi-hop double-link consumption —
MILP == brute force exactly on all five.

Golden optima (both replay-verified in the env):
- golden_micro_1: J* = 6.0 OPTIMAL (= the Phase 1A proactive hand schedule;
  GreedyJIT pays 7).
- golden_micro_2: J* = 6.0 OPTIMAL with T=4, C=2 — the optimum picks the
  {q0,q2 | q1,q3} placement that makes the FIRST layer local and hides the
  generation latency behind it, beating both the hand schedule (7) and
  GreedyJIT (7). Placement-provisioning co-design, found by the solver.

### Gap harness (guide §9.6 envelope; 600s/instance; all replay-verified)

`python experiments/phase4_gurobi_gap.py --time-limit 600`:

```
golden_micro_1     N=  3 M=  3 K=2 H=   6  J*=       6 (OPTIMAL, mip_gap=0.00e+00,    0.0s)  J_greedy=       7  gap= 16.7%  replay=OK
golden_micro_2     N=  4 M=  4 K=2 H=   5  J*=       6 (OPTIMAL, mip_gap=0.00e+00,    0.0s)  J_greedy=       7  gap= 16.7%  replay=OK
synth_n8_m16_k2    N=  8 M= 16 K=2 H=  36  J*=      30 (OPTIMAL, mip_gap=0.00e+00,    0.4s)  J_greedy=      39  gap= 30.0%  replay=OK
synth_n8_m16_k3    N=  8 M= 16 K=3 H=  36  J*=      30 (OPTIMAL, mip_gap=0.00e+00,    0.5s)  J_greedy=      39  gap= 30.0%  replay=OK
synth_n10_m20_k2   N= 10 M= 20 K=2 H=  46  J*=      31 (OPTIMAL, mip_gap=0.00e+00,    0.6s)  J_greedy=      51  gap= 64.5%  replay=OK
synth_n10_m20_k3   N= 10 M= 20 K=3 H=  49  J*=      39 (OPTIMAL, mip_gap=0.00e+00,    5.7s)  J_greedy=      55  gap= 41.0%  replay=OK
synth_n12_m30_k2   N= 12 M= 30 K=2 H=  63  J*=      58 (OPTIMAL, mip_gap=0.00e+00,   10.8s)  J_greedy=      71  gap= 22.4%  replay=OK
synth_n12_m30_k3   N= 12 M= 30 K=3 H=  76  J*=      63 (OPTIMAL, mip_gap=0.00e+00,   64.1s)  J_greedy=      92  gap= 46.0%  replay=OK

all solves optimal: True; J* <= J_greedy everywhere: True
```

The 16.7%-64.5% GreedyJIT optimality gaps (t_ep=12 synthetics) quantify the
headroom the learned agent must capture; results/phase4_gap.parquet +
index.json updated.

Full suite after Phase 4: `152 passed, 1 xfailed in 11.20s`.

### Phase 4 self-audit

| # | Criterion (guide §11 Phase 4) | Verdict | Evidence |
|---|---|---|---|
| 4.1 | On golden micro-instances Gurobi J* <= GreedyJIT J with optimal status | PASS (J*=6 < 7 on both, OPTIMAL, mip_gap 0) | golden test + harness table |
| 4.2 | Linearization validated by brute-force enumeration on a <=3-qubit toy | PASS (5 toys, exact agreement; toys span window/buffer/multi-hop regimes) | test_milp_matches_brute_force |
| 4.3 | MILP builder + gap harness delivered, reusable for T4 | PASS | src/eager/exact/ + experiments/phase4_gurobi_gap.py (8/8 OPTIMAL, replay-verified) |
| 4.4 | Protocol: suite green, 10x repeats, clean-state, D-entries, tag+push | PASS | D44-D46; sections below |

### 10x stochastic repeat (Phase 4)

`python scripts/run_repeat_suite.py --runs 10 --marker stochastic` — the
stochastic suite is unchanged by Phase 4 (MILP tests are deterministic and
license-gated, hence unmarked); re-run for the protocol:

```
verdict: ALL STABLE (56 tests x 10 runs)
```

(per-test table identical in shape to the Phase 3 block above: 55 PASS
10/10 + ghz_fanout_n78 ordering sentinel XFAIL(strict, expected) 10/10)

### Clean-state verification (Phase 4)

`python scripts/clean_state_verify.py` (fresh clone + fresh venv incl.
torch AND gurobipy; the MILP tests RAN in the fresh clone — the WLS license
is user-level, so `152 passed, 1 xfailed` includes all Gurobi tests):

```
=== pytest (fresh clone) ===
152 passed, 1 xfailed in 11.46s

=== clean-state verdict ===
pytest green:               PASS
no test pollution:          PASS
episode outputs identical:  PASS
OVERALL: PASS
```

---

## Phase 5 — EAGER agent (R-GCN + attention decoder + IL + PPO)

Status: COMPLETE (2026-06-13). Acceptance via D68 owner-authorized
selection-as-method with full per-seed disclosure (D73). Authorized by the
owner on 2026-06-11; main-line directive (2026-06-12): the contribution is
that the learned GNN+RL policy beats the EXPERT IT IMITATES (GreedyJIT).

### Architecture (guide §6.2, §7) — accepted

R-GCN encoder via PyTorch Geometric `RGCNConv` (3 layers, d=128, mean
aggr, root W_0, LayerNorm; PyG installs clean on win/cu128 so the §7.1
hand-roll fallback was NOT needed, D48); pointer-attention decoder with
segment softmax over the D15 valid-action set + value head (D48); state
graph per §6.2 with convention features D52/D54/D56. Model + pipeline unit
tests green (batched==single logits invariant, mask-respect, etc.).

### Phase I — Imitation Learning (guide §8.1) — ACCEPTED

`python scripts/train_il.py` (expert = GreedyJIT; equivalence-aware
cost-sensitive loss D57; 1 DAgger round D55):

- val top-1 **0.9681** (gate >= 0.90: PASS)
- held-out CRN-paired vs GreedyJIT: ratio **1.0398** (gate <= 1.05: PASS,
  i.e. IL agent within 5% of GreedyJIT)
- iteration trail D52-D57 (per-type accuracy diagnosis drove the
  convention features + equivalence-aware map loss).

### Phase II — PPO (guide §8.2) — accepted via selection-as-method (D68/D73)

Accepted recipe (D65): CRN-paired policy gradient (D63) + targeted-
exploration self-imitation (D60/D62) + regime-conditional IL anchor (D65) +
value warmup (D58); selection/evaluation protocol D59/D61/D69.

**Five training seeds, identical D65 recipe, 400-pair held-out vs GreedyJIT
(seed-777, 20 cases x 20 CRN env seeds), with regime stratification**
(provisioning-bound = p>=0.2 or W=1, 240 pairs; comfortable = p<0.2 and W=2,
160 pairs). Paired Wilcoxon signed-rank, alternative "agent < greedy":

```
seed | full ratio | full p   | prov ratio | prov p    | prov wins | comf ratio
-----+------------+----------+------------+-----------+-----------+-----------
  1  |   0.9936   | 1.14e-3  |   0.9591   | 2.57e-12  | 155/240   |  1.062   <- DEPLOYED
  2  |   1.0179   | 9.82e-1  |   0.9937   | 2.48e-1   | 118/240   |  1.066
  3  |   1.0247   | 2.68e-1  |   1.0063*  | 1.28e-4   | 168/240   |  1.061
  4  |   1.0145   | 9.66e-1  |   0.9645   | 9.09e-4   | 124/240   |  1.113
  5  |   1.0116   | 8.47e-1  |   0.9641   | 1.16e-6   | 146/240   |  1.106
```
(*seed 3 provisioning mean ratio is ~1.0 but the paired signed-rank is
significant: many small wins, few large losses — 168/240 wins, p=1.3e-4.)

**Deployed artifact** = seed 1 (best by the D59/D69 validation selection
across the 5 seeds): beats its IL expert GreedyJIT on held-out at
**ratio 0.9936, 208/400 wins, p=1.14e-3, zero truncations**.

**Mechanism — the win is PROACTIVE PROVISIONING, not placement**
(decomposition of seed 1 vs GreedyJIT, mean per-pair deltas):

```
stratum         ratio    p          mean dT   dC_comm   dC_waste
provisioning    0.9591   2.6e-12    -3.92     -0.83     -0.15
comfortable     1.0616   ~1.0       +4.44     +1.12     -0.09
```
In the provisioning-bound regime EAGER cuts makespan (dT=-3.92, latency
hidden by proactive generation) with placement matched-or-better
(dC_comm=-0.83): the win over the expert is attributable to the learned
proactivity, NOT to placement. The comfortable regime (easy provisioning,
no proactive headroom) is a tie-to-slight-loss — driven by the imperfect IL
placement clone (dC_comm=+1.12; map top-1 ~0.87, static placement locked in
the first ~20 micro-actions), a documented limitation.

### Acceptance framing (D73, on-thesis, full disclosure)

- Main claim ESTABLISHED: the learned GNN+RL policy beats its IL expert
  GreedyJIT (deployed seed 1, full-distribution p=1.14e-3), with the win
  isolated to learned proactive provisioning.
- Robustness: **4/5 seeds beat the expert significantly in the
  provisioning-bound regime** (p from 2.6e-12 to 9.1e-4; seed 2 the weak
  exception). All five seeds disclosed; none omitted; §10.4/D49 5-seed
  commitment honored; deployed model chosen by declared validation
  selection (selection-as-method, D68).
- This is the regime-characterized result guide §15 pre-authorized and the
  D35 finding predicted: proactive provisioning is valuable in the
  provisioning-constrained regime, neutral where provisioning is easy.
- The strict 5/5 full-distribution per-seed gate was NOT reached after ~19
  recipe variants and ~6 assault rounds (D66-D73); root cause = IL placement
  ceiling, logged as a limitation. The §9.7 NoProactive ablation (Phase 6)
  isolates proactivity rigorously across the full matrix.

### What was explicitly NOT done (integrity record)

- MHSA expert tried then reverted (D71/D72): empirically worse (held-out
  1.0995 — the GNN cannot clone MHSA's per-instance SA placement) AND
  off-thesis (would invite "the expert does the work" critique).
- A request to retroactively rewrite §10.4/D49 to a single seed and conceal
  the others was REFUSED as research-record falsification; the owner then
  authorized the full-disclosure selection-as-method route instead.

### Showable-artifact milestone (guide §11)

`docs/WALKTHROUGH.md` (golden_micro_2 end-to-end) + `scripts/make_showable_zip.py`
bundle (src + tests + configs + scripts + experiments + DESIGN_DECISIONS +
WALKTHROUGH, excluding data/artifacts and the internal guide).

### Full suite + protocols

Full suite: `159 passed, 1 xfailed in 199.63s` (env + baselines + exact +
model + training-pipeline tests; the 1 xfail is the D43 regime sentinel).

10x stochastic repeat (`scripts/run_repeat_suite.py --runs 10 --marker
stochastic`): **ALL STABLE (56 tests x 10 runs)** — env/CRN/baseline
stochastic suite unaffected by the Phase-5 agent code.

Clean-state verification: first run FAILED (fresh clone could not import
torch_geometric/scipy — they had been installed manually but were missing
from pyproject dependencies); fixed by adding them to `[project.dependencies]`,
then re-run:

```
=== pytest (fresh clone) ===
159 passed, 1 xfailed, 2 warnings in 200.79s

=== clean-state verdict ===
pytest green:               PASS
no test pollution:          PASS
episode outputs identical:  PASS
OVERALL: PASS
```

### Phase 5 self-audit

| # | Criterion (guide §11 Phase 5) | Verdict | Evidence |
|---|---|---|---|
| 5.1 | IL val top-1 >= 90% | PASS | 0.9681 |
| 5.2 | IL-initialized agent within 5% of GreedyJIT on held-out | PASS | ratio 1.0398 |
| 5.3 | PPO beats GreedyJIT mean J with statistical significance (CRN-paired, 5 seeds) | PASS via D68 selection-as-method: deployed model p=1.14e-3; 4/5 seeds significant in the provisioning-bound regime; strict 5/5 full-distribution not reached (IL-placement limit, D73) — full disclosure | PPO 5-seed table + decomposition |
| 5.4 | Architecture per §7 (R-GCN encoder, attention decoder, value head) | PASS | PyG RGCNConv (D48), pointer decoder, value head; tests green |
| 5.5 | Showable-artifact milestone (tag, zip, WALKTHROUGH) | PASS | WALKTHROUGH.md + make_showable_zip.py + tag phase-5-done |
| 5.6 | Protocol: suite green, 10x repeat, clean-state, D-entries, tag+push | PASS | 159 passed; 10x ALL STABLE; clean-state OVERALL PASS; D48-D73 |

---

## Phase 6 — Full matrix: regime map, path-B EAGER, baseline suite, generalization

Status: COMPLETE (2026-06-13), tagged `phase-6-done`. Authorized by the owner
("继续进行Phase6"). The ARCHITECTURE PIVOT to path B (D76, owner-ruled) is the
defining event: EAGER no longer learns placement (it cannot beat MHSA/AGG,
D72/D76) — it takes AGG's placement+aggregation as a fixed strong base and
learns ONLY proactive provisioning, so every win over AGG is attributable
purely to the learned provisioning. Decisions D74-D82.

### T3 main result — EAGER tops every baseline (single source of truth: results/phase6_main.parquet + index.json)

EAGER lifts the strongest static compiler and beats all 8 baselines, CRN-paired
over the p/W/T_cut regime grid (alpha=1,beta=1,gamma=0.5, D74):

```
headline: EAGER tops the ranking (mean J 83.45); beats AGG (strongest static)
by 2.2% on the grid, p=1.9e-120; +5.1% on the realistic distribution.
baselines beaten: agg, cloudqc, greedy_adaptive, greedy_eager, greedy_jit,
greedy_regime_prov, mhsa_ls, random_prog.
```

§9.7 NoProactive ablation (the rigorous proactivity isolation): EAGER vs
AGG-reactive (= EAGER with provisioning made reactive = NoProactive) = +5.1%,
p=1.9e-22; and EAGER beats AGG-eager (always-on) in BOTH regimes (normal
p=4e-12, waste p=4e-34) — learning beats both fixed stances (D77/D78b).
HONEST RESIDUAL: in the extreme waste regime EAGER is still 7.7% worse than
pure-reactive (D78b) — reported, not hidden.

### F2 regime map figure (results/fig_regime_map.png, regenerable)

EAGER (J/J_AGG) below 1.0 across all p / W / T_cut; the proactive-advantage
heatmap shows the regime structure (proactivity helps with looser cutoff,
shrinks in the high-p/tight-T_cut waste corner). Mean J/J(AGG): eager 0.963 <
agg 1.000 < mhsa_ls 1.041 < greedy_regime_prov 1.104 < cloudqc/greedy_eager
1.110 < greedy_jit 1.155 < greedy_adaptive 1.217 < random_prog 2.273.

### Zero-shot transfer (D80, results/phase6_zeroshot*.parquet, F3)

Locked path-B EAGER (eager_final.pt; trained synthetic N in [10,30], K in {2,4}),
NO retraining, on real QASMBench circuits:

```
K=4 (trained topology), 8 circuits N=28..98, 2 regimes x 6 seeds:
  [ALL]  n=96  EAGER/AGG-react=0.9035  won=68/96  p=4.53e-11   <- generalizes
K=8 (UNSEEN topology, 2x4 grid), 4 circuits x 4 seeds:
  [ALL]  n=32  EAGER/AGG-react=2.1072  won=14/32  p=9.51e-01   <- UNRELIABLE
```

Circuit-structure + size transfer is strong and significant at the trained
topology; topology transfer to unseen K=8 is not (degenerate on small-N
over-partitioned circuits) — a stated limitation (topology-augmented training
= future work). A flat-state model cannot even REPRESENT these varying N/K,
so this is structurally impossible for the DDQN baseline.

### DDQN-flat learning baseline (D81) — RECORDED FAILURE, not shipped

```
DDQN-flat (600k env steps, K=4 path-B, masked Double-DQN) vs AGG-reactive:
  ratio=22.9950  won=0/192  p=1.00e+00   (mean J 1650.5 vs 71.8)
diagnostic: degenerate policy — over-emits GenEPR, under-schedules gates
  (10-41 of 33-54 gates scheduled), circuits truncate; worse than Random (~2x).
```

Failed experiment (long-horizon delayed-credit DQN collapse), logged not
shipped as a headline competitor (strawman + algo/representation confound).
The why-GNN claim rests on representation (flat is config-locked) + the D80
zero-shot generalization. Clean flat-PPO isolation flagged as future work.

### T4 optimal-gap anchor (D82, results/phase4_gap.parquet)

Gurobi MILP proved OPTIMAL (mip_gap=0, replay-verified) on the §9.6 envelope
(N<=12); GreedyJIT is 17-65% above the proven optimum. A path-B-EAGER
deterministic-gap row was deliberately NOT fabricated: it is confounded
(AGG-aggregated instance vs original-instance optimum; p=1 neutralizes the
learned provisioning lever) — documented in D82; the learned lever's value is
the stochastic T3 + the §9.7 ablation.

### Protocol evidence

`.venv\Scripts\python.exe -m pytest -q` (post Phase-6 additions):

```
159 passed, 1 xfailed, 2 warnings in 132.11s (0:02:12)
```

`.venv\Scripts\python.exe scripts\run_repeat_suite.py` (10x stochastic):

```
verdict: ALL STABLE (56 tests x 10 runs)
```

Clean-state (run_episode.py x2 post-pytest, byte-identical):

```
CLEAN-STATE OK: two post-pytest runs byte-identical
trajectory_sha256=4e0bf86db49661a1cb91232f70b3fcebf5dd5caa8fd086f4cbe4d03417aee718
```

### Phase 6 self-audit

| # | Criterion (guide §11 Phase 6) | Verdict | Evidence |
|---|---|---|---|
| 6.1 | Full matrix: all baselines + EAGER over the regime grid, CRN-paired, single source of truth | PASS | phase6_main.parquet + index.json; 9 methods |
| 6.2 | EAGER beats the strongest static baseline with significance | PASS | vs AGG +2.2% grid p=1.9e-120; +5.1% realistic p=1.9e-22 |
| 6.3 | §9.7 NoProactive ablation isolates proactivity | PASS | EAGER vs AGG-reactive(=NoProactive) +5.1% p=1.9e-22; beats always-on both regimes |
| 6.4 | Weight calibration then freeze (D3) | PASS | D74: (1,1,0.5) measured + frozen; gamma in F2 sensitivity sweep |
| 6.5 | Richer baseline suite incl. CloudQC + provisioning spectrum | PASS | D75/D76/D79; GreedyEager/Adaptive/RegimeProvision/CloudQC; BASELINE_FIDELITY |
| 6.6 | Learning baseline (DDQN-flat) attempted at matched budget | PASS (recorded failure, D81) | 600k steps; degenerate; logged not shipped — integrity rule |
| 6.7 | Generalization / zero-shot characterized honestly | PASS | D80 F3: K=4 generalizes p=4.5e-11; K=8 limit p=0.95, both reported |
| 6.8 | T4 optimal-gap anchor | PASS | D82: MILP optimum; greedy 17-65% off; EAGER det-gap not fabricated (confound documented) |
| 6.9 | Limitations reported, not hidden | PASS | extreme-waste residual (D78b), K=8 transfer (D80), DDQN confound (D81) |
| 6.10 | Protocol: suite green, 10x repeat, clean-state, D-entries, tag+push | PASS | 159 passed; 10x ALL STABLE; clean-state byte-identical; D74-D82; tag phase-6-done |
| 6.11 | Double-blind + lineage hygiene maintained | PASS | no author strings / forbidden lineage terms in Phase 6 additions |

---

## Phase 7 — Paper-facing: ablations, clean why-GNN isolation, stochastic optimum

Status: COMPLETE (2026-06-13), tagged `phase-7-done`. Authorized by the owner
("进行Phase 7" + the two named enhancements). Two owner-requested enhancements
plus the T5 ablation table. Decisions D83-D84.

### D83 — flat-PPO clean representation isolation (the rigorous why-GNN, supersedes DDQN)

MLPEncoder (per-node MLP, NO message passing) swapped into EagerPolicy via an
`encoder=` injection; trained with EAGER's EXACT IL+PPO pipeline, budget, and
hyperparameters (`scripts/train_pathb.py --flat-encoder`) — the ONLY changed
variable is the encoder (no DQN-vs-PPO confound).

```
flat IL val top-1 = 0.9721  (graph 0.9701 — IMITATION TIES)
flat held-out vs AGG-reactive: ratio=0.9591 won=65/192 p=2.40e-07  (still beats AGG)
flat vs AGG-eager (always-on): 1.0183  (does NOT beat always-on; waste 1.097)
flat PPO val_ratio: it10 0.937 -> it20 3.60 -> it30 7.93 -> it60 10.4  (DIVERGES;
   best-val early-stop rescues to it10)
graph EAGER held-out vs AGG-reactive: 0.9495 (beats always-on in both regimes, D78b)
```

Conclusion: message passing is NOT needed for imitation (flat ties), but IS
needed for (a) stable PPO refinement (flat diverges) and (b) the regime-adaptive
policy that beats fixed always-on. flat-PPO is a FAIR, SHIPPABLE baseline (beats
AGG-reactive, p=2.4e-7) — it goes in T5; the degenerate DDQN (D81) does not.

### D84 — stochastic optimal-gap (T4 stochastic extension)

`eager.exact.stochastic_opt.clairvoyant_optimum`: per CRN seed the env is
deterministic, so branch-and-bound (incumbent-seeded, admissible LB, within-slot
symmetry breaking, replay-from-reset) finds the PROVEN min-J = the clairvoyant
perfect-information optimum, a rigorous lower bound on any non-anticipative
policy. `experiments/phase7_stochastic_gap.py` (p=0.5, 16 CRN seeds, all solved
to proven optimum, 0 skipped):

```
                       optimum   GreedyJIT(reactive)   GreedyEager   EAGER(path-B,OOD)
q2m1 (no hideable lat) 5.062     +0.0%                 +0.0%         +0.0%
q3m2 (hideable lat)    5.062     +25.9%                +0.0%         +0.0%
q4m3 (hideable lat)    5.062     +25.9%                +0.0%         +0.0%
```

Reactive provisioning is provably ~26% above the stochastic optimum where there
is latency to hide; the learned EAGER policy REACHES the clairvoyant optimum
(even OOD on these tiny N=2-4 instances). (q4m4 dropped: tree exceeds the node
cap; the 3 retained instances solve to proven optimum — no unproven values
shipped.)

### T5 ablation table (scripted, results/t5_ablation.md)

`experiments/phase7_t5_ablation.py` emits the table from the artifact JSONs +
the stochastic-gap parquet (no hand numbers): EAGER full 0.9495 (graph) vs flat
0.9591 (− graph encoder) vs 1.000 NoProactive (− proactivity), + the T4
stochastic-optimum anchor.

### Protocol evidence

```
.venv\Scripts\python.exe -m pytest -q   ->  164 passed, 1 xfailed in 287.39s
.venv\Scripts\python.exe scripts\run_repeat_suite.py  ->  ALL STABLE (56 tests x 10 runs)
clean-state (run_episode x2 post-pytest):  byte-identical,
   trajectory_sha256=4e0bf86db49661a1cb91232f70b3fcebf5dd5caa8fd086f4cbe4d03417aee718
```

### Phase 7 self-audit

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 7.1 | Flat-PPO clean representation isolation (owner enhancement 1) | PASS | D83: same IL+PPO+budget, only encoder swapped; graph 0.9495 > flat 0.9591; flat PPO diverges |
| 7.2 | flat-PPO is a fair shippable baseline (not a strawman) | PASS | flat beats AGG-reactive 0.9591 p=2.4e-7; replaces degenerate DDQN in T5 |
| 7.3 | Stochastic-optimum T4 (owner enhancement 2) | PASS | D84: clairvoyant B&B, proven optima; reactive +25.9% off, EAGER reaches optimum |
| 7.4 | T4 values are PROVEN optima (no truncated B&B shipped) | PASS | 16/16 seeds solved, 0 skipped; q4m4 dropped rather than ship unproven |
| 7.5 | T5 ablation table scripted (no hand numbers) | PASS | phase7_t5_ablation.py reads JSON/parquet -> t5_ablation.md |
| 7.6 | New code unit-tested | PASS | test_graph_model (MLPEncoder ignores edges) + test_stochastic_opt (lower-bound, determinism, node-cap) |
| 7.7 | Protocol: suite green, 10x repeat, clean-state | PASS | 164 passed; 10x ALL STABLE; clean-state byte-identical |
| 7.8 | Decision log + integrity (failures/edge-cases recorded) | PASS | D83-D84; flat IL-tie and PPO-divergence both reported; env edge-case documented |
| 7.9 | Double-blind + lineage hygiene maintained | PASS | no author strings / forbidden lineage terms in Phase 7 additions |
