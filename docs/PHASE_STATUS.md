# PHASE_STATUS — EAGER (eager-dqc)

> Live state for artifact-based resume. Updated after every major step.
> Acceptance evidence below is REAL pasted command output, never summarized
> from memory.

## Current state

- **Current phase**: Phase 1B (stochastic layer) — starting
- **Last completed step**: Phase 1A closed (62 tests green, golden micros
  matched exactly, cross-process determinism verified)
- **Exact next step**: CRN engine (`eager/env/crn.py`) + stochastic resolve +
  Phase 1B acceptance tests
- **Blockers**: none

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
rootdir: C:\Users\jcshe\AppData\Local\Temp\eager_fresh_p0
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

Status: not started.
