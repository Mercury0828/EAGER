# PHASE_STATUS — EAGER (eager-dqc)

> Live state for artifact-based resume. Updated after every major step.
> Acceptance evidence below is REAL pasted command output, never summarized
> from memory.

## Current state

- **Current phase**: Phase 0 (scaffold) — acceptance evidence collection
- **Last completed step**: pytest green (23 passed) + smoke OK in working tree
- **Exact next step**: fresh-clone acceptance run, then tag phase-0-done; then
  Phase 1A env core
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

Status: not started.

## Phase 1B — Stochastic layer

Status: not started.
