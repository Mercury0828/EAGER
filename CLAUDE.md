# CLAUDE.md — EAGER (eager-dqc)

Learned orchestration of qubit placement, gate scheduling, and proactive
entanglement provisioning for distributed quantum computing over stochastic
quantum networks. Target: IEEE INFOCOM 2027.

## Session-start protocol (do this before any work)

1. Read `docs/guide.md` IN FULL. It is the canonical design document; it wins
   over code comments, this file, and any session prompt unless
   `docs/DESIGN_DECISIONS.md` records an explicit override.
2. Read `docs/PHASE_STATUS.md` (current phase, last completed step, exact next
   step, blockers).
3. Run `git log --oneline -10` and reconcile with PHASE_STATUS.md; if they
   disagree, trust files+git and append a correction note.
4. State the resumed task in one line, then work.

## Non-negotiables

- **Canonical doc**: `docs/guide.md` governs the system model, MDP, agent,
  baselines, experiments, and phase plan. Do not fork or silently rewrite it.
- **Phase gating**: work only on the phase(s) the session is authorized for
  (guide §11). Acceptance criteria are checked with REAL command output pasted
  into `docs/PHASE_STATUS.md` — never summarized from memory, never trimmed.
  Each phase closes with a self-audit table, a git tag `phase-X-done`, and a
  push to the remote.
- **Test hygiene / no test pollution**: tests write only to pytest tmp dirs.
  `results/` is written only by `experiments/`. Clean-state verification
  (script → pytest → script, identical outputs) gates every phase.
- **Stochastic verification**: a single pass is NOT evidence of correctness.
  Flaky-prone/stochastic tests are repeated 10x with per-test pass counts
  reported (must be 10/10).
- **Decision log**: every nontrivial choice becomes a new numbered entry in
  `docs/DESIGN_DECISIONS.md` (`D## | date | decision | rationale |
  alternatives rejected`). Append-only; deviations get a NEW entry.
- **Double-blind hygiene (guide §2.4)**: no author-identifying strings
  anywhere in the repo — no personal names, usernames, lab paths, or hostnames
  in code, comments, commits, configs, or docs.
- **Lineage constraint (guide §2.3)**: the prior-work names/terms listed there
  must NEVER appear in this repo — not in code, comments, commit messages,
  docs, or paper text. Read §2.3 for the list; do not reproduce it elsewhere.
- **Internal-only files**: `docs/guide.md` and `docs/DESIGN_DECISIONS.md` are
  excluded from any public artifact release accompanying the paper.
- **Determinism**: all env stochasticity flows through the counter-based CRN
  engine (guide §6.5), strictly separate from any other RNG. The determinism
  test is permanent, not a one-off.
- **Single metrics implementation**: baselines and agents share the env and
  `eager.env.metrics`; never reimplement costs per method.

## Operating rules

- No mid-run questions to the owner: when blocked, log a documented assumption
  in PHASE_STATUS.md, proceed, and flag it. Halt only for true blockers
  (missing guide.md, broken toolchain).
- Stuck >30 min on one issue: spawn a ONE-SHOT subagent with full context
  embedded, requiring a single decisive recommendation in one run.
- Conventional commits per milestone; push after every milestone/phase.
- Every entry point takes `--seed`.

## Build & run

- Env: `.venv` (Python 3.11+; created with 3.12). Activate or call
  `.venv/Scripts/python.exe` directly.
- Install: `python -m pip install -e .[dev]`
- Tests: `python -m pytest` (markers: `stochastic`, `statistical`)
- Smoke: `python -m eager.smoke`
- Episode demo: `python scripts/run_episode.py --hardware configs/hardware/k2_line.yaml --circuit configs/circuits/golden_micro_1.yaml --seed 0`
- 10x stochastic repeat protocol: `python scripts/run_repeat_suite.py`

## File map

- `docs/guide.md` — constitution | `docs/PHASE_STATUS.md` — live state
- `docs/DESIGN_DECISIONS.md` — decision log | `docs/BASELINE_FIDELITY.md` —
  published-method adaptation disclosures
- `src/eager/` — `config.py`, `circuit.py`, `env/` (simulator, CRN, metrics),
  `expgen/` (generators), `utils/`; later phases add `model/`, `train/`,
  `baselines/`, `exact/` (guide §12)
- `configs/{hardware,circuits}/` — YAML configs | `tests/{unit,integration,statistical}/`
- `scripts/` — runnable helpers | `experiments/` — owns `results/`
