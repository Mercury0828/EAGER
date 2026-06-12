# WALKTHROUGH — one instance, end to end

Focus instance: **golden_micro_2** on **golden_k2_det** (the §11 Phase 5
showable-artifact milestone: math → code references → numerical schedule →
smoke table → comparison against baselines). Everything below is
reproducible from the repo alone; commands are listed with each artifact.

## 1. The instance and the model (math)

Hardware ([configs/hardware/golden_k2_det.yaml](../configs/hardware/golden_k2_det.yaml)):
two QPUs u0, u1 with capacities kappa = [2, 2], one link l0 = (u0, u1) with
W=2 generation channels, buffer B=4, no decoherence cutoff (T_cut = null),
pair cost w = 1, DETERMINISTIC generation taking t_ep = 2 slots per pair per
channel (guide §5.2). Gate durations d_loc = 1, d_rem = 2 slots; objective
weights alpha = beta = 1, gamma = 0.5 (guide §5.1, D3).

Circuit ([configs/circuits/golden_micro_2.yaml](../configs/circuits/golden_micro_2.yaml)):
qubits q0..q3, gates g0=(q0,q2), g1=(q1,q3), g2=(q0,q1), g3=(q2,q3).
Per-qubit serialization (guide §4.1, [src/eager/circuit.py](../src/eager/circuit.py))
derives the DAG

    g0 ─┬─> g2        g0,g1 independent;
    g1 ─┴─> g3        g2 <- {g0,g1}; g3 <- {g0,g1};   depth = 2.

Objective (guide §5.1): J = alpha*T + beta*C_comm + gamma*C_waste, where a
remote gate consumes ONE stored pair on every route link at schedule time
(A6) and Σ ADVANCE rewards = -alpha*T is the potential-based makespan term
(guide §6.4; reward_sum == -J on non-truncated episodes — an invariant the
test suite asserts after every episode).

## 2. Where each mechanic lives (code references)

| Mechanic | Code |
|---|---|
| Micro-action loop, §6.1 resolve order | [src/eager/env/env.py](../src/eager/env/env.py) (`step`, `_resolve`) |
| Validity masks §6.3 (incl. buffer-overflow-safe GenEPR) | `EagerEnv._invalid_reason` |
| Deterministic/stochastic generation, CRN | [src/eager/env/crn.py](../src/eager/env/crn.py), `_draw_generation` |
| Reward §6.4 | `_apply_schedule` (-beta*w per route link), `_resolve` (-alpha per slot, -gamma*w per expiry) |
| Metrics (single implementation) | [src/eager/env/metrics.py](../src/eager/env/metrics.py) |
| GreedyJIT expert §9.1 | [src/eager/baselines/greedy_jit.py](../src/eager/baselines/greedy_jit.py) |
| Exact MILP §5.2 + replay | [src/eager/exact/milp.py](../src/eager/exact/milp.py) |
| State graph §6.2 / R-GCN §7.1 / decoder §7.2 | [src/eager/model/graph.py](../src/eager/model/graph.py), [encoder.py](../src/eager/model/encoder.py), [policy.py](../src/eager/model/policy.py) |
| IL §8.1 (+DAgger, D55/D57) / PPO §8.2 | [src/eager/train/il.py](../src/eager/train/il.py), [ppo.py](../src/eager/train/ppo.py) |

## 3. Numerical schedules (slot by slot)

**GreedyJIT** (placement {q0,q1 | q2,q3}, cut 2 — g0,g1 remote; reproduce:
`python scripts/run_episode.py --hardware configs/hardware/golden_k2_det.yaml
--circuit configs/circuits/golden_micro_2.yaml --seed 0 --policy jit`):

| slot | micro-actions | resolve |
|---|---|---|
| 0 | maps; g0,g1 blocked (no pairs) -> JIT: GenEPR x2 | countdowns 2->1 |
| 1 | blocked | two pairs land (age 1 at slot 2) |
| 2 | Schedule g0 (-1), Schedule g1 (-1), d_rem=2 | running |
| 3 | — | g0,g1 done -> g2,g3 ready |
| 4 | Schedule g2, g3 (local) | all done |

T=5, C_comm=2, C_waste=0 → **J = 7** (the Phase 1A hand-derived golden
value, matched exactly by tests/integration/test_golden_micro.py).

**Exact optimum** (Gurobi, [experiments/phase4_gurobi_gap.py](../experiments/phase4_gurobi_gap.py);
status OPTIMAL, MIP gap 0, replay-verified in the env):

The solver picks the OTHER placement {q0,q2 | q1,q3}: now g0, g1 are LOCAL
and g2, g3 are remote — the first layer needs no pairs, so generation
(tasked at slot 0, pairs land at the end of slot 1) is hidden behind it:

| slot | micro-actions | resolve |
|---|---|---|
| 0 | maps; Schedule g0, g1 (local); GenEPR x2 | g0,g1 done; countdowns 2->1 |
| 1 | (g2,g3 ready but pairs in flight) | pairs land |
| 2 | Schedule g2 (-1), g3 (-1), d_rem=2 | running |
| 3 | — | all done |

T=4, C_comm=2 → **J\* = 6 < 7**: the optimum co-designs placement WITH
provisioning overlap — the latency-hiding the EAGER agent is meant to learn,
already visible in a 4-gate instance.

## 4. Smoke table (this instance across methods)

| method | J | T | C_comm | notes |
|---|---|---|---|---|
| Gurobi exact (det) | **6.0** | 4 | 2.0 | OPTIMAL, replay-verified |
| Phase 1A hand schedule | 7.0 | 5 | 2.0 | golden test, exact match |
| GreedyJIT | 7.0 | 5 | 2.0 | 16.7% above J* |
| AGG (Autocomm-style) | 7.0 | 5 | 2.0 | no bursts on this instance (D40) |
| Random-Progressive (5-seed mean, stochastic default hw) | 41.3 | — | — | panel row `adder_n4`-scale instance; see regenerated Phase 2 panel for the full table |

(The stochastic-mode panel numbers for all 14 instances live in
`results/phase2_panel.parquet`; the deterministic gap table incl. this
instance lives in `results/phase4_gap.parquet`.)

## 5. Versus Random — and what the learned agent adds

On the stochastic default config the Phase 2 panel (PHASE_STATUS, D43) shows
GreedyJIT beating Random-Progressive 12/14 with zero truncations
(exceptions {qft_n63, ghz_fanout_n78} are the characterized
provisioning-throughput-bound regime, D35/D38). The Phase 4 harness bounds
GreedyJIT 16.7–64.5% above the deterministic optimum — that span is the
learnable headroom. Phase 5 closes the loop: the IL-initialized agent
reaches 1.04x GreedyJIT on held-out instances (val top-1 0.968; iteration
trail D52–D57), and PPO fine-tuning pushes past GreedyJIT with CRN-paired
significance (evidence tables in PHASE_STATUS Phase 5).
