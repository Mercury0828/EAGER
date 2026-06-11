# EAGER — Project Guide

**Learning to Orchestrate Computation and Proactive Entanglement Provisioning for Distributed Quantum Computing over Stochastic Quantum Networks**

> **Document status**: Canonical design document. Every Claude Code session working on this
> project MUST read this file in full before writing any code. Where this guide and code
> comments disagree, this guide wins unless DESIGN_DECISIONS.md records an explicit override.
>
> **INTERNAL ONLY**: This file (and DESIGN_DECISIONS.md) must be EXCLUDED from any public
> artifact release accompanying the paper. See §2.3.

---

## 1. Project Identity

**Codename**: EAGER (working title; backronym finalized at writing time, e.g.,
*Entanglement-Aware Generation and Execution oRchestration*). The name encodes the core
trade-off: **eager (proactive) vs. lazy (just-in-time) entanglement provisioning**, mirroring
eager/lazy evaluation in computer science.

**One-line story**: On a quantum network where entanglement generation is *probabilistic*,
*bandwidth-limited*, and *subject to decoherence*, jointly decide (i) qubit placement,
(ii) gate scheduling, and (iii) **proactive entanglement provisioning** with a single learned
policy — a heterogeneous-graph GNN encoder + attention decoder trained via imitation
learning + PPO — and show it beats authoritative static heuristics and prior learning-based
compilers, with zero-shot generalization across circuits, topologies, and network parameters.

**Target venue**: IEEE INFOCOM 2027 (main conference). Double-blind. Page budget: 9 pages of
technical content + references (confirm against the official CFP when it posts; historically
late-July abstract / early-August full-paper deadlines).

**Core claim (frozen wording, do not dilute)**:
> "To the best of our knowledge, this is the first framework to *jointly learn* qubit
> placement, gate scheduling, and *proactive* entanglement provisioning for distributed
> quantum computing over stochastic quantum networks."

The qualifiers "jointly learn" and "proactive" are load-bearing: they differentiate from
(a) network-layer entanglement provisioning that treats computation as fixed demand
(Zhao & Qiao line, INFOCOM'21/'23), (b) allocation-only RL (Pastor et al., Zen et al.,
Russo et al., CO-MAP), and (c) Promponas et al. (ICC'25), whose DDQN compiler handles
stochastic EPR but with flat state representations, no proactive-vs-JIT decision dimension
framed as such, and no cross-instance generalization. Never claim plain "first learning-based
DQC compiler" — that claim is falsifiable.

---

## 2. Strategic Positioning & Constraints

### 2.1 Why this problem is INFOCOM material
Entanglement generation is a *network resource*: probabilistic success, finite parallel
channels per link (bandwidth), finite memory lifetime (decoherence cutoff), per-link cost.
The paper's framing is **"orchestrating computation over a stochastic quantum network"**, not
"compiler optimization". The INFOCOM quantum track (entanglement routing/provisioning:
Zhao & Qiao INFOCOM'21, Zeng et al. INFOCOM'22, asynchronous entanglement provisioning for
DQC INFOCOM'23, Mao et al. INFOCOM'23 qubit allocation) is the home audience.

### 2.2 Why RL is *necessary* here (the defense against "why not search?")
The environment is genuinely stochastic (Bernoulli entanglement generation) and online
(decisions interleave with random outcomes). Offline planning/beam search/MCTS over a
deterministic model is not applicable without replanning at every random outcome; an
adaptive closed-loop policy is the natural object. The proactive-provisioning trade-off
(hide latency vs. risk decoherence waste) depends jointly on future circuit demand and
random network state — exactly what static heuristics cannot anticipate. State this
explicitly in the paper (one paragraph, Section "Why learning").

### 2.3 Lineage constraints (read carefully)
- Do **not** reference, cite, import text from, or structurally mirror any prior internal
  draft or preprint, including UNIQ (arXiv:2512.00401). No mention in code, comments, commit
  messages, docs, or paper text. This project's formulation (stochastic generation +
  decoherence cutoff + proactive provisioning as a decision) is materially different and is
  presented as self-contained and original to this work.
- The experimental design must not mirror prior internal work: baseline suite is anchored on
  Autocomm (MICRO'22), MHSA (INFOCOM'23), and a DDQN-flat agent (ICC'25-style). CloudQC is
  discussed in related work only (optional stretch baseline; see §9.8).
- All "first X" claims use the frozen wording in §1.
- This guide and DESIGN_DECISIONS.md are excluded from any public artifact.

### 2.4 Double-blind hygiene
No author-identifying strings anywhere in the repo (names, lab paths, hostnames in committed
configs). Paper avoids "our prior work" phrasing entirely (true by construction here).

---

## 3. Problem Setting (informal)

A quantum circuit too large for one QPU is executed across K networked QPUs. Remote two-qubit
gates require pre-shared EPR pairs on every link along a route between the two QPUs
(Cat-Comm + entanglement swapping). EPR pairs are produced by *generation attempts* that
succeed only probabilistically, links have a small number of parallel generation channels,
and stored pairs decohere if not consumed within a cutoff window. The controller must decide:

1. **Placement** — which QPU hosts each logical qubit (static, capacity-constrained).
2. **Scheduling** — when each gate executes (respecting DAG precedence and pair availability).
3. **Provisioning** — *when and on which links to start generating EPR pairs*: too early →
   pairs expire (waste); too late → remote gates stall (latency).

Objective: minimize a weighted sum of expected makespan, communication cost, and waste.

---

## 4. System Model (formal)

### 4.1 Logical circuit
- DAG `G_C = (V_G, E_C)` over logical qubits `Q = {q_1..q_N}`.
- `V_G = {g_1..g_M}`: two-qubit gates only (CNOT/CZ-class). Single-qubit gates are ignored
  (Assumption A3). `(g_i, g_j) ∈ E_C` ⇒ g_i must complete before g_j starts.
- `ops(g) = {q_a, q_b}` the operands of g. Gates sharing a qubit are serialized (implied by
  the DAG construction; the instance generator must enforce per-qubit total order).
- Duration: local 2q gate `d_loc = 1` slot. Remote 2q gate `d_rem` slots (default 2: one for
  cat-entangle + classical comm, one for the gate; configurable).

### 4.2 Hardware & network
- Connected undirected graph `G_P = (U, E_P)`, `|U| = K` QPUs.
- QPU u: computing-qubit capacity `κ_u`.
- Link `l = (u,v) ∈ E_P`:
  - `p_l ∈ (0,1]` — per-attempt, per-slot Bernoulli success probability of EPR generation;
  - `W_l ∈ ℕ⁺` — parallel generation channels (bandwidth);
  - `B_l ∈ ℕ⁺` — buffer capacity (max stored pairs);
  - `T_cut ∈ ℕ⁺` — decoherence cutoff: a stored pair expires when its age exceeds T_cut;
  - `w_l ∈ ℝ⁺` — cost per consumed pair (also charged per expired pair, weighted by γ).
- Defaults are global (`p, W, B, T_cut, w` uniform) unless a config overrides per-link.

### 4.3 Communication primitive & routing
- Cat-Comm only (Assumption A6; TP-Comm/qubit migration is future work, Assumption A2).
- Routes are **fixed shortest paths** (hop count; lexicographic tie-break), precomputed per
  hardware config. Executing remote gate g whose operands sit on u ≠ v **consumes one stored
  pair from every link on route R(u,v) simultaneously at schedule time** (end-to-end
  entanglement via swapping at intermediate nodes; swaps deterministic and instantaneous,
  Assumption A4/A5). Per-gate communication cost: `Σ_{l ∈ R(u,v)} w_l`.

### 4.4 Generation semantics
- `GenEPR(l)` tasks ONE free channel of link l into **generate-until-success** mode: each
  subsequent slot it draws Bernoulli(p_l); on success the pair enters l's buffer with age 0
  and the channel frees. No cancel action in v1 (Decision D7).
- `GenEPR(l)` is valid only if `free_channels(l) ≥ 1` AND
  `stored(l) + busy_channels(l) < B_l` (so a success can never overflow the buffer).
- Aging: at end of each slot every stored pair's age += 1; pairs with `age > T_cut` are
  discarded → counted as waste.

### 4.5 Time model
Discrete slots; 1 slot = duration of one local 2q gate. Calibration anchor: with the
literature value t_ep ≈ 12× a CX, set default `p = 1/12 ≈ 0.083` so expected generation time
matches; sensitivity sweeps vary p in [0.05, 0.5].

### 4.6 Assumptions (numbered; reproduce as a table in the paper)
- **A1** Intra-QPU all-to-all (or negligible-cost SWAPs): cost only across QPUs.
- **A2** Static placement; no mid-circuit qubit migration (limitation paragraph in paper).
- **A3** Single-qubit gates negligible; only 2q gates modeled.
- **A4** Entanglement swapping at intermediate nodes is deterministic (q = 1) and folded
  into the remote-gate slot; only *generation* is stochastic in v1.
- **A5** Fixed shortest-path routing (no adaptive routing decisions in v1).
- **A6** Cat-Comm protocol for remote gates; one end-to-end pair (i.e., one pair per route
  link) per remote gate.
- **A7** Classical communication is instantaneous relative to slot granularity (standard).

---

## 5. Optimization Problem

### 5.1 Objective
For a policy π inducing a (random) execution:
```
J(π) = E[ α · T_makespan + β · C_comm + γ · C_waste ]
T_makespan = completion slot of the last gate
C_comm     = Σ_{consumed pairs} w_l
C_waste    = Σ_{expired pairs}  w_l
```
Provisional weights `α = 1.0, β = 1.0, γ = 0.5`; calibrate once in a Phase-6 pilot so the
three terms are the same order of magnitude on medium instances, then FREEZE and record in
DESIGN_DECISIONS.md (Decision D3). A weight sweep appears only in sensitivity analysis.

### 5.2 Deterministic special case (for the exact baseline)
Setting `p_l = 1`, `T_cut = ∞`, generation = deterministic `t_ep` slots per pair per channel
yields a time-indexed MILP (after linearization):
- Variables: `x_{i,k} ∈ {0,1}` (qubit i → QPU k); `s_g ∈ ℤ≥0` (start slot); per-link
  per-slot generation tasking `z_{l,c,t} ∈ {0,1}`; co-location products
  `y_{ab}^{kl} = x_{a,k} x_{b,l}` linearized by McCormick
  (`y ≤ x_{a,k}; y ≤ x_{b,l}; y ≥ x_{a,k}+x_{b,l}−1`).
- Remote indicator: `ρ_g = Σ_{k≠l, (k,l) routes} y_{ab}^{kl}`.
- Constraints: unique mapping; capacity; precedence `s_j ≥ s_i + d_i`; per-link cumulative
  pairs generated by t ≥ cumulative pairs consumed by t; channel occupancy (a tasked channel
  is busy for t_ep slots); horizon H from the GreedyJIT solution.
- Objective: `α·T + β·Σ ρ_g·cost(route)` (no waste in deterministic case).
- NP-hardness: contains RCPSP (β=0) and capacitated graph k-partition (α=0) as special
  cases; write both reductions out properly in the paper (2–4 lines each, not one sentence).

This MILP exists for ONE purpose: Gurobi optimality-gap experiments on small instances
(§10.6, T4). The general stochastic problem is stated natively as the MDP below.

### 5.3 Stochastic problem
The general problem is a finite-horizon stochastic sequential decision problem; no static
program captures it. This motivates the MDP + learning approach (paper transition paragraph).

---

## 6. MDP Specification

### 6.1 Episode & decision loop
An episode = one circuit instance on one hardware config. Within each slot t the agent emits
a *sequence of micro-actions*; each is applied immediately (deterministic bookkeeping). The
slot closes when the agent picks `ADVANCE`, after which the environment resolves randomness:

```
loop over slots t = 0,1,2,...:
    repeat:
        a ← π(s);  a ∈ ValidActions(s)
        apply a    (Map / Schedule / GenEPR bookkeeping)
    until a == ADVANCE
    resolve slot:
        (1) each busy generation channel of link l draws CRN(l, c, t) < p_l;
            success → pair (age 0) into buffer, channel freed
        (2) running gates advance one slot; completed gates update the ready set
        (3) stored pairs age += 1; expired (> T_cut) pairs discarded → waste
        (4) t += 1
until all gates completed, or t > T_budget (truncation)
```
`T_budget = 20·(M + N) + 200` slots (Decision D9). Truncation ⇒ terminal penalty
`P_trunc = α · 10 · (#unfinished gates)`, logged; a sane policy should never truncate.

### 6.2 State: heterogeneous graph `s_t = (G_t, globals)`
**Node types & features** (all features normalized to [0,1] or one-hot):
- **Gate** (unscheduled gates only): [is_remote_known∈{unknown,local,remote} one-hot;
  criticality = longest-path-to-sink / max; ready flag; #unscheduled predecessors (norm);
  depth (norm)].
- **Qubit**: [mapped flag; interaction-graph degree (norm); remaining 2q-gate count (norm)].
- **QPU**: [κ_res/κ; mapped_count/κ; #ready local gates hosted (norm)].
- **Link**: [p_l; free_channels/W_l; stored/B_l; busy_channels/W_l; age histogram of stored
  pairs in 4 remaining-lifetime buckets (each /B_l); pending demand = #ready remote gates
  routed through l (norm)].

**Relations (R-GCN edge types)**:
1. gate→gate (DAG dependency), 2. gate↔qubit (operand), 3. qubit↔QPU (current mapping),
4. QPU↔link (incidence), 5. gate↔link (routed-through; present once both operands mapped),
6. self-loops (handled by W_0 in R-GCN).
Directional relations get distinct types (R-GCN treats direction as separate relations).

**Globals** (concatenated to the readout): [t/T_budget; frac gates done; frac qubits mapped;
frac pairs in buffers (Σstored/ΣB)].

### 6.3 Actions & validity masks
- `Map(q, u)`: q unmapped ∧ κ_res(u) > 0.
- `Schedule(g)`: g unscheduled ∧ all predecessors completed ∧ both operands mapped ∧
  (g local ∨ every link on route has ≥ 1 stored unexpired pair). Applying it: consume one
  pair from each route link immediately; gate runs d_loc or d_rem slots.
- `GenEPR(l)`: per §4.4 validity. Applying it: one channel → busy(generate-until-success).
- `ADVANCE`: always valid (guarantees non-empty action set).

### 6.4 Reward (per micro-action; potential-based makespan term)
- `ADVANCE`: `r = −α · 1` (one slot elapses). Summing gives exactly −α·T_makespan; this is a
  potential-based shaping with Φ(s) = −α·t — cite Ng et al. correctly this time.
- `Schedule(remote g)`: `r = −β · Σ_{l∈route} w_l` at consumption.
- Pair expiry (charged at the resolve step, attached to the ADVANCE transition):
  `r −= γ · w_l` per expired pair.
- Truncation: `r −= P_trunc`.
- **No constant "valid-action bonus"** (it is a constant over fixed-length episodes and does
  nothing; the old draft's r_valid is explicitly abolished).

Return = Σ r (γ_discount = 0.995 for variance control; report undiscounted J in evals).

### 6.5 Determinism & CRN (critical engineering requirement)
All stochasticity flows through a **counter-based RNG**: `outcome(l, c, t) = U(seed, l, c, t)
< p_l`, with U a stable hash (e.g., numpy Philox keyed by run seed, counter = (l_id, c, t),
or xxhash → [0,1)). Properties required (tested):
- Same (seed, l, c, t) → same draw, regardless of query order or policy.
- Two different policies evaluated under the same seed face identical generation luck
  wherever their tasking patterns coincide → valid **common-random-numbers (CRN)** paired
  comparisons (§10.4).

---

## 7. Agent Architecture

### 7.1 Encoder — R-GCN
- L = 3 layers, hidden d = 128, ReLU, per-relation weights + self-loop W_0, mean
  normalization by per-relation in-degree (1/c_{i,r}); LayerNorm between layers.
- Implementation: PyTorch Geometric `RGCNConv` preferred; if PyG install friction on the
  target machine, hand-roll hetero message passing with sparse ops (≈150 lines) — record
  the choice in DESIGN_DECISIONS.md.

### 7.2 Decoder — attention over valid actions (pointer-style)
- Global query: `q = MLP( mean_i h_i ⊕ globals )`, dim d_k = 128.
- Keys per valid action:
  - Map(q_i,u):   `k = W_map [h_{q_i} ∥ h_u]`
  - Schedule(g):  `k = W_sch [h_g]`
  - GenEPR(l):    `k = W_gen [h_l]`
  - ADVANCE:      `k = W_adv · q` (query-conditioned learned key)
- Logits `e_a = qᵀk_a / √d_k`; invalid actions masked to −∞; softmax → π_θ(a|s).
- Value head: `V_φ(s) = MLP(mean_i h_i ⊕ globals)` (shares encoder).
- Paper propositions: permutation invariance (encoder equivariant + mean readout invariant +
  keys permute with nodes) and per-decision complexity O(L·|E|·d + |A|·d) — both carry over;
  re-derive against this exact architecture, do not copy old text.

### 7.3 No (g, τ) time-slot selection
Scheduling is "start now or don't" — the old sinusoidal positional encoding over candidate
start times is removed. Waiting is expressed through ADVANCE; this collapses the action
space and is strictly cleaner under event-driven semantics.

---

## 8. Training Pipeline

### 8.1 Phase I — Imitation Learning (behavioral cloning)
- Expert = GreedyJIT (§9.1) **implemented as a policy over the same env API**, emitting
  micro-action traces (Map/Schedule/GenEPR/ADVANCE) — this is a hard requirement so BC
  targets live in the agent's action vocabulary.
- Dataset: ≥ 50k expert transitions over the training distribution (§10.3), stratified by
  size; 90/10 train/val split.
- Loss: cross-entropy over valid-action sets; train ≤ 20 epochs, early-stop on val accuracy
  plateau. Target: ≥ 90% top-1 imitation accuracy before PPO (sanity gate, not a paper claim).

### 8.2 Phase II — PPO fine-tuning
- Custom PPO loop (do NOT fight stable-baselines3 over variable action sets + masking;
  CleanRL-style single-file trainer adapted to graph batches).
- GAE λ = 0.95, γ = 0.995, clip ε = 0.2, value coef 0.5, entropy coef 0.01 → 0.001 (linear
  anneal), lr 3e-4 with cosine decay, 4 epochs/iter, minibatch 1024 transitions, 16 parallel
  CPU envs × 512-step rollouts, advantage normalization, grad-norm clip 0.5, reward
  normalization by running std (document exact scheme).
- Curriculum: start N ∈ [10,30], unlock [30,60] when mean J beats GreedyJIT on a held-out
  small set for 3 consecutive evals.
- KL early-stop per iteration (target KL 0.02) as instability guard.

### 8.3 Compute
Local RTX 4090 (24 GB): policy/value on GPU, envs on CPU workers. Budget expectation:
IL hours-scale; PPO main config ≤ 24 h. If exceeded, shrink rollout or d before touching L.

---

## 9. Expert & Baselines (the comparison suite)

> Authority layering (Decision D4): top-venue anchor = Autocomm (MICRO'22); home-venue
> anchor = MHSA (INFOCOM'23); learning-SOTA anchor = DDQN-flat (ICC'25-style); self-contained
> expert = GreedyJIT; lower bound = Random; upper bound = Gurobi optimum (deterministic,
> small). CloudQC: related-work discussion only; optional stretch (§9.8).

All baselines are **policies over the same env API** and evaluated under CRN. Every
adaptation from a published method is documented in `docs/BASELINE_FIDELITY.md` and
disclosed in the paper ("X-style, adapted to our stochastic environment").

### 9.1 GreedyJIT (expert + heuristic baseline; fully described in the paper)
1. **Placement**: build qubit interaction graph (edge weight = #2q gates between the pair);
   capacity-constrained balanced k-way partition via METIS/KaHyPar (unit vertex weights,
   parts ≤ κ_u). Cite the hypergraph-partitioning lineage (Andrés-Martínez & Heunen).
2. **Scheduling**: list scheduling — each slot, among ready gates in descending criticality:
   local → Schedule; remote → Schedule if all route links have a stored pair, else register
   pair deficits.
3. **Provisioning (JIT)**: for each deficit link, issue GenEPR up to free channels / buffer
   headroom, prioritizing links serving the most critical blocked gate. Then ADVANCE.

### 9.2 MHSA+LS (Mao et al., INFOCOM'23 — home-venue anchor)
Reimplement Multistage Hybrid Simulated Annealing for *placement* per the paper (greedy
initialization + staged SA minimizing remote-gate count under capacity); pair with the same
list scheduler + JIT provisioning as §9.1 (so the comparison isolates placement quality).
SA budget fixed and reported.

### 9.3 AGG — Autocomm-style aggregation (MICRO'22 — top-venue anchor)
On the METIS placement (same as §9.1 for controlled comparison), detect **burst
communication**: maximal runs of consecutive remote gates between the same QPU pair sharing
an operand qubit, and merge each run to execute under ONE cat-comm channel (one end-to-end
pair serves the whole burst), per Autocomm's aggregation idea. Then list-schedule + JIT.
This reduces pair demand and is the strongest static competitor under scarce entanglement.
First task: check for a public Autocomm artifact on GitHub; if found, follow its burst
detection precisely; otherwise implement from the paper and document deviations.

### 9.4 DDQN-flat (ICC'25-style learning baseline)
Double DQN + target network + replay over a *flat* fixed-size state (per-QPU loads, per-link
[stored, busy, free], top-k ready-gate features, globals; zero-padded) with the same action
space (flattened, max-size masks). Trained **per configuration** with the same env-step
budget as EAGER's PPO phase (fairness documented). Its expected failure to generalize and to
scale is part of the message — the comparison demonstrates the value of the graph encoder.

### 9.5 Random-Progressive
Uniform over valid non-ADVANCE actions; ADVANCE only when nothing else is valid. (Pure
uniform stalls forever; this variant is the honest lower bound.)

### 9.6 Gurobi exact (deterministic special case)
MILP of §5.2 on instances with N ≤ 12, M ≤ 30, K ∈ {2,3}; WLS license available locally.
Time limit 1 h/instance; report MIP gap when not closed. EAGER evaluated on the same
instances in deterministic env mode (p=1, T_cut=∞, t_ep fixed). Deliverable: optimality-gap
table (T4).

### 9.7 EAGER-NoProactive (key ablation)
EAGER with GenEPR masked out; env flag `auto_jit=True` triggers the §9.1 JIT routine
automatically when a ready remote gate lacks pairs. The gap EAGER vs. EAGER-NoProactive *is*
the measured value of proactive provisioning — arguably the single most important number in
the paper.

### 9.8 CloudQC (stretch, optional)
Closest published setting (placement + network scheduling, EPR success probability), but
younger and citation-thin relative to the anchors above; cite and discuss in related work.
Implement only if the matrix finishes early; never let it block submission.

---

## 10. Experimental Design

### 10.1 Instances
- **Synthetic**: random DAG circuits, N ∈ [10,120], 2q-gate density ratio M/N ∈ {1, 3, 6};
  generator enforces per-qubit serialization; seeded.
- **QASMBench subset** (2q-gate skeletons extracted): adder_n4 (tiny sanity), qaoa_n14,
  bv_n30, qft_n50, cat_n60, ghz_n80, vqe_n80, qugan_n71, qpe_n90, bv_n100, plus a
  supremacy-style random circuit n120. License/attribution noted in repo.

### 10.2 Hardware configs
K ∈ {2,4,8}; topologies {line, ring, 2D-grid, connected Erdős–Rényi(p=0.4)};
κ sized so Σκ ≈ 1.25·N (scarcity ablation: 1.0·N); p ∈ [0.05,0.5] (default 1/12);
W ∈ {1..4} (default 2); B ∈ {4..16} (default 8); T_cut ∈ {5..50} (default 20); w_l = 1
uniform (heterogeneous-w sweep optional).

### 10.3 Zero-shot protocol (frozen; reproduce as a table in the paper)
- **Train**: synthetic only, N ∈ [10,60]; topologies {line, grid}; K ∈ {2,4};
  p ∈ {0.08, 0.12, 0.2}; W ∈ {1,2}; T_cut ∈ {10,20}.
- **Test axes** (all zero-shot, no fine-tuning in the main results):
  (a) all QASMBench circuits; (b) synthetic N ∈ [80,120]; (c) topologies {ring, ER}, K = 8;
  (d) p ∈ {0.05, 0.3, 0.5}, T_cut ∈ {5, 50}, W ∈ {3,4}.
- A fine-tuned variant may appear as one extra row, clearly labeled.

### 10.4 Statistical protocol
- ≥ 5 training seeds for every learned method (EAGER, DDQN-flat); heuristics are
  deterministic given placement seed (3 placement seeds for METIS/SA).
- ≥ 20 evaluation episodes per (method, instance, config), CRN-paired via §6.5:
  episode e uses run seed = H(instance_id, config_id, e) for ALL methods.
- Report mean ± std; paired Wilcoxon signed-rank vs. the best baseline per instance;
  significance marks in tables. NO headline number may disagree with the tables —
  Phase-8 audit enforces this (§11).

### 10.5 Metrics (formal; implement exactly once, in one module)
makespan T; C_comm; C_waste; J; **EPR utilization** = consumed/(consumed+expired);
**mean remote-gate stall** = mean(schedule_slot − ready_slot) over remote gates;
inference time per decision and per episode; optimality gap (J−J*)/J* (deterministic mode);
truncation rate (must be ~0).

### 10.6 Planned figures & tables (paper layout target)
- T1 notation; T2 train/test split; **T3 main results** (QASMBench × all methods: J,
  T, C_comm, utilization; mean±std, significance); **T4 Gurobi gap**; T5 ablations
  (NoProactive / IL-only / RL-from-scratch / GCN vs R-GCN / MLP-decoder vs attention).
- F1 system+architecture figure; **F2 sensitivity 2×2** (sweep p, T_cut, W, κ — the
  networking-story centerpiece replacing the old Fig. 8); F3 scaling in N;
  F4 zero-shot heatmap (train config × test axis); **F5 case-study Gantt**: same instance
  + same CRN seed, JIT vs. EAGER timelines showing latency hiding (reviewers love this);
  F6 training curves (appendix if space).

---

## 11. Phase Plan & Acceptance Criteria

> Protocol inherited from prior projects: every phase ends with (i) clean-state run from a
> fresh checkout/venv, (ii) full pytest, (iii) artifact inspection AFTER tests (no test
> pollution: tests write only to tmp paths, never `results/`), (iv) stochastic/flaky-prone
> tests repeated **10×** with failure-rate reporting, (v) PHASE_STATUS.md updated with real
> command output pasted in (never summarized from memory), (vi) git tag `phase-X-done`.

- **Phase 0 — Scaffold**: repo layout (§12), pyproject/env files, configs schema + loader,
  pytest wiring, DESIGN_DECISIONS.md / PHASE_STATUS.md / BASELINE_FIDELITY.md stubs,
  CLAUDE.md. Accept: fresh-env install + `pytest` green + `python -m eager.smoke` prints a
  parsed config.
- **Phase 1A — Deterministic simulator core**: env API (reset/step/masks/info), mapping,
  scheduling, routing, deterministic generation (p=1 mode with t_ep), reward.
  Accept: invariant tests (DAG order never violated; capacity never exceeded; pair
  conservation `generated = consumed + expired + stored + 0` each slot — expired=0 here);
  TWO golden micro-instances (≤4 qubits, ≤6 gates, K=2) with hand-computed J matched
  exactly; same seed → identical trajectory hash.
- **Phase 1B — Stochastic layer**: CRN engine (§6.5), Bernoulli generation, aging, cutoff,
  buffers, waste accounting, auto_jit flag. Accept: CRN property tests (order/policy
  independence); empirical success frequency within 99% CI of p over ≥10⁵ draws;
  conservation invariant incl. expired; aging/expiry golden test; 10× repeat suite stable.
- **Phase 2 — GreedyJIT + Random + trace recorder**: both as env-API policies; expert
  trace dataset writer. Accept: GreedyJIT completes every QASMBench instance on default
  config, zero truncations, J strictly < Random-Progressive on all; traces replayable
  (replay = identical trajectory).
- **Phase 3 — Published-style baselines**: MHSA+LS, AGG (artifact check first), DDQN-flat
  implementation (training deferred to Phase 6). Accept: MHSA placement ≤ METIS remote-gate
  count on ≥70% of a 20-instance panel; AGG strictly reduces consumed pairs vs GreedyJIT on
  burst-heavy circuits (bv, ghz, cat); BASELINE_FIDELITY.md filled.
- **Phase 4 — Gurobi exact**: MILP builder + gap harness. Accept: on golden
  micro-instances Gurobi J* ≤ GreedyJIT J with optimal status; linearization validated by
  brute-force enumeration on a ≤3-qubit toy.
- **Phase 5 — EAGER agent**: encoder/decoder/value, IL, PPO, curriculum. Accept: IL val
  top-1 ≥ 90%; IL-initialized agent within 5% of GreedyJIT J on held-out small set; PPO on
  small config beats GreedyJIT mean J with statistical significance (CRN-paired, 5 seeds).
  → **Showable-artifact milestone**: git tag, zip (src+tests+configs+DESIGN_DECISIONS.md,
  exclude data/artifacts), WALKTHROUGH.md for one focus instance (math → code refs →
  numerical schedule → smoke table → vs-Random comparison).
- **Phase 6 — Full matrix**: train final EAGER (5 seeds) + DDQN-flat per config; run the
  complete evaluation matrix → `results/` parquet + `results/index.json` (single source of
  truth for every number that appears in the paper).
- **Phase 7 — Figures & tables**: scripted generation only (no hand-edited numbers), style
  consistent, every artifact regenerable by one command.
- **Phase 8 — Consistency audit**: a script cross-checks every claimed number in the paper
  draft against results/index.json; abstract/intro/table three-way agreement enforced.
  (Hard lesson from the previous draft: 15% vs 20% vs ~8.5% must never happen again.)

---

## 12. Engineering Conventions

```
eager-dqc/
  CLAUDE.md                  # session entry-point: read docs/guide.md first; core rules
  docs/guide.md              # THIS FILE (canonical)
  docs/DESIGN_DECISIONS.md   # append-only decision log (D-numbers)
  docs/PHASE_STATUS.md       # per-phase status with pasted real outputs
  docs/BASELINE_FIDELITY.md  # published-method adaptation disclosures
  src/eager/
    env/        # simulator: state, masks, transitions, CRN, metrics
    model/      # RGCN encoder, attention decoder, value head
    train/      # il.py, ppo.py, curriculum.py
    baselines/  # greedy_jit.py, mhsa.py, agg.py, ddqn_flat.py, random_prog.py
    exact/      # milp builder + gurobi harness
    expgen/     # instance & hardware generators, qasm skeleton extraction
    utils/
  configs/{hardware,circuits,train,experiments}/*.yaml
  experiments/  # runnable experiment scripts (one per table/figure)
  tests/{unit,integration,statistical}/
  scripts/
  qasm/         # QASMBench subset + attribution
  results/      # gitignored except index.json
  artifacts/    # gitignored (checkpoints, traces)
```

Rules:
- Python 3.11+; `torch` (CUDA), `torch_geometric` (fallback per §7.1), `numpy`, `networkx`,
  `pyyaml`, `gurobipy`, `pytest`, `pandas`/`pyarrow`. User-local env (conda/venv); never
  system-wide installs.
- Every entry point takes `--seed`; env RNG (Philox/CRN) strictly separate from torch RNG.
- Determinism test is a permanent CI-style test, not a one-off.
- Tests never write outside pytest tmp dirs. `results/` is written only by `experiments/`.
- Single metrics module; baselines and agent share the env — no per-method reimplementation
  of costs (classic source of silent unfairness).
- Commit per milestone; tags `phase-X-done`. No author-identifying strings (§2.4).
- DESIGN_DECISIONS.md entries: `D## | date | decision | rationale | alternatives rejected`.

---

## 13. Writing-Phase Guidance (for later; keep in mind while building)

- Section budget (9 pp): Intro 1.25 / Related 0.75 / Model+Problem 1.5 / MDP+Method 2 /
  Evaluation 3 / Conclusion 0.25. NO standalone GNN/attention/RL preliminaries section —
  one paragraph of background folded into the method (hard lesson from the old draft).
- Related-work map (must cite, with one differentiating clause each): Mao INFOCOM'23;
  Autocomm MICRO'22 (+ QuComm MICRO'23); Promponas ICC'25; Chandra TPS-ISA'24; CloudQC;
  Pastor/Escofet multi-core DRL line + FGP-rOEE (Baker CF'20); Zen et al. '25; Russo et al.;
  LeCompte TQE'23; CO-MAP; Zhao & Qiao INFOCOM provisioning line + INFOCOM'23 asynchronous
  provisioning for DQC (boundary: they provision for given demand; we co-decide demand);
  job-level DQC scheduling ('26 arXiv; boundary: job granularity vs gate granularity);
  Iñesta & Wehner cutoff-policy line (source of T_cut); DQC surveys (Caleffi et al.,
  Barral et al.); Cuomo/Ferrari compiler line.
- Claims: §1 frozen wording only. Every number traced to results/index.json (Phase 8).
- Limitations paragraph: A2 (no migration), A4 (deterministic swapping), A5 (fixed routing),
  fidelity not modeled beyond cutoff — each tagged future work.

---

## 14. Decision Log (initial; mirror into DESIGN_DECISIONS.md at Phase 0)

- **D1** Multi-hop remote gates via fixed shortest-path swapping, q=1; generation is the
  only stochastic element in v1. (Networking flavor without routing-decision complexity.)
- **D2** Static placement; no TP-Comm/migration in v1 (limitation, future work).
- **D3** Weights α=1, β=1, γ=0.5 provisional; one calibration pilot in Phase 6, then frozen.
- **D4** Baseline suite per §9; CloudQC demoted to related-work/stretch.
- **D5** Codename EAGER; repo `eager-dqc`; old framework name (GARL) retired — must not
  appear anywhere in this repo.
- **D6** Cat-Comm only; one end-to-end pair (one pair per route link) per remote gate.
- **D7** GenEPR = persistent generate-until-success channel tasking; no cancel action in v1.
- **D8** Counter-based CRN (Philox/hash) for all stochastic draws; mandatory property tests.
- **D9** T_budget = 20(M+N)+200; truncation penalty per §6.1.

---

## 15. Risks & Fallbacks

- **PPO instability** → longer IL, smaller clip (0.1), stronger reward normalization, KL
  early-stop; worst case report IL+light-PPO and lean on NoProactive ablation for the story.
- **PyG on local CUDA stack fails** → hand-rolled R-GCN (§7.1).
- **Autocomm fidelity disputes** → "Autocomm-style" labeling + BASELINE_FIDELITY.md +
  artifact-check first; deviations enumerated in the paper footnote.
- **Gurobi horizon blow-up** → shrink to N ≤ 10, report MIP gap at time limit (still a
  valid bound for the gap table).
- **DDQN-flat embarrassingly weak** → keep equal-budget fairness documented; its weakness
  at scale IS the finding, but verify it is competitive on tiny instances so the comparison
  is honest.
- **Proactive gain smaller than hoped** → widen the regime where it matters (low p, tight
  T_cut, W=1); if the effect is regime-dependent, say so — a characterized regime map is a
  contribution, an oversold universal claim is a rejection.
