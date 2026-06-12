"""Heterogeneous state graph s_t = (G_t, globals) per guide §6.2.

Node types (features normalized to [0,1] / one-hot; dims fixed per type):
  gate (UNSCHEDULED gates only, 7 dims):
      is_remote_known one-hot {unknown, local, remote} (3);
      criticality / max criticality; ready flag;
      #unfinished immediate preds / 2; gate depth / instance depth
  qubit (4 dims): mapped flag; interaction-graph degree / max degree;
      remaining unscheduled 2q-gate count on this qubit / max remaining;
      is-next-to-map flag (1 on the first unmapped qubit in the experts'
      Map-emission order = the partitioner's greedy order, D52/D56; a state
      feature, so the architecture stays permutation-invariant)
  qpu (7 dims): kappa_res/kappa; mapped_count/kappa;
      #ready local gates hosted / max(1, #ready gates);
      QPU id one-hot padded to 4 (the sequential-fill tie-break convention
      the expert uses is id-based, like D52's next-to-map flag; K <= 4 in
      the training distribution; D54)
  link (9 dims): p; free/W; stored/B; busy/W;
      age histogram of stored pairs in 4 REMAINING-LIFETIME buckets, each /B
      (T_cut=null => infinite remaining life => top bucket, D47);
      pending demand = #ready remote gates routed through l / max(1, #ready)

Relations (directional types are distinct; self-loops live in the encoder's
root weight W_0, guide §6.2(6)):
  0 gate->gate (DAG dependency, both endpoints unscheduled)
  1 gate->qubit   2 qubit->gate     (operands)
  3 qubit->qpu    4 qpu->qubit      (current mapping)
  5 qpu->link     6 link->qpu       (incidence)
  7 gate->link    8 link->gate      (route-through; once both operands
                                     mapped and the gate is remote)

Globals (4): t/T_budget; frac gates done; frac qubits mapped;
frac pairs stored (sum stored / sum B).

Node order in the flattened graph: gates (ascending id over the unscheduled
set), qubits, QPUs, links — with index maps kept for the decoder, which
needs h_gate / h_qubit / h_qpu / h_link lookups per valid action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.env import EagerEnv
from ..env.state import UNSCHEDULED

NUM_RELATIONS = 9
GATE_DIM, QUBIT_DIM, QPU_DIM, LINK_DIM = 7, 4, 7, 9
MAX_QPU_ONEHOT = 4
GLOBALS_DIM = 4
AGE_BUCKETS = 4


@dataclass
class GraphSnapshot:
    """numpy graph of one env state (torch-free; the encoder tensorizes)."""

    x_gate: np.ndarray          # [n_gate, GATE_DIM] float32
    x_qubit: np.ndarray         # [n_qubit, QUBIT_DIM]
    x_qpu: np.ndarray           # [n_qpu, QPU_DIM]
    x_link: np.ndarray          # [n_link, LINK_DIM]
    edge_index: np.ndarray      # [2, n_edges] int64 (flattened node ids)
    edge_type: np.ndarray       # [n_edges] int64 in [0, NUM_RELATIONS)
    globals: np.ndarray         # [GLOBALS_DIM] float32
    gate_ids: np.ndarray        # [n_gate] env gate id of each gate node
    gate_row: dict              # env gate id -> gate node row
    n_qubit: int
    n_qpu: int
    n_link: int

    @property
    def n_gate(self) -> int:
        return len(self.gate_ids)

    @property
    def num_nodes(self) -> int:
        return self.n_gate + self.n_qubit + self.n_qpu + self.n_link

    # flattened node id helpers (decoder + encoder share this layout)
    def nid_gate(self, row: int) -> int:
        return row

    def nid_qubit(self, q: int) -> int:
        return self.n_gate + q

    def nid_qpu(self, u: int) -> int:
        return self.n_gate + self.n_qubit + u

    def nid_link(self, l: int) -> int:
        return self.n_gate + self.n_qubit + self.n_qpu + l


def build_graph(env: EagerEnv) -> GraphSnapshot:
    inst, hw = env.instance, env.hardware
    n_q, k_n, n_l = inst.num_qubits, hw.num_qpus, hw.num_links

    unsched = [g for g in range(inst.num_gates)
               if env.gates[g].state == UNSCHEDULED]
    gate_row = {g: i for i, g in enumerate(unsched)}
    n_g = len(unsched)
    ready = env.ready_gates()
    ready_set = set(ready)
    n_ready = max(1, len(ready))

    max_crit = max(1, inst.depth)
    # gate features
    x_gate = np.zeros((n_g, GATE_DIM), dtype=np.float32)
    gate_depth = _gate_depths(inst)
    for i, g in enumerate(unsched):
        gr = env.gates[g]
        if gr.remote is None:
            x_gate[i, 0] = 1.0                       # unknown
        elif gr.remote:
            x_gate[i, 2] = 1.0                       # remote
        else:
            x_gate[i, 1] = 1.0                       # local
        x_gate[i, 3] = inst.criticality[g] / max_crit
        x_gate[i, 4] = 1.0 if g in ready_set else 0.0
        x_gate[i, 5] = gr.n_unfinished_preds / 2.0
        x_gate[i, 6] = gate_depth[g] / max_crit

    # qubit features
    degree = np.zeros(n_q, dtype=np.int64)
    remaining = np.zeros(n_q, dtype=np.int64)
    nbrs: list[set[int]] = [set() for _ in range(n_q)]
    for g, (a, b) in enumerate(inst.gates):
        nbrs[a].add(b)
        nbrs[b].add(a)
        if env.gates[g].state == UNSCHEDULED:
            remaining[a] += 1
            remaining[b] += 1
    for q in range(n_q):
        degree[q] = len(nbrs[q])
    max_deg = max(1, int(degree.max()) if n_q else 1)
    max_rem = max(1, int(remaining.max()) if n_q else 1)
    x_qubit = np.zeros((n_q, QUBIT_DIM), dtype=np.float32)
    next_to_map = -1
    if env._unmapped:
        from ..baselines.greedy_jit import map_emission_order
        next_to_map = next(q for q in map_emission_order(inst)
                           if q in env._unmapped)
    for q in range(n_q):
        x_qubit[q, 0] = 0.0 if env.qubit_qpu[q] is None else 1.0
        x_qubit[q, 1] = degree[q] / max_deg
        x_qubit[q, 2] = remaining[q] / max_rem
        x_qubit[q, 3] = 1.0 if q == next_to_map else 0.0

    # qpu features
    mapped_per = [0] * k_n
    for u in env.qubit_qpu:
        if u is not None:
            mapped_per[u] += 1
    ready_local = [0] * k_n
    for g in ready:
        gr = env.gates[g]
        if gr.remote is False:
            a, _ = inst.gates[g]
            ready_local[env.qubit_qpu[a]] += 1
    x_qpu = np.zeros((k_n, QPU_DIM), dtype=np.float32)
    for u in range(k_n):
        x_qpu[u, 0] = env.kappa_res[u] / hw.kappa[u]
        x_qpu[u, 1] = mapped_per[u] / hw.kappa[u]
        x_qpu[u, 2] = ready_local[u] / n_ready
        if u < MAX_QPU_ONEHOT:
            x_qpu[u, 3 + u] = 1.0

    # link features
    demand, _ = env.deficit_demand()
    x_link = np.zeros((n_l, LINK_DIM), dtype=np.float32)
    for l in range(n_l):
        lc = hw.links[l]
        ls = env.links[l]
        x_link[l, 0] = lc.p
        x_link[l, 1] = ls.free_channels / lc.W
        x_link[l, 2] = ls.stored / lc.B
        x_link[l, 3] = ls.busy_channels / lc.W
        for age in ls.stored_ages:
            if lc.T_cut is None:
                bucket = AGE_BUCKETS - 1                 # infinite remaining
            else:
                remaining_life = lc.T_cut - age          # in [0, T_cut)
                frac = remaining_life / lc.T_cut
                bucket = min(AGE_BUCKETS - 1, int(frac * AGE_BUCKETS))
            x_link[l, 4 + bucket] += 1.0 / lc.B
        x_link[l, 8] = _remote_ready_demand(env, ready, l) / n_ready

    snap = GraphSnapshot(
        x_gate=x_gate, x_qubit=x_qubit, x_qpu=x_qpu, x_link=x_link,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros(0, dtype=np.int64),
        globals=np.array([
            env.t / env.t_budget,
            env.done_count / inst.num_gates,
            (n_q - len(env._unmapped)) / n_q,
            sum(ls.stored for ls in env.links)
            / max(1, sum(lc.B for lc in hw.links)),
        ], dtype=np.float32),
        gate_ids=np.array(unsched, dtype=np.int64),
        gate_row=gate_row, n_qubit=n_q, n_qpu=k_n, n_link=n_l,
    )

    src, dst, rel = [], [], []

    def add(s: int, d: int, r: int) -> None:
        src.append(s)
        dst.append(d)
        rel.append(r)

    for i, g in enumerate(unsched):                       # 0: gate->gate
        for s_g in inst.succs[g]:
            j = gate_row.get(s_g)
            if j is not None:
                add(snap.nid_gate(i), snap.nid_gate(j), 0)
    for i, g in enumerate(unsched):                       # 1/2: gate<->qubit
        for q in inst.gates[g]:
            add(snap.nid_gate(i), snap.nid_qubit(q), 1)
            add(snap.nid_qubit(q), snap.nid_gate(i), 2)
    for q in range(n_q):                                  # 3/4: qubit<->qpu
        u = env.qubit_qpu[q]
        if u is not None:
            add(snap.nid_qubit(q), snap.nid_qpu(u), 3)
            add(snap.nid_qpu(u), snap.nid_qubit(q), 4)
    for l in range(n_l):                                  # 5/6: qpu<->link
        lc = hw.links[l]
        for u in (lc.u, lc.v):
            add(snap.nid_qpu(u), snap.nid_link(l), 5)
            add(snap.nid_link(l), snap.nid_qpu(u), 6)
    for i, g in enumerate(unsched):                       # 7/8: gate<->link
        gr = env.gates[g]
        if gr.remote:
            for l in gr.route:
                add(snap.nid_gate(i), snap.nid_link(l), 7)
                add(snap.nid_link(l), snap.nid_gate(i), 8)

    if src:
        snap.edge_index = np.array([src, dst], dtype=np.int64)
        snap.edge_type = np.array(rel, dtype=np.int64)
    return snap


def _gate_depths(inst) -> list[int]:
    """Longest path FROM sources (in gates, counting itself); cached on the
    instance object."""
    cached = getattr(inst, "_eager_gate_depths", None)
    if cached is not None:
        return cached
    depth = [1] * inst.num_gates
    for g in range(inst.num_gates):                       # list order is topo
        for s in inst.succs[g]:
            depth[s] = max(depth[s], depth[g] + 1)
    object.__setattr__(inst, "_eager_gate_depths", depth)
    return depth


def _remote_ready_demand(env: EagerEnv, ready, l: int) -> int:
    count = 0
    for g in ready:
        gr = env.gates[g]
        if gr.remote and l in gr.route:
            count += 1
    return count
