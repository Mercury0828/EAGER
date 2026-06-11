"""Attention decoder over valid actions + value head (guide §7.2).

Global query q = MLP(mean_i h_i  ++  globals), d_k = 128. Keys per valid
action: Map(q_i,u): W_map[h_q ++ h_u]; Schedule(g): W_sch h_g; GenEPR(l):
W_gen h_l; ADVANCE: W_adv q (query-conditioned). Logits e_a = q.k_a/sqrt(d_k)
over the VALID set only (invalid actions are never enumerated, which equals
masking to -inf); per-state softmax is a segment softmax over variable-size
action sets. Value head V(s) = MLP(mean_i h_i ++ globals), sharing the
encoder.

Action spec codes (CPU-side, built per state with the snapshot's gate-row
mapping): 0 Map(idx1=qubit, idx2=qpu) | 1 Schedule(idx1=gate row) |
2 GenEPR(idx1=link) | 3 ADVANCE. The order of actions inside a state is the
env's deterministic D15 enumeration of the valid set, so a position index
fully identifies an action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..env.actions import ADVANCE, Action, Advance, GenEPR, Map, Schedule
from ..env.env import EagerEnv
from .encoder import HIDDEN, BatchedGraphs, RGCNEncoder, mean_readout
from .graph import GLOBALS_DIM, GraphSnapshot

A_MAP, A_SCH, A_GEN, A_ADV = 0, 1, 2, 3


@dataclass
class ActionSet:
    actions: list[Action]       # env Action objects, D15 valid-set order
    spec: np.ndarray            # [n, 3] int64: (code, idx1, idx2)


def build_action_set(env: EagerEnv, snap: GraphSnapshot) -> ActionSet:
    actions = env.valid_actions()
    spec = np.zeros((len(actions), 3), dtype=np.int64)
    for i, a in enumerate(actions):
        if isinstance(a, Map):
            spec[i] = (A_MAP, a.qubit, a.qpu)
        elif isinstance(a, Schedule):
            spec[i] = (A_SCH, snap.gate_row[a.gate], 0)
        elif isinstance(a, GenEPR):
            spec[i] = (A_GEN, a.link, 0)
        else:
            spec[i] = (A_ADV, 0, 0)
    return ActionSet(actions=actions, spec=spec)


@dataclass
class PolicyOutput:
    logits: torch.Tensor        # [total_actions] flat across the batch
    act_graph: torch.Tensor     # [total_actions] graph id per action
    ptr: list[int]              # start offset of each graph's action block
    value: torch.Tensor         # [n_graphs]

    def log_softmax(self) -> torch.Tensor:
        return self.logits - segment_logsumexp(self.logits, self.act_graph,
                                               len(self.ptr))[self.act_graph]

    def log_prob_of(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: [n_graphs] index within each graph's action block."""
        flat = torch.tensor(self.ptr, device=positions.device) + positions
        return self.log_softmax()[flat]

    def entropy(self) -> torch.Tensor:
        logp = self.log_softmax()
        p = logp.exp()
        n_graphs = len(self.ptr)
        out = torch.zeros(n_graphs, device=logp.device)
        out.index_add_(0, self.act_graph, -p * logp)
        return out

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Gumbel-max segment sampling -> per-graph position indices."""
        u = torch.rand(self.logits.shape, device=self.logits.device,
                       generator=generator).clamp_(1e-10, 1 - 1e-10)
        z = self.logits - torch.log(-torch.log(u))
        return _segment_argmax_positions(z, self.act_graph, self.ptr)

    def greedy(self) -> torch.Tensor:
        return _segment_argmax_positions(self.logits, self.act_graph, self.ptr)


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int
                      ) -> torch.Tensor:
    m = torch.full((n,), -torch.inf, device=x.device)
    m = m.scatter_reduce(0, seg, x, reduce="amax", include_self=True)
    s = torch.zeros(n, device=x.device)
    s.index_add_(0, seg, (x - m[seg]).exp())
    return m + s.clamp(min=1e-30).log()


def _segment_argmax_positions(x: torch.Tensor, seg: torch.Tensor,
                              ptr: list[int]) -> torch.Tensor:
    m = torch.full((len(ptr),), -torch.inf, device=x.device)
    m = m.scatter_reduce(0, seg, x, reduce="amax", include_self=True)
    is_max = (x == m[seg])
    idx = torch.arange(x.shape[0], device=x.device)
    first = torch.full((len(ptr),), x.shape[0], device=x.device,
                       dtype=torch.long)
    first = first.scatter_reduce(0, seg[is_max], idx[is_max], reduce="amin",
                                 include_self=True)
    return first - torch.tensor(ptr, device=x.device)


class EagerPolicy(nn.Module):
    """Encoder + pointer decoder + value head (guide §7)."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.encoder = RGCNEncoder(hidden=hidden)
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden + GLOBALS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.w_map = nn.Linear(2 * hidden, hidden)
        self.w_sch = nn.Linear(hidden, hidden)
        self.w_gen = nn.Linear(hidden, hidden)
        self.w_adv = nn.Linear(hidden, hidden)
        self.v_mlp = nn.Sequential(
            nn.Linear(hidden + GLOBALS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, batch: BatchedGraphs,
                action_sets: list[ActionSet]) -> PolicyOutput:
        h = self.encoder(batch)
        readout = mean_readout(h, batch)
        ctx = torch.cat([readout, batch.globals], dim=1)
        query = self.q_mlp(ctx)                              # [G, d]
        value = self.v_mlp(ctx).squeeze(-1)                  # [G]

        device = h.device
        ptr, act_graph_parts = [], []
        spec_parts = []
        total = 0
        for i, aset in enumerate(action_sets):
            n = aset.spec.shape[0]
            ptr.append(total)
            total += n
            act_graph_parts.append(np.full(n, i, dtype=np.int64))
            spec_parts.append(aset.spec)
        act_graph = torch.from_numpy(np.concatenate(act_graph_parts)).to(device)
        spec = torch.from_numpy(np.concatenate(spec_parts, axis=0)).to(device)

        # resolve node rows per action using the batch's per-type position maps
        snaps = batch.snaps
        gate_off, qubit_off, qpu_off, link_off = [], [], [], []
        g_acc = q_acc = u_acc = l_acc = 0
        for s in snaps:
            gate_off.append(g_acc)
            qubit_off.append(q_acc)
            qpu_off.append(u_acc)
            link_off.append(l_acc)
            g_acc += s.n_gate
            q_acc += s.n_qubit
            u_acc += s.n_qpu
            l_acc += s.n_link
        gate_off = torch.tensor(gate_off, device=device)[act_graph]
        qubit_off = torch.tensor(qubit_off, device=device)[act_graph]
        qpu_off = torch.tensor(qpu_off, device=device)[act_graph]
        link_off = torch.tensor(link_off, device=device)[act_graph]

        h_gate_all = h[batch.gate_pos]
        h_qubit_all = h[batch.qubit_pos]
        h_qpu_all = h[batch.qpu_pos]
        h_link_all = h[batch.link_pos]

        code = spec[:, 0]
        keys = torch.zeros(total, self.hidden, device=device)
        m_map = code == A_MAP
        if m_map.any():
            hq = h_qubit_all[qubit_off[m_map] + spec[m_map, 1]]
            hu = h_qpu_all[qpu_off[m_map] + spec[m_map, 2]]
            keys[m_map] = self.w_map(torch.cat([hq, hu], dim=1))
        m_sch = code == A_SCH
        if m_sch.any():
            keys[m_sch] = self.w_sch(h_gate_all[gate_off[m_sch] + spec[m_sch, 1]])
        m_gen = code == A_GEN
        if m_gen.any():
            keys[m_gen] = self.w_gen(h_link_all[link_off[m_gen] + spec[m_gen, 1]])
        m_adv = code == A_ADV
        if m_adv.any():
            keys[m_adv] = self.w_adv(query[act_graph[m_adv]])

        logits = (query[act_graph] * keys).sum(dim=1) / math.sqrt(self.hidden)
        return PolicyOutput(logits=logits, act_graph=act_graph, ptr=ptr,
                            value=value)


def act_greedy(policy: EagerPolicy, env: EagerEnv, device) -> Action:
    """Single-env greedy action (evaluation path)."""
    from .graph import build_graph
    snap = build_graph(env)
    aset = build_action_set(env, snap)
    batch = BatchedGraphs([snap], device)
    with torch.no_grad():
        out = policy(batch, [aset])
    pos = int(out.greedy()[0].item())
    return aset.actions[pos]
