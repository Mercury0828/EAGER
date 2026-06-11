"""R-GCN encoder (guide §7.1): per-type input projections to d=128, then
L=3 RGCNConv layers (per-relation weights + root/self-loop weight W_0,
mean normalization by per-relation in-degree), ReLU, LayerNorm between
layers. PyTorch Geometric RGCNConv (the guide's preferred path; verified
working on this machine — D48).

Batched encoding uses flattened block-diagonal graphs with a `node2graph`
segment vector (PyG-style batching without requiring Data objects).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import RGCNConv

from .graph import (
    GATE_DIM,
    GLOBALS_DIM,
    LINK_DIM,
    NUM_RELATIONS,
    QPU_DIM,
    QUBIT_DIM,
    GraphSnapshot,
)

HIDDEN = 128
LAYERS = 3


class BatchedGraphs:
    """Block-diagonal batch of GraphSnapshots, tensorized once."""

    def __init__(self, snaps: list[GraphSnapshot], device: torch.device):
        self.snaps = snaps
        self.device = device
        offsets, n2g = [], []
        total = 0
        for i, s in enumerate(snaps):
            offsets.append(total)
            n2g.extend([i] * s.num_nodes)
            total += s.num_nodes
        self.offsets = offsets
        self.num_nodes = total
        self.node2graph = torch.tensor(n2g, dtype=torch.long, device=device)
        self.x_gate = _cat([s.x_gate for s in snaps], GATE_DIM, device)
        self.x_qubit = _cat([s.x_qubit for s in snaps], QUBIT_DIM, device)
        self.x_qpu = _cat([s.x_qpu for s in snaps], QPU_DIM, device)
        self.x_link = _cat([s.x_link for s in snaps], LINK_DIM, device)
        eis, ets = [], []
        for s, off in zip(snaps, offsets):
            if s.edge_index.shape[1]:
                eis.append(s.edge_index + off)
                ets.append(s.edge_type)
        if eis:
            self.edge_index = torch.from_numpy(
                np.concatenate(eis, axis=1)).to(device)
            self.edge_type = torch.from_numpy(np.concatenate(ets)).to(device)
        else:
            self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            self.edge_type = torch.zeros(0, dtype=torch.long, device=device)
        self.globals = torch.from_numpy(
            np.stack([s.globals for s in snaps])).to(device)
        # per-type node positions within the flattened batch
        gate_pos, qubit_pos, qpu_pos, link_pos = [], [], [], []
        for s, off in zip(snaps, offsets):
            gate_pos.append(np.arange(s.n_gate) + off)
            qubit_pos.append(np.arange(s.n_qubit) + off + s.n_gate)
            qpu_pos.append(np.arange(s.n_qpu) + off + s.n_gate + s.n_qubit)
            link_pos.append(np.arange(s.n_link) + off + s.n_gate
                            + s.n_qubit + s.n_qpu)
        self.gate_pos = torch.from_numpy(np.concatenate(gate_pos)).to(device)
        self.qubit_pos = torch.from_numpy(np.concatenate(qubit_pos)).to(device)
        self.qpu_pos = torch.from_numpy(np.concatenate(qpu_pos)).to(device)
        self.link_pos = torch.from_numpy(np.concatenate(link_pos)).to(device)


def _cat(arrs: list[np.ndarray], dim: int, device) -> torch.Tensor:
    if not arrs or all(a.shape[0] == 0 for a in arrs):
        return torch.zeros(0, dim, device=device)
    return torch.from_numpy(np.concatenate(arrs, axis=0)).to(device)


class RGCNEncoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.hidden = hidden
        self.in_gate = nn.Linear(GATE_DIM, hidden)
        self.in_qubit = nn.Linear(QUBIT_DIM, hidden)
        self.in_qpu = nn.Linear(QPU_DIM, hidden)
        self.in_link = nn.Linear(LINK_DIM, hidden)
        self.convs = nn.ModuleList([
            RGCNConv(hidden, hidden, num_relations=NUM_RELATIONS,
                     aggr="mean", root_weight=True)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, batch: BatchedGraphs) -> torch.Tensor:
        """Return node embeddings [num_nodes, hidden] in batch layout."""
        h = torch.zeros(batch.num_nodes, self.hidden,
                        device=batch.node2graph.device)
        if batch.x_gate.shape[0]:
            h[batch.gate_pos] = self.in_gate(batch.x_gate)
        h[batch.qubit_pos] = self.in_qubit(batch.x_qubit)
        h[batch.qpu_pos] = self.in_qpu(batch.x_qpu)
        h[batch.link_pos] = self.in_link(batch.x_link)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(torch.relu(
                conv(h, batch.edge_index, batch.edge_type)))
        return h


def mean_readout(h: torch.Tensor, batch: BatchedGraphs) -> torch.Tensor:
    """Per-graph mean over ALL nodes [n_graphs, hidden] (permutation
    invariant, guide §7.2)."""
    n_graphs = batch.globals.shape[0]
    out = torch.zeros(n_graphs, h.shape[1], device=h.device)
    out.index_add_(0, batch.node2graph, h)
    counts = torch.bincount(batch.node2graph, minlength=n_graphs
                            ).clamp(min=1).unsqueeze(1)
    return out / counts
