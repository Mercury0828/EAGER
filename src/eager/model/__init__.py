"""EAGER agent: heterogeneous state graph, R-GCN encoder, attention decoder,
value head (guide §6.2, §7)."""

from .graph import GraphSnapshot, build_graph

__all__ = ["GraphSnapshot", "build_graph"]
