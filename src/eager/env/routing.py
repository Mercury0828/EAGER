"""Fixed shortest-path routing, precomputed per hardware config (guide §4.3).

Routes are hop-count shortest paths with a lexicographic tie-break: among all
shortest paths, the lexicographically smallest node sequence (achieved by
greedily taking the lowest-indexed next hop that stays on a shortest path).

One canonical route per *unordered* QPU pair {u, v}, computed from the
lower-indexed endpoint (D17), so pair consumption is independent of gate
operand order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..config import HardwareConfig


@dataclass(frozen=True)
class RoutingTable:
    """Precomputed canonical routes. Keys are (u, v) with u < v."""

    link_routes: dict[tuple[int, int], tuple[int, ...]]   # link ids along route
    node_routes: dict[tuple[int, int], tuple[int, ...]]   # node sequence u..v

    def route(self, u: int, v: int) -> tuple[int, ...]:
        """Link ids of the canonical route between u and v (empty if u == v)."""
        if u == v:
            return ()
        key = (u, v) if u < v else (v, u)
        return self.link_routes[key]

    def cost(self, u: int, v: int, link_weights: dict[int, float]) -> float:
        return sum(link_weights[l] for l in self.route(u, v))


def _bfs_distances(adj: list[list[int]], source: int) -> list[int]:
    dist = [-1] * len(adj)
    dist[source] = 0
    q = deque([source])
    while q:
        cur = q.popleft()
        for n in adj[cur]:
            if dist[n] == -1:
                dist[n] = dist[cur] + 1
                q.append(n)
    return dist


def build_routing(hw: HardwareConfig) -> RoutingTable:
    k = hw.num_qpus
    adj: list[list[int]] = [[] for _ in range(k)]
    edge_to_id: dict[tuple[int, int], int] = {}
    for i, l in enumerate(hw.links):
        adj[l.u].append(l.v)
        adj[l.v].append(l.u)
        edge_to_id[(l.u, l.v)] = i
    for nbrs in adj:
        nbrs.sort()

    link_routes: dict[tuple[int, int], tuple[int, ...]] = {}
    node_routes: dict[tuple[int, int], tuple[int, ...]] = {}
    dist_from: dict[int, list[int]] = {}
    for u in range(k):
        for v in range(u + 1, k):
            if v not in dist_from:
                dist_from[v] = _bfs_distances(adj, v)
            dist_v = dist_from[v]
            # Greedy lowest-index next hop along shortest paths u -> v gives the
            # lexicographically smallest shortest node sequence.
            path = [u]
            cur = u
            while cur != v:
                cur = min(n for n in adj[cur] if dist_v[n] == dist_v[cur] - 1)
                path.append(cur)
            links = []
            for a, b in zip(path, path[1:]):
                e = (a, b) if a < b else (b, a)
                links.append(edge_to_id[e])
            link_routes[(u, v)] = tuple(links)
            node_routes[(u, v)] = tuple(path)

    return RoutingTable(link_routes=link_routes, node_routes=node_routes)
