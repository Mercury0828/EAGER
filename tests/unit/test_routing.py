"""Phase 1A: fixed shortest-path routing with lexicographic tie-break."""

import pytest

from eager.config import load_hardware_config
from eager.env.routing import build_routing


def hw(**over):
    base = {"name": "t", "qpus": 4, "topology": "line", "kappa": 4,
            "link_defaults": {"p": 1.0, "W": 1, "B": 4, "T_cut": None, "w": 1.0},
            "mode": "deterministic", "t_ep": 1}
    base.update(over)
    return load_hardware_config(base)


def test_line_multi_hop():
    cfg = hw(qpus=4)                      # links: 0=(0,1) 1=(1,2) 2=(2,3)
    rt = build_routing(cfg)
    assert rt.route(0, 3) == (0, 1, 2)
    assert rt.route(1, 3) == (1, 2)
    assert rt.node_routes[(0, 3)] == (0, 1, 2, 3)


def test_ring_lexicographic_tie_break():
    cfg = hw(qpus=4, topology="ring")     # links: 0=(0,1) 1=(0,3) 2=(1,2) 3=(2,3)
    rt = build_routing(cfg)
    # 0->2 has two shortest paths: 0-1-2 and 0-3-2; lex smallest is 0-1-2.
    assert rt.node_routes[(0, 2)] == (0, 1, 2)
    assert rt.route(0, 2) == (0, 2)
    # 1->3: 1-0-3 vs 1-2-3; lex smallest is 1-0-3.
    assert rt.node_routes[(1, 3)] == (1, 0, 3)
    assert rt.route(1, 3) == (0, 1)


def test_grid_routes():
    cfg = hw(qpus=4, topology="grid", grid_dims=[2, 2])
    # links: 0=(0,1) 1=(0,2) 2=(1,3) 3=(2,3)
    rt = build_routing(cfg)
    assert rt.node_routes[(0, 3)] == (0, 1, 3)   # lex beats 0-2-3
    assert rt.route(0, 3) == (0, 2)
    assert rt.route(2, 1) == rt.route(1, 2)      # canonical unordered route


def test_same_node_empty_route():
    rt = build_routing(hw())
    assert rt.route(2, 2) == ()


def test_route_cost():
    cfg = hw(qpus=3, topology="line",
             link_overrides={"1-2": {"w": 2.5}})
    rt = build_routing(cfg)
    weights = {i: l.w for i, l in enumerate(cfg.links)}
    assert rt.cost(0, 2, weights) == pytest.approx(3.5)
