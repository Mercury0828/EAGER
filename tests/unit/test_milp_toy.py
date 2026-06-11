"""Phase 4 acceptance: the MILP linearization is validated by brute-force
enumeration on <= 3-qubit toys (guide §11), across generation-timing,
channel-window, and buffer regimes; every MILP solution also replays in the
env to the exact same J."""

import pytest

gp = pytest.importorskip("gurobipy")

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import EnvParams
from eager.exact.brute_force import brute_force_optimum
from eager.exact.milp import replay_exact, solve_exact


@pytest.fixture(scope="module", autouse=True)
def _license_available():
    try:
        with gp.Env(params={"OutputFlag": 0}):
            pass
    except gp.GurobiError as exc:        # pragma: no cover
        pytest.skip(f"no usable Gurobi license: {exc}")


def det_hw(**over):
    base = {"name": "toy", "qpus": 2, "topology": "line", "kappa": [2, 2],
            "mode": "deterministic", "t_ep": 2,
            "link_defaults": {"p": 1.0, "W": 1, "B": 2, "T_cut": None,
                              "w": 1.0}}
    base.update(over)
    return load_hardware_config(base)


TOYS = {
    # micro1-shaped chain, one forced split (kappa total 4 >= 3 qubits)
    "chain3": (det_hw(),
               instance_from_gates("chain3", 3, ((0, 1), (1, 2), (0, 1)))),
    # both gates forced remote on one link, W=1: sequential pair generation
    "remote2": (det_hw(kappa=[1, 1]),
                instance_from_gates("remote2", 2, ((0, 1), (0, 1)))),
    # tight buffer B=1 with fast generation and two channels
    "buffer1": (det_hw(kappa=[1, 1], t_ep=1,
                       link_defaults={"p": 1.0, "W": 2, "B": 1,
                                      "T_cut": None, "w": 1.0}),
                instance_from_gates("buffer1", 2, ((0, 1), (0, 1)))),
    # all-local possible: optimum must avoid the link entirely
    "local3": (det_hw(kappa=[3, 1]),
               instance_from_gates("local3", 3, ((0, 1), (1, 2)))),
    # 3-QPU line: multi-hop route consumes BOTH links
    "hop3": (det_hw(qpus=3, topology="line", kappa=[1, 1, 1]),
             instance_from_gates("hop3", 3, ((0, 2), (1, 2)))),
}


@pytest.mark.parametrize("toy", sorted(TOYS))
def test_milp_matches_brute_force(toy):
    hw, inst = TOYS[toy]
    params = EnvParams()
    horizon = 12
    bf = brute_force_optimum(hw, inst, params, horizon=horizon)
    res = solve_exact(hw, inst, params, horizon=horizon, time_limit=120)
    assert res.status == "OPTIMAL", (toy, res.status)
    assert res.j_star == pytest.approx(bf["J"], abs=1e-9), (
        toy, res.j_star, bf["J"])
    # the solution must replay in the env to the same J (semantics match)
    m = replay_exact(res, hw, inst, params)
    assert m["J"] == pytest.approx(res.j_star, abs=1e-9)
    assert not m["truncated"]


def test_local3_avoids_communication():
    hw, inst = TOYS["local3"]
    res = solve_exact(hw, inst, EnvParams(), horizon=10, time_limit=60)
    assert res.c_comm == 0.0
    assert res.t_makespan == 2          # two serial local gates
    assert res.j_star == 2.0


def test_remote2_hand_value():
    """W=1, t_ep=2: pair 1 tasked at 0 (available slot 2) -> g0 at 2..3;
    pair 2 tasked at 2 (window allows) -> available 4 -> g1 at 4..5.
    T=6, C=2 -> J=8."""
    hw, inst = TOYS["remote2"]
    res = solve_exact(hw, inst, EnvParams(), horizon=12, time_limit=60)
    assert res.status == "OPTIMAL" and res.j_star == 8.0


def test_horizon_from_greedy_is_lossless():
    """Solving with the GreedyJIT horizon must give the same optimum as a
    generously larger horizon."""
    hw, inst = TOYS["chain3"]
    res_default = solve_exact(hw, inst, EnvParams(), time_limit=60)  # greedy H
    res_wide = solve_exact(hw, inst, EnvParams(), horizon=14, time_limit=60)
    assert res_default.j_star == pytest.approx(res_wide.j_star, abs=1e-9)
