"""Phase 7 (D84): clairvoyant stochastic optimum — the per-seed env is
CRN-deterministic, so branch-and-bound finds the PROVEN minimum J, which lower-
bounds every policy's J on that seed (perfect-information bound)."""

import pytest

from eager.baselines.greedy_jit import GreedyEagerPolicy, GreedyJITPolicy
from eager.baselines.traces import run_episode
from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import EagerEnv
from eager.env.env import EnvParams
from eager.exact.stochastic_opt import NodeCapExceeded, clairvoyant_optimum


def hw(kappa):
    return load_hardware_config(
        {"name": "tiny", "qpus": 2, "topology": "line", "kappa": kappa,
         "mode": "stochastic", "t_ep": 2,
         "link_defaults": {"p": 0.5, "W": 1, "B": 2, "T_cut": 20, "w": 1.0}})


def test_optimum_lower_bounds_every_policy():
    """On instances with hideable latency the optimum is strictly below
    reactive JIT, and never above ANY policy (perfect-information bound)."""
    inst = instance_from_gates("q4m3", 4, ((0, 1), (2, 3), (1, 2)))
    h = hw(2)
    for seed in range(4):
        opt = clairvoyant_optimum(h, inst, seed, EnvParams())["J"]
        jit = run_episode(EagerEnv(h, inst, EnvParams()),
                          GreedyJITPolicy(placement_seed=0), seed)[0]["metrics"]["J"]
        eag = run_episode(EagerEnv(h, inst, EnvParams()),
                          GreedyEagerPolicy(), seed)[0]["metrics"]["J"]
        assert opt <= jit + 1e-9        # optimum cannot exceed any policy
        assert opt <= eag + 1e-9
    # at least one seed: reactive strictly worse than optimum (hideable latency)
    gaps = [run_episode(EagerEnv(h, inst, EnvParams()),
                        GreedyJITPolicy(placement_seed=0), s)[0]["metrics"]["J"]
            - clairvoyant_optimum(h, inst, s, EnvParams())["J"]
            for s in range(4)]
    assert max(gaps) > 0


def test_deterministic_per_seed():
    inst = instance_from_gates("q2m1", 2, ((0, 1),))
    h = hw(1)
    a = clairvoyant_optimum(h, inst, 3, EnvParams())
    b = clairvoyant_optimum(h, inst, 3, EnvParams())
    assert a["J"] == b["J"] and a["nodes"] == b["nodes"]


def test_node_cap_raises():
    inst = instance_from_gates("q4m3", 4, ((0, 1), (2, 3), (1, 2)))
    with pytest.raises(NodeCapExceeded):
        clairvoyant_optimum(hw(2), inst, 0, EnvParams(), node_cap=50)


def test_deterministic_hardware_rejected():
    inst = instance_from_gates("q2m1", 2, ((0, 1),))
    det = load_hardware_config(
        {"name": "det", "qpus": 2, "topology": "line", "kappa": 1,
         "mode": "deterministic", "t_ep": 2,
         "link_defaults": {"p": 1.0, "W": 1, "B": 2, "T_cut": None, "w": 1.0}})
    with pytest.raises(ValueError):
        clairvoyant_optimum(det, inst, 0, EnvParams())
