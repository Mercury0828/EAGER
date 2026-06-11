"""Phase 1B: stochastic mode with p=1 is trajectory-identical to
deterministic mode with t_ep=1 (the two semantics' anchor point), and the
deterministic switch remains available (Gurobi experiments depend on it)."""

import pytest

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import EagerEnv
from eager.utils.hashing import TrajectoryHasher
from eager.utils.scripted_policies import simple_jit_policy

pytestmark = pytest.mark.stochastic


def hw(mode, t_ep, p):
    return load_hardware_config(
        {"name": f"eq_{mode}", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": mode, "t_ep": t_ep,
         "link_defaults": {"p": p, "W": 2, "B": 4, "T_cut": 8, "w": 1.0}})


def trajectory_hash(hardware, seed):
    inst = instance_from_gates(
        "mix", 5, ((0, 1), (2, 3), (1, 2), (3, 4), (0, 1), (2, 4)))
    env = EagerEnv(hardware, inst)
    hasher = TrajectoryHasher()
    obs = env.reset(seed)
    hasher.update_reset(obs)
    done = False
    while not done:
        action = simple_jit_policy(env)
        obs, r, done, _ = env.step(action)
        hasher.update(action, obs, r, done)
    return hasher.hexdigest(), env.metrics()


def test_stochastic_p1_equals_deterministic_tep1():
    h_det, m_det = trajectory_hash(hw("deterministic", t_ep=1, p=1.0), seed=9)
    h_sto, m_sto = trajectory_hash(hw("stochastic", t_ep=1, p=1.0), seed=9)
    assert h_det == h_sto
    assert m_det["J"] == m_sto["J"]


def test_deterministic_mode_still_available_with_cutoff_inf():
    """The §5.2 deterministic special case (p=1 semantics, T_cut=inf, fixed
    t_ep) stays reachable via config switch."""
    cfg = load_hardware_config(
        {"name": "det12", "qpus": 2, "topology": "line", "kappa": 6,
         "mode": "deterministic", "t_ep": 12,
         "link_defaults": {"p": 1.0, "W": 2, "B": 8, "T_cut": None, "w": 1.0}})
    assert cfg.deterministic and cfg.t_ep == 12 and cfg.links[0].T_cut is None
    inst = instance_from_gates("g", 2, ((0, 1),))
    env = EagerEnv(cfg, inst)
    env.reset(0)
    assert env._crn is None, "no CRN engine in deterministic mode"
