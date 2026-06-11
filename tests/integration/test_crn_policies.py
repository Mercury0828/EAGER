"""Phase 1B acceptance: two DIFFERENT scripted policies under the same run
seed observe IDENTICAL draws at identical (link, channel, t) coordinates
(guide §6.5 — the property that makes CRN-paired comparisons valid)."""

import pytest

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import ADVANCE, EagerEnv, EnvParams, GenEPR

pytestmark = pytest.mark.stochastic

N_SLOTS = 30
SEED = 5


def k3_hw():
    return load_hardware_config(
        {"name": "k3s", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": "stochastic",
         "link_defaults": {"p": 0.3, "W": 1, "B": 4, "T_cut": None, "w": 1.0}})


def make_env():
    inst = instance_from_gates("idle", 2, ((0, 1),))
    return EagerEnv(k3_hw(), inst, EnvParams(record_draws=True))


def run_policy_a(seed):
    """Policy A: tasks link 0 only, lets it generate-until-success, then idles."""
    env = make_env()
    env.reset(seed)
    env.step(GenEPR(0))
    for _ in range(N_SLOTS):
        env.step(ADVANCE)
    return env


def run_policy_b(seed):
    """Policy B (different): tasks link 0 AND link 1, re-tasks link 1 after
    every success; never touches the circuit either."""
    env = make_env()
    env.reset(seed)
    env.step(GenEPR(0))
    env.step(GenEPR(1))
    for _ in range(N_SLOTS):
        if (env.is_valid(GenEPR(1))):
            env.step(GenEPR(1))
        env.step(ADVANCE)
    return env


def test_identical_draws_at_identical_coordinates():
    env_a = run_policy_a(SEED)
    env_b = run_policy_b(SEED)
    draws_a, draws_b = env_a.draw_log, env_b.draw_log

    common = set(draws_a) & set(draws_b)
    assert common, "policies must coincide on some (l,c,t) coordinates"
    for coord in common:
        assert draws_a[coord] == draws_b[coord], (
            f"draw mismatch at {coord}: {draws_a[coord]} vs {draws_b[coord]}")

    # Both policies tasked link 0 channel 0 from slot 0 in
    # generate-until-success mode, so their link-0 tasking patterns coincide
    # exactly: same draws -> same success slot -> identical draw sets.
    a_l0 = {k: v for k, v in draws_a.items() if k[0] == 0}
    b_l0 = {k: v for k, v in draws_b.items() if k[0] == 0}
    assert a_l0 == b_l0
    assert any(v for v in a_l0.values()), "link 0 should succeed within horizon"
    assert env_a.links[0].generated == env_b.links[0].generated

    # Policy B's extra activity on link 1 must not perturb link-0 luck.
    assert any(k[0] == 1 for k in draws_b), "B did draw on link 1"
    assert not any(k[0] == 1 for k in draws_a)


def test_same_policy_same_seed_identical_log():
    e1, e2 = run_policy_b(SEED), run_policy_b(SEED)
    assert e1.draw_log == e2.draw_log


def test_different_seed_changes_luck():
    e1, e2 = run_policy_a(0), run_policy_a(1)
    l0_1 = {k: v for k, v in e1.draw_log.items() if k[0] == 0}
    l0_2 = {k: v for k, v in e2.draw_log.items() if k[0] == 0}
    assert l0_1 != l0_2, "different run seeds should change generation luck"
