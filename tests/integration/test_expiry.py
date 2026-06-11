"""Phase 1B acceptance: buffer aging, decoherence cutoff, waste accounting.

Includes the golden aging/expiry derivation (a pair generated at slot t with
T_cut=3 and never consumed is discarded exactly at the resolve of slot
t + T_cut and charged to waste exactly once) and the consumability-window
edges (D13: consumable during slots t+1 .. t+T_cut).
"""

import pytest

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import ADVANCE, EagerEnv, GenEPR, Map, Schedule

pytestmark = pytest.mark.stochastic


def stoch_hw(**over):
    base = {"name": "exp", "qpus": 2, "topology": "line", "kappa": 4,
            "mode": "stochastic",
            "link_defaults": {"p": 1.0, "W": 1, "B": 4, "T_cut": 3, "w": 1.0}}
    base.update(over)
    return load_hardware_config(base)


def test_expiry_golden_derivation():
    """Hand derivation (p=1, T_cut=3, w=1; alpha=1, gamma=0.5):

      slot 0: Map(q0,u0) Map(q1,u0); GenEPR(l0); ADVANCE
              resolve 0: draw(l0,c0,t=0) succeeds (p=1) -> pair age 0;
                         aging -> age 1.   [pair 'generated at slot 0']
      slot 1: ADVANCE -> age 2
      slot 2: ADVANCE -> age 3  (3 <= T_cut: still stored)
      slot 3: ADVANCE -> age 4 > T_cut=3 -> DISCARDED at resolve of slot
              3 = t_gen + T_cut, charged exactly once:
              reward(ADVANCE_3) = -alpha - gamma*w = -1.5
      slot 4: Schedule(g0) [local]; ADVANCE -> done.

      Totals: T=5; C_comm=0; C_waste=1.0; J = 5 + 0 + 0.5 = 5.5;
              reward_sum = -5 - 0.5 = -5.5 = -J;
              pairs: generated=1, consumed=0, expired=1, stored=0.
    """
    env = EagerEnv(stoch_hw(), instance_from_gates("one_local", 2, ((0, 1),)))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 0))
    env.step(GenEPR(0))

    _, r, _, _ = env.step(ADVANCE)                 # resolve slot 0
    assert r == -1.0
    assert env.links[0].stored_ages == [1] and env.links[0].generated == 1

    _, r, _, _ = env.step(ADVANCE)                 # slot 1 -> age 2
    assert r == -1.0 and env.links[0].stored_ages == [2]

    _, r, _, _ = env.step(ADVANCE)                 # slot 2 -> age 3 (kept)
    assert r == -1.0 and env.links[0].stored_ages == [3]
    assert env.links[0].expired == 0 and env.c_waste == 0.0

    _, r, _, _ = env.step(ADVANCE)                 # slot 3 -> age 4 > 3: waste
    assert r == -1.5, "expiry charged on THIS advance: -alpha - gamma*w"
    assert env.links[0].stored_ages == [] and env.links[0].expired == 1
    assert env.c_waste == 1.0

    env.step(Schedule(0))
    _, r, done, info = env.step(ADVANCE)
    assert done and r == -1.0, "expiry charged exactly once, never again"

    m = info["metrics"]
    assert m["T"] == 5 and m["C_comm"] == 0.0 and m["C_waste"] == 1.0
    assert m["J"] == 5.5 and m["reward_sum"] == -5.5
    assert m["pairs"] == {"generated": 1, "consumed": 0, "expired": 1, "stored": 0}
    assert m["epr_utilization"] == 0.0
    # conservation including expiry:
    ls = env.links[0]
    assert ls.generated == ls.consumed + ls.expired + ls.stored


def remote_env():
    env = EagerEnv(stoch_hw(), instance_from_gates("one_remote", 2, ((0, 1),)))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 1))                            # remote across link 0
    env.step(GenEPR(0))
    env.step(ADVANCE)                              # pair generated at slot 0
    return env


def test_pair_consumable_on_last_window_slot():
    """Generated at slot 0, T_cut=3 -> consumable during slots 1..3; consuming
    at slot 3 (age exactly T_cut) must succeed with zero waste."""
    env = remote_env()
    env.step(ADVANCE)                              # slot 1
    env.step(ADVANCE)                              # slot 2
    assert env.links[0].stored_ages == [3]         # slot 3 micro time
    assert env.is_valid(Schedule(0))
    _, r, _, _ = env.step(Schedule(0))
    assert r == -1.0 and env.links[0].consumed == 1
    env.step(ADVANCE)
    _, _, done, info = env.step(ADVANCE)
    assert done
    m = info["metrics"]
    assert m["C_waste"] == 0.0 and m["pairs"]["expired"] == 0


def test_pair_gone_one_slot_after_window():
    env = remote_env()
    for _ in range(3):                             # slots 1, 2, 3 idle
        env.step(ADVANCE)
    # slot 4: expired at resolve of slot 3; remote gate is blocked again
    assert env.links[0].stored == 0 and env.links[0].expired == 1
    assert not env.is_valid(Schedule(0))


def test_conservation_with_interleaved_expiry_and_consumption():
    """Generate continuously with a tight cutoff; consume occasionally; the
    per-link conservation must hold after every slot."""
    hw = stoch_hw(link_defaults={"p": 0.5, "W": 2, "B": 3, "T_cut": 2, "w": 1.0})
    env = EagerEnv(hw, instance_from_gates(
        "two_remote", 4, ((0, 1), (2, 3), (0, 2))))
    env.reset(3)
    env.step(Map(0, 0))
    env.step(Map(1, 1))
    env.step(Map(2, 0))
    env.step(Map(3, 1))
    done = False
    micro = 0
    while not done:
        ls = env.links[0]
        if env.is_valid(Schedule(0)):
            action = Schedule(0)
        elif env.is_valid(Schedule(1)):
            action = Schedule(1)
        elif env.is_valid(Schedule(2)):
            action = Schedule(2)
        elif env.is_valid(GenEPR(0)):
            action = GenEPR(0)
        else:
            action = ADVANCE
        _, _, done, info = env.step(action)
        assert ls.generated == ls.consumed + ls.expired + ls.stored, (
            f"conservation broken at t={env.t}")
        micro += 1
        assert micro < 100_000
    m = info["metrics"]
    assert m["done"] and not m["truncated"]
    p = m["pairs"]
    assert p["generated"] == p["consumed"] + p["expired"] + p["stored"]
