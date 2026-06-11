"""Phase 1A: deterministic-mode timing, rewards, and multi-hop consumption
(conventions frozen in D13)."""

import pytest

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import ADVANCE, EagerEnv, EnvParams, GenEPR, Map, Schedule


def det_hw(**over):
    base = {"name": "t", "qpus": 2, "topology": "line", "kappa": 4,
            "mode": "deterministic", "t_ep": 2,
            "link_defaults": {"p": 1.0, "W": 2, "B": 4, "T_cut": None, "w": 1.0}}
    base.update(over)
    return load_hardware_config(base)


def test_deterministic_generation_timing():
    """Channel tasked in slot t delivers at the resolve of slot t+t_ep-1;
    the pair is visible (age 1) from slot t+t_ep onward."""
    env = EagerEnv(det_hw(), instance_from_gates("g", 2, ((0, 1),)))
    env.reset(0)
    obs, *_ = env.step(GenEPR(0))
    assert obs["links"][0] == {"stored_ages": [], "busy": 1, "free": 1}
    obs, *_ = env.step(ADVANCE)          # resolve slot 0: countdown 2->1
    assert obs["links"][0]["busy"] == 1 and obs["links"][0]["stored_ages"] == []
    obs, *_ = env.step(ADVANCE)          # resolve slot 1: pair lands, ages to 1
    assert obs["links"][0] == {"stored_ages": [1], "busy": 0, "free": 2}
    assert env.links[0].generated == 1


def test_single_local_gate_episode():
    env = EagerEnv(det_hw(), instance_from_gates("g", 2, ((0, 1),)))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 0))
    obs, r, done, info = env.step(Schedule(0))
    assert r == 0.0 and not done
    obs, r, done, info = env.step(ADVANCE)
    assert done and r == -1.0
    m = info["metrics"]
    assert m["T"] == 1 and m["C_comm"] == 0.0 and m["J"] == 1.0
    assert m["reward_sum"] == -1.0
    assert env.gates[0].schedule_slot == 0 and env.gates[0].remote is False


def test_remote_gate_duration_and_reward():
    env = EagerEnv(det_hw(t_ep=1), instance_from_gates("g", 2, ((0, 1),)))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 1))                  # remote across link 0
    env.step(GenEPR(0))
    env.step(ADVANCE)                    # pair lands (t_ep=1)
    obs, r, done, info = env.step(Schedule(0))
    assert r == -1.0, "consumption charged -beta*w at schedule time"
    assert env.links[0].stored == 0 and env.links[0].consumed == 1
    obs, r, done, info = env.step(ADVANCE)   # d_rem=2: 1 slot left
    assert not done
    obs, r, done, info = env.step(ADVANCE)
    assert done
    m = info["metrics"]
    # T=3 slots (gen, run, run), C_comm=1 -> J=4; reward_sum=-4
    assert m["T"] == 3 and m["C_comm"] == 1.0 and m["J"] == 4.0
    assert m["reward_sum"] == -4.0
    assert m["epr_utilization"] == 1.0
    assert m["mean_remote_stall"] == 1.0  # ready slot 0, scheduled slot 1


def test_multi_hop_consumes_every_route_link():
    hw = load_hardware_config(
        {"name": "k3", "qpus": 3, "topology": "line", "kappa": [1, 1, 1],
         "mode": "deterministic", "t_ep": 1,
         "link_defaults": {"p": 1.0, "W": 1, "B": 2, "T_cut": None, "w": 1.0}})
    env = EagerEnv(hw, instance_from_gates("g", 2, ((0, 1),)))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 2))                  # route u0->u2 = links (0, 1)
    assert env.gates[0].route == (0, 1)
    env.step(GenEPR(0))
    env.step(GenEPR(1))
    env.step(ADVANCE)
    obs, r, done, info = env.step(Schedule(0))
    assert r == -2.0, "one pair consumed per route link, simultaneously"
    assert env.links[0].consumed == 1 and env.links[1].consumed == 1
    assert env.c_comm == 2.0


def test_truncation_penalty_and_flag():
    env = EagerEnv(det_hw(), instance_from_gates("g", 2, ((0, 1),)),
                   EnvParams(t_budget=2))
    env.reset(0)
    total = 0.0
    for _ in range(3):                   # slots 0,1 resolve; slot 2 -> t=3 > 2
        obs, r, done, info = env.step(ADVANCE)
        total += r
    assert done and info["truncated"]
    m = info["metrics"]
    assert m["truncated"] and m["unfinished_gates"] == 1
    # reward = -3 (slots) - 1*10*1 (penalty); J = T=3
    assert total == pytest.approx(-13.0)
    assert m["J"] == pytest.approx(3.0)
    with pytest.raises(RuntimeError, match="finished"):
        env.step(ADVANCE)


def test_gate_becomes_ready_next_slot_after_pred():
    env = EagerEnv(det_hw(), instance_from_gates("c", 2, ((0, 1), (0, 1))))
    env.reset(0)
    env.step(Map(0, 0))
    env.step(Map(1, 0))
    env.step(Schedule(0))
    assert env.gates[1].ready_slot is None
    env.step(ADVANCE)                    # g0 completes at resolve of slot 0
    assert env.gates[1].ready_slot == 1
    assert env.is_valid(Schedule(1))
