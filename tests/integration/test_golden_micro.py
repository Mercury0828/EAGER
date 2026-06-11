"""Phase 1A acceptance: TWO golden micro-instances with hand-computed
makespan, C_comm, and J reproduced EXACTLY by a scripted action sequence.

Shared setup — hardware configs/hardware/golden_k2_det.yaml:
    K=2 QPUs (u0, u1), one link l0=(u0,u1); kappa=[2,2];
    deterministic mode, t_ep=2 (a tasked channel delivers its pair at the
    resolve of the SECOND slot after tasking, i.e. tasked in slot t ->
    pair lands at resolve of slot t+1, visible/consumable from slot t+2);
    W=2 channels, B=4, T_cut=null (no expiry), w=1.0 per pair.
    EnvParams defaults: alpha=1, beta=1, gamma=0.5, d_loc=1, d_rem=2.

Conventions under test (D13): makespan T = number of resolved slots when the
last gate completes = number of ADVANCE actions; reward_sum = -J exactly on
non-truncated episodes (Σ ADVANCE = -alpha*T, Schedule(remote) = -beta*Σw,
no waste here).
"""

import pytest

from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import ADVANCE, EagerEnv, GenEPR, Map, Schedule


@pytest.fixture()
def golden_hw(hardware_dir):
    return load_hardware_config(hardware_dir / "golden_k2_det.yaml")


def run_script(env, script):
    """Apply a scripted action sequence; return (rewards, infos)."""
    rewards, infos = [], []
    done = False
    for action in script:
        assert not done, "script continues past episode end"
        obs, r, done, info = env.step(action)
        rewards.append(r)
        infos.append(info)
    return rewards, done, infos[-1]


def test_golden_micro_1(golden_hw, circuits_dir):
    """Golden micro 1: 3 qubits, 3 gates, serial chain, ONE remote gate.

    Circuit (golden_micro_1.yaml): g0=(q0,q1), g1=(q1,q2), g2=(q0,q1)
    DAG (per-qubit serialization): g0 -> g1 (q1); g1 -> g2 (q1); g0 -> g2 (q0)
    Placement (scripted): q0,q1 -> u0 (fills kappa=2); q2 -> u1
      => g0 local(u0), g1 REMOTE (q1@u0, q2@u1, route=[l0]), g2 local(u0)

    Hand-derived timeline (t_ep=2, d_loc=1, d_rem=2):
      slot 0: Map(q0,u0) Map(q1,u0) Map(q2,u1); Schedule(g0) [local, d=1];
              GenEPR(l0) [channel 0 busy, countdown 2]; ADVANCE
              resolve 0: countdown 2->1 (no pair yet); g0 completes
                         -> g1 ready (ready_slot=1); t=1
      slot 1: g1 ready but l0 has NO stored pair (pair lands at resolve of
              slot 1) -> Schedule(g1) must be INVALID here; ADVANCE
              resolve 1: countdown 1->0 -> pair into buffer (age 0);
                         aging -> age 1; t=2
      slot 2: Schedule(g1): consumes the pair on l0 (reward -beta*w = -1),
              C_comm=1, runs d_rem=2 (slots 2,3); stall = 2-1 = 1; ADVANCE
              resolve 2: g1 remaining 2->1; t=3
      slot 3: ADVANCE; resolve 3: g1 completes -> g2 ready (ready_slot=4); t=4
      slot 4: Schedule(g2) [local, d=1]; ADVANCE
              resolve 4: g2 completes -> ALL DONE; t=5

    Totals: T = 5 slots; C_comm = 1.0; C_waste = 0.0
            J = 1*5 + 1*1 + 0.5*0 = 6.0
            reward_sum = 5*(-1) [ADVANCE] + (-1) [Schedule g1] = -6.0 = -J
            pairs: generated=1, consumed=1, expired=0, stored=0
            EPR utilization = 1/(1+0) = 1.0; mean remote stall = 1.0
    """
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    env = EagerEnv(golden_hw, inst)
    env.reset(0)

    # slot 0
    for a in (Map(0, 0), Map(1, 0), Map(2, 1)):
        _, r, _, _ = env.step(a)
        assert r == 0.0
    _, r, _, _ = env.step(Schedule(0))
    assert r == 0.0, "local gate costs nothing at schedule"
    env.step(GenEPR(0))
    _, r, done, _ = env.step(ADVANCE)
    assert r == -1.0 and not done

    # slot 1: the remote gate must NOT be schedulable yet (pair in flight)
    assert env.gates[1].remote is True and env.gates[1].route == (0,)
    assert not env.is_valid(Schedule(1))
    env.step(ADVANCE)

    # slot 2: pair now stored (age 1); schedule the remote gate
    assert env.links[0].stored_ages == [1]
    _, r, _, _ = env.step(Schedule(1))
    assert r == -1.0, "remote schedule consumes the pair: -beta*w"
    env.step(ADVANCE)

    # slot 3: g1 still running
    assert not env.is_valid(Schedule(2)), "g2 predecessor g1 not done"
    env.step(ADVANCE)

    # slot 4: finish
    env.step(Schedule(2))
    _, r, done, info = env.step(ADVANCE)
    assert done and not info["truncated"]

    m = info["metrics"]
    assert m["T"] == 5
    assert m["C_comm"] == 1.0
    assert m["C_waste"] == 0.0
    assert m["J"] == 6.0
    assert m["reward_sum"] == -6.0
    assert m["pairs"] == {"generated": 1, "consumed": 1, "expired": 0, "stored": 0}
    assert m["epr_utilization"] == 1.0
    assert m["mean_remote_stall"] == 1.0


def test_golden_micro_2(golden_hw, circuits_dir):
    """Golden micro 2: 4 qubits, 4 gates, TWO parallel remote gates sharing
    one link (exercises W=2 parallel channels + FIFO consumption), then two
    parallel local gates on different QPUs.

    Circuit (golden_micro_2.yaml): g0=(q0,q2), g1=(q1,q3), g2=(q0,q1), g3=(q2,q3)
    DAG: g2 <- {g0 (q0), g1 (q1)}; g3 <- {g0 (q2), g1 (q3)}
    Placement (scripted): q0,q1 -> u0; q2,q3 -> u1
      => g0 REMOTE, g1 REMOTE (both route=[l0]); g2 local(u0), g3 local(u1)

    Hand-derived timeline:
      slot 0: 4 Maps; GenEPR(l0) twice (channels 0 and 1 busy, countdown 2;
              validity: stored+busy=0+1 < B=4, then 1+1 < 4); ADVANCE
              resolve 0: both countdowns 2->1; t=1
      slot 1: nothing schedulable (no pairs yet); ADVANCE
              resolve 1: both channels deliver -> 2 pairs (age 0) -> age 1; t=2
      slot 2: Schedule(g0): consumes oldest pair (-1); Schedule(g1): consumes
              the other (-1); both run d_rem=2 (slots 2,3); stalls = 2-0 = 2
              (both were DAG-ready at reset, ready_slot=0); ADVANCE
              resolve 2: g0,g1 remaining 2->1; t=3
      slot 3: ADVANCE; resolve 3: g0,g1 complete -> g2,g3 ready (slot 4); t=4
      slot 4: Schedule(g2) [local u0], Schedule(g3) [local u1] — disjoint
              qubits, run concurrently, d_loc=1; ADVANCE
              resolve 4: both complete -> ALL DONE; t=5

    Totals: T = 5; C_comm = 2.0; C_waste = 0.0
            J = 1*5 + 1*2 + 0.5*0 = 7.0
            reward_sum = 5*(-1) + 2*(-1) = -7.0 = -J
            pairs: generated=2, consumed=2, expired=0, stored=0
            utilization = 1.0; mean remote stall = (2+2)/2 = 2.0
    """
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_2.yaml"))
    env = EagerEnv(golden_hw, inst)
    env.reset(0)

    # slot 0
    for a in (Map(0, 0), Map(1, 0), Map(2, 1), Map(3, 1)):
        env.step(a)
    env.step(GenEPR(0))
    assert env.links[0].busy_channels == 1
    env.step(GenEPR(0))
    assert env.links[0].busy_channels == 2
    assert not env.is_valid(Schedule(0)), "remote, no stored pair yet"
    env.step(ADVANCE)

    # slot 1
    assert env.links[0].stored == 0
    env.step(ADVANCE)

    # slot 2: two pairs available
    assert env.links[0].stored_ages == [1, 1]
    _, r0, _, _ = env.step(Schedule(0))
    _, r1, _, _ = env.step(Schedule(1))
    assert r0 == -1.0 and r1 == -1.0
    assert env.links[0].stored == 0
    env.step(ADVANCE)

    # slot 3
    env.step(ADVANCE)

    # slot 4: both local gates in parallel (different QPUs, disjoint qubits)
    assert env.is_valid(Schedule(2)) and env.is_valid(Schedule(3))
    env.step(Schedule(2))
    env.step(Schedule(3))
    _, _, done, info = env.step(ADVANCE)
    assert done and not info["truncated"]

    m = info["metrics"]
    assert m["T"] == 5
    assert m["C_comm"] == 2.0
    assert m["C_waste"] == 0.0
    assert m["J"] == 7.0
    assert m["reward_sum"] == -7.0
    assert m["pairs"] == {"generated": 2, "consumed": 2, "expired": 0, "stored": 0}
    assert m["epr_utilization"] == 1.0
    assert m["mean_remote_stall"] == 2.0
