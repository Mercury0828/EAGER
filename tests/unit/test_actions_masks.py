"""Phase 1A: action validity rules and masks, exactly per guide §6.3."""

import pytest

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import ADVANCE, ActionSpace, EagerEnv, EnvParams, GenEPR, Map, Schedule


def det_hw(**over):
    base = {"name": "t", "qpus": 3, "topology": "line", "kappa": [1, 1, 1],
            "mode": "deterministic", "t_ep": 1,
            "link_defaults": {"p": 1.0, "W": 1, "B": 1, "T_cut": None, "w": 1.0}}
    base.update(over)
    return load_hardware_config(base)


def make_env(**hw_over):
    inst = instance_from_gates("chain3", 3, ((0, 1), (1, 2)))
    return EagerEnv(det_hw(**hw_over), inst)


def test_action_space_round_trip():
    space = ActionSpace(num_qubits=3, num_qpus=3, num_gates=2, num_links=2)
    assert space.size == 3 * 3 + 2 + 2 + 1
    for i in range(space.size):
        assert space.index_of(space.action_at(i)) == i
    assert space.action_at(space.size - 1) == ADVANCE


def test_map_validity():
    env = make_env()
    env.reset(0)
    assert env.is_valid(Map(0, 0))
    env.step(Map(0, 0))
    assert not env.is_valid(Map(0, 1)), "qubit already mapped"
    assert not env.is_valid(Map(1, 0)), "QPU 0 capacity exhausted (kappa=1)"
    assert env.is_valid(Map(1, 1))


def test_schedule_requires_preds_and_mapping_and_pairs():
    env = make_env(kappa=[2, 1, 1])
    env.reset(0)
    # g0=(q0,q1), g1=(q1,q2); g1 depends on g0 via q1
    assert not env.is_valid(Schedule(0)), "operands not mapped yet"
    env.step(Map(0, 0))
    env.step(Map(1, 0))
    assert env.is_valid(Schedule(0)), "local gate, preds done, mapped"
    assert not env.is_valid(Schedule(1)), "predecessor g0 not done"
    env.step(Schedule(0))
    env.step(Map(2, 2))                  # q1@u0, q2@u2 -> remote, route links (0,1)
    env.step(ADVANCE)                    # g0 (d_loc=1) completes
    assert env.gates[0].state == 2
    # g1 ready but no pairs anywhere
    assert not env.is_valid(Schedule(1))
    env.step(GenEPR(0))
    env.step(ADVANCE)                    # pair on link 0 only (t_ep=1)
    assert not env.is_valid(Schedule(1)), "needs a pair on EVERY route link"
    env.step(GenEPR(1))
    env.step(ADVANCE)                    # pair on link 1
    assert env.is_valid(Schedule(1))


def test_gen_epr_channel_and_buffer_overflow_safety():
    env = make_env()                     # W=1, B=1
    env.reset(0)
    assert env.is_valid(GenEPR(0))
    env.step(GenEPR(0))
    assert not env.is_valid(GenEPR(0)), "no free channel (W=1)"
    env.step(ADVANCE)                    # t_ep=1 -> pair stored
    assert env.links[0].stored == 1
    # stored(1) + busy(0) >= B(1): tasking again could overflow on success
    assert not env.is_valid(GenEPR(0))
    reason = env._invalid_reason(GenEPR(0))
    assert "buffer-overflow-unsafe" in reason


def test_in_flight_counts_toward_buffer_safety():
    env = make_env(link_defaults={"p": 1.0, "W": 2, "B": 2, "T_cut": None,
                                  "w": 1.0})
    env.reset(0)
    env.step(GenEPR(0))
    assert env.is_valid(GenEPR(0)), "stored(0)+busy(1) < B(2)"
    env.step(GenEPR(0))
    assert not env.is_valid(GenEPR(0)), "stored(0)+busy(2) reaches B(2)"


def test_advance_always_valid_and_mask_alignment():
    env = make_env()
    env.reset(0)
    acts = env.valid_actions()
    assert acts[-1] == ADVANCE
    mask = env.valid_action_mask()
    assert mask[env.action_space.size - 1], "ADVANCE bit set"
    assert mask.sum() == len(acts)
    for a in acts:
        assert mask[env.action_space.index_of(a)]


def test_invalid_step_raises():
    env = make_env()
    env.reset(0)
    with pytest.raises(ValueError, match="already mapped|capacity|no residual"):
        env.step(Map(0, 0))
        env.step(Map(0, 1))
    env2 = make_env()
    with pytest.raises(RuntimeError, match="reset"):
        env2.step(ADVANCE)
    with pytest.raises(ValueError, match="unfinished predecessor|not fully mapped"):
        env.step(Schedule(1))


def test_unmappable_instance_rejected():
    inst = instance_from_gates("big", 4, ((0, 1), (2, 3)))
    with pytest.raises(ValueError, match="unmappable"):
        EagerEnv(det_hw(), inst)        # kappa total 3 < N=4
