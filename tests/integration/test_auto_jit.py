"""Phase 1B acceptance: auto_jit smoke (guide §9.7, D21).

With auto_jit=ON, a scripted Map+Schedule-only policy (it NEVER issues
GenEPR) must complete a small stochastic instance containing remote gates
without truncation. Negative control: the same setup with auto_jit=OFF
truncates — proving the env-level JIT routine is what provisions the pairs.
"""

import pytest
from util_invariants import run_checked_episode

from eager.config import SynthParams, load_hardware_config
from eager.env import EagerEnv, EnvParams
from eager.expgen.synthetic import generate_instance
from eager.utils.scripted_policies import map_schedule_only_policy

pytestmark = pytest.mark.stochastic


def k3_hw():
    return load_hardware_config(
        {"name": "k3aj", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": "stochastic",
         "link_defaults": {"p": 0.3, "W": 2, "B": 4, "T_cut": 10, "w": 1.0}})


def instance():
    return generate_instance(SynthParams(num_qubits=6, num_gates=12, seed=None),
                             seed=3)


def has_remote_gates(env):
    return any(g.remote for g in env.gates)


def test_auto_jit_completes_map_schedule_only_policy():
    env = EagerEnv(k3_hw(), instance(), EnvParams(auto_jit=True))
    m = run_checked_episode(env, map_schedule_only_policy, seed=11)
    assert has_remote_gates(env), "instance must actually exercise remote gates"
    assert m["done"] and not m["truncated"]
    assert m["pairs"]["consumed"] > 0, "remote gates consumed env-provisioned pairs"


def test_without_auto_jit_same_policy_truncates():
    env = EagerEnv(k3_hw(), instance(), EnvParams(auto_jit=False))
    m = run_checked_episode(env, map_schedule_only_policy, seed=11)
    assert has_remote_gates(env)
    assert m["truncated"], (
        "without env-level JIT the GenEPR-less policy must starve")
    assert m["pairs"]["generated"] == 0


def test_auto_jit_respects_channel_and_buffer_limits():
    """Invariants (incl. stored+busy <= B, W channels) hold throughout an
    auto_jit episode; run_checked_episode asserts them every micro-action."""
    hw = load_hardware_config(
        {"name": "k3tight", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": "stochastic",
         "link_defaults": {"p": 0.5, "W": 1, "B": 1, "T_cut": 5, "w": 1.0}})
    env = EagerEnv(hw, instance(), EnvParams(auto_jit=True))
    m = run_checked_episode(env, map_schedule_only_policy, seed=4)
    assert m["done"] and not m["truncated"]
