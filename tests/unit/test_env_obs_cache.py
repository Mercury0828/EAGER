"""Phase 2 perf pass guard: the incrementally maintained obs caches and index
sets must equal a fresh recomputation from the source-of-truth gate objects
after every micro-action."""

import numpy as np
import pytest
from util_invariants import random_policy_step

from eager.config import SynthParams, load_hardware_config
from eager.env import EagerEnv
from eager.env.state import RUNNING, UNSCHEDULED
from eager.expgen.synthetic import generate_instance
from eager.utils.scripted_policies import simple_jit_policy

pytestmark = pytest.mark.stochastic


def reference_views(env):
    state = [g.state for g in env.gates]
    remaining = [g.remaining for g in env.gates]
    ready = [int(g.state == UNSCHEDULED and g.n_unfinished_preds == 0)
             for g in env.gates]
    ready_set = {g for g in range(env.instance.num_gates)
                 if env.gates[g].state == UNSCHEDULED
                 and env.gates[g].n_unfinished_preds == 0}
    running_set = {g for g in range(env.instance.num_gates)
                   if env.gates[g].state == RUNNING}
    unmapped = {q for q in range(env.instance.num_qubits)
                if env.qubit_qpu[q] is None}
    return state, remaining, ready, ready_set, running_set, unmapped


def assert_caches_consistent(env):
    state, remaining, ready, ready_set, running_set, unmapped = reference_views(env)
    assert env._c_state == state
    assert env._c_remaining == remaining
    assert env._c_ready == ready
    assert env._ready == ready_set
    assert env._running == running_set
    assert env._unmapped == unmapped


@pytest.mark.parametrize("policy_kind", ["jit", "random"])
def test_obs_caches_match_reference(policy_kind):
    hw = load_hardware_config(
        {"name": "cachechk", "qpus": 3, "topology": "line", "kappa": [3, 3, 3],
         "mode": "stochastic",
         "link_defaults": {"p": 0.3, "W": 2, "B": 3, "T_cut": 2, "w": 1.0}})
    inst = generate_instance(SynthParams(8, 24, None), seed=5)
    env = EagerEnv(hw, inst)
    rng = np.random.default_rng(1)
    policy = (simple_jit_policy if policy_kind == "jit"
              else lambda e: random_policy_step(e, rng))
    env.reset(2)
    assert_caches_consistent(env)
    done = False
    while not done:
        _, _, done, _ = env.step(policy(env))
        assert_caches_consistent(env)
