"""Phase 1A: same (config, seed, action sequence) -> identical trajectory
hash, in-process (the cross-process variant lives in
tests/integration/test_determinism_process.py)."""

from util_invariants import random_policy_step

import numpy as np

from eager.circuit import instance_from_gates
from eager.config import load_hardware_config
from eager.env import EagerEnv
from eager.utils.hashing import TrajectoryHasher
from eager.utils.scripted_policies import simple_jit_policy


def det_hw():
    return load_hardware_config(
        {"name": "k3det", "qpus": 3, "topology": "line", "kappa": [2, 2, 2],
         "mode": "deterministic", "t_ep": 2,
         "link_defaults": {"p": 1.0, "W": 1, "B": 2, "T_cut": None, "w": 1.0}})


def inst():
    return instance_from_gates(
        "mix", 5, ((0, 1), (2, 3), (1, 2), (3, 4), (0, 1), (2, 4)))


def run_hash(policy_factory, seed):
    env = EagerEnv(det_hw(), inst())
    hasher = TrajectoryHasher()
    obs = env.reset(seed)
    hasher.update_reset(obs)
    policy = policy_factory()
    done = False
    while not done:
        action = policy(env)
        obs, r, done, _ = env.step(action)
        hasher.update(action, obs, r, done)
    return hasher.hexdigest()


def test_jit_policy_trajectory_hash_stable():
    h1 = run_hash(lambda: simple_jit_policy, seed=42)
    h2 = run_hash(lambda: simple_jit_policy, seed=42)
    assert h1 == h2


def test_seeded_random_policy_trajectory_hash_stable():
    def factory():
        rng = np.random.default_rng(7)
        return lambda env: random_policy_step(env, rng)
    h1 = run_hash(factory, seed=42)
    h2 = run_hash(factory, seed=42)
    assert h1 == h2


def test_different_action_sequences_differ():
    def factory():
        rng = np.random.default_rng(8)
        return lambda env: random_policy_step(env, rng)
    assert run_hash(factory, seed=42) != run_hash(lambda: simple_jit_policy, 42)
