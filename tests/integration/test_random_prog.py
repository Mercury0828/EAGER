"""Phase 2: Random-Progressive baseline — §9.5 semantics (ADVANCE only when
forced), seeded reproducibility."""

import pytest
from util_invariants import InvariantChecker

from eager.baselines.random_prog import RandomProgressivePolicy
from eager.circuit import build_instance
from eager.config import SynthParams, load_circuit_config
from eager.env import Advance, EagerEnv
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_instance
from eager.utils.hashing import TrajectoryHasher

pytestmark = pytest.mark.stochastic


def make_env(n=8, m=16, inst_seed=2):
    hw = default_panel_hardware(num_qubits=n)
    inst = generate_instance(SynthParams(n, m, None), seed=inst_seed)
    return EagerEnv(hw, inst)


def run_with_hash(env, policy_seed, env_seed, watch_advance=False):
    policy = RandomProgressivePolicy(policy_seed=policy_seed)
    checker = InvariantChecker(env)
    hasher = TrajectoryHasher()
    obs = env.reset(env_seed)
    hasher.update_reset(obs)
    done = False
    while not done:
        if watch_advance:
            n_valid = len(env.valid_actions())
        action = policy(env)
        if watch_advance and isinstance(action, Advance):
            assert n_valid == 1, ("Random-Progressive must pick ADVANCE only "
                                  "when it is the sole valid action")
        obs, r, done, _ = env.step(action)
        hasher.update(action, obs, r, done)
        checker.after_step(action, obs, r, done)
    checker.final_dag_check()
    return hasher.hexdigest(), env.metrics()


def test_advance_only_when_forced():
    env = make_env()
    _, m = run_with_hash(env, policy_seed=5, env_seed=0, watch_advance=True)
    assert m["done"]


def test_seeded_reproducibility_and_policy_seed_sensitivity():
    h1, _ = run_with_hash(make_env(), policy_seed=5, env_seed=0)
    h2, _ = run_with_hash(make_env(), policy_seed=5, env_seed=0)
    h3, _ = run_with_hash(make_env(), policy_seed=6, env_seed=0)
    assert h1 == h2
    assert h1 != h3


def test_panel_circuit_completes(repo_root):
    cfg = load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / "adder_n4.yaml")
    inst = build_instance(cfg)
    env = EagerEnv(default_panel_hardware(inst.num_qubits), inst)
    _, m = run_with_hash(env, policy_seed=9000, env_seed=0)
    assert m["done"] and not m["truncated"]
