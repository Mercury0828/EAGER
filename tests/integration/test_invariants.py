"""Phase 1A acceptance: invariants hold after EVERY micro-action under both
scripted-JIT and seeded random policies, across hardware shapes (incl.
multi-hop K=3) — DAG precedence, capacity, pair conservation, buffer safety,
reward accounting (see tests/util_invariants.py)."""

import numpy as np
import pytest
from util_invariants import random_policy_step, run_checked_episode

from eager.circuit import build_instance, instance_from_gates
from eager.config import SynthParams, load_circuit_config, load_hardware_config
from eager.expgen.synthetic import generate_instance
from eager.utils.scripted_policies import simple_jit_policy


def det_hw(qpus, topology, kappa, t_ep, W, B, **extra):
    cfg = {"name": f"det_k{qpus}_{topology}", "qpus": qpus, "topology": topology,
           "kappa": kappa, "mode": "deterministic", "t_ep": t_ep,
           "link_defaults": {"p": 1.0, "W": W, "B": B, "T_cut": None, "w": 1.0}}
    cfg.update(extra)
    return load_hardware_config(cfg)


HW_CASES = {
    "k2_line": det_hw(2, "line", 6, 2, 2, 4),
    "k3_line": det_hw(3, "line", [2, 2, 2], 1, 1, 2),
    "k4_grid": det_hw(4, "grid", 4, 3, 2, 8, grid_dims=[2, 2]),
}


def circuits_for(hw):
    total = sum(hw.kappa)
    insts = []
    if total >= 3:
        insts.append(instance_from_gates("micro1", 3, ((0, 1), (1, 2), (0, 1))))
    if total >= 6:
        insts.append(generate_instance(SynthParams(6, 12, None), seed=3))
    if total >= 8:
        insts.append(generate_instance(SynthParams(8, 24, None), seed=5))
    return insts


@pytest.mark.parametrize("hw_name", sorted(HW_CASES))
def test_jit_policy_invariants(hw_name):
    from eager.env import EagerEnv
    hw = HW_CASES[hw_name]
    for inst in circuits_for(hw):
        env = EagerEnv(hw, inst)
        m = run_checked_episode(env, simple_jit_policy, seed=0)
        assert m["done"] and not m["truncated"], (hw_name, inst.name, m)
        assert m["pairs"]["expired"] == 0, "no expiry in deterministic/no-cutoff"


@pytest.mark.parametrize("hw_name", sorted(HW_CASES))
@pytest.mark.parametrize("policy_seed", [0, 1, 2])
def test_random_policy_invariants(hw_name, policy_seed):
    from eager.env import EagerEnv
    hw = HW_CASES[hw_name]
    for inst in circuits_for(hw):
        env = EagerEnv(hw, inst)
        rng = np.random.default_rng(policy_seed)
        m = run_checked_episode(env, lambda e: random_policy_step(e, rng), seed=0)
        assert m["done"], (hw_name, inst.name)


def stoch_hw(name, p, T_cut, W, B):
    return load_hardware_config(
        {"name": name, "qpus": 3, "topology": "line", "kappa": [3, 3, 3],
         "mode": "stochastic",
         "link_defaults": {"p": p, "W": W, "B": B, "T_cut": T_cut, "w": 1.0}})


STOCH_CASES = {
    "p30_cut2_tight": stoch_hw("s1", p=0.3, T_cut=2, W=2, B=3),
    "p083_cut20": stoch_hw("s2", p=1.0 / 12.0, T_cut=20, W=2, B=8),
    "p50_cut1_w1b1": stoch_hw("s3", p=0.5, T_cut=1, W=1, B=1),
}


@pytest.mark.stochastic
@pytest.mark.parametrize("case", sorted(STOCH_CASES))
def test_stochastic_jit_policy_invariants_with_expiry(case):
    """Phase 1B: conservation including expiry (generated == consumed +
    expired + stored) holds every slot under tight cutoffs; reward identity
    holds with waste charges."""
    from eager.env import EagerEnv
    hw = STOCH_CASES[case]
    for inst in (generate_instance(SynthParams(6, 12, None), seed=3),
                 generate_instance(SynthParams(8, 20, None), seed=5)):
        for run_seed in (0, 1):
            env = EagerEnv(hw, inst)
            m = run_checked_episode(env, simple_jit_policy, seed=run_seed)
            assert m["done"] and not m["truncated"], (case, inst.name, run_seed)
            p = m["pairs"]
            assert p["generated"] == p["consumed"] + p["expired"] + p["stored"]


@pytest.mark.stochastic
@pytest.mark.parametrize("policy_seed", [0, 1])
def test_stochastic_random_policy_invariants(policy_seed):
    from eager.env import EagerEnv
    hw = STOCH_CASES["p30_cut2_tight"]
    inst = generate_instance(SynthParams(6, 12, None), seed=3)
    env = EagerEnv(hw, inst)
    rng = np.random.default_rng(policy_seed)
    m = run_checked_episode(env, lambda e: random_policy_step(e, rng), seed=7)
    assert m["done"], (policy_seed, m)


def test_golden_configs_with_jit_policy(hardware_dir, circuits_dir):
    """The shipped golden configs complete under the scripted JIT policy with
    conservation generated == consumed + stored (expired == 0)."""
    from eager.env import EagerEnv
    hw = load_hardware_config(hardware_dir / "golden_k2_det.yaml")
    for fname in ("golden_micro_1.yaml", "golden_micro_2.yaml"):
        inst = build_instance(load_circuit_config(circuits_dir / fname))
        env = EagerEnv(hw, inst)
        m = run_checked_episode(env, simple_jit_policy, seed=0)
        assert m["done"] and not m["truncated"]
        p = m["pairs"]
        assert p["generated"] == p["consumed"] + p["stored"]
