"""Phase 2 acceptance: trace record/replay — replay must reproduce the
recorded trajectory hash exactly; tampering must be detected; traces write
only to pytest tmp dirs."""

import pytest

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.random_prog import RandomProgressivePolicy
from eager.baselines.traces import (
    load_traces,
    record_episode,
    replay_episode,
    save_traces,
)
from eager.circuit import build_instance
from eager.config import SynthParams, load_circuit_config, load_hardware_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware
from eager.expgen.synthetic import generate_instance

pytestmark = pytest.mark.stochastic


def make_env():
    hw = default_panel_hardware(num_qubits=10)
    inst = generate_instance(SynthParams(10, 30, None), seed=6)
    return EagerEnv(hw, inst)


def test_greedy_trace_replays_identically(tmp_path):
    env = make_env()
    traces = [record_episode(env, GreedyJITPolicy(placement_seed=0), seed=s)
              for s in (0, 1, 2)]
    path = tmp_path / "greedy.jsonl"
    save_traces(traces, path)
    for trace in load_traces(path):
        verdict = replay_episode(env, trace)
        assert verdict["match"] and verdict["done"]
        assert verdict["J"] == trace["metrics"]["J"]


def test_random_trace_replays_identically(tmp_path):
    env = make_env()
    trace = record_episode(env, RandomProgressivePolicy(policy_seed=9001), seed=4)
    path = tmp_path / "random.jsonl"
    save_traces([trace], path)
    verdict = replay_episode(env, load_traces(path)[0])
    assert verdict["match"]


def test_tampered_trace_detected():
    env = make_env()
    trace = record_episode(env, GreedyJITPolicy(placement_seed=0), seed=0)
    tampered = dict(trace)
    tampered["actions"] = list(trace["actions"])
    tampered["actions"][len(tampered["actions"]) // 2] = (
        env.action_space.size - 1)              # swap a mid-episode action
    try:
        verdict = replay_episode(env, tampered)
        assert not verdict["match"]
    except (ValueError, RuntimeError):
        pass                                    # invalid action / early end


def test_wrong_binding_rejected(circuits_dir):
    env = make_env()
    trace = record_episode(env, GreedyJITPolicy(placement_seed=0), seed=0)
    other = build_instance(load_circuit_config(circuits_dir / "golden_micro_1.yaml"))
    other_env = EagerEnv(
        load_hardware_config({"name": "other", "qpus": 2, "topology": "line",
                              "kappa": 4, "mode": "stochastic",
                              "link_defaults": {"p": 0.5, "W": 2, "B": 4,
                                                "T_cut": 10, "w": 1.0}}),
        other)
    with pytest.raises(ValueError, match="recorded on"):
        replay_episode(other_env, trace)


def test_trace_records_expert_vocabulary():
    """BC requirement (§8.1): trace actions are indices into the SAME action
    space the agent will use."""
    env = make_env()
    trace = record_episode(env, GreedyJITPolicy(placement_seed=0), seed=0)
    space = env.action_space
    assert all(0 <= a < space.size for a in trace["actions"])
    assert trace["actions"][-1] == space.size - 1, "episode ends on ADVANCE"
