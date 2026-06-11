"""Phase 3: DDQN-flat implementation (guide §9.4) — featurizer shape/padding,
mask-respecting action selection, Double-DQN update mechanics, and a tiny
training smoke (full per-config training happens in Phase 6)."""

import numpy as np
import pytest

from eager.baselines.ddqn_flat import (
    DDQNConfig,
    DDQNFlatAgent,
    FlatFeaturizer,
    train_ddqn,
)
from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import EagerEnv

pytestmark = pytest.mark.stochastic


def make_env(circuits_dir):
    hw = load_hardware_config(
        {"name": "ddqn_hw", "qpus": 2, "topology": "line", "kappa": 4,
         "mode": "stochastic",
         "link_defaults": {"p": 0.5, "W": 2, "B": 4, "T_cut": 10, "w": 1.0}})
    inst = build_instance(load_circuit_config(circuits_dir / "golden_micro_2.yaml"))
    return EagerEnv(hw, inst)


def test_featurizer_shape_and_padding(circuits_dir):
    env = make_env(circuits_dir)
    feat = FlatFeaturizer(env)
    env.reset(0)
    x = feat(env)
    assert x.shape == (feat.dim,) and x.dtype == np.float32
    assert np.all(np.isfinite(x))
    # only 4 gates ready at most -> top-k block is zero-padded
    k_block = x[3 * feat.k + 4 * feat.n_links:
                3 * feat.k + 4 * feat.n_links + 4 * feat.top_k]
    assert np.count_nonzero(k_block) <= 4 * 4


def test_action_selection_respects_mask(circuits_dir):
    env = make_env(circuits_dir)
    agent = DDQNFlatAgent(env, seed=1)
    env.reset(0)
    done = False
    checked = 0
    while not done and checked < 300:
        for eps in (0.0, 1.0):          # greedy and exploratory paths
            action, idx, _ = agent.select_action(env, eps=eps)
            assert env.valid_action_mask()[idx], (eps, action)
        action, idx, _ = agent.select_action(env, eps=0.5)
        _, _, done, _ = env.step(action)
        checked += 1
    assert done, "episode should finish under masked random play"


def test_double_dqn_update_runs_and_targets_sync(circuits_dir):
    env = make_env(circuits_dir)
    cfg = DDQNConfig(train_start=32, batch_size=16, target_sync_every=8)
    history = train_ddqn(env, total_env_steps=300, config=cfg, seed=0)
    assert history["losses"], "updates must have run"
    assert all(np.isfinite(l) for l in history["losses"])
    assert history["episodes"] >= 1


def test_save_load_roundtrip(tmp_path, circuits_dir):
    env = make_env(circuits_dir)
    agent = DDQNFlatAgent(env, seed=3)
    path = tmp_path / "ddqn.pt"
    agent.save(path)
    agent2 = DDQNFlatAgent(env, seed=4)
    agent2.load(path)
    env.reset(0)
    x = agent.featurizer(env)
    import torch
    with torch.no_grad():
        q1 = agent.online(torch.from_numpy(x))
        q2 = agent2.online(torch.from_numpy(x))
    assert torch.allclose(q1, q2)
