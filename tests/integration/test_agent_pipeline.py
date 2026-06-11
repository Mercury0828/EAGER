"""Phase 5: IL/PPO pipeline smokes — tiny-budget runs proving the full
data->train->act loop works end to end on CPU (full-budget acceptance runs
live in scripts/, evidence in PHASE_STATUS)."""

import numpy as np
import pytest
import torch

from eager.env import EagerEnv
from eager.model.policy import EagerPolicy, act_greedy
from eager.train.distribution import held_out_cases, sample_case
from eager.train.evaluate import paired_eval
from eager.train.il import collect_expert_dataset, split_episodes, train_il
from eager.train.ppo import PPOConfig, train_ppo

DEVICE = torch.device("cpu")


def test_il_overfit_tiny():
    """BC must be able to overfit a small expert set (learnability of the
    architecture + losses); also exercises dataset collection."""
    episodes, stats = collect_expert_dataset(min_transitions=400, seed=5,
                                             log_every=10_000)
    assert stats["transitions"] >= 400
    train_data, val_data = split_episodes(episodes, val_frac=0.2, seed=6)
    policy = EagerPolicy(hidden=64)
    result = train_il(policy, train_data[:300], train_data[:300], DEVICE,
                      max_epochs=15, batch_size=64, lr=1e-3, patience=15,
                      seed=0, log=lambda *_: None)
    assert result["best_val_top1"] > 0.85, result


def test_agent_episode_and_paired_eval_smoke():
    rng = np.random.default_rng(2)
    case = sample_case(rng)
    policy = EagerPolicy(hidden=32)
    env = EagerEnv(case.hardware, case.instance)
    # untrained agent still completes (T_budget bounds it; ADVANCE included)
    from eager.train.evaluate import run_agent_episode
    m = run_agent_episode(policy, env, env_seed=0, device=DEVICE)
    assert m["T"] > 0
    ev = paired_eval(policy, held_out_cases(2, seed=9), [0], DEVICE)
    assert ev["n_pairs"] == 2
    assert 0.0 <= ev["wilcoxon_p_less"] <= 1.0


def test_ppo_smoke_two_iters():
    torch.manual_seed(0)
    policy = EagerPolicy(hidden=32)
    cfg = PPOConfig(n_envs=4, rollout_steps=24, total_iters=2, minibatch=48,
                    update_epochs=2)
    result = train_ppo(policy, cfg, DEVICE, seed=3, log=lambda *_: None)
    assert len(result["history"]) == 2
    for row in result["history"]:
        assert np.isfinite(row["pi_loss"]) and np.isfinite(row["v_loss"])
        assert np.isfinite(row["approx_kl"])
