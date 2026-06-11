"""Episode trace recording and replay (guide §11 Phase 2; feeds §8.1 IL).

A trace stores the env binding (hardware/circuit names, env seed, params),
the action sequence as ActionSpace indices (the agent's action vocabulary),
per-step rewards, final metrics, and the trajectory SHA-256. Replay re-runs
the actions on a freshly reset env and must reproduce the hash bit-for-bit
(acceptance: replay = identical trajectory). States are NOT stored: the env
is deterministic given (config, seed, actions), so replay regenerates them —
the IL dataset builder (Phase 5) derives (state, action) pairs the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..env.env import EagerEnv
from ..utils.hashing import TrajectoryHasher

TRACE_FORMAT = 1


def run_episode(env: EagerEnv, policy: Callable, seed: int,
                max_micro_steps: int = 5_000_000) -> tuple[dict, list[int], list[float]]:
    """Run one episode; return (final info, action indices, rewards)."""
    space = env.action_space
    hasher = TrajectoryHasher()
    obs = env.reset(seed)
    hasher.update_reset(obs)
    actions: list[int] = []
    rewards: list[float] = []
    done = False
    info: dict = {}
    while not done:
        action = policy(env)
        obs, r, done, info = env.step(action)
        hasher.update(action, obs, r, done)
        actions.append(space.index_of(action))
        rewards.append(float(r))
        if len(actions) > max_micro_steps:
            raise RuntimeError("micro-step guard tripped while recording")
    info = dict(info)
    info["trajectory_sha256"] = hasher.hexdigest()
    return info, actions, rewards


def record_episode(env: EagerEnv, policy: Callable, seed: int,
                   policy_name: str | None = None) -> dict:
    info, actions, rewards = run_episode(env, policy, seed)
    m = info["metrics"]
    return {
        "format": TRACE_FORMAT,
        "hardware": env.hardware.name,
        "circuit": env.instance.name,
        "env_seed": seed,
        "mode": env.hardware.mode,
        "auto_jit": env.params.auto_jit,
        "policy": policy_name or getattr(policy, "name", type(policy).__name__),
        "actions": actions,
        "rewards": rewards,
        "metrics": {
            "T": m["T"], "C_comm": m["C_comm"], "C_waste": m["C_waste"],
            "J": m["J"], "truncated": m["truncated"],
            "pairs": m["pairs"], "reward_sum": m["reward_sum"],
        },
        "trajectory_sha256": info["trajectory_sha256"],
    }


def replay_episode(env: EagerEnv, trace: dict) -> dict:
    """Re-run a trace's action sequence; return the recomputed fingerprint."""
    if trace["format"] != TRACE_FORMAT:
        raise ValueError(f"unsupported trace format {trace['format']!r}")
    if (env.hardware.name != trace["hardware"]
            or env.instance.name != trace["circuit"]):
        raise ValueError(
            f"trace was recorded on ({trace['hardware']}, {trace['circuit']}), "
            f"env is ({env.hardware.name}, {env.instance.name})")
    space = env.action_space
    hasher = TrajectoryHasher()
    obs = env.reset(trace["env_seed"])
    hasher.update_reset(obs)
    done = False
    for idx in trace["actions"]:
        if done:
            raise ValueError("trace continues past episode end")
        action = space.action_at(idx)
        obs, r, done, info = env.step(action)
        hasher.update(action, obs, r, done)
    digest = hasher.hexdigest()
    return {
        "trajectory_sha256": digest,
        "match": digest == trace["trajectory_sha256"],
        "done": done,
        "J": info["metrics"]["J"] if done else None,
    }


def save_traces(traces: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for tr in traces:
            fh.write(json.dumps(tr, sort_keys=True) + "\n")


def load_traces(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
