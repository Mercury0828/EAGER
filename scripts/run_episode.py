#!/usr/bin/env python
"""Run one scripted-policy episode end to end; print metrics + trajectory hash.

Used by the clean-state verification protocol (guide §11) and the
cross-process determinism acceptance test. Output is fully deterministic for a
given (hardware, circuit, seed, policy, auto_jit): no timestamps, no absolute
paths. Writes nothing to disk.

Example:
    python scripts/run_episode.py --hardware configs/hardware/k2_line.yaml \
        --circuit configs/circuits/golden_micro_1.yaml --seed 0 --policy jit
"""

from __future__ import annotations

import argparse
import sys

from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import EagerEnv, EnvParams
from eager.utils.hashing import TrajectoryHasher
from eager.utils.scripted_policies import POLICIES

MICRO_STEP_GUARD = 10_000_000


def fmt(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="jit")
    parser.add_argument("--auto-jit", action="store_true",
                        help="enable env-level JIT provisioning (guide §9.7)")
    parser.add_argument("--trace", action="store_true",
                        help="print every micro-action and reward")
    args = parser.parse_args(argv)

    hw = load_hardware_config(args.hardware)
    inst = build_instance(load_circuit_config(args.circuit), seed=args.seed)
    env = EagerEnv(hw, inst, EnvParams(auto_jit=args.auto_jit))
    policy = POLICIES[args.policy]

    hasher = TrajectoryHasher()
    obs = env.reset(args.seed)
    hasher.update_reset(obs)

    done = False
    micro_steps = 0
    while not done:
        action = policy(env)
        obs, reward, done, info = env.step(action)
        hasher.update(action, obs, reward, done)
        if args.trace:
            print(f"t={info['t']} {action!r} r={fmt(reward)} done={done}")
        micro_steps += 1
        if micro_steps > MICRO_STEP_GUARD:
            print("ERROR: micro-step guard tripped; aborting", file=sys.stderr)
            return 2

    m = env.metrics()
    pairs = m["pairs"]
    print(f"episode hardware={hw.name} circuit={inst.name} seed={args.seed} "
          f"policy={args.policy} mode={hw.mode} auto_jit={env.params.auto_jit}")
    print(f"T={m['T']} C_comm={fmt(m['C_comm'])} C_waste={fmt(m['C_waste'])} "
          f"J={fmt(m['J'])} truncated={m['truncated']}")
    print(f"pairs generated={pairs['generated']} consumed={pairs['consumed']} "
          f"expired={pairs['expired']} stored={pairs['stored']}")
    print(f"epr_utilization={fmt(m['epr_utilization'])} "
          f"mean_remote_stall={fmt(m['mean_remote_stall'])}")
    print(f"reward_sum={fmt(m['reward_sum'])} micro_steps={micro_steps}")
    print(f"trajectory_sha256={hasher.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
