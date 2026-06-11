#!/usr/bin/env python
"""Record scripted-policy episode traces to a JSONL file (default under the
gitignored artifacts/ tree), and verify each trace replays to an identical
trajectory before writing.

Usage (from the repo root):
    python scripts/record_traces.py --hardware configs/hardware/k2_line.yaml \
        --circuit configs/circuits/golden_micro_1.yaml --policy greedy \
        --episodes 5 --seed0 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.random_prog import RandomProgressivePolicy
from eager.baselines.traces import record_episode, replay_episode, save_traces
from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import EagerEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--policy", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=0,
                        help="env seed of episode e is seed0 + e")
    parser.add_argument("--placement-seed", type=int, default=0)
    parser.add_argument("--out", default=None,
                        help="output JSONL (default artifacts/traces/<circuit>_<policy>.jsonl)")
    args = parser.parse_args(argv)

    hw = load_hardware_config(args.hardware)
    inst = build_instance(load_circuit_config(args.circuit), seed=args.seed0)
    env = EagerEnv(hw, inst)

    traces = []
    for e in range(args.episodes):
        seed = args.seed0 + e
        if args.policy == "greedy":
            policy = GreedyJITPolicy(placement_seed=args.placement_seed)
        else:
            policy = RandomProgressivePolicy(policy_seed=9000 + seed)
        trace = record_episode(env, policy, seed)
        verdict = replay_episode(env, trace)
        if not verdict["match"]:
            print(f"ERROR: episode seed={seed} failed replay verification",
                  file=sys.stderr)
            return 1
        m = trace["metrics"]
        print(f"episode seed={seed}: T={m['T']} J={m['J']:.6g} "
              f"truncated={m['truncated']} steps={len(trace['actions'])} replay=OK")
        traces.append(trace)

    out = Path(args.out) if args.out else (
        Path("artifacts") / "traces" / f"{inst.name}_{args.policy}.jsonl")
    save_traces(traces, out)
    print(f"wrote {len(traces)} traces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
