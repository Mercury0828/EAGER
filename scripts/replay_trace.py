#!/usr/bin/env python
"""Replay traces recorded by scripts/record_traces.py and verify each
reproduces its trajectory hash exactly (Phase 2 acceptance: replay =
identical trajectory).

Usage (from the repo root):
    python scripts/replay_trace.py --trace artifacts/traces/foo.jsonl \
        --hardware configs/hardware/k2_line.yaml \
        --circuit configs/circuits/golden_micro_1.yaml
"""

from __future__ import annotations

import argparse
import sys

from eager.baselines.traces import load_traces, replay_episode
from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import EagerEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--circuit", required=True)
    args = parser.parse_args(argv)

    hw = load_hardware_config(args.hardware)
    traces = load_traces(args.trace)
    inst = build_instance(load_circuit_config(args.circuit),
                          seed=traces[0]["env_seed"])
    env = EagerEnv(hw, inst)

    failures = 0
    for i, trace in enumerate(traces):
        verdict = replay_episode(env, trace)
        status = "OK" if verdict["match"] else "MISMATCH"
        print(f"trace {i} (env_seed={trace['env_seed']}, "
              f"policy={trace['policy']}): replay {status}")
        failures += 0 if verdict["match"] else 1

    print(f"replayed {len(traces)} traces, {failures} mismatches")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
