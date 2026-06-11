"""Smoke entry point: parse a hardware + circuit config and print a summary.

Usage (from the repo root):
    python -m eager.smoke
    python -m eager.smoke --hardware configs/hardware/k4_grid.yaml \
                          --circuit configs/circuits/golden_micro_2.yaml --seed 0
"""

from __future__ import annotations

import argparse
import sys

from .circuit import build_instance
from .config import ConfigError, load_circuit_config, load_hardware_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eager.smoke",
                                     description=__doc__)
    parser.add_argument("--hardware", default="configs/hardware/k2_line.yaml")
    parser.add_argument("--circuit", default="configs/circuits/golden_micro_1.yaml")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for synthetic circuit configs")
    args = parser.parse_args(argv)

    try:
        hw = load_hardware_config(args.hardware)
        circ_cfg = load_circuit_config(args.circuit)
        inst = build_instance(circ_cfg, seed=args.seed)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 1

    print(hw.summary())
    print(inst.summary())
    n, m = inst.num_qubits, inst.num_gates
    print(f"derived: T_budget=20*(M+N)+200={20 * (m + n) + 200} slots "
          f"(guide D9); total capacity={sum(hw.kappa)} for N={n} qubits")
    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
