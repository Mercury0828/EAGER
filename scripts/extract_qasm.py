#!/usr/bin/env python
"""Extract 2q-gate skeletons from qasm/qasmbench/*.qasm into explicit circuit
YAMLs under configs/circuits/qasmbench/, plus the generated supremacy-style
n120 instance (guide §10.1). Deterministic; safe to re-run (overwrites).

Usage (from the repo root):
    python scripts/extract_qasm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from eager.expgen.qasm_skeleton import instance_from_qasm
from eager.expgen.synthetic import generate_layered_random_instance

REPO = Path(__file__).resolve().parent.parent
QASM_DIR = REPO / "qasm" / "qasmbench"
OUT_DIR = REPO / "configs" / "circuits" / "qasmbench"

SUPREMACY = {"num_qubits": 120, "num_layers": 10, "seed": 2027}


def write_instance_yaml(inst, out_path: Path, source_note: str) -> None:
    lines = [
        f"# {source_note}",
        f"# N={inst.num_qubits} qubits, M={inst.num_gates} two-qubit gates, "
        f"depth={inst.depth}",
        f"name: {inst.name}",
        "kind: explicit",
        f"num_qubits: {inst.num_qubits}",
        "gates:",
    ]
    lines.extend(f"  - [{a}, {b}]" for a, b in inst.gates)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    commit = (QASM_DIR / "SOURCE_COMMIT.txt").read_text().strip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for qasm_path in sorted(QASM_DIR.glob("*.qasm")):
        inst = instance_from_qasm(qasm_path)
        note = (f"extracted 2q skeleton from QASMBench {qasm_path.name} "
                f"@ {commit[:12]} (see qasm/README.md)")
        write_instance_yaml(inst, OUT_DIR / f"{qasm_path.stem}.yaml", note)
        rows.append((inst.name, inst.num_qubits, inst.num_gates, inst.depth))

    sup = generate_layered_random_instance(**SUPREMACY)
    note = (f"constructed supremacy-style random circuit: "
            f"{SUPREMACY['num_layers']} layers of random perfect matchings, "
            f"seed={SUPREMACY['seed']} (guide §10.1; not a QASMBench file)")
    write_instance_yaml(sup, OUT_DIR / f"{sup.name}.yaml", note)
    rows.append((sup.name, sup.num_qubits, sup.num_gates, sup.depth))

    width = max(len(r[0]) for r in rows)
    print(f"{'instance':<{width}}  {'N':>4}  {'M':>6}  {'depth':>5}")
    for name, n, m, d in rows:
        print(f"{name:<{width}}  {n:>4}  {m:>6}  {d:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
