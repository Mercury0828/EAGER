"""OPENQASM 2.0 -> two-qubit-gate skeleton extraction (guide §10.1).

Per Assumption A3 the model keeps only two-qubit interactions: single-qubit
gates, barriers, measurements, and classical registers are dropped; every
two-qubit instruction (cx, cz, cp/cu1, crz, swap, rzz, ...) contributes ONE
gate on its qubit pair (instruction-level semantics; see DESIGN_DECISIONS).
Three-qubit standard gates are expanded into their textbook two-qubit
constructions:

    ccx(a,b,c)   -> (b,c) (a,c) (b,c) (a,c) (a,b) (a,b)     [6 CNOTs]
    cswap(a,b,c) -> (b,c) + ccx(a,b,c) pairs + (b,c)        [8 gates]

In-file `gate ... { ... }` definitions are inlined recursively. Anything
unrecognized raises (nothing is silently dropped). The gate-list order
follows the program order, so per-qubit serialization (eager.circuit)
derives the dependency DAG exactly as for every other instance source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..circuit import CircuitInstance, instance_from_gates
from ..config import ConfigError

ONE_QUBIT = {
    "u1", "u2", "u3", "u", "p", "id", "x", "y", "z", "h", "s", "sdg",
    "t", "tdg", "rx", "ry", "rz", "sx", "sxdg",
}
TWO_QUBIT = {
    "cx", "cz", "cy", "ch", "swap", "cp", "cu1", "cu3", "cu", "crx", "cry",
    "crz", "rzz", "rxx", "ryy", "iswap",
}
SKIP_STMT = ("barrier", "measure", "reset", "creg")

_GATE_DEF_RE = re.compile(
    r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([\w\s,]+?)\s*\{([^}]*)\}", re.S)
_CALL_RE = re.compile(r"^(\w+)\s*(?:\(((?:[^()]|\([^()]*\))*)\))?\s*(.*)$", re.S)
_QREG_RE = re.compile(r"^qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_ARG_RE = re.compile(r"^(\w+)(?:\[\s*(\d+)\s*\])?$")


@dataclass(frozen=True)
class _GateDef:
    formals: tuple[str, ...]      # formal qubit argument names
    body: tuple[str, ...]         # ';'-separated statements


def _ccx_pairs(a: int, b: int, c: int) -> list[tuple[int, int]]:
    return [(b, c), (a, c), (b, c), (a, c), (a, b), (a, b)]


def parse_qasm_skeleton(text: str, name: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Return (num_qubits, ordered two-qubit gate list) for an OPENQASM 2.0
    program."""
    # strip comments
    lines = [ln.split("//", 1)[0] for ln in text.splitlines()]
    src = "\n".join(lines)
    if "if(" in src.replace(" ", "") or re.search(r"\bif\s*\(", src):
        raise ConfigError(f"qasm '{name}': classical conditionals (if) are "
                          "not supported by the skeleton extractor")

    # collect and remove in-file gate definitions
    defs: dict[str, _GateDef] = {}
    def grab(m: re.Match) -> str:
        gname = m.group(1)
        formals = tuple(t.strip() for t in m.group(3).split(",") if t.strip())
        body = tuple(s.strip() for s in m.group(4).split(";") if s.strip())
        defs[gname] = _GateDef(formals=formals, body=body)
        return ""
    src = _GATE_DEF_RE.sub(grab, src)

    regs: dict[str, tuple[int, int]] = {}   # name -> (offset, size)
    n_qubits = 0
    gates: list[tuple[int, int]] = []

    def resolve(token: str, env: dict[str, str] | None) -> int:
        token = token.strip()
        if env is not None and token in env:
            token = env[token]
        m = _ARG_RE.match(token)
        if not m:
            raise ConfigError(f"qasm '{name}': cannot parse qubit operand {token!r}")
        reg, idx = m.group(1), m.group(2)
        if reg not in regs:
            raise ConfigError(f"qasm '{name}': unknown register {reg!r} in {token!r}")
        if idx is None:
            raise ConfigError(f"qasm '{name}': whole-register operand {token!r} "
                              "in a multi-qubit gate is not supported")
        off, size = regs[reg]
        i = int(idx)
        if i >= size:
            raise ConfigError(f"qasm '{name}': index {token!r} outside register "
                              f"size {size}")
        return off + i

    def emit_pair(a: int, b: int, gname: str) -> None:
        if a == b:
            raise ConfigError(f"qasm '{name}': gate '{gname}' acts twice on "
                              f"the same qubit after flattening")
        gates.append((a, b))

    def handle(stmt: str, env: dict[str, str] | None) -> None:
        nonlocal n_qubits
        stmt = " ".join(stmt.split())
        if not stmt:
            return
        if stmt.startswith(("OPENQASM", "include")):
            return
        if stmt.startswith(SKIP_STMT) or "->" in stmt:
            return
        m = _QREG_RE.match(stmt)
        if m:
            if env is not None:
                raise ConfigError(f"qasm '{name}': qreg inside a gate body")
            reg, size = m.group(1), int(m.group(2))
            regs[reg] = (n_qubits, size)
            n_qubits += size
            return
        m = _CALL_RE.match(stmt)
        if not m:
            raise ConfigError(f"qasm '{name}': cannot parse statement {stmt!r}")
        gname, args_raw = m.group(1), m.group(3)
        args = [a for a in (t.strip() for t in args_raw.split(",")) if a]

        if gname in ONE_QUBIT:
            return
        if gname in TWO_QUBIT:
            if len(args) != 2:
                raise ConfigError(f"qasm '{name}': gate '{gname}' expects 2 "
                                  f"operands, got {len(args)} in {stmt!r}")
            emit_pair(resolve(args[0], env), resolve(args[1], env), gname)
            return
        if gname in ("ccx", "ccz"):
            a, b, c = (resolve(t, env) for t in args)
            for pair in _ccx_pairs(a, b, c):
                emit_pair(*pair, gname)
            return
        if gname == "cswap":
            a, b, c = (resolve(t, env) for t in args)
            emit_pair(c, b, gname)
            for pair in _ccx_pairs(a, b, c):
                emit_pair(*pair, gname)
            emit_pair(c, b, gname)
            return
        if gname in defs:
            d = defs[gname]
            if len(args) != len(d.formals):
                raise ConfigError(
                    f"qasm '{name}': call to '{gname}' with {len(args)} "
                    f"operands; definition has {len(d.formals)}")
            # bind formal names to the CALLER-resolved tokens
            inner_env = {}
            for formal, actual in zip(d.formals, args):
                tok = actual.strip()
                if env is not None and tok in env:
                    tok = env[tok]
                inner_env[formal] = tok
            for body_stmt in d.body:
                handle(body_stmt, inner_env)
            return
        raise ConfigError(f"qasm '{name}': unsupported gate '{gname}' "
                          f"(statement {stmt!r}); extend the extractor tables")

    for stmt in src.split(";"):
        handle(stmt, None)

    if n_qubits == 0:
        raise ConfigError(f"qasm '{name}': no qreg declarations found")
    if not gates:
        raise ConfigError(f"qasm '{name}': no two-qubit gates found")
    return n_qubits, tuple(gates)


def instance_from_qasm(path: str | Path, name: str | None = None) -> CircuitInstance:
    path = Path(path)
    n, gates = parse_qasm_skeleton(path.read_text(encoding="utf-8"),
                                   name or path.stem)
    return instance_from_gates(name or path.stem, n, gates)
