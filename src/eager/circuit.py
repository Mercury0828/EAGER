"""Circuit instances: gate lists with a derived dependency DAG.

Semantics (guide §4.1): the circuit DAG ``G_C`` contains two-qubit gates only.
A gate list (explicit YAML or synthetic) is interpreted as a temporal order;
the DAG is derived by **per-qubit serialization**: each gate depends on the
immediately preceding gate that touches each of its operands. This enforces a
per-qubit total order by construction, as the guide requires of the instance
generator.

Criticality (used for features and provisioning priority, guide §6.2/§9.1) is
the longest path to a sink in *gate counts* (each gate counts 1, itself
included). Actual durations (local=1, remote=2 slots) depend on the mapping
and are therefore not part of this static quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import CircuitConfig, ConfigError


@dataclass(frozen=True)
class CircuitInstance:
    name: str
    num_qubits: int
    gates: tuple[tuple[int, int], ...]          # gate id -> (q_a, q_b)
    preds: tuple[tuple[int, ...], ...]          # immediate predecessor gate ids
    succs: tuple[tuple[int, ...], ...]          # immediate successor gate ids
    criticality: tuple[int, ...]                # longest path to sink, in gates
    depth: int                                  # max criticality

    @property
    def num_gates(self) -> int:
        return len(self.gates)

    @property
    def dag_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple((i, j) for i in range(self.num_gates) for j in self.succs[i])

    def summary(self) -> str:
        lines = [
            f"circuit '{self.name}': N={self.num_qubits} qubits, "
            f"M={self.num_gates} two-qubit gates, depth={self.depth}",
        ]
        for g, (a, b) in enumerate(self.gates):
            lines.append(
                f"  g{g}: ({a},{b}) preds={list(self.preds[g])} "
                f"crit={self.criticality[g]}"
            )
        return "\n".join(lines)


def instance_from_gates(name: str, num_qubits: int,
                        gates: tuple[tuple[int, int], ...]) -> CircuitInstance:
    """Build an instance from an ordered gate list (per-qubit serialization)."""
    m = len(gates)
    for i, (a, b) in enumerate(gates):
        if a == b or not (0 <= a < num_qubits and 0 <= b < num_qubits):
            raise ConfigError(
                f"circuit '{name}': gates[{i}] = ({a},{b}) is not a valid "
                f"two-qubit gate over 0..{num_qubits - 1}")

    last_toucher: dict[int, int] = {}
    preds: list[tuple[int, ...]] = []
    succ_sets: list[set[int]] = [set() for _ in range(m)]
    for g, (a, b) in enumerate(gates):
        ps = sorted({last_toucher[q] for q in (a, b) if q in last_toucher})
        preds.append(tuple(ps))
        for p in ps:
            succ_sets[p].add(g)
        last_toucher[a] = g
        last_toucher[b] = g

    succs = tuple(tuple(sorted(s)) for s in succ_sets)

    # Dependencies only point from earlier to later list positions, so the
    # list order is already topological; sweep in reverse for longest paths.
    crit = [1] * m
    for g in range(m - 1, -1, -1):
        if succs[g]:
            crit[g] = 1 + max(crit[s] for s in succs[g])

    return CircuitInstance(
        name=name, num_qubits=num_qubits, gates=tuple(gates),
        preds=tuple(preds), succs=succs,
        criticality=tuple(crit), depth=max(crit) if crit else 0,
    )


def build_instance(cfg: CircuitConfig, seed: int | None = None) -> CircuitInstance:
    """Materialize a CircuitConfig into a CircuitInstance.

    For synthetic configs, ``seed`` overrides ``params.seed``; one of the two
    must be present.
    """
    if cfg.kind == "explicit":
        return instance_from_gates(cfg.name, cfg.num_qubits, cfg.gates)
    from .expgen.synthetic import generate_instance
    used = seed if seed is not None else cfg.params.seed
    if used is None:
        raise ConfigError(
            f"circuit config '{cfg.name}': synthetic instance needs a seed "
            "(set params.seed or pass build_instance(cfg, seed=...))")
    return generate_instance(cfg.params, seed=used, name=cfg.name)
