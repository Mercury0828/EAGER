"""Micro-actions and the fixed action-space enumeration (guide §6.3, D15).

Enumeration order (stable contract for masks and flat agents):
    [Map(q,u) for q in 0..N-1 for u in 0..K-1]   indices 0 .. N*K-1
    [Schedule(g) for g in 0..M-1]                indices N*K .. N*K+M-1
    [GenEPR(l) for l in 0..L-1]                  indices N*K+M .. N*K+M+L-1
    ADVANCE                                      index  N*K+M+L (last)
"""

from __future__ import annotations

from dataclasses import dataclass


class Action:
    """Base class for micro-actions."""
    __slots__ = ()


@dataclass(frozen=True)
class Map(Action):
    qubit: int
    qpu: int


@dataclass(frozen=True)
class Schedule(Action):
    gate: int


@dataclass(frozen=True)
class GenEPR(Action):
    link: int


@dataclass(frozen=True)
class Advance(Action):
    pass


ADVANCE = Advance()


@dataclass(frozen=True)
class ActionSpace:
    num_qubits: int
    num_qpus: int
    num_gates: int
    num_links: int

    @property
    def size(self) -> int:
        return (self.num_qubits * self.num_qpus + self.num_gates
                + self.num_links + 1)

    def index_of(self, action: Action) -> int:
        n, k, m = self.num_qubits, self.num_qpus, self.num_gates
        if isinstance(action, Map):
            if not (0 <= action.qubit < n and 0 <= action.qpu < k):
                raise ValueError(f"{action} outside action space {self}")
            return action.qubit * k + action.qpu
        if isinstance(action, Schedule):
            if not 0 <= action.gate < m:
                raise ValueError(f"{action} outside action space {self}")
            return n * k + action.gate
        if isinstance(action, GenEPR):
            if not 0 <= action.link < self.num_links:
                raise ValueError(f"{action} outside action space {self}")
            return n * k + m + action.link
        if isinstance(action, Advance):
            return self.size - 1
        raise TypeError(f"not an Action: {action!r}")

    def action_at(self, index: int) -> Action:
        if not 0 <= index < self.size:
            raise ValueError(f"action index {index} outside 0..{self.size - 1}")
        n, k, m = self.num_qubits, self.num_qpus, self.num_gates
        if index < n * k:
            return Map(qubit=index // k, qpu=index % k)
        index -= n * k
        if index < m:
            return Schedule(gate=index)
        index -= m
        if index < self.num_links:
            return GenEPR(link=index)
        return ADVANCE
