"""Mutable runtime state containers for the simulator (guide §4, §6)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Gate lifecycle states
UNSCHEDULED = 0
RUNNING = 1
DONE = 2


@dataclass
class ChannelState:
    """One generation channel of a link.

    A tasked (busy) channel is in generate-until-success mode (D7): it makes a
    Bernoulli(p) attempt at every slot resolve until success, then frees. In
    deterministic mode (guide §5.2) it instead counts down exactly t_ep slots.
    """

    busy: bool = False
    tasked_slot: int = -1
    remaining: int = 0          # deterministic-mode countdown; unused otherwise


@dataclass
class LinkState:
    """Buffer + channels + conservation counters for one link.

    ``stored_ages`` is FIFO: index 0 is the oldest pair (largest age).
    Consumption pops from the front (oldest-first, D16). Ages are incremented
    at resolve step (3), so at micro-action time every stored pair has age
    >= 1, and a pair generated at the resolve of slot t is consumable during
    slots t+1 .. t+T_cut (D13).
    """

    stored_ages: list[int] = field(default_factory=list)
    channels: list[ChannelState] = field(default_factory=list)
    generated: int = 0
    consumed: int = 0
    expired: int = 0

    @property
    def stored(self) -> int:
        return len(self.stored_ages)

    @property
    def busy_channels(self) -> int:
        return sum(1 for c in self.channels if c.busy)

    @property
    def free_channels(self) -> int:
        return sum(1 for c in self.channels if not c.busy)


@dataclass
class GateRuntime:
    """Per-gate runtime bookkeeping.

    ``ready_slot`` is the first slot in which the gate could be scheduled
    (all predecessors DONE): 0 for source gates, completion-resolve slot + 1
    otherwise (D13). ``remote``/``route`` are resolved once both operands are
    mapped and never change (static placement, A2).
    """

    state: int = UNSCHEDULED
    remaining: int = 0
    n_unfinished_preds: int = 0
    ready_slot: int | None = None
    schedule_slot: int | None = None
    remote: bool | None = None
    route: tuple[int, ...] | None = None
