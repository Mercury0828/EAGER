"""Core simulator environment (guide §6).

Episode loop (guide §6.1): within each slot the agent emits a sequence of
micro-actions (Map / Schedule / GenEPR), each applied immediately as
deterministic bookkeeping; the slot closes on ADVANCE, after which the
environment resolves, in this exact order:

    (1) every busy generation channel attempts (stochastic: CRN draw;
        deterministic: t_ep countdown); success -> pair (age 0) into the
        buffer, channel freed
    (2) running gates advance one slot; completed gates update the ready set
    (3) stored pairs age += 1; pairs with age > T_cut are discarded -> waste
    (4) t += 1; terminate when all gates are done, truncate when t > T_budget

Reward (guide §6.4): ADVANCE costs -α (Σ over the episode = -α·T, a
potential-based shaping with Φ(s) = -α·t); Schedule of a remote gate costs
-β·Σ_route w_l at consumption; each expired pair costs -γ·w_l, attached to the
ADVANCE that resolved it; truncation costs -α·10·(#unfinished gates). There is
NO valid-action bonus. Hence reward_sum == -J exactly on non-truncated
episodes.

Timing conventions are frozen in DESIGN_DECISIONS.md D13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from .actions import ADVANCE, Action, ActionSpace, Advance, GenEPR, Map, Schedule
from .metrics import episode_metrics
from .routing import build_routing
from .state import DONE, RUNNING, UNSCHEDULED, ChannelState, GateRuntime, LinkState


@dataclass(frozen=True)
class EnvParams:
    """Objective weights and episode parameters (guide §5.1, D3, D9)."""

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5          # waste weight in J (not the RL discount)
    d_loc: int = 1
    d_rem: int = 2
    t_budget: int | None = None  # None -> 20*(M+N)+200 (D9)
    auto_jit: bool = False       # guide §9.7; default OFF
    record_draws: bool = False   # debug: log CRN draws {(l,c,t): bool}


class EagerEnv:
    """Simulator with the micro-action API: reset(seed) / step / masks / info.

    Configs are bound at construction (D14). Public read-only runtime state
    (used by scripted policies, tests, and feature builders): ``t``,
    ``qubit_qpu``, ``kappa_res``, ``gates``, ``links``, ``done``,
    ``truncated``, ``c_comm``, ``c_waste``, ``reward_sum``, ``draw_log``.

    ``step`` raises ValueError on an invalid action: agents must respect
    ``valid_action_mask()`` / ``valid_actions()``; silent penalties would hide
    bugs (D14).
    """

    def __init__(self, hardware: HardwareConfig, circuit: CircuitInstance,
                 params: EnvParams | None = None):
        self.hardware = hardware
        self.instance = circuit
        self.params = params or EnvParams()

        if circuit.num_qubits > sum(hardware.kappa):
            raise ValueError(
                f"instance '{circuit.name}' has N={circuit.num_qubits} qubits "
                f"but hardware '{hardware.name}' offers total capacity "
                f"{sum(hardware.kappa)}; unmappable")

        self.routing = build_routing(hardware)
        self.action_space = ActionSpace(
            num_qubits=circuit.num_qubits, num_qpus=hardware.num_qpus,
            num_gates=circuit.num_gates, num_links=hardware.num_links)

        # qubit -> gate ids touching it (for remoteness resolution on Map)
        self._qubit_gates: list[list[int]] = [[] for _ in range(circuit.num_qubits)]
        for g, (a, b) in enumerate(circuit.gates):
            self._qubit_gates[a].append(g)
            self._qubit_gates[b].append(g)

        m, n = circuit.num_gates, circuit.num_qubits
        self.t_budget = (self.params.t_budget if self.params.t_budget is not None
                         else 20 * (m + n) + 200)

        self._is_reset = False

    # ------------------------------------------------------------------ API

    def reset(self, seed: int):
        """Start a fresh episode. ``seed`` keys the CRN engine (guide §6.5)."""
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed
        self.t = 0
        self.done = False
        self.truncated = False
        self.c_comm = 0.0
        self.c_waste = 0.0
        self.reward_sum = 0.0
        self.done_count = 0
        self.qubit_qpu: list[int | None] = [None] * self.instance.num_qubits
        self.kappa_res: list[int] = list(self.hardware.kappa)
        self.gates: list[GateRuntime] = []
        for g in range(self.instance.num_gates):
            n_preds = len(self.instance.preds[g])
            self.gates.append(GateRuntime(
                state=UNSCHEDULED, remaining=0, n_unfinished_preds=n_preds,
                ready_slot=0 if n_preds == 0 else None))
        self.links: list[LinkState] = [
            LinkState(stored_ages=[],
                      channels=[ChannelState() for _ in range(lc.W)])
            for lc in self.hardware.links
        ]
        self.draw_log: dict[tuple[int, int, int], bool] = {}
        self._crn = None
        if not self.hardware.deterministic:
            self._crn = self._make_crn(seed)
        self._is_reset = True
        return self._obs()

    def step(self, action: Action):
        if not self._is_reset:
            raise RuntimeError("call reset(seed) before step()")
        if self.done:
            raise RuntimeError("episode finished; call reset(seed)")
        if not isinstance(action, Action):
            raise TypeError(f"not an Action: {action!r}")
        reason = self._invalid_reason(action)
        if reason is not None:
            raise ValueError(f"invalid action {action}: {reason}")

        reward = 0.0
        if isinstance(action, Map):
            self._apply_map(action.qubit, action.qpu)
        elif isinstance(action, Schedule):
            reward += self._apply_schedule(action.gate)
        elif isinstance(action, GenEPR):
            self._task_channel(action.link)
        else:  # Advance
            reward -= self.params.alpha
            if self.params.auto_jit:
                self._auto_jit_provision()
            reward += self._resolve()

        self.reward_sum += reward
        return self._obs(), reward, self.done, self._info()

    def is_valid(self, action: Action) -> bool:
        return self._invalid_reason(action) is None

    def valid_actions(self) -> list[Action]:
        """Valid micro-actions in the fixed enumeration order (D15)."""
        out: list[Action] = []
        if self._is_reset and not self.done:
            for q in range(self.instance.num_qubits):
                if self.qubit_qpu[q] is None:
                    out.extend(Map(q, u) for u in range(self.hardware.num_qpus)
                               if self.kappa_res[u] > 0)
            out.extend(Schedule(g) for g in range(self.instance.num_gates)
                       if self._invalid_reason(Schedule(g)) is None)
            out.extend(GenEPR(l) for l in range(self.hardware.num_links)
                       if self._invalid_reason(GenEPR(l)) is None)
            out.append(ADVANCE)
        return out

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.size, dtype=bool)
        for a in self.valid_actions():
            mask[self.action_space.index_of(a)] = True
        return mask

    def metrics(self) -> dict:
        return episode_metrics(self)

    def link_deficits(self) -> list[int]:
        """Per-link pair deficit of ready-but-blocked remote gates (§9.1(3))."""
        deficits, _ = self._deficit_demand()
        return deficits

    # ------------------------------------------------------------ validity

    def _invalid_reason(self, action: Action) -> str | None:
        if isinstance(action, Map):
            q, u = action.qubit, action.qpu
            if not 0 <= q < self.instance.num_qubits:
                return f"qubit {q} outside 0..{self.instance.num_qubits - 1}"
            if not 0 <= u < self.hardware.num_qpus:
                return f"QPU {u} outside 0..{self.hardware.num_qpus - 1}"
            if self.qubit_qpu[q] is not None:
                return f"qubit {q} already mapped to QPU {self.qubit_qpu[q]}"
            if self.kappa_res[u] <= 0:
                return f"QPU {u} has no residual capacity"
            return None
        if isinstance(action, Schedule):
            g = action.gate
            if not 0 <= g < self.instance.num_gates:
                return f"gate {g} outside 0..{self.instance.num_gates - 1}"
            gr = self.gates[g]
            if gr.state != UNSCHEDULED:
                return f"gate {g} already scheduled"
            if gr.n_unfinished_preds > 0:
                return (f"gate {g} has {gr.n_unfinished_preds} unfinished "
                        "predecessor(s)")
            a, b = self.instance.gates[g]
            if self.qubit_qpu[a] is None or self.qubit_qpu[b] is None:
                return f"gate {g} operands not fully mapped"
            if gr.remote:
                for l in gr.route:
                    if self.links[l].stored < 1:
                        return (f"gate {g} is remote but link {l} has no "
                                "stored pair")
            return None
        if isinstance(action, GenEPR):
            l = action.link
            if not 0 <= l < self.hardware.num_links:
                return f"link {l} outside 0..{self.hardware.num_links - 1}"
            ls = self.links[l]
            lc = self.hardware.links[l]
            if ls.free_channels < 1:
                return f"link {l} has no free generation channel"
            if ls.stored + ls.busy_channels >= lc.B:
                return (f"link {l} buffer-overflow-unsafe: stored="
                        f"{ls.stored} + busy={ls.busy_channels} >= B={lc.B}")
            return None
        if isinstance(action, Advance):
            return None
        return f"unknown action type {type(action).__name__}"

    # ---------------------------------------------------------- application

    def _apply_map(self, q: int, u: int) -> None:
        self.qubit_qpu[q] = u
        self.kappa_res[u] -= 1
        for g in self._qubit_gates[q]:
            gr = self.gates[g]
            if gr.remote is None:
                a, b = self.instance.gates[g]
                ua, ub = self.qubit_qpu[a], self.qubit_qpu[b]
                if ua is not None and ub is not None:
                    gr.remote = ua != ub
                    gr.route = self.routing.route(ua, ub)

    def _apply_schedule(self, g: int) -> float:
        reward = 0.0
        gr = self.gates[g]
        if gr.remote:
            # Consume one stored pair from EVERY link on the route,
            # simultaneously, at schedule time (guide §4.3, A6).
            for l in gr.route:
                ls = self.links[l]
                ls.stored_ages.pop(0)            # FIFO oldest-first (D16)
                ls.consumed += 1
                w = self.hardware.links[l].w
                self.c_comm += w
                reward -= self.params.beta * w
        gr.state = RUNNING
        gr.remaining = self.params.d_rem if gr.remote else self.params.d_loc
        gr.schedule_slot = self.t
        return reward

    def _task_channel(self, l: int) -> None:
        ls = self.links[l]
        for ch in ls.channels:
            if not ch.busy:
                ch.busy = True
                ch.tasked_slot = self.t
                ch.remaining = self.hardware.t_ep if self.hardware.deterministic else 0
                return
        raise AssertionError(f"no free channel on link {l} (validity bug)")

    # ------------------------------------------------------------- resolve

    def _resolve(self) -> float:
        reward = 0.0
        t = self.t

        # (1) generation attempts on busy channels
        for lid, ls in enumerate(self.links):
            p = self.hardware.links[lid].p
            for c, ch in enumerate(ls.channels):
                if not ch.busy:
                    continue
                if self.hardware.deterministic:
                    ch.remaining -= 1
                    success = ch.remaining == 0
                else:
                    success = self._draw_generation(lid, c, t, p)
                if success:
                    ls.stored_ages.append(0)
                    ls.generated += 1
                    ch.busy = False
                    ch.tasked_slot = -1
                    ch.remaining = 0

        # (2) running gates advance; completions update the ready set
        for gid, gr in enumerate(self.gates):
            if gr.state == RUNNING:
                gr.remaining -= 1
                if gr.remaining == 0:
                    gr.state = DONE
                    self.done_count += 1
                    for s in self.instance.succs[gid]:
                        sg = self.gates[s]
                        sg.n_unfinished_preds -= 1
                        if sg.n_unfinished_preds == 0:
                            sg.ready_slot = t + 1

        # (3) aging and decoherence cutoff
        for lid, ls in enumerate(self.links):
            lc = self.hardware.links[lid]
            kept: list[int] = []
            for age in ls.stored_ages:
                age += 1
                if lc.T_cut is not None and age > lc.T_cut:
                    ls.expired += 1
                    self.c_waste += lc.w
                    reward -= self.params.gamma * lc.w
                else:
                    kept.append(age)
            ls.stored_ages = kept

        # (4) advance time; terminal checks
        self.t += 1
        if self.done_count == self.instance.num_gates:
            self.done = True
        elif self.t > self.t_budget:
            self.truncated = True
            self.done = True
            unfinished = self.instance.num_gates - self.done_count
            reward -= self.params.alpha * 10.0 * unfinished
        return reward

    def _draw_generation(self, link: int, channel: int, t: int, p: float) -> bool:
        raise NotImplementedError(
            "stochastic generation arrives with the Phase 1B CRN engine; "
            "use a deterministic-mode hardware config")

    def _make_crn(self, seed: int):
        raise NotImplementedError(
            "stochastic mode arrives in Phase 1B; "
            "use a deterministic-mode hardware config")

    # ------------------------------------------------------------- auto-JIT

    def _deficit_demand(self) -> tuple[list[int], list[int | None]]:
        nl = self.hardware.num_links
        demand = [0] * nl
        max_crit: list[int | None] = [None] * nl
        for gid, gr in enumerate(self.gates):
            if gr.state != UNSCHEDULED or gr.n_unfinished_preds > 0:
                continue
            if gr.remote is not True:
                continue
            if all(self.links[l].stored >= 1 for l in gr.route):
                continue  # schedulable now -> not blocked
            crit = self.instance.criticality[gid]
            for l in gr.route:
                demand[l] += 1
                if max_crit[l] is None or crit > max_crit[l]:
                    max_crit[l] = crit
        deficits = [
            max(0, demand[l] - self.links[l].stored - self.links[l].busy_channels)
            for l in range(nl)
        ]
        return deficits, max_crit

    def _auto_jit_provision(self) -> None:
        """§9.1(3) JIT routine, applied when the agent yields the slot (D21)."""
        deficits, max_crit = self._deficit_demand()
        order = sorted((l for l in range(self.hardware.num_links) if deficits[l] > 0),
                       key=lambda l: (-(max_crit[l] or 0), l))
        for l in order:
            ls = self.links[l]
            lc = self.hardware.links[l]
            n = min(deficits[l], ls.free_channels,
                    lc.B - ls.stored - ls.busy_channels)
            for _ in range(max(0, n)):
                self._task_channel(l)

    # ------------------------------------------------------------ obs/info

    def _obs(self) -> dict:
        """Integer-only structured snapshot (graph features arrive in Phase 5)."""
        return {
            "t": self.t,
            "qubit_qpu": [-1 if u is None else u for u in self.qubit_qpu],
            "kappa_res": list(self.kappa_res),
            "gate_state": [g.state for g in self.gates],
            "gate_remaining": [g.remaining for g in self.gates],
            "gate_ready": [
                int(g.state == UNSCHEDULED and g.n_unfinished_preds == 0)
                for g in self.gates
            ],
            "links": [
                {"stored_ages": list(l.stored_ages), "busy": l.busy_channels,
                 "free": l.free_channels}
                for l in self.links
            ],
            "done_gates": self.done_count,
        }

    def _info(self) -> dict:
        generated = sum(l.generated for l in self.links)
        consumed = sum(l.consumed for l in self.links)
        expired = sum(l.expired for l in self.links)
        stored = sum(l.stored for l in self.links)
        return {
            "t": self.t,
            "truncated": self.truncated,
            "counters": {"generated": generated, "consumed": consumed,
                         "expired": expired, "stored": stored},
            "metrics": episode_metrics(self),
        }
