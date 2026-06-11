"""Exhaustive optimum for TINY deterministic instances (guide §11 Phase 4:
linearization validated by brute-force enumeration on a <= 3-qubit toy).

Enumerates every capacity-feasible placement and every DAG-feasible start
assignment within the horizon; for each, pair logistics are checked exactly
by latest-fit tasking:

  Each consumption of link l at slot t needs a tasking at some slot
  tau <= t - t_ep, subject to the rolling-window channel capacity (at most
  W_l taskings in any t_ep window) and the buffer bound (cumulative taskings
  minus cumulative consumptions <= B_l at every slot). Assigning taskings
  latest-first (largest deadlines first, each placed at the latest feasible
  slot) is exact: for identical-duration jobs it is feasibility-optimal for
  the window constraint (exchange argument), and it minimizes cumulative
  taskings pointwise, hence is optimal for the buffer bound too.

Complexity is exponential — guard rails reject anything beyond toy scale.
"""

from __future__ import annotations

from itertools import product

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from ..env.env import EnvParams
from ..env.routing import build_routing

MAX_QUBITS = 4
MAX_GATES = 5
MAX_HORIZON = 14


def _latest_fit_feasible(consumptions: list[int], t_ep: int, w_cap: int,
                         b_cap: int, horizon: int) -> bool:
    """Exact feasibility of serving pair consumptions at the given slots."""
    deadlines = sorted((t - t_ep for t in consumptions), reverse=True)
    if deadlines and deadlines[-1] < 0:
        return False
    tasked: list[int] = []                     # chosen tasking slots

    def window_load(slot: int) -> int:
        return sum(1 for x in tasked if slot - t_ep < x <= slot)

    for d in deadlines:
        slot = d
        while slot >= 0:
            # placing at `slot` adds one tasking to every window covering it
            if all(window_load(s) < w_cap
                   for s in range(slot, min(slot + t_ep, horizon))):
                tasked.append(slot)
                break
            slot -= 1
        else:
            return False
    # buffer: cumulative taskings - cumulative consumptions <= B at every t
    for t in range(horizon):
        cum_task = sum(1 for x in tasked if x <= t)
        cum_cons = sum(1 for c in consumptions if c <= t)
        if cum_task - cum_cons > b_cap:
            return False
    return True


def brute_force_optimum(hardware: HardwareConfig, instance: CircuitInstance,
                        params: EnvParams | None = None,
                        horizon: int = MAX_HORIZON) -> dict:
    params = params or EnvParams()
    if not hardware.deterministic:
        raise ValueError("brute force covers the deterministic case only")
    if (instance.num_qubits > MAX_QUBITS or instance.num_gates > MAX_GATES
            or horizon > MAX_HORIZON):
        raise ValueError("instance beyond toy scale for exhaustive search")

    n_q, m_g = instance.num_qubits, instance.num_gates
    k_n = hardware.num_qpus
    t_ep = hardware.t_ep
    routing = build_routing(hardware)

    topo = list(range(m_g))                      # gate list order is topological
    best = {"J": float("inf")}
    n_evaluated = 0

    for placement in product(range(k_n), repeat=n_q):
        loads = [0] * k_n
        for q, u in enumerate(placement):
            loads[u] += 1
        if any(loads[k] > hardware.kappa[k] for k in range(k_n)):
            continue
        durations, gate_links, c_comm = [], [], 0.0
        for a, b in instance.gates:
            if placement[a] == placement[b]:
                durations.append(params.d_loc)
                gate_links.append(())
            else:
                route = routing.route(placement[a], placement[b])
                durations.append(params.d_rem)
                gate_links.append(route)
                c_comm += sum(hardware.links[e].w for e in route)

        starts = [0] * m_g

        def feasible_logistics() -> bool:
            for e in range(hardware.num_links):
                cons = [starts[g] for g in range(m_g) if e in gate_links[g]]
                if cons and not _latest_fit_feasible(
                        cons, t_ep, hardware.links[e].W, hardware.links[e].B,
                        horizon):
                    return False
            return True

        def rec(idx: int) -> None:
            nonlocal n_evaluated
            if idx == m_g:
                n_evaluated += 1
                if feasible_logistics():
                    t_mk = max(starts[g] + durations[g] for g in range(m_g))
                    j = params.alpha * t_mk + params.beta * c_comm
                    if j < best["J"]:
                        best.update(J=j, T=t_mk, C_comm=c_comm,
                                    placement=list(placement),
                                    starts=list(starts))
                return
            g = topo[idx]
            earliest = 0
            for p in instance.preds[g]:
                earliest = max(earliest, starts[p] + durations[p])
            for t in range(earliest, horizon - durations[g] + 1):
                starts[g] = t
                # bound: the partial makespan grows monotonically in t, so
                # once it can no longer beat the incumbent, stop this branch
                lower_t = max(starts[gg] + durations[gg]
                              for gg in topo[: idx + 1])
                if params.alpha * lower_t + params.beta * c_comm >= best["J"]:
                    break
                rec(idx + 1)
            starts[g] = 0

        rec(0)

    if best["J"] == float("inf"):
        raise RuntimeError("no feasible solution within the horizon")
    best["n_evaluated"] = n_evaluated
    return best
