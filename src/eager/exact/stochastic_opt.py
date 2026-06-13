"""Clairvoyant (perfect-information) optimum for TINY stochastic instances
(guide §10.6 T4, stochastic extension — D84).

The deterministic MILP/brute-force (eager.exact.milp / brute_force) neutralizes
the learned lever: at p=1 provisioning is trivial, so a deterministic optimal
gap cannot measure proactive provisioning (D82). This module gives a
STOCHASTIC optimum anchor instead.

Key fact: the env is CRN-driven (guide §6.5), so for a FIXED seed every
generation outcome is predetermined — the seeded env is deterministic. The
minimum J achievable on that seeded env (found by exhaustive branch-and-bound
over the micro-action tree) is therefore the CLAIRVOYANT optimum for that
scenario: an agent that knew the future. Averaged over CRN seeds it is the
expected-value-of-perfect-information bound, which is a rigorous LOWER BOUND on
the expected cost of ANY non-anticipative policy (including the true MDP
optimum):  E[clairvoyant] <= E[optimal policy] <= E[EAGER].  So EAGER's gap to
this bound over-states (never under-states) its distance to the achievable
stochastic optimum — a conservative, honest anchor.

Branch-and-bound:
- incumbent pre-seeded by the best heuristic on the same seed (tight bound);
- admissible lower bound from a partial state:
  alpha*(t + unscheduled_critical_depth * d_loc) + beta*c_comm + gamma*c_waste
  (every term is monotonic non-decreasing along any continuation, and each
  unscheduled critical-path gate needs >= d_loc more time, chained);
- the search MUST complete (no node-cap truncation) for the returned value to
  be the proven optimum — callers assert completion and keep instances tiny.
"""

from __future__ import annotations

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from ..env.actions import Advance, GenEPR, Map, Schedule
from ..env.env import EagerEnv, EnvParams
from ..env.state import UNSCHEDULED


class NodeCapExceeded(RuntimeError):
    """Raised when the B&B tree exceeds the node cap before completing — the
    returned value would NOT be a proven optimum, so callers must enlarge the
    cap or shrink the instance rather than trust a truncated search."""


def _unscheduled_depth(env: EagerEnv) -> int:
    """Max criticality (longest gate-chain) over UNSCHEDULED gates — an
    admissible count of chained gates still to run."""
    crit = env.instance.criticality
    d = 0
    for g in range(env.instance.num_gates):
        if env.gates[g].state == UNSCHEDULED and crit[g] > d:
            d = crit[g]
    return d


def _lower_bound(env: EagerEnv, p: EnvParams) -> float:
    return (p.alpha * (env.t + _unscheduled_depth(env) * p.d_loc)
            + p.beta * env.c_comm + p.gamma * env.c_waste)


# Canonical within-slot action key (Map -> Schedule -> GenEPR), used for
# SYMMETRY BREAKING: all non-ADVANCE actions issued between two ADVANCEs commute
# (a pair tasked this slot is not ready until +t_ep, so it cannot enable a
# same-slot Schedule; Map precedes Schedule so dependencies still flow), hence
# the resulting next-slot state is order-independent. Forcing issuance in
# increasing key order eliminates the factorial of within-slot permutations.
_RANK = {Map: 0, Schedule: 1, GenEPR: 2, Advance: 3}


def _key(a) -> tuple:
    if isinstance(a, Map):
        return (0, a.qubit, a.qpu)
    if isinstance(a, Schedule):
        return (1, a.gate, 0)
    if isinstance(a, GenEPR):
        return (2, a.link, 0)
    return (3, 0, 0)                                # Advance


def _greedy_incumbent(hardware, instance, seed, params) -> float:
    """A cheap completing rollout (take the first canonical non-ADVANCE action
    each step, else ADVANCE) to seed a FINITE incumbent. Pruning against it
    keeps the B&B out of the pathological deep-unpruned region, so a heuristic
    incumbent is always present even when the caller passes none."""
    env = EagerEnv(hardware, instance, params)
    env.reset(seed)
    done = False
    while not done:
        acts = sorted(env.valid_actions(), key=_key)
        a = next((x for x in acts if not isinstance(x, Advance)), acts[-1])
        _, _, done, info = env.step(a)
    return info["metrics"]["J"] if not env.truncated else float("inf")


def clairvoyant_optimum(hardware: HardwareConfig, instance: CircuitInstance,
                        seed: int, params: EnvParams | None = None,
                        incumbent: float = float("inf"),
                        node_cap: int = 2_000_000) -> dict:
    """Proven minimum J on the seeded (hence deterministic) env via B&B.

    Returns {J, T, C_comm, C_waste, nodes, truncated}. Raises NodeCapExceeded
    if the cap is hit before the tree is exhausted (value would be unproven).
    If no incumbent is supplied, one is seeded from a cheap greedy rollout so
    the search is always pruned (unpruned deep search can reach env edge
    states; pruning keeps it in the normally-reachable region)."""
    if hardware.deterministic:
        raise ValueError("use brute_force/milp for the deterministic case")
    params = params or EnvParams()
    if incumbent == float("inf"):
        incumbent = _greedy_incumbent(hardware, instance, seed, params) + 1e-9
    best = {"J": incumbent, "T": None, "C_comm": None, "C_waste": None,
            "truncated": True}
    nodes = 0
    # one reusable env; state is reconstructed by replaying the action prefix
    # from reset(seed). The env is CRN-deterministic given the seed, so replay
    # reproduces the exact state — robust where deepcopy is not (the env may
    # transitively reference non-copyable objects).
    env = EagerEnv(hardware, instance, params)

    def at(prefix: list) -> EagerEnv:
        env.reset(seed)
        for a in prefix:
            env.step(a)
        return env

    def dfs(prefix: list, floor: tuple) -> None:
        """floor = the minimum canonical key allowed for the NEXT non-ADVANCE
        action this slot (symmetry breaking); ADVANCE resets it to (0,0,0)."""
        nonlocal nodes, best
        e = at(prefix)
        if e.done:
            if not e.truncated:
                m = e._info()["metrics"]
                if m["J"] < best["J"]:
                    best = {"J": m["J"], "T": m["T"], "C_comm": m["C_comm"],
                            "C_waste": m["C_waste"], "truncated": False}
            return
        if _lower_bound(e, params) >= best["J"]:
            return                                  # prune: cannot beat best
        branches = []
        for a in sorted(e.valid_actions(), key=_key):
            k = _key(a)
            if isinstance(a, Advance):
                branches.append((a, (0, 0, 0)))
            elif k >= floor:
                branches.append((a, k))             # else would re-order slot
        for a, next_floor in branches:
            nodes += 1
            if nodes > node_cap:
                raise NodeCapExceeded(f"{nodes} nodes, instance too large")
            dfs(prefix + [a], next_floor)

    dfs([], (0, 0, 0))
    best["nodes"] = nodes
    if best["T"] is None:
        raise RuntimeError("no feasible completing trajectory found")
    return best
