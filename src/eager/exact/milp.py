"""Time-indexed MILP for the deterministic special case (guide §5.2).

Exactly mirrors the env semantics in deterministic mode (D13): a channel
tasked in slot tau delivers its pair at the resolve of slot tau + t_ep - 1
(consumable from slot tau + t_ep); a gate started in slot s occupies slots
s .. s + d - 1 (d = d_loc local, d_rem remote); a remote gate consumes one
stored pair from EVERY link of the canonical route at its start slot;
makespan T = max(s_g + d_g); J = alpha*T + beta*C_comm (no waste:
deterministic mode requires T_cut = null).

Variables (linearized per the guide; products via McCormick):
  x[i,k]        qubit i on QPU k                               (binary)
  y[g,(k,l)]    x[a_g,k] * x[b_g,l] for ordered k != l         (continuous,
                exact at binary x via McCormick)
  sigma[g,t]    gate g starts at slot t                        (binary)
  w[g,l,t]      usage[g,l] * sigma[g,t]                        (continuous,
                McCormick; usage[g,l] = sum of y over QPU pairs routed
                through l, in {0,1} at integer points)
  n[l,t]        channels of link l tasked at slot t            (integer,
                0..W_l; channels are interchangeable, so the per-channel
                z[l,c,t] of the guide aggregates exactly to n[l,t] with the
                rolling-window occupancy constraint)
  T             makespan                                       (integer)

Constraints: unique mapping; capacity; precedence
s_j >= s_i + d_loc + (d_rem - d_loc) * rho_i; pair availability per (l, t):
cumulative consumption through t <= taskings through t - t_ep; buffer:
cumulative taskings - cumulative consumption <= B_l at every t (with the
env's consume-before-task micro-order, this is exactly the overflow-safe
GenEPR rule); channel window: taskings within any t_ep window <= W_l.

The horizon H is taken from the GreedyJIT makespan on the same instance
(an optimal schedule satisfies T <= T_greedy, so the restriction is lossless).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..circuit import CircuitInstance
from ..config import HardwareConfig
from ..env.actions import ADVANCE, GenEPR, Map, Schedule
from ..env.env import EagerEnv, EnvParams
from ..env.routing import build_routing


@dataclass(frozen=True)
class ExactResult:
    status: str                 # "OPTIMAL" | "TIME_LIMIT" | ...
    j_star: float
    t_makespan: int
    c_comm: float
    mip_gap: float
    runtime_s: float
    horizon: int
    placement: tuple[int, ...]
    starts: tuple[int, ...]
    remote: tuple[bool, ...]
    taskings: dict = field(hash=False, default_factory=dict)  # (l, t) -> count


def _check_deterministic(hardware: HardwareConfig) -> None:
    if not hardware.deterministic:
        raise ValueError("the exact MILP covers the deterministic special "
                         "case only (guide §5.2); set mode: deterministic")
    for i, lc in enumerate(hardware.links):
        if lc.T_cut is not None:
            raise ValueError(f"deterministic MILP assumes no cutoff; link {i} "
                             f"has T_cut={lc.T_cut} (use T_cut: null)")


def greedy_horizon(hardware: HardwareConfig, instance: CircuitInstance,
                   params: EnvParams) -> int:
    """GreedyJIT makespan on the same instance = lossless MILP horizon."""
    from ..baselines.greedy_jit import GreedyJITPolicy
    env = EagerEnv(hardware, instance, params)
    env.reset(0)
    policy = GreedyJITPolicy(placement_seed=0)
    done = False
    while not done:
        _, _, done, info = env.step(policy(env))
    m = info["metrics"]
    if m["truncated"]:
        raise RuntimeError("GreedyJIT truncated; no finite horizon available")
    return m["T"]


def solve_exact(hardware: HardwareConfig, instance: CircuitInstance,
                params: EnvParams | None = None, horizon: int | None = None,
                time_limit: float = 3600.0, mip_gap: float = 1e-6,
                log: bool = False, threads: int = 0) -> ExactResult:
    import gurobipy as gp
    from gurobipy import GRB

    params = params or EnvParams()
    _check_deterministic(hardware)
    if horizon is None:
        horizon = greedy_horizon(hardware, instance, params)

    n_q, k_n = instance.num_qubits, hardware.num_qpus
    m_g, n_l = instance.num_gates, hardware.num_links
    h = horizon
    t_ep = hardware.t_ep
    d_loc, d_rem = params.d_loc, params.d_rem
    routing = build_routing(hardware)

    pairs = [(k, l) for k in range(k_n) for l in range(k_n) if k != l]
    route_of = {p: routing.route(*p) for p in pairs}
    cost_of = {p: sum(hardware.links[e].w for e in route_of[p]) for p in pairs}

    env_pars = {"OutputFlag": 1 if log else 0, "TimeLimit": time_limit,
                "MIPGap": mip_gap, "Seed": 0, "Threads": threads}
    with gp.Env(params=env_pars) as genv, gp.Model(env=genv) as model:
        x = model.addVars(n_q, k_n, vtype=GRB.BINARY, name="x")
        y_keys = [(g, k, l) for g in range(m_g) for (k, l) in pairs]
        y = model.addVars(y_keys, lb=0.0, ub=1.0, name="y")
        sigma = model.addVars(m_g, h, vtype=GRB.BINARY, name="sigma")
        w = model.addVars(m_g, n_l, h, lb=0.0, ub=1.0, name="w")
        n_task = model.addVars(n_l, h, vtype=GRB.INTEGER, lb=0, name="n")
        for e in range(n_l):
            for t in range(h):
                n_task[e, t].UB = hardware.links[e].W
        t_mk = model.addVar(vtype=GRB.INTEGER, lb=0, ub=h, name="T")

        # mapping
        model.addConstrs((x.sum(i, "*") == 1 for i in range(n_q)), "unique")
        model.addConstrs(
            (x.sum("*", k) <= hardware.kappa[k] for k in range(k_n)), "cap")

        # McCormick: y[g,(k,l)] = x[a,k] * x[b,l]
        for g, (a, b) in enumerate(instance.gates):
            for (k, l) in pairs:
                model.addConstr(y[g, k, l] <= x[a, k])
                model.addConstr(y[g, k, l] <= x[b, l])
                model.addConstr(y[g, k, l] >= x[a, k] + x[b, l] - 1)

        rho = {g: gp.quicksum(y[g, k, l] for (k, l) in pairs)
               for g in range(m_g)}
        usage = {
            (g, e): gp.quicksum(y[g, k, l] for (k, l) in pairs
                                if e in route_of[(k, l)])
            for g in range(m_g) for e in range(n_l)
        }

        # one start per gate
        model.addConstrs((sigma.sum(g, "*") == 1 for g in range(m_g)), "start")
        s = {g: gp.quicksum(t * sigma[g, t] for t in range(h))
             for g in range(m_g)}

        # McCormick: w[g,e,t] = usage[g,e] * sigma[g,t]
        for g in range(m_g):
            for e in range(n_l):
                for t in range(h):
                    model.addConstr(w[g, e, t] <= usage[(g, e)])
                    model.addConstr(w[g, e, t] <= sigma[g, t])
                    model.addConstr(
                        w[g, e, t] >= usage[(g, e)] + sigma[g, t] - 1)

        # precedence
        for i in range(m_g):
            for j in instance.succs[i]:
                model.addConstr(
                    s[j] >= s[i] + d_loc + (d_rem - d_loc) * rho[i],
                    name=f"prec_{i}_{j}")

        # pair availability, buffer, channel window
        for e in range(n_l):
            lc = hardware.links[e]
            for t in range(h):
                consumed = gp.quicksum(w[g, e, tau] for g in range(m_g)
                                       for tau in range(t + 1))
                generated = gp.quicksum(n_task[e, tau]
                                        for tau in range(max(0, t - t_ep + 1)))
                tasked = gp.quicksum(n_task[e, tau] for tau in range(t + 1))
                model.addConstr(consumed <= generated, f"avail_{e}_{t}")
                model.addConstr(tasked - consumed <= lc.B, f"buf_{e}_{t}")
                model.addConstr(
                    gp.quicksum(n_task[e, tau]
                                for tau in range(max(0, t - t_ep + 1), t + 1))
                    <= lc.W, f"win_{e}_{t}")

        # makespan
        for g in range(m_g):
            model.addConstr(
                t_mk >= s[g] + d_loc + (d_rem - d_loc) * rho[g], f"mk_{g}")

        comm = gp.quicksum(cost_of[(k, l)] * y[g, k, l]
                           for g in range(m_g) for (k, l) in pairs)
        model.setObjective(params.alpha * t_mk + params.beta * comm,
                           GRB.MINIMIZE)
        model.optimize()

        status_name = {GRB.OPTIMAL: "OPTIMAL", GRB.TIME_LIMIT: "TIME_LIMIT",
                       GRB.INFEASIBLE: "INFEASIBLE"}.get(
                           model.Status, str(model.Status))
        if model.SolCount == 0:
            raise RuntimeError(f"no MILP solution (status {status_name})")

        placement = tuple(
            next(k for k in range(k_n) if x[i, k].X > 0.5)
            for i in range(n_q))
        starts = tuple(
            next(t for t in range(h) if sigma[g, t].X > 0.5)
            for g in range(m_g))
        remote = tuple(
            sum(y[g, k, l].X for (k, l) in pairs) > 0.5 for g in range(m_g))
        taskings = {(e, t): int(round(n_task[e, t].X))
                    for e in range(n_l) for t in range(h)
                    if n_task[e, t].X > 0.5}
        c_comm = sum(cost_of[(k, l)] * y[g, k, l].X
                     for g in range(m_g) for (k, l) in pairs)

        return ExactResult(
            status=status_name, j_star=float(model.ObjVal),
            t_makespan=int(round(t_mk.X)), c_comm=float(round(c_comm, 9)),
            mip_gap=float(model.MIPGap), runtime_s=float(model.Runtime),
            horizon=h, placement=placement, starts=starts, remote=remote,
            taskings=taskings)


def replay_exact(result: ExactResult, hardware: HardwareConfig,
                 instance: CircuitInstance,
                 params: EnvParams | None = None) -> dict:
    """Re-execute the MILP solution as env micro-actions (consume-before-
    task order within each slot) and verify the env reproduces J* exactly."""
    params = params or EnvParams()
    env = EagerEnv(hardware, instance, params)
    env.reset(0)
    for q in range(instance.num_qubits):
        env.step(Map(q, result.placement[q]))
    gates_at = {}
    for g, t in enumerate(result.starts):
        gates_at.setdefault(t, []).append(g)
    done = False
    info = {}
    for t in range(result.t_makespan):
        for g in sorted(gates_at.get(t, [])):
            env.step(Schedule(g))
        for e in range(hardware.num_links):
            for _ in range(result.taskings.get((e, t), 0)):
                env.step(GenEPR(e))
        _, _, done, info = env.step(ADVANCE)
    if not done:
        raise AssertionError("replay did not finish at the MILP makespan")
    m = info["metrics"]
    if abs(m["J"] - result.j_star) > 1e-6 or m["T"] != result.t_makespan:
        raise AssertionError(
            f"replay mismatch: env J={m['J']} T={m['T']} vs "
            f"MILP J*={result.j_star} T={result.t_makespan}")
    return m
