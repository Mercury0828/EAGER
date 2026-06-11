"""Single metrics implementation (guide §10.5, §12).

Baselines and agents share these definitions through the env; no method may
reimplement costs. All quantities are derived from env counters:

- T            makespan: number of resolved slots when the last gate completes
               (equivalently the ADVANCE count, so Σ ADVANCE rewards = -α·T)
- C_comm       Σ w_l over consumed pairs
- C_waste      Σ w_l over expired pairs
- J            α·T + β·C_comm + γ·C_waste  (truncation penalty is reward
               shaping, not part of J; see `truncated`)
- epr_utilization      consumed / (consumed + expired); None if no pair ever
                       finished generating-and-resolving (denominator 0)
- mean_remote_stall    mean(schedule_slot - ready_slot) over scheduled remote
                       gates; None if no remote gate was scheduled
- truncated            episode hit T_budget with unfinished gates
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .state import DONE

if TYPE_CHECKING:  # pragma: no cover
    from .env import EagerEnv


def objective(alpha: float, beta: float, gamma: float,
              t_makespan: int, c_comm: float, c_waste: float) -> float:
    """J = α·T + β·C_comm + γ·C_waste (guide §5.1)."""
    return alpha * t_makespan + beta * c_comm + gamma * c_waste


def episode_metrics(env: "EagerEnv") -> dict[str, Any]:
    generated = sum(l.generated for l in env.links)
    consumed = sum(l.consumed for l in env.links)
    expired = sum(l.expired for l in env.links)
    stored = sum(l.stored for l in env.links)

    stalls = [
        g.schedule_slot - g.ready_slot
        for g in env.gates
        if g.remote and g.schedule_slot is not None
    ]
    attempts = consumed + expired
    unfinished = env.instance.num_gates - env.done_count

    return {
        "T": env.t,
        "C_comm": env.c_comm,
        "C_waste": env.c_waste,
        "J": objective(env.params.alpha, env.params.beta, env.params.gamma,
                       env.t, env.c_comm, env.c_waste),
        "epr_utilization": (consumed / attempts) if attempts > 0 else None,
        "mean_remote_stall": (sum(stalls) / len(stalls)) if stalls else None,
        "truncated": env.truncated,
        "done": all(g.state == DONE for g in env.gates),
        "unfinished_gates": unfinished,
        "pairs": {"generated": generated, "consumed": consumed,
                  "expired": expired, "stored": stored},
        "per_link": [
            {"generated": l.generated, "consumed": l.consumed,
             "expired": l.expired, "stored": l.stored}
            for l in env.links
        ],
        "reward_sum": env.reward_sum,
    }
