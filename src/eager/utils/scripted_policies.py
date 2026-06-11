"""Tiny scripted policies for tests, demos, and scripts/run_episode.py (D20).

These are deterministic pure functions of the env state, shared between the
test suite and the episode script so the two cannot drift. They are NOT the
Phase-2 GreedyJIT baseline (no balanced partitioning, no criticality-ordered
list scheduling) — just the minimum competent behavior needed to exercise the
simulator end to end.
"""

from __future__ import annotations

from ..env.actions import ADVANCE, Action, GenEPR, Map, Schedule
from ..env.env import EagerEnv


def _first_fit_map(env: EagerEnv) -> Action | None:
    for q in range(env.instance.num_qubits):
        if env.qubit_qpu[q] is None:
            for u in range(env.hardware.num_qpus):
                if env.kappa_res[u] > 0:
                    return Map(q, u)
    return None


def _first_valid_schedule(env: EagerEnv) -> Action | None:
    for g in range(env.instance.num_gates):
        if env.is_valid(Schedule(g)):
            return Schedule(g)
    return None


def _jit_gen(env: EagerEnv) -> Action | None:
    deficits = env.link_deficits()
    for l in range(env.hardware.num_links):
        if deficits[l] > 0 and env.is_valid(GenEPR(l)):
            return GenEPR(l)
    return None


def simple_jit_policy(env: EagerEnv) -> Action:
    """Map first-fit; schedule lowest-id valid gate; JIT-generate deficits;
    else ADVANCE."""
    for picker in (_first_fit_map, _first_valid_schedule, _jit_gen):
        action = picker(env)
        if action is not None:
            return action
    return ADVANCE


def map_schedule_only_policy(env: EagerEnv) -> Action:
    """Never issues GenEPR (relies on env auto_jit for remote gates)."""
    for picker in (_first_fit_map, _first_valid_schedule):
        action = picker(env)
        if action is not None:
            return action
    return ADVANCE


POLICIES = {
    "jit": simple_jit_policy,
    "map-schedule": map_schedule_only_policy,
}
