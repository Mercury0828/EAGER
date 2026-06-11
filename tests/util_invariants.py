"""Shared invariant checker + seeded random-policy driver for the test suite.

Invariants asserted after EVERY micro-action (guide §11 Phase 1A/1B):
- DAG precedence: a gate is scheduled only when all predecessors are DONE;
  final cross-check schedule_slot(j) >= schedule_slot(i) + d_i per edge.
- QPU capacity never exceeded; kappa_res bookkeeping consistent.
- Pair conservation per link, every step: generated == consumed + expired +
  stored (expired == 0 in deterministic/no-cutoff runs).
- Buffer safety: stored + busy <= B at all times; ages in [1, T_cut].
- Reward accounting: at episode end, reward_sum == -J (minus the truncation
  penalty when truncated).
"""

import math

import numpy as np

from eager.env.actions import ADVANCE, Advance, Schedule
from eager.env.state import DONE, RUNNING


class InvariantChecker:
    def __init__(self, env):
        self.env = env
        self.reward_total = 0.0

    def after_step(self, action, obs, reward, done):
        env = self.env
        hw, inst = env.hardware, env.instance

        if isinstance(action, Schedule):
            for p in inst.preds[action.gate]:
                assert env.gates[p].state == DONE, (
                    f"gate {action.gate} scheduled before predecessor {p} done")

        counts = [0] * hw.num_qpus
        for u in env.qubit_qpu:
            if u is not None:
                counts[u] += 1
        for u in range(hw.num_qpus):
            assert counts[u] <= hw.kappa[u], f"QPU {u} capacity exceeded"
            assert env.kappa_res[u] == hw.kappa[u] - counts[u], (
                f"QPU {u} kappa_res bookkeeping broken")

        for lid, ls in enumerate(env.links):
            lc = hw.links[lid]
            assert ls.generated == ls.consumed + ls.expired + ls.stored, (
                f"link {lid} conservation broken at t={env.t}: "
                f"gen={ls.generated} cons={ls.consumed} exp={ls.expired} "
                f"stored={ls.stored}")
            assert ls.stored + ls.busy_channels <= lc.B, (
                f"link {lid} buffer overflow risk: stored={ls.stored} "
                f"busy={ls.busy_channels} B={lc.B}")
            assert len(ls.channels) == lc.W
            for age in ls.stored_ages:
                assert age >= 1, "pair visible with age < 1 at micro time"
                if lc.T_cut is not None:
                    assert age <= lc.T_cut, f"zombie pair age {age} > {lc.T_cut}"

        for gr in env.gates:
            if gr.state == RUNNING:
                assert gr.remaining >= 1

        self.reward_total += reward
        if done:
            m = env.metrics()
            expected = -m["J"]
            if m["truncated"]:
                expected -= env.params.alpha * 10.0 * m["unfinished_gates"]
            assert math.isclose(self.reward_total, expected, abs_tol=1e-9), (
                f"reward accounting broken: sum={self.reward_total} "
                f"expected={expected}")

    def final_dag_check(self):
        env = self.env
        inst = env.instance
        for i in range(inst.num_gates):
            for j in inst.succs[i]:
                gi, gj = env.gates[i], env.gates[j]
                if gj.schedule_slot is not None:
                    assert gi.schedule_slot is not None
                    d_i = env.params.d_rem if gi.remote else env.params.d_loc
                    assert gj.schedule_slot >= gi.schedule_slot + d_i, (
                        f"DAG violated: gate {j} at {gj.schedule_slot} vs "
                        f"gate {i} at {gi.schedule_slot} + d={d_i}")


def random_policy_step(env, rng: np.random.Generator):
    """Uniform over valid non-ADVANCE actions with probability 0.7, else
    ADVANCE (pure-uniform stalls; this drives episodes to completion)."""
    acts = env.valid_actions()
    non_adv = [a for a in acts if not isinstance(a, Advance)]
    if non_adv and rng.random() < 0.7:
        return non_adv[int(rng.integers(len(non_adv)))]
    return ADVANCE


def run_checked_episode(env, policy, seed, max_micro_steps=2_000_000):
    """Run an episode under `policy(env) -> action`, asserting invariants
    after every micro-action. Returns final metrics."""
    checker = InvariantChecker(env)
    env.reset(seed)
    done = False
    steps = 0
    while not done:
        action = policy(env)
        obs, reward, done, info = env.step(action)
        checker.after_step(action, obs, reward, done)
        steps += 1
        assert steps <= max_micro_steps, "micro-step guard tripped"
    checker.final_dag_check()
    return env.metrics()
