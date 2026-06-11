"""Phase 4 acceptance: on the golden micro-instances Gurobi reaches OPTIMAL
status with J* <= GreedyJIT's J, and the solution replays in the env to
exactly J* (guide §11).

Hand-checked optima (golden_k2_det: t_ep=2, W=2, B=4, kappa=[2,2]):

  golden_micro_1 (g0=(0,1), g1=(1,2), g2=(0,1); serial chain):
    placement {q0,q1 | q2} (cut 1); GenEPR at slot 0 -> pair available at
    slot 2; g0@0 (local), g1@2..3 (remote), g2@4 -> T=5, C=1, J*=6
    (equals the proactive golden schedule of Phase 1A; GreedyJIT gets 7).

  golden_micro_2 (g0=(0,2), g1=(1,3), g2=(0,1), g3=(2,3)):
    the OPTIMAL placement is {q0,q2 | q1,q3} — it makes the FIRST layer
    (g0,g1) local and the SECOND layer (g2,g3) remote, so generation
    latency (pairs tasked at slot 0, available at slot 2) is hidden behind
    the local layer: g0,g1@0 (local), g2,g3@2..3 (remote, consume the two
    pairs) -> T=4, C=2, J*=6. Both the Phase 1A hand schedule and GreedyJIT
    use the {q0,q1 | q2,q3} placement and pay J=7 — the optimum beats them
    by co-designing placement WITH provisioning overlap.
"""

import pytest

gp = pytest.importorskip("gurobipy")

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import load_circuit_config, load_hardware_config
from eager.env import EagerEnv, EnvParams
from eager.exact.milp import replay_exact, solve_exact


@pytest.fixture(scope="module", autouse=True)
def _license_available():
    try:
        with gp.Env(params={"OutputFlag": 0}):
            pass
    except gp.GurobiError as exc:        # pragma: no cover
        pytest.skip(f"no usable Gurobi license: {exc}")


def greedy_j(hw, inst):
    env = EagerEnv(hw, inst)
    info, _, _ = run_episode(env, GreedyJITPolicy(placement_seed=0), 0)
    m = info["metrics"]
    assert not m["truncated"]
    return m["J"]


EXPECTED = {"golden_micro_1": 6.0, "golden_micro_2": 6.0}


@pytest.mark.parametrize("stem", sorted(EXPECTED))
def test_golden_optimal_and_replay(stem, hardware_dir, circuits_dir):
    hw = load_hardware_config(hardware_dir / "golden_k2_det.yaml")
    inst = build_instance(load_circuit_config(circuits_dir / f"{stem}.yaml"))
    res = solve_exact(hw, inst, EnvParams(), time_limit=300)
    assert res.status == "OPTIMAL"
    assert res.mip_gap <= 1e-6
    assert res.j_star == pytest.approx(EXPECTED[stem], abs=1e-9)
    j_g = greedy_j(hw, inst)
    assert res.j_star <= j_g + 1e-9, (stem, res.j_star, j_g)
    m = replay_exact(res, hw, inst, EnvParams())
    assert m["J"] == pytest.approx(res.j_star, abs=1e-9)


def test_golden_micro_2_optimum_beats_greedy_strictly(hardware_dir,
                                                      circuits_dir):
    """The placement-provisioning co-design gap: J*=6 < GreedyJIT's 7."""
    hw = load_hardware_config(hardware_dir / "golden_k2_det.yaml")
    inst = build_instance(
        load_circuit_config(circuits_dir / "golden_micro_2.yaml"))
    res = solve_exact(hw, inst, EnvParams(), time_limit=300)
    assert res.t_makespan == 4 and res.c_comm == 2.0
    assert res.j_star < greedy_j(hw, inst)
