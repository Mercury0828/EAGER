"""Phase 2: CRN-paired J ordering on a small panel subset — GreedyJIT never
truncates and beats Random-Progressive in mean J (the full 13-instance
acceptance run lives in experiments/phase2_panel.py; evidence in
PHASE_STATUS.md)."""

import pytest

from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.random_prog import RandomProgressivePolicy
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import load_circuit_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware

pytestmark = pytest.mark.stochastic

SEEDS = [0, 1, 2, 3, 4]          # matches the panel protocol (5 CRN seeds)

# Known regime exceptions at the default p=1/12 (D35/D38/D43): on
# provisioning-throughput-bound serialized circuits, Random-Progressive's
# ADVANCE-only-when-forced semantics make it an accidental always-on
# (maximally proactive) provisioner, and reactive JIT cannot hide the
# ~1/(2p)-slot generation latency per serialized remote gate. strict=True
# keeps the finding visible: if the regime flips, the xfail FAILS loudly and
# the set must be re-measured — exactly what happened when the D41
# sequential-fill tie-break landed (bv_n30 and qaoa_n6 left the set;
# ghz_fanout_n78 entered it; qft_n63 remains but is too slow for the test
# subset — the panel covers it).
REGIME_EXCEPTIONS = {"ghz_fanout_n78"}

CASES = [
    pytest.param(stem,
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason="D35/D43 regime finding: provisioning-throughput-"
                            "bound at default p=1/12; see PHASE_STATUS")
                 if stem in REGIME_EXCEPTIONS else ())
    for stem in ["adder_n4", "qaoa_n6", "bv_n30", "ghz_fanout_n78"]
]


def paired_js(stem, repo_root, p_override=None):
    cfg = load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / f"{stem}.yaml")
    inst = build_instance(cfg)
    hw = default_panel_hardware(inst.num_qubits)
    if p_override is not None:
        from eager.config import load_hardware_config
        import math
        kappa = math.ceil(1.25 * inst.num_qubits / 4)
        hw = load_hardware_config({
            "name": f"panel_p{p_override}", "qpus": 4, "topology": "grid",
            "grid_dims": [2, 2], "kappa": kappa, "mode": "stochastic",
            "link_defaults": {"p": p_override, "W": 2, "B": 8, "T_cut": 20,
                              "w": 1.0}})
    jg, jr = [], []
    for e in SEEDS:
        env = EagerEnv(hw, inst)
        info, _, _ = run_episode(env, GreedyJITPolicy(placement_seed=0), e)
        m = info["metrics"]
        assert not m["truncated"], f"GreedyJIT truncated on {stem} seed {e}"
        jg.append(m["J"])
        env = EagerEnv(hw, inst)
        info, _, _ = run_episode(
            env, RandomProgressivePolicy(policy_seed=9000 + e), e)
        jr.append(info["metrics"]["J"])
    return sum(jg) / len(SEEDS), sum(jr) / len(SEEDS)


@pytest.mark.parametrize("stem", CASES)
def test_greedy_beats_random_crn_paired(stem, repo_root):
    mean_g, mean_r = paired_js(stem, repo_root)
    assert mean_g < mean_r, (stem, mean_g, mean_r)


@pytest.mark.parametrize("stem,p", [("bv_n30", 0.3), ("qaoa_n6", 0.5)])
def test_regime_boundary_greedy_wins_at_higher_p(stem, p, repo_root):
    """Mechanism guard for D35: once generation is fast enough that JIT
    latency stops dominating, GreedyJIT must beat Random-Progressive on the
    same instances (measured crossovers: bv_n30 by p=0.2, qaoa_n6 by p=0.5)."""
    mean_g, mean_r = paired_js(stem, repo_root, p_override=p)
    assert mean_g < mean_r, (stem, p, mean_g, mean_r)
