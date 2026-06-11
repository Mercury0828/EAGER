"""Phase 3 acceptance core: AGG strictly reduces consumed pairs vs GreedyJIT
on burst-carrying circuits, CRN-paired; chain-form ghz stays burst-free with
identical consumption (the structural D40 result)."""

import pytest

from eager.baselines.agg import make_agg_method
from eager.baselines.greedy_jit import GreedyJITPolicy
from eager.baselines.traces import run_episode
from eager.circuit import build_instance
from eager.config import load_circuit_config
from eager.env import EagerEnv
from eager.expgen.hardware import default_panel_hardware

pytestmark = pytest.mark.stochastic

SEEDS = [0, 1]


def paired_consumed(stem, repo_root):
    inst = build_instance(load_circuit_config(
        repo_root / "configs" / "circuits" / "qasmbench" / f"{stem}.yaml"))
    hw = default_panel_hardware(inst.num_qubits)
    transformed, agg_policy, placement, stats = make_agg_method(inst, hw)

    consumed = {"greedy": [], "agg": []}
    j = {"greedy": [], "agg": []}
    for e in SEEDS:
        env = EagerEnv(hw, inst)
        info, _, _ = run_episode(env, GreedyJITPolicy(placement_seed=0), e)
        m = info["metrics"]
        assert not m["truncated"]
        consumed["greedy"].append(m["pairs"]["consumed"])
        j["greedy"].append(m["J"])

        env = EagerEnv(hw, transformed)
        info, _, _ = run_episode(env, agg_policy, e)
        m = info["metrics"]
        assert not m["truncated"]
        consumed["agg"].append(m["pairs"]["consumed"])
        j["agg"].append(m["J"])
    return consumed, j, stats


@pytest.mark.parametrize("stem", ["bv_n30", "ghz_fanout_n78"])
def test_agg_strictly_reduces_consumed_pairs(stem, repo_root):
    consumed, j, stats = paired_consumed(stem, repo_root)
    assert stats["gates_aggregated"] > 0, f"{stem} should carry bursts"
    for cg, ca in zip(consumed["greedy"], consumed["agg"]):
        assert ca < cg, (stem, consumed)
    # aggregation also helps J here (fewer pairs to wait for)
    assert sum(j["agg"]) <= sum(j["greedy"]), (stem, j)


def test_chain_ghz_unchanged(repo_root):
    consumed, j, stats = paired_consumed("ghz_n78", repo_root)
    assert stats["n_bursts"] == 0
    assert consumed["agg"] == consumed["greedy"]
