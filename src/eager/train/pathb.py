"""Path B (D76): provisioning-only EAGER on a fixed strong (AGG) base.

The agent does NOT learn placement — it cannot beat MHSA/AGG at it (D72/D76).
Instead each episode starts on the AGG-aggregated instance with the AGG
placement PRE-APPLIED, so the only decisions the agent (and its expert) make
are Schedule / GenEPR / ADVANCE — the proactive-provisioning policy. The
expert is GreedyRegimeProvision (eager normally, reactive in the waste
regime), which beats AGG-reactive by ~2% (up to ~4.6% oracle) with the gain
attributable purely to provisioning (placement+aggregation matched to AGG).

Training distribution spans the FULL regime grid (p/W/T_cut incl. the waste
regime) so the agent learns the regime-adaptive switch from the link
features. Reuses the model, IL, and PPO machinery (the pre-mapped env yields
no Map actions, so the action space is naturally provisioning-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..baselines.agg import transform_instance
from ..baselines.greedy_jit import compute_placement, map_emission_order
from ..circuit import CircuitInstance
from ..config import HardwareConfig, SynthParams, load_hardware_config
from ..env.actions import Map
from ..env.env import EagerEnv
from ..expgen.synthetic import generate_instance

P_GRID = (0.05, 0.08, 0.12, 0.2, 0.3, 0.5)
W_GRID = (1, 2, 4)
TCUT_GRID = (5, 20, 50)


@dataclass(frozen=True)
class PathBCase:
    hardware: HardwareConfig
    agg_instance: CircuitInstance     # AGG-transformed instance to run
    placement: tuple[int, ...]        # AGG placement, pre-applied each episode
    label: str


def _hardware(qpus: int, n_qubits: int, p: float, w_ch: int, t_cut: int):
    kappa = math.ceil(1.25 * n_qubits / qpus)
    topo = ({"qpus": 2, "topology": "line"} if qpus == 2
            else {"qpus": 4, "topology": "grid", "grid_dims": [2, 2]})
    return load_hardware_config({
        "name": f"k{qpus}_p{p}_w{w_ch}_c{t_cut}", **topo, "kappa": kappa,
        "mode": "stochastic", "t_ep": 12,
        "link_defaults": {"p": p, "W": w_ch, "B": 8, "T_cut": t_cut,
                          "w": 1.0}})


def _waste_regime_params(rng):
    """The waste corner (D75): sparse circuit (d=1) + high p + W>=2 + tight
    T_cut=5, where always-on over-generates and pairs expire — the regime the
    single network learns imperfectly (the D78b 7.7%-vs-reactive residual)."""
    return 1, float(rng.choice([0.3, 0.5])), int(rng.choice([2, 4])), 5


def sample_pathb_case(rng: np.random.Generator, qpus_choices=(2, 4),
                      waste_frac: float = 0.0) -> PathBCase:
    """waste_frac (curriculum, D85): with this probability the case is drawn
    from the waste corner instead of the uniform p/W/T_cut grid, so the policy
    sees the under-represented hold-back regime far more often."""
    n = int(rng.integers(10, 31))
    is_waste = waste_frac > 0.0 and rng.random() < waste_frac
    if is_waste:
        d, p, w_ch, t_cut = _waste_regime_params(rng)
    else:
        d = int(rng.choice([1, 3]))
        p, w_ch, t_cut = (float(rng.choice(P_GRID)), int(rng.choice(W_GRID)),
                          int(rng.choice(TCUT_GRID)))
    inst = generate_instance(SynthParams(n, n * d, None),
                             seed=int(rng.integers(0, 100000)))
    qpus = int(rng.choice(qpus_choices))
    placement = tuple(compute_placement(
        inst, _hardware(qpus, n, 0.12, 2, 20), seed=0))
    agg_inst, _ = transform_instance(inst, list(placement))
    hw = _hardware(qpus, n, p, w_ch, t_cut)
    return PathBCase(hardware=hw, agg_instance=agg_inst, placement=placement,
                     label=f"{agg_inst.name}@{hw.name}")


def held_out_pathb_cases(n_cases: int = 20, seed: int = 777,
                         waste: bool = False) -> list[PathBCase]:
    """waste=True forces every case into the waste corner (D85 eval) — used to
    measure the reactive-residual shrinkage from the waste curriculum without
    it being diluted by the ~96%-normal random held-out."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        n = int(rng.integers(10, 31))
        if waste:
            d, p, w_ch, t_cut = _waste_regime_params(rng)
        else:
            d = int(rng.choice([1, 3]))
            p, w_ch, t_cut = (float(rng.choice(P_GRID)), int(rng.choice(W_GRID)),
                              int(rng.choice(TCUT_GRID)))
        inst = generate_instance(SynthParams(n, n * d, None),
                                 seed=10_000 + int(rng.integers(0, 100000)))
        qpus = int(rng.choice((2, 4)))
        placement = tuple(compute_placement(
            inst, _hardware(qpus, n, 0.12, 2, 20), seed=0))
        agg_inst, _ = transform_instance(inst, list(placement))
        hw = _hardware(qpus, n, p, w_ch, t_cut)
        cases.append(PathBCase(hardware=hw, agg_instance=agg_inst,
                               placement=placement,
                               label=f"{agg_inst.name}@{hw.name}"))
    return cases


def premapped_env(case: PathBCase, seed: int) -> EagerEnv:
    """Reset an env on the AGG instance and apply the AGG placement, so the
    agent acts from the fully-mapped state (provisioning-only)."""
    env = EagerEnv(case.hardware, case.agg_instance)
    env.reset(seed)
    for q in map_emission_order(case.agg_instance):
        if q in env._unmapped:
            env.step(Map(q, case.placement[q]))
    return env
