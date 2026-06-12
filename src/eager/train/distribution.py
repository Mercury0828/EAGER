"""Training/held-out case distributions (guide §10.3 restricted to the
Phase 5 SMALL CONFIG; D49).

Train (stage A): synthetic N in [10,30], M/N in {1,3}; hardware sampled from
topologies {K=2 line, K=4 2x2 grid} x p in {0.08, 0.12, 0.2} x W in {1,2}
x T_cut in {10,20}; B=8, w=1, kappa = ceil(1.25*N/K), stochastic mode.
Stage B (curriculum unlock, guide §8.2): N in [30,60].

Held-out evaluation cases draw from the SAME distribution with a disjoint
seed range (generator seeds >= 10_000), so they are unseen instances of the
training distribution (zero-shot across distributions is Phase 6 material).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..circuit import CircuitInstance
from ..config import HardwareConfig, SynthParams, load_hardware_config
from ..expgen.synthetic import generate_instance

P_CHOICES = (0.08, 0.12, 0.2)
W_CHOICES = (1, 2)
TCUT_CHOICES = (10, 20)
RATIO_CHOICES = (1, 3)
STAGE_N = {"A": (10, 30), "B": (30, 60)}


@dataclass(frozen=True)
class Case:
    hardware: HardwareConfig
    instance: CircuitInstance
    label: str


def make_hardware(qpus: int, n_qubits: int, p: float, w_ch: int,
                  t_cut: int) -> HardwareConfig:
    kappa = math.ceil(1.25 * n_qubits / qpus)
    topo = {"qpus": 2, "topology": "line"} if qpus == 2 else {
        "qpus": 4, "topology": "grid", "grid_dims": [2, 2]}
    return load_hardware_config({
        "name": f"k{qpus}_p{p}_w{w_ch}_c{t_cut}_kap{kappa}",
        **topo, "kappa": kappa, "mode": "stochastic", "t_ep": 12,
        "link_defaults": {"p": p, "W": w_ch, "B": 8, "T_cut": t_cut,
                          "w": 1.0}})


def sample_case(rng: np.random.Generator, stage: str = "A",
                regime: str = "full") -> Case:
    """regime='provisioning' restricts hardware draws to the provisioning-
    bound part of the grid (p=0.2 or W=1 — the combos where proactive
    provisioning has measured headroom, 4/6 of the (p, W) grid; D64), used
    by the regime-staged curriculum. 'full' is the D49 distribution."""
    lo, hi = STAGE_N[stage]
    n = int(rng.integers(lo, hi + 1))
    ratio = int(rng.choice(RATIO_CHOICES))
    gen_seed = int(rng.integers(0, 10_000))
    inst = generate_instance(SynthParams(n, n * ratio, None), seed=gen_seed)
    while True:
        p = float(rng.choice(P_CHOICES))
        w_ch = int(rng.choice(W_CHOICES))
        if regime == "full" or p == 0.2 or w_ch == 1:
            break
    hw = make_hardware(
        qpus=int(rng.choice((2, 4))), n_qubits=n,
        p=p, w_ch=w_ch, t_cut=int(rng.choice(TCUT_CHOICES)))
    return Case(hardware=hw, instance=inst,
                label=f"{inst.name}@{hw.name}")


def held_out_cases(n_cases: int = 20, seed: int = 777,
                   stage: str = "A") -> list[Case]:
    """Unseen instances (generator seeds >= 10_000) from the stage
    distribution; deterministic given seed."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n_cases):
        lo, hi = STAGE_N[stage]
        n = int(rng.integers(lo, hi + 1))
        ratio = int(rng.choice(RATIO_CHOICES))
        gen_seed = 10_000 + int(rng.integers(0, 10_000))
        inst = generate_instance(SynthParams(n, n * ratio, None),
                                 seed=gen_seed)
        hw = make_hardware(
            qpus=int(rng.choice((2, 4))), n_qubits=n,
            p=float(rng.choice(P_CHOICES)), w_ch=int(rng.choice(W_CHOICES)),
            t_cut=int(rng.choice(TCUT_CHOICES)))
        cases.append(Case(hardware=hw, instance=inst,
                          label=f"{inst.name}@{hw.name}"))
    return cases
