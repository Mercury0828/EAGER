"""Hardware config generators for experiment panels (guide §10.2).

The Phase 2 default panel hardware (D32): K=4 2x2 grid with per-QPU capacity
kappa_u = ceil(1.25*N/K) (guide §10.2 sizing) and the §10.2 link defaults
(p=1/12, W=2, B=8, T_cut=20, w=1), stochastic mode.
"""

from __future__ import annotations

import math

from ..config import HardwareConfig, load_hardware_config


def default_panel_hardware(num_qubits: int, qpus: int = 4) -> HardwareConfig:
    if qpus != 4:
        raise ValueError("the Phase 2 default panel is fixed at K=4 (D32); "
                         "topology sweeps arrive with the Phase 6 matrix")
    kappa = math.ceil(1.25 * num_qubits / qpus)
    return load_hardware_config({
        "name": f"panel_k4_grid_kap{kappa}",
        "qpus": 4, "topology": "grid", "grid_dims": [2, 2],
        "kappa": kappa, "mode": "stochastic", "t_ep": 12,
        "link_defaults": {"p": 1.0 / 12.0, "W": 2, "B": 8, "T_cut": 20,
                          "w": 1.0},
    })
