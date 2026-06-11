"""Counter-based common-random-numbers engine (guide §6.5, D8, D18).

Every stochastic draw in the environment is a pure function of
(run seed, link id, channel id, slot):

    uniform(l, c, t) = first float64 of Philox keyed by the run seed with
                       counter (t, c, l, STREAM_GENERATION)

Properties (tested in tests/unit/test_crn.py and
tests/integration/test_crn_policies.py):
- same (seed, l, c, t) -> same draw, regardless of query order or policy;
- two policies evaluated under the same seed face identical generation luck
  wherever their channel-tasking patterns coincide -> valid CRN-paired
  comparisons (guide §10.4).

The engine is strictly separate from every other RNG in the project (the
synthetic-instance generator uses PCG64; torch RNG arrives in Phase 5).
Future stochastic elements (e.g. probabilistic swapping) must use a new
stream constant, never this one.
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1
# Fixed key-tweak / stream constants (arbitrary but frozen; see D18).
_KEY_TWEAK = 0x45414745522D4451       # ASCII "EAGER-DQ"
STREAM_GENERATION = 0x0EA6E12


class CRNEngine:
    """Stateless per-draw Philox: construction is cheap, draws are pure."""

    def __init__(self, seed: int):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        self.seed = seed
        lo = seed & _MASK64
        hi = ((seed >> 64) & _MASK64) ^ _KEY_TWEAK
        self._key = np.array([lo, hi], dtype=np.uint64)

    def uniform(self, link: int, channel: int, t: int) -> float:
        """Deterministic uniform in [0, 1) for coordinate (link, channel, t)."""
        for name, v in (("link", link), ("channel", channel), ("t", t)):
            if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < 2**63:
                raise ValueError(f"CRN coordinate '{name}' must be an integer "
                                 f"in [0, 2^63), got {v!r}")
        counter = np.array([t, channel, link, STREAM_GENERATION],
                           dtype=np.uint64)
        bg = np.random.Philox(key=self._key, counter=counter)
        return float(np.random.Generator(bg).random())

    def success(self, link: int, channel: int, t: int, p: float) -> bool:
        """Bernoulli(p) generation outcome at (link, channel, t)."""
        return self.uniform(link, channel, t) < p
