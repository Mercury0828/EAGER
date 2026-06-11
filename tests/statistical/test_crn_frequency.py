"""Phase 1B acceptance: empirical Bernoulli success frequency within the 99%
CI of p over >= 1e5 draws, for p in {0.05, 1/12, 0.3} (guide §11).

The draws are CRN-deterministic given the seed, so this test has a fixed
outcome: it verifies the generator's frequency behavior once, and the 10x
repeat protocol confirms stability by construction.
"""

import math

import pytest

from eager.env.crn import CRNEngine

pytestmark = [pytest.mark.stochastic, pytest.mark.statistical]

N_DRAWS = 100_000
Z_99 = 2.5758293035489004      # two-sided 99% normal quantile


@pytest.mark.parametrize("p", [0.05, 1.0 / 12.0, 0.3])
def test_success_frequency_within_99ci(p):
    eng = CRNEngine(seed=20270101)
    hits = 0
    # Sweep the slot coordinate on one (link, channel): exactly how a busy
    # channel consumes the stream during an episode.
    for t in range(N_DRAWS):
        hits += eng.success(0, 0, t, p)
    phat = hits / N_DRAWS
    half_width = Z_99 * math.sqrt(p * (1.0 - p) / N_DRAWS)
    assert abs(phat - p) <= half_width, (
        f"p={p}: empirical {phat:.6f} outside 99% CI half-width "
        f"{half_width:.6f}")


def test_frequency_across_links_and_channels():
    """Frequency also holds when sweeping link/channel coordinates (no
    stream correlation across the counter words)."""
    eng = CRNEngine(seed=314159)
    p = 0.3
    n = 0
    hits = 0
    for l in range(10):
        for c in range(10):
            for t in range(1_000):
                hits += eng.success(l, c, t, p)
                n += 1
    phat = hits / n
    half_width = Z_99 * math.sqrt(p * (1.0 - p) / n)
    assert abs(phat - p) <= half_width
