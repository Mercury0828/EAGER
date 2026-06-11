"""Phase 1B acceptance: CRN engine property tests (guide §6.5, D8/D18).

Same (seed, l, c, t) -> same draw regardless of query order; engines with the
same seed agree everywhere; different seeds disagree somewhere; coordinates
are validated; success() is the documented threshold test.
"""

import random

import pytest

from eager.env.crn import CRNEngine

pytestmark = pytest.mark.stochastic

COORDS = [(l, c, t) for l in range(4) for c in range(3) for t in range(25)]


def test_query_order_independence():
    eng = CRNEngine(seed=123)
    first = {coord: eng.uniform(*coord) for coord in COORDS}
    shuffled = COORDS.copy()
    random.Random(99).shuffle(shuffled)
    second = {coord: eng.uniform(*coord) for coord in shuffled}
    assert first == second


def test_same_seed_same_draws_across_engines():
    a, b = CRNEngine(seed=2027), CRNEngine(seed=2027)
    for coord in COORDS:
        assert a.uniform(*coord) == b.uniform(*coord)


def test_different_seeds_differ_somewhere():
    a, b = CRNEngine(seed=0), CRNEngine(seed=1)
    diffs = sum(a.uniform(*c) != b.uniform(*c) for c in COORDS)
    assert diffs > len(COORDS) * 0.9, "independent streams expected"


def test_uniform_range_and_threshold_semantics():
    eng = CRNEngine(seed=7)
    for coord in COORDS:
        u = eng.uniform(*coord)
        assert 0.0 <= u < 1.0
        assert eng.success(*coord, p=1.0) is True, "p=1 always succeeds"
        assert eng.success(*coord, p=0.5) == (u < 0.5)


def test_coordinate_separation():
    """Distinct coordinates must give (essentially always) distinct draws —
    guards against accidentally ignoring a counter word."""
    eng = CRNEngine(seed=11)
    vals = {}
    for coord in COORDS:
        vals.setdefault(eng.uniform(*coord), []).append(coord)
    collisions = {v: cs for v, cs in vals.items() if len(cs) > 1}
    assert not collisions, f"distinct coords collided: {collisions}"


def test_input_validation():
    with pytest.raises(ValueError, match="seed"):
        CRNEngine(seed=-1)
    eng = CRNEngine(seed=0)
    with pytest.raises(ValueError, match="coordinate"):
        eng.uniform(-1, 0, 0)
    with pytest.raises(ValueError, match="coordinate"):
        eng.uniform(0, 0, 2**63)


def test_large_seed_supported():
    eng = CRNEngine(seed=2**80 + 5)   # seeds beyond 64 bits fold into the key
    assert 0.0 <= eng.uniform(0, 0, 0) < 1.0
