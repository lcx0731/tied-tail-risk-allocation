from fractions import Fraction as F
from itertools import product
from random import Random

import pytest

from tail_allocation import Portfolio, TailRisk
from tail_allocation.demo import fixture


def vertices(probabilities, alpha):
    """Independent exact oracle: intersect one equality with box edges."""
    caps = [p / (1 - alpha) for p in probabilities]
    found = set()
    for free in range(len(caps)):
        others = [i for i in range(len(caps)) if i != free]
        for bits in product((0, 1), repeat=len(others)):
            q = [F()] * len(caps)
            for i, bit in zip(others, bits):
                q[i] = bit * caps[i]
            q[free] = 1 - sum(q)
            if 0 <= q[free] <= caps[free]:
                found.add(tuple(q))
    return found


def objective(q, z):
    return sum(qi * zi for qi, zi in zip(q, z))


def test_exact_enumeration_checks_es_and_entire_optimal_face():
    rng = Random(817)
    for _ in range(32):
        n = rng.randrange(2, 7)
        raw = [rng.randrange(1, 6) for _ in range(n)]
        p = [F(x, sum(raw)) for x in raw]
        z = [F(rng.randrange(-4, 5)) for _ in range(n)]
        a = F(rng.randrange(0, 10), 10)
        scores = [F(rng.randrange(-8, 9)) for _ in range(n)]
        candidates = vertices(p, a)
        best = max(objective(q, z) for q in candidates)
        optimal = [q for q in candidates if objective(q, z) == best]
        t = TailRisk(z, p, a)
        assert t.es == best
        assert t.certify(t.representative())
        assert t.score_bounds(scores) == (min(objective(q, scores) for q in optimal),
                                           max(objective(q, scores) for q in optimal))
        # Independent primal formula: minimum over all loss breakpoints.
        primal = min(v + sum(pi * max(zi - v, 0) for pi, zi in zip(p, z)) / (1 - a)
                     for v in z)
        assert primal == best


def test_tied_fixture_ranges_are_sharp_but_not_a_cartesian_product():
    p = fixture()
    assert p.tail.cutoff == 12 and p.tail.es == 14
    assert p.contributions() == (F(17, 4), F(9, 2), F(21, 4))
    assert p.contribution_bounds() == ((F(2), F(13, 2)), (F(9, 4), F(27, 4)),
                                      (F(3), F(15, 2)))
    assert sum(b[1] for b in p.contribution_bounds()) == F(83, 4) > p.tail.es
    optimal = [q for q in vertices(p.tail.probabilities, p.tail.alpha)
               if objective(q, p.tail.losses) == p.tail.es]
    assert len(optimal) == 6
    for q in optimal:
        assert sum(p.contributions(q)) == 14
    for j, (lower, upper) in enumerate(p.contribution_bounds()):
        values = [p.contributions(q)[j] for q in optimal]
        assert (min(values), max(values)) == (lower, upper)


def test_left_right_slopes_and_subgradient_inequality():
    p, d = fixture(), (1, -1, 0)
    assert p.directional_bounds(d) == (F(-19, 4), F(17, 4))
    for h in (F(1, 10000), F(1, 100), F(1, 10)):
        assert (p.shift(d, h).tail.es - p.tail.es) / h == F(17, 4)
        assert (p.shift(d, -h).tail.es - p.tail.es) / -h == F(-19, 4)
    g = p.gradient()
    for direction in product((-2, 0, 3), repeat=3):
        shifted = p.shift(direction, 1)
        assert shifted.tail.es >= p.tail.es + objective(g, direction)


def test_negative_and_zero_exposures_have_correct_contribution_bounds():
    p = Portfolio([(2, 1, 9), (4, 3, -5), (1, 0, 2)], [1, -1, 0],
                  [F(1, 3)] * 3, F(1, 2))
    assert p.tail.es == 1
    assert p.contribution_bounds() == ((F(4, 3), F(10, 3)),
                                      (F(-7, 3), F(-1, 3)), (0, 0))
    assert sum(p.contributions()) == 1


def test_probability_splitting_and_permutation_preserve_allocations():
    p = fixture()
    # Split one tied scenario into identical rows with unequal masses.
    rows = list(p.assets) + [p.assets[4]]
    probs = list(p.tail.probabilities) + [F(3, 100)]
    probs[4] -= F(3, 100)
    split = Portfolio(rows, p.weights, probs, p.tail.alpha)
    order = [7, 3, 1, 6, 0, 4, 2, 5]
    permuted = Portfolio([p.assets[i] for i in order], p.weights,
                         [p.tail.probabilities[i] for i in order], p.tail.alpha)
    for model in (split, permuted):
        assert model.tail.es == p.tail.es
        assert model.contributions() == p.contributions()
        assert model.contribution_bounds() == p.contribution_bounds()
        assert model.directional_bounds((1, -1, 0)) == p.directional_bounds((1, -1, 0))


def test_boundary_confidence_levels_and_full_boundary_fill():
    z, probs = [0, 1, 1, 4], [F(1, 4)] * 4
    mean = TailRisk(z, probs, 0)
    assert mean.representative() == tuple(probs) and mean.es == F(3, 2)
    assert mean.score_bounds([9, 2, -5, 8]) == (F(7, 2), F(7, 2))
    maximum = TailRisk(z, probs, "0.999")
    assert maximum.es == 4 and maximum.representative() == (0, 0, 0, 1)
    full = TailRisk(z, probs, F(1, 4))
    assert full.representative() == (0, F(1, 3), F(1, 3), F(1, 3))
    assert full.score_bounds([9, 2, -5, 8]) == (F(5, 3), F(5, 3))


def test_risk_translation_positive_scaling_and_subadditivity():
    p, a = [F(1, 5), F(3, 10), F(1, 2)], F(3, 5)
    x, y = [-2, 8, 3], [9, -1, 4]
    base = TailRisk(x, p, a).es
    assert TailRisk([v + 7 for v in x], p, a).es == base + 7
    assert TailRisk([v * 3 for v in x], p, a).es == base * 3
    assert TailRisk([v + u for v, u in zip(x, y)], p, a).es <= base + TailRisk(y, p, a).es


def test_nonunique_scenario_weights_can_still_give_unique_gradient():
    p = Portfolio([(3, -1), (3, -1)], [1, 2], [F(1, 2)] * 2, F(3, 4))
    assert p.tail.extreme([1, -1], maximize=False)[1] != p.tail.extreme([1, -1])[1]
    assert p.contribution_bounds() == ((3, 3), (-2, -2))
    assert p.directional_bounds([2, 3]) == (3, 3)


def test_zero_exposures_have_zero_allocations_without_a_unique_derivative():
    p = Portfolio([(1, 0), (0, 1)], [0, 0], [F(1, 2)] * 2, F(1, 2))
    assert p.tail.es == 0
    assert p.contribution_bounds() == ((0, 0), (0, 0))
    assert p.directional_bounds([1, -1]) == (-1, 1)


def test_feasible_but_nonoptimal_witness_is_rejected():
    p = fixture()
    with pytest.raises(ValueError, match="does not attain"):
        p.tail.certify(p.tail.probabilities)
    with pytest.raises(ValueError, match="bounds"):
        p.tail.certify([0] * 7 + [1])
    with pytest.raises(ValueError, match="sum to one"):
        p.tail.certify([0] * 8)
    with pytest.raises(ValueError):
        p.contributions([1])


@pytest.mark.parametrize("losses,probs,alpha,error", [
    ([], [], "0.8", ValueError), ([1], [1], 1, ValueError),
    ([1], [1], -1, ValueError), ([1], ["0.9"], "0.8", ValueError),
    ([1, 2], [0, 1], "0.8", ValueError), ([1], [1, 1], "0.8", ValueError),
    ([1.0], [1], "0.8", TypeError), ([1], [1], 0.8, TypeError),
    ([True], [1], "0.8", TypeError), (["NaN"], [1], "0.8", ValueError),
])
def test_invalid_inputs(losses, probs, alpha, error):
    with pytest.raises(error):
        TailRisk(losses, probs, alpha)


def test_portfolio_shape_and_exact_decimal_strings():
    for rows, weights in (([], [1]), ([[1], [2, 3]], [1]), ([[1]], [])):
        with pytest.raises(ValueError):
            Portfolio(rows, weights, [1], "0.8")
    p = Portfolio([["0.1", "0.2"], ["0.3", 0]], [1, 1], ["0.5", "0.5"], "0.9")
    assert p.tail.losses == (F(3, 10), F(3, 10))
    with pytest.raises(ValueError):
        p.shift([1], 1)
    with pytest.raises(ValueError):
        p.directional_bounds([1])
    with pytest.raises(ValueError):
        p.tail.extreme([1])
