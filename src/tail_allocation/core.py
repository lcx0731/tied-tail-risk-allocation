"""Exact finite-scenario Expected Shortfall and its allocation face.

Inputs are integers, Fraction objects, or rational/decimal strings. Binary
floats are rejected so that equality at a tail boundary is a stated choice.
"""
from dataclasses import dataclass
from fractions import Fraction


def exact(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is int or isinstance(value, str):
        return Fraction(value)
    raise TypeError("use int, Fraction, or a rational/decimal string; not float")


def vector(values):
    result = tuple(exact(x) for x in values)
    if not result:
        raise ValueError("vectors must be nonempty")
    return result


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("vector lengths differ")
    return sum((x * y for x, y in zip(left, right)), Fraction())


@dataclass(frozen=True, init=False)
class TailRisk:
    losses: tuple
    probabilities: tuple
    alpha: Fraction
    cutoff: Fraction
    es: Fraction
    caps: tuple
    fixed: tuple
    boundary: tuple
    remaining: Fraction

    def __init__(self, losses, probabilities, alpha):
        z, p, a = vector(losses), vector(probabilities), exact(alpha)
        if len(z) != len(p) or any(x <= 0 for x in p) or sum(p) != 1:
            raise ValueError("positive probabilities must match losses and sum to one")
        if not 0 <= a < 1:
            raise ValueError("alpha must be in [0, 1)")
        mass = Fraction()
        cutoff = min(z)
        for i in sorted(range(len(z)), key=z.__getitem__, reverse=True):
            mass += p[i]
            if mass >= 1 - a:
                cutoff = z[i]
                break
        caps = tuple(x / (1 - a) for x in p)
        fixed = tuple(cap if loss > cutoff else Fraction() for loss, cap in zip(z, caps))
        boundary = tuple(i for i, loss in enumerate(z) if loss == cutoff)
        remaining = 1 - sum(fixed)
        es = dot(fixed, z) + remaining * cutoff
        for name, value in dict(losses=z, probabilities=p, alpha=a, cutoff=cutoff,
                                es=es, caps=caps, fixed=fixed, boundary=boundary,
                                remaining=remaining).items():
            object.__setattr__(self, name, value)

    def representative(self):
        """An optimum distributing boundary mass proportionally to probability."""
        q = list(self.fixed)
        denominator = sum(self.probabilities[i] for i in self.boundary)
        for i in self.boundary:
            q[i] = self.remaining * self.probabilities[i] / denominator
        return tuple(q)

    def certify(self, witness):
        """Reject weights unless they are an exactly feasible ES maximizer."""
        q = vector(witness)
        if len(q) != len(self.losses) or sum(q) != 1:
            raise ValueError("witness must have one weight per scenario and sum to one")
        if any(not 0 <= v <= cap for v, cap in zip(q, self.caps)):
            raise ValueError("witness exceeds the risk-envelope bounds")
        if dot(q, self.losses) != self.es:
            raise ValueError("witness is feasible but does not attain Expected Shortfall")
        return q

    def extreme(self, scores, *, maximize=True):
        """Optimize a linear score over the entire ES-optimal face.

        Only boundary mass is variable. Fill its bounded slots in score order.
        Tied scores use input order for the witness; the optimal value is unique.
        """
        scores = vector(scores)
        if len(scores) != len(self.losses) or type(maximize) is not bool:
            raise ValueError("scores must match scenarios and maximize must be bool")
        q, remaining = list(self.fixed), self.remaining
        for i in sorted(self.boundary, key=scores.__getitem__, reverse=maximize):
            q[i] = min(remaining, self.caps[i])
            remaining -= q[i]
        assert remaining == 0
        witness = self.certify(q)
        return dot(witness, scores), witness

    def score_bounds(self, scores):
        scores = tuple(scores)
        return (self.extreme(scores, maximize=False)[0], self.extreme(scores)[0])


@dataclass(frozen=True, init=False)
class Portfolio:
    """Fixed-probability linear scenario losses; exposures may be negative."""
    assets: tuple
    weights: tuple
    tail: TailRisk

    def __init__(self, assets, weights, probabilities, alpha):
        rows, w = tuple(vector(row) for row in assets), vector(weights)
        if not rows or any(len(row) != len(w) for row in rows):
            raise ValueError("scenario matrix must be nonempty and match exposures")
        object.__setattr__(self, "assets", rows)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "tail", TailRisk(tuple(dot(row, w) for row in rows),
                                               probabilities, alpha))

    def gradient(self, witness=None):
        """One ES subgradient; it need not be the unique derivative."""
        q = self.tail.representative() if witness is None else self.tail.certify(witness)
        return tuple(dot(q, column) for column in zip(*self.assets))

    def contributions(self, witness=None):
        return tuple(w * g for w, g in zip(self.weights, self.gradient(witness)))

    def contribution_bounds(self):
        """Sharp coordinate ranges, not independently attainable choices."""
        return tuple(self.tail.score_bounds(tuple(w * x for x in column))
                     for w, column in zip(self.weights, zip(*self.assets)))

    def directional_bounds(self, direction):
        """Left and right slopes of ES(weights + h * direction) at h=0."""
        d = vector(direction)
        return self.tail.score_bounds(tuple(dot(row, d) for row in self.assets))

    def shift(self, direction, step):
        d, h = vector(direction), exact(step)
        if len(d) != len(self.weights):
            raise ValueError("direction must match exposures")
        return Portfolio(self.assets, tuple(w + h * v for w, v in zip(self.weights, d)),
                         self.tail.probabilities, self.tail.alpha)
