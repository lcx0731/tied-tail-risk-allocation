"""Deterministic synthetic portfolio; no market or customer data."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path

from .core import Portfolio, dot


def fixture():
    return Portfolio(
        [(-2, 1, 0), (1, 0, 1), (2, 3, 1), (6, 3, 1),
         (10, 1, 1), (1, 10, 1), (1, 1, 10), (5, 6, 9)],
        [1, 1, 1], [F(x, 100) for x in (10, 15, 20, 20, 10, 10, 10, 5)], F(4, 5))


def report():
    p = fixture()
    t = p.tail
    # This conditional mean is intentionally wrong for the requested ES.
    full_tail = [i for i, z in enumerate(t.losses) if z >= t.cutoff]
    naive = sum(t.probabilities[i] * t.losses[i] for i in full_tail) / sum(
        t.probabilities[i] for i in full_tail)
    direction = (1, -1, 0)
    scores = tuple(dot(row, direction) for row in p.assets)
    witnesses = [t.extreme(scores, maximize=flag)[1] for flag in (False, True)]
    h = F(1, 1000)
    left, right = p.directional_bounds(direction)
    assert (p.shift(direction, -h).tail.es - t.es) / -h == left
    assert (p.shift(direction, h).tail.es - t.es) / h == right
    assert sum(p.contributions()) == t.es
    return {
        "data": "Eight explicitly specified synthetic scenarios; arbitrary loss units.",
        "alpha": t.alpha,
        "scenario_assets": p.assets,
        "probabilities": t.probabilities,
        "weights": p.weights,
        "portfolio_losses": t.losses,
        "cutoff": t.cutoff,
        "expected_shortfall": t.es,
        "naive_conditional_mean_at_or_above_cutoff": naive,
        "representative_witness": t.representative(),
        "representative_contributions": p.contributions(),
        "contribution_bounds": p.contribution_bounds(),
        "sum_of_coordinate_upper_bounds": sum(b[1] for b in p.contribution_bounds()),
        "direction": direction,
        "directional_slopes_left_right": (left, right),
        "representative_directional_slope": dot(p.gradient(), direction),
        "directional_extreme_witnesses": witnesses,
        "directional_extreme_contributions": [p.contributions(q) for q in witnesses],
        "exact_difference_step": h,
        "stress_curve": [
            {"step": F(k, 20), "es": p.shift(direction, F(k, 20)).tail.es}
            for k in range(-20, 21)
        ],
    }


def encode(value):
    if isinstance(value, F):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write exact rational JSON; otherwise stdout")
    args = parser.parse_args()
    text = json.dumps(report(), default=encode, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
