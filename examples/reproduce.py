"""Reproduce exact JSON and a static figure from the synthetic fixture."""
import argparse
import json
from fractions import Fraction as F
from itertools import permutations
from math import atan2
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from tail_allocation.core import dot
from tail_allocation.demo import encode, fixture, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, p = report(), fixture()
    (args.output / "metrics.json").write_text(json.dumps(data, default=encode, indent=2) + "\n")
    teal, orange, navy, gray = "#007f86", "#d4722c", "#243b53", "#d7dfe6"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.labelcolor": navy, "text.color": navy})
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6), layout="constrained")
    fig.suptitle("One tail risk, several valid allocations", fontsize=20, fontweight="bold")
    fig.supxlabel("Deterministic synthetic scenarios · confidence level 80% · arbitrary loss units", fontsize=10)

    ax = axs[0, 0]
    x = list(range(1, 9))
    ax.bar(x, [float(v) for v in p.tail.probabilities], color=gray, label="Scenario probability")
    ax.bar(x, [float(q * (1 - p.tail.alpha)) for q in p.tail.representative()],
           color=teal, label="Selected tail mass (pro rata)")
    ax.set(title="A  Select exactly 20% probability mass", xlabel="Scenario", ylabel="Probability")
    ax.set_xticks(x)
    ax.set_ylim(0, 0.30)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for i, loss, probability in zip(x, p.tail.losses, p.tail.probabilities):
        ax.text(i, float(probability) + 0.007, f"L={loss}", ha="center", fontsize=9)

    ax = axs[0, 1]
    bounds = p.contribution_bounds()
    for i, ((lo, hi), representative) in enumerate(zip(bounds, p.contributions())):
        ax.plot([float(lo), float(hi)], [i, i], color=teal, lw=6, solid_capstyle="round")
        ax.scatter(float(representative), i, s=80, color=orange, edgecolor="white", zorder=3)
        ax.text(float(lo), i + .17, f"{float(lo):.2f}", ha="center", fontsize=9)
        ax.text(float(hi), i + .17, f"{float(hi):.2f}", ha="center", fontsize=9)
    ax.set_yticks(range(3), ["Line A", "Line B", "Line C"])
    ax.set(title="B  Sharp individual contribution ranges", xlabel="Contribution to ES", ylim=(-.5, 2.65), xlim=(1, 8.3))
    ax.scatter([], [], color=orange, label="Probability-proportional choice")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.15)

    ax = axs[1, 0]
    steps = [float(row["step"]) for row in data["stress_curve"]]
    values = [float(row["es"]) for row in data["stress_curve"]]
    ax.plot(steps, values, color=teal, lw=2.5, label="Recomputed ES")
    representative_slope = float(dot(p.gradient(), (1, -1, 0)))
    ax.plot(steps, [14 + h * representative_slope for h in steps], color=orange,
            linestyle="--", label="One subgradient's affine lower bound")
    ax.axvline(0, color=gray, lw=1)
    ax.set(title="C  Opposite slopes at the boundary tie", xlabel="Exposure shift h: w + h × (1, −1, 0)", ylabel="Expected Shortfall", ylim=(13, 19.5))
    ax.text(.03, .92, "Left slope −4.75; right slope +4.25", transform=ax.transAxes, fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=.15)

    # For this fixture the boundary has three caps 1/2 and total mass 3/4.
    # Its six vertices are the permutations of (0, 1/4, 1/2).
    points = []
    for shares in permutations((F(0), F(1, 4), F(1, 2))):
        q = list(p.tail.fixed)
        for i, share in zip(p.tail.boundary, shares):
            q[i] = share
        p.tail.certify(q)
        c = p.contributions(q)
        assert sum(c) == 14
        points.append((float(c[0]), float(c[1])))
    center = (float(p.contributions()[0]), float(p.contributions()[1]))
    points.sort(key=lambda point: atan2(point[1] - center[1], point[0] - center[0]))
    ax = axs[1, 1]
    lo_a, hi_a = map(float, bounds[0])
    lo_b, hi_b = map(float, bounds[1])
    ax.add_patch(Rectangle((lo_a, lo_b), hi_a - lo_a, hi_b - lo_b, fill=False,
                           linestyle="--", edgecolor=orange, label="Coordinate range box"))
    ax.add_patch(Polygon(points, color=teal, alpha=.22, label="Jointly attainable allocations"))
    xs, ys = zip(*(points + points[:1]))
    ax.plot(xs, ys, color=teal, lw=2)
    ax.scatter(*center, color=orange, s=70, zorder=3)
    ax.scatter(hi_a, hi_b, marker="x", color=orange, s=90, zorder=3)
    ax.annotate("Not jointly attainable", (hi_a, hi_b), xytext=(3.0, 7.1), fontsize=9,
                arrowprops={"arrowstyle": "->", "color": orange})
    ax.set(title="D  Line C must equal 14 − A − B", xlabel="Line A contribution", ylabel="Line B contribution",
           xlim=(1.4, 7.2), ylim=(1.6, 7.7))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=.15)
    fig.savefig(args.output / "allocation_geometry.png", dpi=180, facecolor="white",
                metadata={"Software": "tied-tail-risk-allocation"})
    plt.close(fig)
    print(f"ES={data['expected_shortfall']}; exact JSON and figure saved in {args.output}")


if __name__ == "__main__":
    main()
