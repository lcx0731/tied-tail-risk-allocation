import json
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from tail_allocation import Portfolio
from tail_allocation.demo import fixture


def test_scipy_primal_and_dual_face_references():
    rng = np.random.default_rng(791)
    cases = [fixture()]
    for _ in range(12):
        rows = rng.integers(-8, 9, size=(7, 3)).tolist()
        weights = rng.integers(-2, 3, size=3).tolist()
        masses = rng.integers(1, 8, size=7).tolist()
        cases.append(Portfolio(rows, weights, [F(x, sum(masses)) for x in masses], F(4, 5)))
    for p in cases:
        z = np.asarray(p.tail.losses, dtype=float)
        probs = np.asarray(p.tail.probabilities, dtype=float)
        a, n = float(p.tail.alpha), len(z)
        # Free cutoff + n nonnegative slack variables: primal RU formulation.
        result = linprog(np.r_[1, probs / (1 - a)],
                         A_ub=np.column_stack((-np.ones(n), -np.eye(n))), b_ub=-z,
                         bounds=[(None, None)] + [(0, None)] * n, method="highs")
        assert result.success
        assert abs(result.fun - float(p.tail.es)) < 1e-8
        # Add the ES equality to the envelope, then optimize each allocation.
        for j, (low, high) in enumerate(p.contribution_bounds()):
            scores = np.array([float(row[j] * p.weights[j]) for row in p.assets])
            for sign, expected in ((1, low), (-1, high)):
                fit = linprog(sign * scores, A_eq=np.stack([np.ones(n), z]),
                              b_eq=[1, float(p.tail.es)],
                              bounds=list(zip(np.zeros(n), probs / (1 - a))), method="highs")
                assert fit.success
                assert abs(sign * fit.fun - float(expected)) < 1e-8


def test_installed_command_outside_repo(tmp_path):
    result = subprocess.run([str(Path(sys.executable).with_name('tail-allocation-demo'))],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["expected_shortfall"] == "14"
    assert data["directional_slopes_left_right"] == ["-19/4", "17/4"]
    target = tmp_path / "nested" / "metrics.json"
    subprocess.run([sys.executable, "-m", "tail_allocation.demo", "--output", str(target)],
                   cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(target.read_text()) == data
