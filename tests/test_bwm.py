import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swc.bwm import bayesian_bwm, input_based_cr, linear_bwm, simulate_expert, validate

# The thesis's five decision-makers (criteria: cost, social, carbon emission, waste emission)
THESIS = [
    dict(best=0, worst=3, bo=[1, 4, 5, 6], ow=[6, 2, 2, 1]),
    dict(best=0, worst=3, bo=[1, 3, 3, 3], ow=[4, 1, 1, 1]),
    dict(best=1, worst=2, bo=[3, 2, 2, 4], ow=[2, 2, 1, 1]),
    dict(best=1, worst=2, bo=[2, 1, 2, 3], ow=[2, 3, 1, 1]),
    dict(best=1, worst=0, bo=[3, 1, 2, 3], ow=[1, 3, 1, 1]),
]


def test_validation_catches_thesis_dm3_and_accepts_dm1():
    assert validate(**THESIS[0]) == []
    issues = validate(**THESIS[2])
    assert any("best vs itself must be 1" in s for s in issues)
    assert any("largest value" in s for s in issues)
    assert any("differs between the two vectors" in s for s in validate(**THESIS[1]))


def test_input_based_cr_matches_excel_solver():
    # BWM-Solver-5.xlsx reported CR^I = 0.1333 (threshold 0.199) for DM1
    cr, thr, ok = input_based_cr(**THESIS[0])
    assert cr == pytest.approx(0.1333, abs=1e-3) and thr == pytest.approx(0.199) and ok


def test_linear_bwm_reproduces_thesis_dm1_weights():
    w, xi = linear_bwm(**THESIS[0])
    assert w == pytest.approx([0.6093, 0.1656, 0.1325, 0.0927], abs=2e-3)
    assert xi == pytest.approx(0.0530, abs=2e-3)


def test_bayesian_bwm_recovers_true_weights():
    rng = np.random.default_rng(1)
    w_true = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
    resp = [simulate_expert(w_true, rng) for _ in range(12)]
    BO = np.array([r[2] for r in resp]); OW = np.array([r[3] for r in resp])
    res = bayesian_bwm(BO, OW, draws=1500, burn=3000, seed=0)
    assert np.argmax(res.mean) == 0 and np.argmin(res.mean) == len(w_true) - 1
    assert np.abs(res.mean - w_true).max() < 0.06
    cred = res.credal()
    assert cred[0, -1] > 0.95
