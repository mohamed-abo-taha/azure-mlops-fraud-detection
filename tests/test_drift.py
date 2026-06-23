import numpy as np
import pandas as pd

from src.monitoring import drift_report, psi


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(3, 1, 5000)) > 0.2


def test_drift_report_flags_shifted_feature():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f1": rng.normal(0, 1, 3000), "f2": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"f1": rng.normal(0, 1, 3000), "f2": rng.normal(2, 1, 3000)})
    rep = drift_report(ref, cur, ["f1", "f2"])
    assert rep["overall_drift"] is True
    assert rep["n_drifted"] >= 1
