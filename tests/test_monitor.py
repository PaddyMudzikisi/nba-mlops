import numpy as np
import pandas as pd

from nba.monitor import psi, status


def test_identical_distributions_have_zero_psi():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(s, s) < 1e-9


def test_shifted_distribution_raises_alert():
    rng = np.random.default_rng(0)
    ref, cur = pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(1.0, size=5000))
    assert status(psi(ref, cur), 0.1, 0.25) == "alert"


def test_categorical_psi_handles_unseen_mix():
    ref = pd.Series(["SMS"] * 80 + ["App"] * 20)
    cur = pd.Series(["SMS"] * 20 + ["App"] * 80)
    assert psi(ref, cur) > 0.25
    assert psi(ref, ref) < 1e-9
