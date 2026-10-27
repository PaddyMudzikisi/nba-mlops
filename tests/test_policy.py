import numpy as np
import pandas as pd

from nba.policy import calibrate


def test_calibrate_is_identity_without_weighting():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(calibrate(p, 1.0), p)


def test_calibrate_divides_odds_by_weight():
    p = np.array([0.2, 0.5, 0.8])
    q = calibrate(p, 4.0)
    assert np.allclose((p / (1 - p)) / (q / (1 - q)), 4.0)


def test_recommends_highest_expected_value(toy_policy):
    frame = pd.DataFrame({"x": [3.0, -3.0], "band": ["a", "b"], "consent_marketing": [1, 1]})
    proba, ev = toy_policy.score(frame)
    assert np.allclose(ev, proba * [100.0, 10.0])
    assert (toy_policy.recommend(frame) == ev.idxmax(axis=1)).all()


def test_no_recommendation_without_consent(toy_policy):
    frame = pd.DataFrame({"x": [3.0], "band": ["a"], "consent_marketing": [0]})
    assert toy_policy.recommend(frame).isna().all()
