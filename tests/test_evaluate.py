import numpy as np
import pandas as pd

from nba.evaluate import estimate, one_hot, ope_contributions

ACTIONS = ["a", "b"]


def toy_log():
    logged = pd.Series(["a", "b", "a", "b"])
    reward = pd.Series([1.0, 0.0, 0.0, 2.0])
    propensity = pd.Series([0.5, 0.25, 0.5, 0.25])
    return logged, reward, propensity


def test_ips_matches_hand_computation():
    logged, reward, propensity = toy_log()
    target = one_hot(pd.Series(["a", "a", "a", "b"]), ACTIONS)  # agrees on rows 0, 2, 3
    q = pd.DataFrame(0.0, index=logged.index, columns=ACTIONS)
    w, r, contrib = ope_contributions(target, logged, reward, propensity, q)
    assert np.allclose(w, [2, 0, 2, 4])
    assert np.isclose(contrib["IPS"].mean(), (2 * 1 + 2 * 0 + 4 * 2) / 4)
    assert np.isclose(estimate(w, r, contrib, 10, 0)["SNIPS"]["value"], 10 / 8)


def test_dr_equals_dm_when_model_is_exact():
    logged, reward, propensity = toy_log()
    target = pd.DataFrame(0.5, index=logged.index, columns=ACTIONS)
    q = pd.DataFrame({"a": reward.where(logged == "a", 0.7), "b": reward.where(logged == "b", 0.3)})
    _, _, contrib = ope_contributions(target, logged, reward, propensity, q)
    assert np.allclose(contrib["DR"], contrib["DM"])


def test_logging_policy_is_recovered_by_ips():
    rng = np.random.default_rng(0)
    n = 20000
    logged = pd.Series(rng.choice(ACTIONS, n, p=[0.8, 0.2]))
    propensity = logged.map({"a": 0.8, "b": 0.2})
    reward = pd.Series(np.where(logged == "b", rng.random(n) < 0.5, rng.random(n) < 0.1).astype(float))
    target = one_hot(pd.Series("b", index=logged.index), ACTIONS)  # always offer b: true value 0.5
    q = pd.DataFrame(0.0, index=logged.index, columns=ACTIONS)
    w, r, contrib = ope_contributions(target, logged, reward, propensity, q)
    assert abs(contrib["IPS"].mean() - 0.5) < 0.03
