import numpy as np
import pandas as pd
import xgboost as xgb

from nba import insights


def test_importance_table_gives_shares_mean_and_rank():
    t = insights.importance_table({"loan": {"income": 3.0, "age": 1.0}, "card": {"income": 1.0, "age": 1.0}})
    assert list(t.index) == ["income", "age"]
    assert np.isclose(t.loc["income", "loan"], 75) and np.isclose(t.loc["income", "mean share"], 62.5)
    assert t["rank"].tolist() == [1, 2]
    assert insights.summary(t)[0].startswith("Across all offers the strongest predictors are income")


def test_directions_recover_the_sign_of_each_effect():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"tenure": rng.uniform(0, 100, 2000), "risk": rng.uniform(0, 1, 2000),
                      "band": pd.Categorical(rng.choice(["low", "high"], 2000))})
    logit = 0.04 * X.tenure - 3 * X.risk + np.where(X.band == "high", 1.5, 0) - 1
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-logit))).astype(int)
    model = xgb.XGBClassifier(n_estimators=60, max_depth=3, enable_categorical=True, random_state=0).fit(X, y)
    d = insights.directions(X, insights.shap_values(model, X), ["band"]).set_index("feature")
    assert d.loc["tenure", "raises acceptance"] == "higher values"
    assert d.loc["risk", "raises acceptance"] == "lower values"
    assert d.loc["band", "raises acceptance"] == "high" and d.loc["band", "lowers acceptance"] == "low"
    dep = insights.dependence(X, insights.shap_values(model, X), "band", ["band"])
    assert dep.loc["high", "mean SHAP"] > dep.loc["low", "mean SHAP"]
