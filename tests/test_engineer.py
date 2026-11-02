import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi.testclient import TestClient

from nba import engineer, models
from nba.cleaning import clean
from nba.policy import Policy
from nba.serve import create_app

RECIPE = [
    {"op": "combine", "name": "avg_txn", "a": "value", "b": "count", "how": "divide"},
    {"op": "log", "name": "log_value", "column": "value"},
    {"op": "bin", "name": "count_band", "column": "count", "edges": [2, 5]},
    {"op": "binarize", "name": "busy", "column": "count", "threshold": 3},
    {"op": "date_part", "name": "offer_weekend", "part": "weekend"},
    {"op": "log", "name": "log_avg_txn", "column": "avg_txn"},  # built on an engineered feature
]


def frame():
    return pd.DataFrame({"value": [100.0, 50.0, 0.0, np.nan], "count": [4, 0, 2, 6],
                         "offer_date": pd.to_datetime(["2026-11-07", "2026-11-09", "2026-11-10", "2026-11-11"])})


def test_apply_computes_each_technique():
    out = engineer.apply(frame(), RECIPE)
    assert out.avg_txn.iloc[0] == 25 and np.isnan(out.avg_txn.iloc[1])  # division by zero -> missing
    assert np.isclose(out.log_value.iloc[0], np.log(101))
    assert out.count_band.tolist() == ["2 to < 5", "< 2", "2 to < 5", ">= 5"]
    assert out.busy.tolist() == [1, 0, 0, 1]
    assert out.offer_weekend.tolist() == [1, 0, 0, 0]  # 7 Nov 2026 is a Saturday
    assert np.isclose(out.log_avg_txn.iloc[0], np.log(26))


def test_raw_inputs_replace_engineered_features_by_their_sources():
    feats = ["region", "log_avg_txn", "offer_weekend", "busy"]
    assert engineer.raw_inputs(feats, RECIPE) == ["region", "value", "count"]


def test_check_rejects_bad_names_and_non_numeric_sources():
    f = frame().assign(region="a")
    assert engineer.check({"op": "log", "name": "Bad name", "column": "value"}, f, [])
    assert engineer.check({"op": "log", "name": "value", "column": "value"}, f, [])  # already exists
    assert engineer.check({"op": "log", "name": "log_region", "column": "region"}, f, [])
    assert engineer.check({"op": "log", "name": "log_value", "column": "value"}, f, []) == []


def test_api_computes_engineered_features_from_raw_inputs(tmp_path):
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({"value": rng.uniform(1, 100, 300), "count": rng.integers(1, 10, 300),
                        "offer_date": pd.Timestamp("2026-01-01")})
    X = engineer.apply(raw, RECIPE[:2])[["avg_txn", "log_value"]]
    y = (X.avg_txn > X.avg_txn.median()).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=2).fit(X, y)
    meta = {"train_run_id": "fe-run", "features": ["avg_txn", "log_value"], "categories": {},
            "engineered": RECIPE[:2], "inputs": engineer.raw_inputs(["avg_txn", "log_value"], RECIPE[:2]),
            "actions": {"loan": {"value": 1.0, "scale_pos_weight": 1.0, "model_version": "1"}}}
    c = TestClient(create_app(Policy(meta, {"loan": model}), tmp_path / "d.jsonl"))
    assert c.get("/schema").json()["features"] == ["value", "count"]
    r = c.post("/recommend", json={"customer_key": "SYN1", "features": {"value": 90.0, "count": 1}})
    assert r.status_code == 200
    expected = model.predict_proba(X.iloc[:0].pipe(lambda _: engineer.apply(
        pd.DataFrame({"value": [90.0], "count": [1]}), RECIPE[:2])[["avg_txn", "log_value"]]))[0, 1]
    assert np.isclose(r.json()["ranking"][0]["probability"], expected, atol=1e-6)


def test_preprocessing_options_build_the_numeric_pipeline():
    params = {"seed": 0, "preprocessing": {"impute": "mean", "scaling": "minmax", "pca_components": 2}}
    steps = models.numeric_steps(params, "random_forest", 5)
    assert [type(s).__name__ for _, s in steps.steps] == ["SimpleImputer", "MinMaxScaler", "PCA"]
    assert steps.steps[0][1].strategy == "mean"
    # Recorded runs without the options keep their original pipelines.
    assert type(models.numeric_steps({"seed": 0}, "random_forest", 5)).__name__ == "SimpleImputer"
    lr = models.numeric_steps({"seed": 0}, "logistic_regression", 5)
    assert [type(s).__name__ for _, s in lr.steps] == ["SimpleImputer", "StandardScaler"]


def test_cleaning_mean_fill_and_standard_deviation_outliers():
    f = pd.DataFrame({"k": [f"S{i}" for i in range(40)], "d": "2026-01-01", "o": "loan", "y": [0, 1] * 20,
                      "x": [1.0] * 38 + [np.nan, 1000.0]})
    schema = {"columns": {"customer_key": "k", "offer_date": "d", "action": "o", "accepted": "y"},
              "categorical": [], "numeric": ["x"]}
    filled, _, _ = clean(f, schema, {"fill_numeric": "mean"})
    assert np.isclose(filled.x.iloc[38], f.x.mean())
    capped, _, log = clean(f, schema, {"outliers": "drop", "outlier_columns": ["x"], "outlier_method": "zscore",
                                       "outlier_z": 3})
    assert 1000.0 not in capped.x.tolist() and "standard deviations" in log[-1]["detail"]
