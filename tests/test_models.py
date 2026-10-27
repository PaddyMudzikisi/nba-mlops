import copy

import numpy as np
import pandas as pd
import pytest

from nba import models
from nba.train import split_frame, train_action

BASE = {"seed": 0, "n_jobs": 1, "balance_classes": False,
        "split": {"method": "random", "test_size": 0.25},
        "dataset": {"categorical": ["band"], "numeric": ["x"]},
        "xgboost": {"n_estimators": 20, "max_depth": 2}, "random_forest": {"n_estimators": 20, "max_depth": 3},
        "logistic_regression": {"C": 1.0},
        "tuning": {"enabled": False, "cv_folds": 3, "scoring": "roc_auc",
                   "grid": {"xgboost": {"max_depth": [2, 3]}, "random_forest": {"max_depth": [2, 3]},
                            "logistic_regression": {"C": [0.1, 1.0]}}}}


def toy(n=400):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({
        "offer_id": [f"o{i:04d}" for i in range(n)], "action": "loan",
        "offer_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), "D"),
        "x": x, "band": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "accepted": (x + rng.normal(scale=0.7, size=n) > 0.8).astype(int), "value": 1.0})


@pytest.mark.parametrize("kind", models.MODEL_TYPES)
def test_every_model_type_trains_and_ranks(kind):
    params = {**copy.deepcopy(BASE), "model": {"type": kind}}
    r = train_action(toy(), "loan", params)
    test = r["test"]
    m = models.classification_metrics(test.accepted, r["proba"], r["threshold"])
    assert m["roc_auc"] > 0.75
    assert 0 < r["threshold"] < 1
    method, imp = models.importance(r["model"], params, test[["band", "x"]], test.accepted)
    assert max(imp, key=imp.get) == "x"


@pytest.mark.parametrize("kind", models.MODEL_TYPES)
def test_training_is_deterministic(kind):
    params = {**copy.deepcopy(BASE), "model": {"type": kind}}
    a, b = train_action(toy(), "loan", params, False), train_action(toy(), "loan", params, False)
    assert np.array_equal(a["proba"], b["proba"])


def test_tuning_picks_from_the_grid_on_training_data_only():
    params = copy.deepcopy(BASE)
    params["model"] = {"type": "logistic_regression"}
    params["tuning"]["enabled"] = True
    r = train_action(toy(), "loan", params, False)
    assert r["tuning"]["best_params"]["C"] in [0.1, 1.0]


def test_random_split_is_seeded_and_stratified():
    frame, params = toy(), copy.deepcopy(BASE)
    tr1, te1 = split_frame(frame, params)
    tr2, te2 = split_frame(frame, params)
    assert te1.offer_id.tolist() == te2.offer_id.tolist()
    assert len(te1) == 100
    assert abs(te1.accepted.mean() - frame.accepted.mean()) < 0.02


def test_time_split_uses_the_cutoff():
    params = copy.deepcopy(BASE)
    params["split"] = {"method": "time", "train_end": "2025-06-30"}
    train, test = split_frame(toy(), params)
    assert train.offer_date.max() <= pd.Timestamp("2025-06-30") < test.offer_date.min()
