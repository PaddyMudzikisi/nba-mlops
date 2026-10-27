import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from nba.policy import Policy

ACTIONS = ["loan", "savings"]


@pytest.fixture
def toy_policy():
    """Two tiny models on random data: 'loan' acceptance rises with x, 'savings' is flat."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400), "band": pd.Categorical(rng.choice(["a", "b"], 400))})
    y = {"loan": (X.x + rng.normal(scale=0.5, size=400) > 0.5).astype(int),
         "savings": rng.integers(0, 2, 400)}
    models = {a: xgb.XGBClassifier(n_estimators=20, max_depth=2, enable_categorical=True,
                                   tree_method="hist", random_state=0).fit(X, y[a]) for a in ACTIONS}
    meta = {"train_run_id": "test-run", "features": ["x", "band"], "categories": {"band": ["a", "b"]},
            "actions": {"loan": {"value": 100.0, "scale_pos_weight": 1.0, "model_version": "1"},
                        "savings": {"value": 10.0, "scale_pos_weight": 1.0, "model_version": "1"}}}
    return Policy(meta, models)
