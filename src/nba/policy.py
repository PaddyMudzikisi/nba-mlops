"""Next-best-action policy: score every action for a customer and recommend the
eligible action with the highest expected value (calibrated probability x value)."""
import mlflow
import numpy as np
import pandas as pd

from nba.models import load_model


def calibrate(proba, scale_pos_weight):
    """Undo scale_pos_weight. Weighting positives by w multiplies the fitted odds
    by w, so divide the odds back by w; without this, probabilities are inflated
    by a different amount for each action and cannot be compared across actions."""
    return proba / (proba + (1 - proba) * scale_pos_weight)


class Policy:
    def __init__(self, meta, models):
        self.meta = meta
        self.models = models
        self.actions = sorted(meta["actions"])

    @classmethod
    def from_run(cls, run_id):
        meta = mlflow.artifacts.load_dict(f"runs:/{run_id}/policy.json")
        models = {a: load_model(m["model_uri"], m.get("flavor", "xgboost")) for a, m in meta["actions"].items()}
        return cls(meta, models)

    def encode(self, frame):
        X = frame[self.meta["features"]].copy()
        for col, categories in self.meta["categories"].items():
            X[col] = pd.Categorical(X[col].astype("string").fillna("missing"), categories=categories)
        return X

    def score(self, frame):
        """Calibrated acceptance probability and expected value, one column per action."""
        X = self.encode(frame)
        proba = pd.DataFrame(index=frame.index, columns=self.actions, dtype=float)
        for a in self.actions:
            raw = self.models[a].predict_proba(X)[:, 1]
            proba[a] = calibrate(raw, self.meta["actions"][a]["scale_pos_weight"])
        values = pd.Series({a: self.meta["actions"][a]["value"] for a in self.actions})
        return proba, proba * values

    def recommend(self, frame):
        """Best action per row; NaN where the eligibility column (e.g. marketing consent) is not 1."""
        _, ev = self.score(frame)
        best = ev.idxmax(axis=1)
        # Policies recorded before the column was configurable used consent_marketing.
        col = self.meta.get("eligibility_column", "consent_marketing")
        eligible = frame[col] == 1 if col and col in frame else True
        return best.where(eligible, np.nan)
