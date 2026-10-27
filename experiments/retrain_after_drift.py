"""Does retraining close the outcome drift found by nba.monitor? (run inside the job image):
    docker compose run --rm --no-deps app python experiments/retrain_after_drift.py
Champion: trained up to split.train_end. Challenger: retrained with three more
months. Both are scored on the same later window, which neither saw in training.
"""
import copy

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from nba.config import load_params
from nba.features import load_data
from nba.train import fit_action

params = load_params()
data = load_data(params)
champion_end = pd.Timestamp(params["split"]["train_end"])
challenger_end = champion_end + pd.DateOffset(months=3)
challenger = copy.deepcopy(params)
challenger["split"]["train_end"] = str(challenger_end.date())

print(f"Scored on offers after {challenger_end.date()}; champion trained to {champion_end.date()}, "
      f"challenger to {challenger_end.date()}")
print(f"{'action':26s} {'model':10s} {'n':>5s} {'AUC':>6s} {'Brier':>7s} {'pred':>6s} {'obs':>6s} {'gap':>6s}")
for action in sorted(data.action.unique()):
    for name, p in [("champion", params), ("challenger", challenger)]:
        _, _, test, proba = fit_action(data, action, p)
        later = (test.offer_date > challenger_end).to_numpy()
        y, q = test.accepted[later], proba[later]
        print(f"{action:26s} {name:10s} {len(y):5d} {roc_auc_score(y, q):6.3f} {brier_score_loss(y, q):7.4f} "
              f"{q.mean():6.3f} {y.mean():6.3f} {y.mean() / q.mean() - 1:+6.0%}")
