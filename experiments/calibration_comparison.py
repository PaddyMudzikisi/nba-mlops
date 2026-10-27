"""Calibration comparison behind params.yaml balance_classes (run inside the job image):
    docker compose run --rm --no-deps app python experiments/calibration_comparison.py
Trains each action four ways on the training window and reports test AUC,
mean predicted vs observed acceptance, and Brier score."""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from nba.config import load_params
from nba.features import load_data, build_action_frame, features
from nba.policy import calibrate
p = load_params(); d = load_data(p); X = features(p); seed = p["seed"]
end = pd.Timestamp(p["split"]["train_end"]); cal_start = end - pd.DateOffset(months=3) + pd.Timedelta(days=1)
def fit(tr, spw):
    pos = tr.accepted.sum()
    m = xgb.XGBClassifier(**p["xgboost"], scale_pos_weight=(len(tr)-pos)/pos if spw else 1.0,
        random_state=seed, n_jobs=p["n_jobs"], enable_categorical=True).fit(tr[X], tr.accepted)
    return m
logit = lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
print(f"{'action':26s} {'variant':14s} {'AUC':>6s} {'pred':>6s} {'obs':>6s} {'brier':>7s}")
for a in sorted(d.action.unique()):
    f = build_action_frame(d, a); tr, te = f[f.offer_date<=end], f[f.offer_date>end]
    out = {}
    m = fit(tr, True); w = m.get_params()["scale_pos_weight"]; raw = m.predict_proba(te[X])[:,1]
    out["spw raw"] = raw; out["spw odds-corr"] = calibrate(raw, w)
    out["no spw"] = fit(tr, False).predict_proba(te[X])[:,1]
    fit_tr, cal = tr[tr.offer_date<cal_start], tr[tr.offer_date>=cal_start]
    m2 = fit(fit_tr, True)
    lr = LogisticRegression(C=1e6).fit(logit(m2.predict_proba(cal[X])[:,1]).reshape(-1,1), cal.accepted)
    out["spw + Platt"] = lr.predict_proba(logit(m2.predict_proba(te[X])[:,1]).reshape(-1,1))[:,1]
    for k, q in out.items():
        print(f"{a:26s} {k:14s} {roc_auc_score(te.accepted,q):6.3f} {q.mean():6.3f} {te.accepted.mean():6.3f} {brier_score_loss(te.accepted,q):7.4f}")
    print(f"{'':26s} train acceptance rate {tr.accepted.mean():.3f}")
