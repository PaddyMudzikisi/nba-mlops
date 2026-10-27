"""Configure a training run (split, features, model, tuning) and start it as a
background job. The choices become a parameter set that the run records, so
the run can be regenerated exactly."""
import copy
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yaml

from nba import datasets, jobs, models
from nba.config import REPO, load_params
from nba.explore import load
from nba.portal import common

common.page("3 · Train")
name = common.active_dataset("train")
if name is None:
    st.stop()
base = load_params()
schema = datasets.params_for(name, base)["dataset"]
_, data = load(name)
st.caption(f"Dataset **{name}** · {len(data):,} offers · {data.action.nunique()} offers to model · "
           f"{data.offer_date.min().date()} to {data.offer_date.max().date()}")

st.subheader("Split")
method = st.radio("Method", ["time", "random"], horizontal=True,
                  format_func={"time": "Out-of-time (train on earlier offers, test on later ones)",
                               "random": "Random (percentage held out)"}.get)
split = {"method": method}
if method == "time":
    dates = data.offer_date.sort_values()
    default = (pd.Timestamp(base["split"]["train_end"]) if name == base["dataset"].get("name")
               else dates.iloc[int(len(dates) * 0.7)].normalize())
    cutoff = st.date_input("Train on offers up to and including", default.date(),
                           min_value=dates.iloc[0].date(), max_value=dates.iloc[-1].date())
    split["train_end"] = str(cutoff)
    n_train = int((data.offer_date <= pd.Timestamp(cutoff)).sum())
else:
    pct = st.slider("Test set (% of each offer's rows)", 10, 50, int(base["split"].get("test_size", 0.2) * 100), 5)
    split["test_size"] = pct / 100
    n_train = int(len(data) * (1 - split["test_size"]))
    st.warning("A random split mixes earlier and later offers, so the model is tested partly on the past it "
               "learned from. Scores are usually optimistic compared with an out-of-time split; use it for "
               "comparison, and prefer out-of-time results for conclusions.")
n_test = len(data) - n_train
st.write(f"Train **{n_train:,}** offers ({n_train / len(data):.0%}) · test **{n_test:,}** ({n_test / len(data):.0%})")
if min(n_train, n_test) < 200:
    st.error("Too few offers on one side of the split.")

st.subheader("Features")
c1, c2 = st.columns(2)
categorical = c1.multiselect("Categorical", schema["categorical"], schema["categorical"])
numeric = c2.multiselect("Numeric", schema["numeric"], schema["numeric"])
if schema.get("protected"):
    st.caption(f"Protected attributes ({', '.join(schema['protected'])}) are never used as features; "
               "they are kept for the fairness check.")

st.subheader("Model")
kind = st.selectbox("Model family", models.MODEL_TYPES,
                    format_func={"xgboost": "XGBoost (gradient-boosted trees)", "random_forest": "Random forest",
                                 "logistic_regression": "Logistic regression"}.get)
hyper = dict(base.get(kind, {}))
c = st.columns(4)
if kind == "xgboost":
    hyper["n_estimators"] = c[0].number_input("Trees", 20, 3000, int(hyper.get("n_estimators", 300)), 20)
    hyper["max_depth"] = c[1].number_input("Max depth", 1, 12, int(hyper.get("max_depth", 4)))
    hyper["learning_rate"] = c[2].number_input("Learning rate", 0.005, 0.5, float(hyper.get("learning_rate", 0.05)),
                                               0.005, format="%.3f")
    hyper["min_child_weight"] = c[3].number_input("Min child weight", 0.0, 50.0, float(hyper.get("min_child_weight", 5)))
    hyper["subsample"] = c[0].slider("Row subsample", 0.3, 1.0, float(hyper.get("subsample", 0.8)), 0.05)
    hyper["colsample_bytree"] = c[1].slider("Column subsample", 0.3, 1.0, float(hyper.get("colsample_bytree", 0.8)), 0.05)
    hyper["reg_lambda"] = c[2].number_input("L2 regularisation", 0.0, 100.0, float(hyper.get("reg_lambda", 1.0)))
elif kind == "random_forest":
    hyper["n_estimators"] = c[0].number_input("Trees", 20, 3000, int(hyper.get("n_estimators", 300)), 20)
    depth = c[1].number_input("Max depth (0 = unlimited)", 0, 50, int(hyper.get("max_depth") or 0))
    hyper["max_depth"] = None if depth == 0 else int(depth)
    hyper["min_samples_leaf"] = c[2].number_input("Min samples per leaf", 1, 200, int(hyper.get("min_samples_leaf", 5)))
else:
    hyper["C"] = c[0].select_slider("Inverse regularisation C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                    float(hyper.get("C", 1.0)))
balance = st.checkbox("Weight classes to balance acceptances and rejections", value=bool(base.get("balance_classes")),
                      help="Off by default: weighting inflated the probabilities that the expected-value ranking "
                           "compares across offers (see experiments/calibration_comparison.py).")

st.subheader("Tuning")
tune_cfg = copy.deepcopy(base.get("tuning", {}))
tune_cfg["enabled"] = st.checkbox("Tune hyperparameters with a grid search",
                                  help="Cross-validation on the training data only; the test set is never used. "
                                       "Time-ordered folds for an out-of-time split.")
if tune_cfg["enabled"]:
    c1, c2 = st.columns(2)
    tune_cfg["cv_folds"] = c1.number_input("Folds", 2, 10, int(tune_cfg.get("cv_folds", 3)))
    tune_cfg["scoring"] = c2.selectbox("Score to optimise", ["average_precision", "roc_auc", "neg_brier_score"])
    grid_text = st.text_area("Grid (YAML: parameter: [values])", yaml.safe_dump(tune_cfg["grid"][kind]), height=110)
    try:
        grid = yaml.safe_load(grid_text) or {}
        assert isinstance(grid, dict) and all(isinstance(v, list) and v for v in grid.values())
        tune_cfg["grid"][kind] = grid
        n_fits = tune_cfg["cv_folds"] * pd.Series([len(v) for v in grid.values()]).prod()
        st.caption(f"{n_fits} fits per offer.")
    except Exception:  # noqa: BLE001
        st.error("The grid must be YAML mapping each parameter to a list of values.")
        st.stop()

st.subheader("Start")
label = st.text_input("Label for this run", f"{kind} · {method} split")
params = copy.deepcopy(base)
params["dataset"] = {**schema, "categorical": categorical, "numeric": numeric}
params.update({"split": split, "model": {"type": kind}, kind: hyper, "balance_classes": balance, "tuning": tune_cfg})
with st.expander("Parameter set to be recorded"):
    st.code(yaml.safe_dump({k: params[k] for k in ["split", "model", kind, "balance_classes", "tuning"]},
                           sort_keys=False), language="yaml")
ready = bool(categorical + numeric) and min(n_train, n_test) >= 200
if st.button("Start training", type="primary", disabled=not ready):
    folder = REPO / "logs" / "jobs" / "params"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{name}.yaml"
    path.write_text(yaml.safe_dump(params, sort_keys=False))
    jobs.start("train", ["--params", path.relative_to(REPO)], label=f"{name} · {label}")
    st.success("Training started. Follow it below; results appear on the Results page when it finishes.")

common.job_panel(match="train", title="Training jobs")
