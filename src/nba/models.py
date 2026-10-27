"""The model families the pipeline can train, behind one interface: build,
tune, explain, and log/load through MLflow."""
import mlflow
import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_TYPES = ["xgboost", "random_forest", "logistic_regression"]


def model_type(params):
    # Runs recorded before model selection existed were XGBoost.
    return params.get("model", {}).get("type", "xgboost")


def build(params, categorical, numeric, scale_pos_weight):
    kind = model_type(params)
    hyper = params.get(kind, {})
    seed, n_jobs = params["seed"], params["n_jobs"]
    if kind == "xgboost":
        return xgb.XGBClassifier(**hyper, scale_pos_weight=scale_pos_weight, random_state=seed, n_jobs=n_jobs,
                                 enable_categorical=True)
    weight = None if scale_pos_weight == 1.0 else {0: 1.0, 1: scale_pos_weight}
    numeric_prep = SimpleImputer(strategy="median")
    if kind == "logistic_regression":
        numeric_prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                              ("num", numeric_prep, numeric)])
    if kind == "random_forest":
        clf = RandomForestClassifier(**hyper, class_weight=weight, random_state=seed, n_jobs=n_jobs)
    elif kind == "logistic_regression":
        clf = LogisticRegression(**hyper, class_weight=weight, random_state=seed, max_iter=5000)
    else:
        raise ValueError(f"unknown model type {kind!r}; expected one of {MODEL_TYPES}")
    return Pipeline([("prep", prep), ("clf", clf)])


def tune(model, params, X, y, dates):
    """Grid search on the training data only. Time-ordered folds for a time split,
    stratified folds for a random split. Returns the refitted best model."""
    cfg = params.get("tuning", {})
    if not cfg.get("enabled", False):
        return model, {}
    kind = model_type(params)
    prefix = "" if kind == "xgboost" else "clf__"
    grid = {prefix + k: v for k, v in cfg["grid"][kind].items()}
    folds = cfg.get("cv_folds", 3)
    if params["split"].get("method", "time") == "time":
        order = np.argsort(dates.to_numpy(), kind="stable")
        X, y = X.iloc[order], y.iloc[order]
        cv = TimeSeriesSplit(n_splits=folds)
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=params["seed"])
    search = GridSearchCV(model, grid, scoring=cfg.get("scoring", "average_precision"), cv=cv, n_jobs=1,
                          refit=True)
    search.fit(X, y)
    best = {k.removeprefix(prefix): v for k, v in search.best_params_.items()}
    return search.best_estimator_, {"best_params": best, "best_cv_score": float(search.best_score_),
                                    "scoring": cfg.get("scoring", "average_precision")}


def choose_threshold(model, params, X, y):
    """Decision threshold maximising F1 on out-of-fold training predictions (the
    test set is never used to choose it)."""
    cv = StratifiedKFold(n_splits=params.get("tuning", {}).get("cv_folds", 3), shuffle=True,
                         random_state=params["seed"])
    oof = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]
    candidates = np.unique(np.quantile(oof, np.linspace(0.01, 0.99, 99)))
    scores = [f1_score(y, oof >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, proba)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, proba)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def importance(model, params, X, y):
    """Mean |SHAP| for XGBoost (native), permutation importance on ROC-AUC otherwise."""
    if model_type(params) == "xgboost":
        contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
        values = np.abs(contribs[:, :-1]).mean(axis=0)  # last column is the bias term
        method = "mean |SHAP|"
    else:
        result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5,
                                        random_state=params["seed"], n_jobs=1)
        values = np.clip(result.importances_mean, 0, None)
        method = "permutation (ROC-AUC drop)"
    return method, dict(sorted(zip(X.columns, np.round(values, 6).tolist()), key=lambda kv: -kv[1]))


def log_model(model, params, registered_name):
    if model_type(params) == "xgboost":
        return "xgboost", mlflow.xgboost.log_model(model, artifact_path="model",
                                                   registered_model_name=registered_name)
    return "sklearn", mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=registered_name)


def load_model(uri, flavor="xgboost"):
    return mlflow.xgboost.load_model(uri) if flavor == "xgboost" else mlflow.sklearn.load_model(uri)
