"""Train one XGBoost propensity model per action and record everything needed
to regenerate it: data hash, git commit, image id, params and test predictions."""
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from nba.config import CUSTOMERS_PATH, OFFERS_PATH, PARAMS_PATH, REPO, load_params
from nba.features import FEATURES, build_action_frame, dataset_sha256


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def provenance():
    return {
        "git_commit": git("rev-parse", "HEAD") or "none",
        "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no"))),
        "data_sha256": dataset_sha256(CUSTOMERS_PATH, OFFERS_PATH),
        "image_id": os.environ.get("IMAGE_ID", "unknown"),
        "platform": platform.machine(),
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def load_data():
    return pd.read_parquet(CUSTOMERS_PATH), pd.read_parquet(OFFERS_PATH)


def fit_action(customers, offers, action, params):
    frame = build_action_frame(customers, offers, action)
    in_train = frame.offer_date <= pd.Timestamp(params["split"]["train_end"])
    train, test = frame[in_train], frame[~in_train]

    y_train = train.accepted
    pos = int(y_train.sum())
    model = xgb.XGBClassifier(
        **params["xgboost"],
        scale_pos_weight=(len(y_train) - pos) / pos,
        random_state=params["seed"],
        n_jobs=params["n_jobs"],
        enable_categorical=True,
    )
    model.fit(train[FEATURES], y_train)
    proba = model.predict_proba(test[FEATURES])[:, 1]
    return model, train, test, proba


def mean_abs_shap(model, X):
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    values = np.abs(contribs[:, :-1]).mean(axis=0)  # last column is the bias term
    return dict(sorted(zip(X.columns, values.round(6).tolist()), key=lambda kv: -kv[1]))


def main():
    params = load_params()
    customers, offers = load_data()
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run(run_name="train") as parent:
        prov = provenance()
        mlflow.set_tags(prov)
        mlflow.log_artifact(str(PARAMS_PATH))
        mlflow.log_params({"seed": params["seed"], "train_end": params["split"]["train_end"],
                           **{f"xgb_{k}": v for k, v in params["xgboost"].items()}})
        if prov["git_dirty"] == "True":
            print("WARNING: uncommitted changes; this run cannot be tied to an exact commit.")

        for action in params["actions"]:
            with mlflow.start_run(run_name=action, nested=True):
                model, train, test, proba = fit_action(customers, offers, action, params)
                y_test = test.accepted
                metrics = {
                    "roc_auc": roc_auc_score(y_test, proba),
                    "pr_auc": average_precision_score(y_test, proba),
                    "n_train": len(train),
                    "n_test": len(test),
                    "pos_rate_train": train.accepted.mean(),
                    "pos_rate_test": y_test.mean(),
                }
                mlflow.set_tags({"action": action, **prov})
                mlflow.log_metrics(metrics)
                mlflow.log_dict(mean_abs_shap(model, test[FEATURES]), "shap_mean_abs.json")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "test_predictions.csv"
                    pd.DataFrame({"offer_id": test.offer_id, "proba": proba}).to_csv(
                        path, index=False, float_format="%.17g")
                    mlflow.log_artifact(str(path))
                mlflow.xgboost.log_model(model, artifact_path="model",
                                         registered_model_name=f"nba_{action}")
                print(f"{action:16s} ROC-AUC {metrics['roc_auc']:.3f}  PR-AUC {metrics['pr_auc']:.3f}  "
                      f"test n={metrics['n_test']:,}")

        print(f"\nParent run id: {parent.info.run_id}")


if __name__ == "__main__":
    main()
