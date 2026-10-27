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

from nba.config import PARAMS_PATH, REPO, load_params
from nba.features import build_action_frame, dataset_sha256, features, load_data


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def provenance(params):
    return {
        "git_commit": git("rev-parse", "HEAD") or "none",
        "git_dirty": str(bool(git("status", "--porcelain", "--untracked-files=no"))),
        "data_sha256": dataset_sha256(REPO / params["dataset"]["path"]),
        "image_id": os.environ.get("IMAGE_ID", "unknown"),
        "platform": platform.machine(),
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def fit_action(data, action, params):
    frame = build_action_frame(data, action)
    X = features(params)
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
    model.fit(train[X], y_train)
    proba = model.predict_proba(test[X])[:, 1]
    return model, train, test, proba


def mean_abs_shap(model, X):
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    values = np.abs(contribs[:, :-1]).mean(axis=0)  # last column is the bias term
    return dict(sorted(zip(X.columns, values.round(6).tolist()), key=lambda kv: -kv[1]))


def main():
    params = load_params()
    data = load_data(params)
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run(run_name="train") as parent:
        prov = provenance(params)
        mlflow.set_tags(prov)
        mlflow.log_artifact(str(PARAMS_PATH))
        mlflow.log_params({"seed": params["seed"], "train_end": params["split"]["train_end"],
                           **{f"xgb_{k}": v for k, v in params["xgboost"].items()}})
        if prov["git_dirty"] == "True":
            print("WARNING: uncommitted changes; this run cannot be tied to an exact commit.")

        policy = {
            "train_run_id": parent.info.run_id,
            "features": features(params),
            "categories": {col: data[col].cat.categories.tolist() for col in params["dataset"]["categorical"]},
            "actions": {},
        }
        for action in sorted(data.action.unique()):
            with mlflow.start_run(run_name=action, nested=True) as child:
                model, train, test, proba = fit_action(data, action, params)
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
                mlflow.log_dict(mean_abs_shap(model, test[features(params)]), "shap_mean_abs.json")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "test_predictions.csv"
                    pd.DataFrame({"offer_id": test.offer_id, "proba": proba}).to_csv(
                        path, index=False, float_format="%.17g")
                    mlflow.log_artifact(str(path))
                info = mlflow.xgboost.log_model(model, artifact_path="model",
                                                registered_model_name=f"nba_{action}")
                policy["actions"][action] = {
                    "value": float(train.value.median()),
                    "scale_pos_weight": model.get_params()["scale_pos_weight"],
                    "model_uri": f"runs:/{child.info.run_id}/model",
                    "model_version": str(info.registered_model_version),
                }
                print(f"{action:26s} ROC-AUC {metrics['roc_auc']:.3f}  PR-AUC {metrics['pr_auc']:.3f}  "
                      f"test n={metrics['n_test']:,}")

        # Everything the scoring step needs to turn these models into a policy.
        mlflow.log_dict(policy, "policy.json")
        print(f"\nParent run id: {parent.info.run_id}")


if __name__ == "__main__":
    main()
