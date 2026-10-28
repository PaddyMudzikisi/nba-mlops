"""Train one propensity model per action and record everything needed to
regenerate it: data hash, git commit, image id, the full effective params and
test predictions.

  python -m nba.train                    # params.yaml
  python -m nba.train --params run.yaml  # a parameter set written by the portal
"""
import argparse
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
from sklearn.model_selection import train_test_split

from nba import models
from nba.config import REPO, load_params
from nba.features import build_action_frame, dataset_sha256, features, load_data


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


# What a model depends on besides its data and parameters, which are pinned
# separately (data_sha256, the logged params.yaml). Datasets added and reports
# written from the portal therefore do not make a run "dirty".
CODE_PATHS = ["src", "requirements.lock", "Dockerfile"]


def code_changes():
    return [line[2:].strip() for line in git("status", "--porcelain", "--", *CODE_PATHS).splitlines()]


def provenance(params):
    changes = code_changes()
    return {
        "git_commit": git("rev-parse", "HEAD") or "none",
        "git_dirty": str(bool(changes)),
        "git_changes": ", ".join(changes)[:500] or "none",
        "data_sha256": dataset_sha256(REPO / params["dataset"]["path"]),
        "image_id": os.environ.get("IMAGE_ID", "unknown"),
        "platform": platform.machine(),
        "python": platform.python_version(),
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def split_frame(frame, params):
    """Out-of-time split on a date (default), or a seeded random split stratified on the outcome."""
    split = params["split"]
    if split.get("method", "time") == "time":
        in_train = frame.offer_date <= pd.Timestamp(split["train_end"])
        return frame[in_train], frame[~in_train]
    stratify = frame.accepted if frame.accepted.value_counts().min() >= 2 else None
    train, test = train_test_split(frame, test_size=split["test_size"], stratify=stratify,
                                   random_state=params["seed"])
    return train.sort_values("offer_id"), test.sort_values("offer_id")


def test_rows(data, params):
    """The held-out offers of a run, across all actions: the test period for a
    time split, the union of each action's held-out rows for a random split."""
    if params["split"].get("method", "time") == "time":
        in_train = data.offer_date <= pd.Timestamp(params["split"]["train_end"])
        return data[in_train], data[~in_train].reset_index(drop=True)
    parts = [split_frame(build_action_frame(data, a), params) for a in sorted(data.action.unique())]
    return (pd.concat([p[0] for p in parts]).reset_index(drop=True),
            pd.concat([p[1] for p in parts]).reset_index(drop=True))


def train_action(data, action, params, with_diagnostics=True):
    frame = build_action_frame(data, action)
    X = features(params)
    train, test = split_frame(frame, params)
    y_train = train.accepted
    pos = int(y_train.sum())
    # Runs recorded before balance_classes existed were trained balanced.
    weight = (len(y_train) - pos) / pos if params.get("balance_classes", True) else 1.0
    model = models.build(params, params["dataset"]["categorical"], params["dataset"]["numeric"], weight)
    model, tuning = models.tune(model, params, train[X], y_train, train.offer_date)
    if not tuning:
        model.fit(train[X], y_train)
    out = {"model": model, "train": train, "test": test, "proba": model.predict_proba(test[X])[:, 1],
           "tuning": tuning, "scale_pos_weight": weight}
    if with_diagnostics:
        out["threshold"] = models.choose_threshold(model, params, train[X], y_train)
        out["proba_train"] = model.predict_proba(train[X])[:, 1]
    return out


def fit_action(data, action, params):
    r = train_action(data, action, params, with_diagnostics=False)
    return r["model"], r["train"], r["test"], r["proba"]


def registered_name(params, action):
    name = params["dataset"].get("name")
    return f"nba_{name}_{action}" if name else f"nba_{action}"


def flat_params(params):
    kind = models.model_type(params)
    split = params["split"]
    out = {"seed": params["seed"], "dataset": params["dataset"].get("name", "default"), "model_type": kind,
           "split_method": split.get("method", "time"), "balance_classes": params.get("balance_classes", True),
           "tuning": params.get("tuning", {}).get("enabled", False), "n_features": len(features(params))}
    out.update({"train_end": split["train_end"]} if out["split_method"] == "time" else {"test_size": split["test_size"]})
    out.update({f"{kind}_{k}": v for k, v in params.get(kind, {}).items()})
    return out


def run(params, run_name="train"):
    data = load_data(params)
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=run_name) as parent:
        prov = provenance(params)
        mlflow.set_tags({**prov, "dataset": params["dataset"].get("name", "default"),
                         "model_type": models.model_type(params),
                         "split_method": params["split"].get("method", "time")})
        mlflow.log_dict(params, "params.yaml")
        mlflow.log_params(flat_params(params))
        if prov["git_dirty"] == "True":
            print("WARNING: uncommitted changes; this run cannot be tied to an exact commit.")

        policy = {
            "train_run_id": parent.info.run_id,
            "features": features(params),
            "categories": {col: data[col].cat.categories.tolist() for col in params["dataset"]["categorical"]},
            "eligibility_column": params["dataset"].get("eligibility_column"),
            "model_type": models.model_type(params),
            "actions": {},
        }
        X = features(params)
        for action in sorted(data.action.unique()):
            with mlflow.start_run(run_name=action, nested=True) as child:
                r = train_action(data, action, params)
                train, test = r["train"], r["test"]
                test_m = models.classification_metrics(test.accepted, r["proba"], r["threshold"])
                train_m = models.classification_metrics(train.accepted, r["proba_train"], r["threshold"])
                metrics = {**test_m, **{f"train_{k}": v for k, v in train_m.items()},
                           "threshold": r["threshold"], "n_train": len(train), "n_test": len(test),
                           "pos_rate_train": train.accepted.mean(), "pos_rate_test": test.accepted.mean(),
                           "baseline_accuracy": max(test.accepted.mean(), 1 - test.accepted.mean())}
                mlflow.set_tags({"action": action, **prov})
                mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
                if r["tuning"]:
                    mlflow.log_dict(r["tuning"], "tuning.json")
                method, imp = models.importance(r["model"], params, test[X], test.accepted)
                mlflow.log_dict({"method": method, "importance": imp}, "importance.json")
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "test_predictions.csv"
                    pd.DataFrame({"offer_id": test.offer_id, "proba": r["proba"]}).to_csv(
                        path, index=False, float_format="%.17g")
                    mlflow.log_artifact(str(path))
                flavor, info = models.log_model(r["model"], params, registered_name(params, action))
                policy["actions"][action] = {
                    "value": float(train.value.median()),
                    "scale_pos_weight": r["scale_pos_weight"],
                    "model_uri": f"runs:/{child.info.run_id}/model",
                    "model_version": str(info.registered_model_version),
                    "registered_model": registered_name(params, action),
                    "flavor": flavor,
                }
                print(f"{action:26s} ROC-AUC {test_m['roc_auc']:.3f} (train {train_m['roc_auc']:.3f})  "
                      f"PR-AUC {test_m['pr_auc']:.3f}  F1 {test_m['f1']:.3f}  test n={len(test):,}")

        # Everything the scoring step needs to turn these models into a policy.
        mlflow.log_dict(policy, "policy.json")
        print(f"\nParent run id: {parent.info.run_id}")
        return parent.info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="parameter set to train with (default: params.yaml)")
    args = parser.parse_args()
    run(load_params(args.params) if args.params else load_params())


if __name__ == "__main__":
    main()
