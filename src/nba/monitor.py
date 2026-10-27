"""Drift monitoring: Population Stability Index (PSI) of each feature and each
model score, comparing the training window (reference) with the period after it
(current), overall and month by month.

PSI = sum over bins of (cur - ref) * ln(cur / ref). Common reading: < 0.1 stable,
0.1-0.25 moderate shift, > 0.25 significant shift (retrain recommended).
"""
import argparse
import json
import tempfile

import mlflow
import numpy as np
import pandas as pd

from nba.config import REPO, load_params
from nba.features import load_data
from nba.policy import Policy
from nba.train import provenance

REPORTS = REPO / "reports"
EPS = 1e-4  # floor for empty bins, so ln() stays finite


def psi(reference, current, bins=10):
    """PSI of a numeric or categorical series; numeric bins are reference deciles."""
    if isinstance(reference.dtype, pd.CategoricalDtype) or reference.dtype == object:
        ref = reference.value_counts(normalize=True)
        cur = current.value_counts(normalize=True).reindex(ref.index, fill_value=0)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = pd.Series(np.histogram(reference, edges)[0] / len(reference))
        cur = pd.Series(np.histogram(current, edges)[0] / len(current))
    ref, cur = ref.clip(lower=EPS), cur.clip(lower=EPS)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def status(value, warn, alert):
    return "alert" if value > alert else "warn" if value > warn else "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="parent training run whose models are monitored")
    run_id = parser.parse_args().run_id

    with tempfile.TemporaryDirectory() as tmp:
        params = load_params(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="params.yaml", dst_path=tmp))
    warn, alert = params["monitoring"]["psi_warn"], params["monitoring"]["psi_alert"]
    data = load_data(params)
    policy = Policy.from_run(run_id)
    proba, _ = policy.score(data)
    columns = {**{f: data[f] for f in policy.meta["features"]},
               **{f"score:{a}": proba[a] for a in policy.actions}}

    in_ref = data.offer_date <= pd.Timestamp(params["split"]["train_end"])
    month = data.offer_date.dt.to_period("M").astype(str)
    overall = {name: psi(s[in_ref], s[~in_ref]) for name, s in columns.items()}
    monthly = {m: {name: psi(s[in_ref], s[month == m]) for name, s in columns.items()}
               for m in sorted(month[~in_ref].unique())}

    report = {
        "train_run_id": run_id,
        "thresholds": {"warn": warn, "alert": alert},
        "psi": {name: {"value": v, "status": status(v, warn, alert)}
                for name, v in sorted(overall.items(), key=lambda kv: -kv[1])},
        "psi_monthly": monthly,
    }
    report["retrain_recommended"] = any(v["status"] == "alert" for v in report["psi"].values())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "drift.json").write_text(json.dumps(report, indent=2))

    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="monitor"):
        mlflow.set_tags({"monitors": run_id, "retrain_recommended": str(report["retrain_recommended"]),
                         **provenance(params)})
        mlflow.log_metrics({f"psi_{n.replace(':', '_')}": v for n, v in overall.items()})
        mlflow.log_dict(report, "drift.json")

    print(f"PSI, training window vs after (warn > {warn}, alert > {alert}):")
    for name, v in report["psi"].items():
        print(f"  {name:36s} {v['value']:.4f}  {v['status']}")
    worst = max(max(m.values()) for m in monthly.values())
    print(f"\nWorst single-month PSI: {worst:.4f}")
    print("Retrain recommended:", report["retrain_recommended"])


if __name__ == "__main__":
    main()
