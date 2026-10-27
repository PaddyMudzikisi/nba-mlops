"""Reproducibility test (proposal section 2.9).

Regenerate every model of a recorded training run from its recorded params,
then compare the regenerated test-set predictions with the originals.
Exits non-zero if any action differs by more than the stated tolerance.
"""
import argparse
import sys
import tempfile

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from nba.config import load_params
from nba.train import fit_action, load_data, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="parent training run to reproduce")
    run_id = parser.parse_args().run_id

    client = MlflowClient()
    parent = client.get_run(run_id)
    recorded = parent.data.tags
    current = provenance()

    with tempfile.TemporaryDirectory() as tmp:
        params = load_params(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="params.yaml", dst_path=tmp))
    tolerance = float(params["reproducibility"]["tolerance"])

    print("Artefact checks (recorded vs now):")
    for key in ["data_sha256", "git_commit", "image_id", "platform", "xgboost"]:
        status = "match" if recorded.get(key) == current[key] else "DIFFERS"
        print(f"  {key:12s} {status}")
    if recorded.get("data_sha256") != current["data_sha256"]:
        sys.exit("Dataset snapshot differs from the recorded one; check out the matching DVC version first.")

    children = client.search_runs([parent.info.experiment_id],
                                  filter_string=f"tags.mlflow.parentRunId = '{run_id}'")
    customers, offers = load_data()
    results = {}
    for child in children:
        action = child.data.tags["action"]
        with tempfile.TemporaryDirectory() as tmp:
            original = pd.read_csv(mlflow.artifacts.download_artifacts(
                run_id=child.info.run_id, artifact_path="test_predictions.csv", dst_path=tmp),
                float_precision="round_trip")
        _, _, test, proba = fit_action(customers, offers, action, params)
        regenerated = pd.DataFrame({"offer_id": test.offer_id.to_numpy(), "proba_new": proba})
        merged = original.merge(regenerated, on="offer_id", how="outer", validate="1:1")
        if merged.isna().any().any():
            results[action] = np.inf  # different test rows: not reproduced
        else:
            results[action] = float(np.abs(merged.proba - merged.proba_new).max())

    passed = all(diff <= tolerance for diff in results.values())
    print(f"\nPrediction comparison (tolerance {tolerance:g}):")
    for action, diff in sorted(results.items()):
        print(f"  {action:16s} max |diff| = {diff:.3g}  {'PASS' if diff <= tolerance else 'FAIL'}")

    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="reproduce"):
        mlflow.set_tags({"reproduces": run_id, "passed": str(passed), **current})
        mlflow.log_metrics({f"max_abs_diff_{a}": d for a, d in results.items() if np.isfinite(d)})

    print(f"\nReproducibility test: {'PASSED' if passed else 'FAILED'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
