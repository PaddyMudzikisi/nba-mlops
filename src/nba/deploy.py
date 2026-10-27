"""Deployment gate: a training run can be served only if it finished, was
trained from committed code, and passed the reproducibility test. Deploying
sets the registry alias `production` on its models and records the run in
deploy/current.json, which the scoring API loads."""
import json
from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from nba.config import REPO

DEPLOY = REPO / "deploy"
CURRENT = DEPLOY / "current.json"


def checks(run_id):
    """Gate checks as (name, passed, detail)."""
    client = MlflowClient()
    run = client.get_run(run_id)
    tags = run.data.tags
    repro = client.search_runs([run.info.experiment_id],
                               filter_string=f"tags.reproduces = '{run_id}'",
                               order_by=["attributes.start_time DESC"], max_results=1)
    evaluated = client.search_runs([run.info.experiment_id], filter_string=f"tags.evaluates = '{run_id}'",
                                   max_results=1)
    return [
        ("Training finished", run.info.status == "FINISHED", run.info.status),
        ("Trained from committed code", tags.get("git_dirty") == "False",
         f"commit {tags.get('git_commit', '?')[:7]}" + (" with uncommitted changes" if tags.get("git_dirty") == "True" else "")),
        ("Reproducibility test passed", bool(repro) and repro[0].data.tags.get("passed") == "True",
         "latest test passed" if repro and repro[0].data.tags.get("passed") == "True"
         else ("latest test failed" if repro else "not run yet")),
        ("Policy evaluated (recommended)", bool(evaluated), "done" if evaluated else "not run yet"),
    ]


def deploy(run_id, by="portal"):
    required = [c for c in checks(run_id) if not c[0].endswith("(recommended)")]
    failed = [name for name, ok, _ in required if not ok]
    if failed:
        raise ValueError("cannot deploy: " + "; ".join(failed))
    policy = mlflow.artifacts.load_dict(f"runs:/{run_id}/policy.json")
    client = MlflowClient()
    versions = {}
    for action, meta in policy["actions"].items():
        name = meta.get("registered_model", f"nba_{action}")
        client.set_registered_model_alias(name, "production", meta["model_version"])
        versions[action] = f"{name} v{meta['model_version']}"
    record = {"run_id": run_id, "deployed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "by": by, "models": versions}
    DEPLOY.mkdir(exist_ok=True)
    CURRENT.write_text(json.dumps(record, indent=2))
    with open(DEPLOY / "history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def current():
    return json.loads(CURRENT.read_text()) if CURRENT.exists() else None
