"""Scoring service: POST a customer's features, get the next best action.

Serves the policy of one training run: SERVE_RUN_ID if set, otherwise the run
recorded in deploy/current.json by the deployment gate (POST /reload picks up
a new deployment). Every decision is appended, with the run and model versions
that made it, to a JSON-lines decision log.

  uvicorn --factory nba.serve:from_env --host 0.0.0.0 --port 8000
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nba.policy import Policy


class Customer(BaseModel):
    customer_key: str  # pseudonymous key only; never a real customer identifier
    consent_marketing: int = 1  # value of the policy's eligibility column (e.g. marketing consent)
    features: dict[str, str | float | int]


def create_app(policy, log_path, loader=None):
    """loader() returns the policy to serve after POST /reload (None: reload unavailable)."""
    app = FastAPI(title="NBA scoring service")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"policy": policy}

    def versions():
        return {a: m["model_version"] for a, m in state["policy"].meta["actions"].items()}

    def serving():
        if state["policy"] is None:
            raise HTTPException(503, "no model deployed yet; deploy a run from the portal")
        return state["policy"]

    @app.get("/health")
    def health():
        if state["policy"] is None:
            return {"status": "no model deployed"}
        return {"status": "ok", "train_run_id": state["policy"].meta["train_run_id"], "model_versions": versions()}

    @app.get("/schema")
    def schema():
        p = serving()
        return {"features": p.meta["features"], "categories": p.meta["categories"],
                "eligibility_column": p.meta.get("eligibility_column", "consent_marketing")}

    @app.post("/reload")
    def reload():
        if loader is None:
            raise HTTPException(400, "this server was started with a fixed run")
        state["policy"] = loader()
        return health()

    @app.post("/recommend")
    def recommend(customer: Customer):
        policy = serving()
        missing = sorted(set(policy.meta["features"]) - set(customer.features))
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        for col, categories in policy.meta["categories"].items():
            if str(customer.features[col]) not in categories:
                raise HTTPException(422, f"unknown {col} {customer.features[col]!r}; expected one of {categories}")

        eligibility = policy.meta.get("eligibility_column", "consent_marketing") or "consent_marketing"
        frame = pd.DataFrame([{**customer.features, eligibility: customer.consent_marketing}])
        proba, ev = policy.score(frame)
        action = policy.recommend(frame).iloc[0]
        ranking = sorted(({"action": a, "probability": round(float(proba.at[0, a]), 6),
                           "expected_value": round(float(ev.at[0, a]), 4)} for a in policy.actions),
                         key=lambda r: -r["expected_value"])
        decision = {
            "decision_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_key": customer.customer_key,
            "action": None if pd.isna(action) else action,
            "reason": "not eligible (e.g. no marketing consent)" if pd.isna(action) else "highest expected value",
            "ranking": ranking,
            "train_run_id": policy.meta["train_run_id"],
            "model_versions": versions(),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(decision) + "\n")
        return decision

    return app


def from_env():
    log = os.environ.get("DECISION_LOG", "logs/decisions.jsonl")
    if os.environ.get("SERVE_RUN_ID"):
        return create_app(Policy.from_run(os.environ["SERVE_RUN_ID"]), log)

    def deployed():
        from nba.deploy import current
        record = current()
        return Policy.from_run(record["run_id"]) if record else None

    return create_app(deployed(), log, loader=deployed)
