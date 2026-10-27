"""Scoring service: POST a customer's features, get the next best action.

Serves the policy of one training run (SERVE_RUN_ID) and appends every decision,
with the run and model versions that made it, to a JSON-lines decision log.

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
    consent_marketing: int = 1
    features: dict[str, str | float | int]


def create_app(policy, log_path):
    app = FastAPI(title="NBA scoring service")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    versions = {a: m["model_version"] for a, m in policy.meta["actions"].items()}

    @app.get("/health")
    def health():
        return {"status": "ok", "train_run_id": policy.meta["train_run_id"], "model_versions": versions}

    @app.post("/recommend")
    def recommend(customer: Customer):
        missing = sorted(set(policy.meta["features"]) - set(customer.features))
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        for col, categories in policy.meta["categories"].items():
            if customer.features[col] not in categories:
                raise HTTPException(422, f"unknown {col} {customer.features[col]!r}; expected one of {categories}")

        frame = pd.DataFrame([{**customer.features, "consent_marketing": customer.consent_marketing}])
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
            "reason": "no marketing consent" if pd.isna(action) else "highest expected value",
            "ranking": ranking,
            "train_run_id": policy.meta["train_run_id"],
            "model_versions": versions,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(decision) + "\n")
        return decision

    return app


def from_env():
    return create_app(Policy.from_run(os.environ["SERVE_RUN_ID"]),
                      os.environ.get("DECISION_LOG", "logs/decisions.jsonl"))
