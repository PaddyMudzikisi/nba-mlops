import json

from fastapi.testclient import TestClient

from nba.serve import create_app


def client(toy_policy, tmp_path):
    return TestClient(create_app(toy_policy, tmp_path / "decisions.jsonl"))


def test_recommend_returns_ranked_actions_and_logs_decision(toy_policy, tmp_path):
    c = client(toy_policy, tmp_path)
    assert c.get("/health").json()["train_run_id"] == "test-run"
    resp = c.post("/recommend", json={"customer_key": "SYN1", "features": {"x": 3.0, "band": "a"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == body["ranking"][0]["action"]
    assert body["ranking"][0]["expected_value"] >= body["ranking"][1]["expected_value"]
    logged = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert logged[0]["decision_id"] == body["decision_id"]
    assert logged[0]["model_versions"] == {"loan": "1", "savings": "1"}


def test_no_consent_gives_no_action(toy_policy, tmp_path):
    resp = client(toy_policy, tmp_path).post(
        "/recommend", json={"customer_key": "SYN1", "consent_marketing": 0, "features": {"x": 3.0, "band": "a"}})
    assert resp.json()["action"] is None


def test_bad_input_is_rejected(toy_policy, tmp_path):
    c = client(toy_policy, tmp_path)
    assert c.post("/recommend", json={"customer_key": "SYN1", "features": {"x": 1.0}}).status_code == 422
    assert c.post("/recommend", json={"customer_key": "SYN1",
                                      "features": {"x": 1.0, "band": "zzz"}}).status_code == 422
