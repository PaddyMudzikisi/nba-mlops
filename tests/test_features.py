import pandas as pd

from nba.features import build_action_frame, load_data


def params_for(path):
    return {"dataset": {
        "path": str(path),
        "columns": {"customer_key": "cid", "offer_date": "dt", "action": "offer", "accepted": "y",
                    "value": "usd", "propensity": "p"},
        "categorical": ["channel"], "numeric": ["tenure"]}}


def write(tmp_path, rows):
    path = tmp_path / "offers.csv"
    pd.DataFrame(rows, columns=["cid", "dt", "offer", "y", "usd", "p", "channel", "tenure"]).to_csv(path, index=False)
    return path


def test_columns_are_mapped_to_roles(tmp_path):
    path = write(tmp_path, [["SYN1", "2025-01-01", "loan", 1, 100, 0.5, "SMS", 3],
                            ["SYN2", "2025-01-02", "card", 0, 20, 0.3, "App", 9]])
    data = load_data(params_for(path))
    assert {"customer_key", "offer_date", "action", "accepted", "value", "propensity", "offer_id"} <= set(data)
    assert data.offer_id.tolist() == ["SYN1|2025-01-01|loan", "SYN2|2025-01-02|card"]
    assert list(data.channel.cat.categories) == ["App", "SMS"]
    assert build_action_frame(data, "loan").customer_key.tolist() == ["SYN1"]


def test_duplicate_offers_get_a_counter(tmp_path):
    row = ["SYN1", "2025-01-01", "loan", 1, 100, 0.5, "SMS", 3]
    data = load_data(params_for(write(tmp_path, [row, row])))
    assert data.offer_id.tolist() == ["SYN1|2025-01-01|loan", "SYN1|2025-01-01|loan|1"]


def test_optional_value_and_propensity_and_text_outcome(tmp_path):
    path = tmp_path / "offers.csv"
    pd.DataFrame({"cid": ["A", "B"], "dt": ["2025-01-01", "2025-01-02"], "offer": ["loan", "loan"],
                  "y": ["Yes", "no"], "channel": ["SMS", None], "tenure": [3, "x"]}).to_csv(path, index=False)
    params = params_for(path)
    params["dataset"]["columns"] = {"customer_key": "cid", "offer_date": "dt", "action": "offer", "accepted": "y"}
    data = load_data(params)
    assert data.accepted.tolist() == [1, 0]
    assert (data.value == 1.0).all() and data.propensity.isna().all()
    assert list(data.channel.cat.categories) == ["SMS", "missing"]
    assert data.tenure.isna().tolist() == [False, True]
