import numpy as np
import pandas as pd

from nba.cleaning import clean, quality
from nba.datasets import guess_schema
from nba.privacy import scan

SCHEMA = {"columns": {"customer_key": "client_id", "offer_date": "offer_date", "action": "offer_name",
                      "accepted": "took_up", "value": None, "propensity": None},
          "categorical": ["channel"], "numeric": ["balance", "sparse"], "protected": [], "eligibility_column": None,
          "date_format": None}


def frame():
    n = 100
    return pd.DataFrame({
        "client_id": [f"SYN{i}" for i in range(n)],
        "offer_date": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "offer_name": ["Loan", "loan ", "Card", "Card"] * (n // 4),
        "took_up": ["yes", "no"] * (n // 2),
        "channel": ["Mobile", "mobile", "Branch", " branch"] * (n // 4),
        "balance": list(range(n - 1)) + [10_000],
        "sparse": [np.nan] * (n - 5) + [1.0] * 5,
    })


def test_defaults_standardise_text_and_drop_incomplete_and_duplicate_rows():
    f = frame()
    f.loc[3, "took_up"] = "maybe"
    f.loc[4, "offer_date"] = "not a date"
    f = pd.concat([f, f.iloc[[0]]], ignore_index=True)
    out, schema, log = clean(f, SCHEMA, {})
    assert set(out.offer_name) == {"Loan", "Card"} and set(out.channel) == {"Mobile", "Branch"}
    assert len(out) == 98  # one unreadable outcome, one unreadable date, one duplicate
    assert [s["step"] for s in log] == ["trim and standardise text", "drop incomplete rows", "drop duplicate rows"]
    assert schema == {**SCHEMA, "categorical": ["channel"], "numeric": ["balance", "sparse"]}


def test_sparse_features_outliers_dates_and_rare_categories():
    recipe = {"max_missing_share": 0.5, "outliers": "cap", "outlier_columns": ["balance"],
              "outlier_quantiles": [0.0, 0.95], "date_from": "2025-01-11", "rare_category_min": 1000}
    out, schema, log = clean(frame(), SCHEMA, recipe)
    assert "sparse" not in out and schema["numeric"] == ["balance"]
    assert out.balance.max() < 10_000 and len(out) == 90
    assert set(out.channel) == {"other"}


def test_fill_and_drop_missing_features():
    f = frame()
    f.loc[0, "balance"] = np.nan
    f.loc[1, "channel"] = np.nan
    s = {**SCHEMA, "numeric": ["balance"]}
    filled, _, _ = clean(f, s, {"fill_numeric": "median", "fill_categorical": "mode"})
    assert filled[["balance", "channel"]].notna().all().all()
    dropped, _, _ = clean(f, s, {"fill_numeric": "drop", "fill_categorical": "drop"})
    assert len(dropped) == len(f) - 2


def test_quality_counts_duplicates_and_case_variants():
    profile, duplicates = quality(pd.concat([frame(), frame().head(2)]), SCHEMA)
    assert duplicates == 2
    assert profile.set_index("column").loc["channel", "extra spellings"] == 2


def test_offer_name_is_not_personal_data_and_offers_are_never_counts():
    f = frame().assign(products_held_count=1, customer_name="x")
    assert scan(f)["sensitive_columns"] == ["customer_name"]
    assert guess_schema(f.drop(columns="offer_name"))["columns"]["action"] is None
    assert guess_schema(f)["columns"]["action"] == "offer_name"
