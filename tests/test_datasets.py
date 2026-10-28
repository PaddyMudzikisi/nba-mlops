import pandas as pd

from nba.datasets import guess_schema, validate
from nba.privacy import scan


def frame():
    n = 60
    return pd.DataFrame({
        "client_id": [f"SYN{i}" for i in range(n)], "offer_date": ["2025-01-01"] * n,
        "product": ["loan", "card"] * (n // 2), "target_accepted": [0, 1, 0] * (n // 3),
        "region": ["a", "b"] * (n // 2), "gender": ["F", "M"] * (n // 2), "tenure": range(n),
        "consent_marketing": [1] * n})


def test_guess_schema_finds_roles_and_keeps_protected_and_consent_out_of_features():
    s = guess_schema(frame())
    assert s["columns"]["customer_key"] == "client_id"
    assert s["columns"]["action"] == "product"
    assert s["columns"]["accepted"] == "target_accepted"
    assert s["protected"] == ["gender"] and "gender" not in s["categorical"]
    assert s["eligibility_column"] == "consent_marketing" and "consent_marketing" not in s["numeric"]
    assert validate(frame(), s) == []


def test_validate_reports_missing_roles_and_thin_offers():
    s = guess_schema(frame())
    s["columns"]["accepted"] = None
    assert any("accepted" in p for p in validate(frame(), s))
    small = frame().head(12)
    assert any("fewer than 10" in p for p in validate(small, guess_schema(small)))


def test_privacy_scan_blocks_personal_data_and_flags_column_names():
    f = frame()
    assert scan(f)["clean"]
    f["contact"] = ["someone@example.com"] + ["x"] * (len(f) - 1)
    f["full_name"] = "anon"
    result = scan(f)
    assert not result["clean"] and result["pattern_matches"]["contact"] == {"email": 1}
    assert result["sensitive_columns"] == ["full_name"]


def test_delete_removes_the_folder_and_protects_the_reference_dataset(tmp_path, monkeypatch):
    import pytest
    import yaml

    from nba import datasets, deploy
    monkeypatch.setattr(datasets, "DATASETS", tmp_path)
    monkeypatch.setattr(deploy, "current", lambda: None)
    for name, path in [("extra", "data/datasets/extra/data.csv"), ("reference", "data/raw/mydata.csv")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "schema.yaml").write_text(yaml.safe_dump({"name": name, "path": path}))
    datasets.delete("extra")
    assert not (tmp_path / "extra").exists()
    with pytest.raises(ValueError, match="reference"):
        datasets.delete("reference")
    with pytest.raises(ValueError, match="no dataset"):
        datasets.delete("missing")
