"""Upload a CSV, check it for personal data, map its columns to the pipeline's
roles, clean it and register it as a DVC-versioned dataset; or clean a
registered dataset into a new one."""
import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from nba import datasets
from nba.config import REPO
from nba.portal import cleaning_ui, common
from nba.privacy import scan

common.page("1 · Data")
common.active_dataset("data")
registered, upload, reclean = st.tabs(["Registered datasets", "Upload a new dataset", "Clean a registered dataset"])

with registered:
    rows = []
    for d in datasets.list_datasets():
        path = REPO / d["path"]
        n = sum(1 for _ in open(path)) - 1 if path.exists() else None
        rows.append({"name": d["name"], "description": d.get("description", ""), "offers (rows)": n,
                     "features": len(d["categorical"]) + len(d["numeric"]), "path": d["path"],
                     "cleaned": "yes" if d.get("cleaning") else "", "created": d.get("created", "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Choose the dataset to work with in the sidebar; every page uses it.")

with reclean:
    names = [d["name"] for d in datasets.list_datasets()]
    source = st.selectbox("Dataset to clean", names, index=names.index(st.session_state["active_dataset"])
                          if st.session_state.get("active_dataset") in names else 0)
    schema = datasets.load_schema(source)
    path = REPO / schema["path"]
    original = pd.read_csv(path) if path.exists() else None
    if original is None:
        st.error(f"{schema['path']} is not on this machine; run `dvc pull` first.")
    else:
        st.caption(f"The cleaned data is saved as a new dataset, so {source!r} and the runs trained on it are unchanged.")
        base = {k: schema.get(k) for k in ["columns", "categorical", "numeric", "protected", "eligibility_column",
                                           "date_format"]}
        cleaned, new_schema, recipe, log = cleaning_ui.controls(original, base, "reclean")
        new_schema["cleaning"] = {"source": source, "rows_in": len(original), "rows_out": len(cleaned),
                                  "recipe": recipe, "log": log}
        problems = datasets.validate(cleaned, new_schema)
        for p in problems:
            st.error(p)
        c1, c2 = st.columns(2)
        new_name = c1.text_input("New dataset name", f"{source}_clean"[:40])
        description = c2.text_input("Description", f"{source} cleaned", key="reclean-desc")
        exists = new_name in names
        if exists:
            st.error(f"A dataset called {new_name!r} already exists.")
        if st.button("Save the cleaned dataset", type="primary", key="reclean-save",
                     disabled=bool(problems) or exists or not datasets.NAME.match(new_name)):
            with st.spinner("Saving and versioning with DVC..."):
                saved, out = datasets.save(new_name, cleaned.to_csv(index=False).encode(), new_schema, description)
            st.session_state["active_dataset"] = new_name
            st.success(f"Saved {saved['path']} and versioned it with DVC. It is now the active dataset.")
            with st.expander("DVC output"):
                st.code(out, language=None)
            st.page_link("pages/2_Explore.py", label="Next: explore it")

with upload:
    st.markdown("One row per offer made to a customer. Required: a pseudonymous customer key, the offer "
                "date, the offer (product), and whether it was accepted. Optional: the offer's value and "
                "the logged propensity. **Only anonymised, masked or synthetic data.**")
    file = st.file_uploader("CSV file", type="csv")
    if file is None:
        st.stop()
    raw = file.getvalue()
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file as CSV: {e}")
        st.stop()
    st.write(f"**{len(frame):,} rows × {frame.shape[1]} columns**")

    st.subheader("Privacy check")
    result = scan(frame)
    if not result["clean"]:
        st.error("Blocked: values that look like personal data were found. Remove or mask them and upload again.")
        st.table(pd.DataFrame([{"column": c, **hits} for c, hits in result["pattern_matches"].items()]).fillna(0))
        st.stop()
    st.success("No e-mail addresses, phone numbers, national ID numbers or long account-like numbers found.")
    if result["sensitive_columns"]:
        st.warning("These column names suggest personal data. They are removed unless you confirm they are safe.")
        keep = st.multiselect("Keep these columns (only if they hold no personal data)", result["sensitive_columns"])
        frame = frame.drop(columns=[c for c in result["sensitive_columns"] if c not in keep])

    st.subheader("Map the columns")
    guess = datasets.guess_schema(frame)
    cols = list(frame.columns)
    none = "(none)"
    labels = {"customer_key": "Customer key (pseudonymous)", "offer_date": "Offer date", "action": "Offer / product",
              "accepted": "Accepted (0/1 or yes/no)", "value": "Offer value (optional)",
              "propensity": "Logged propensity (optional)"}
    roles = {}
    c = st.columns(3)
    for i, role in enumerate(datasets.ROLES):
        options = [none] + cols
        g = guess["columns"].get(role)
        roles[role] = c[i % 3].selectbox(labels[role], options, index=options.index(g) if g in options else 0,
                                         key=f"role-{role}")
    roles = {k: (None if v == none else v) for k, v in roles.items()}
    left = [x for x in cols if x not in roles.values()]
    c1, c2 = st.columns(2)
    categorical = c1.multiselect("Categorical features", left, [x for x in guess["categorical"] if x in left])
    numeric = c2.multiselect("Numeric features", [x for x in left if x not in categorical],
                             [x for x in guess["numeric"] if x in left and x not in categorical])
    rest = [x for x in left if x not in categorical + numeric]
    c1, c2, c3 = st.columns(3)
    protected = c1.multiselect("Protected attributes (fairness check only, never features)", rest,
                               [x for x in guess["protected"] if x in rest])
    elig_opts = [none] + [x for x in rest if x not in protected]
    eligibility = c2.selectbox("Eligibility column (1 = may be offered, e.g. consent)", elig_opts,
                               index=elig_opts.index(guess["eligibility_column"])
                               if guess["eligibility_column"] in elig_opts else 0)
    date_format = c3.text_input("Date format (blank = automatic)", placeholder="%d/%m/%Y")
    unused = [x for x in rest if x not in protected and x != eligibility]
    if unused:
        st.caption(f"Not used: {', '.join(unused)}")

    schema = {"columns": roles, "categorical": categorical, "numeric": numeric, "protected": protected,
              "eligibility_column": None if eligibility == none else eligibility,
              "date_format": date_format or None}

    st.subheader("Clean")
    rows_in = len(frame)
    frame, schema, recipe, log = cleaning_ui.controls(frame, schema, "upload")
    schema["cleaning"] = {"rows_in": rows_in, "rows_out": len(frame), "recipe": recipe, "log": log}

    st.subheader("Check")
    problems = datasets.validate(frame, schema)
    if problems:
        for p in problems:
            st.error(p)
    else:
        y = frame[roles["accepted"]]
        from nba.features import to_binary
        summary = (frame.assign(_y=to_binary(y)).groupby(roles["action"])._y.agg(["size", "mean"])
                   .rename(columns={"size": "offers", "mean": "acceptance rate"}))
        st.success("The mapping is valid.")
        st.dataframe(summary.style.format({"acceptance rate": "{:.1%}", "offers": "{:,}"}))
        if not roles["propensity"]:
            st.info("No logged propensity: policy evaluation will use model-based (direct method) estimates only.")
        if not roles["value"]:
            st.info("No offer value: every acceptance counts as 1, so the policy recommends the most likely offer.")

    st.subheader("Save")
    c1, c2 = st.columns(2)
    default = re.sub(r"[^a-z0-9_]+", "_", Path(file.name).stem.lower()).strip("_") or "dataset"
    if not re.match(r"^[a-z]", default):
        default = "d_" + default
    name = c1.text_input("Dataset name", default[:40])
    description = c2.text_input("Description", "")
    confirmed = st.checkbox("I confirm this dataset is anonymised, masked or synthetic and contains no client "
                            "personal data.")
    exists = name in [d["name"] for d in datasets.list_datasets()]
    if exists:
        st.error(f"A dataset called {name!r} already exists.")
    if st.button("Save and version the dataset", type="primary",
                 disabled=bool(problems) or not confirmed or exists or not datasets.NAME.match(name)):
        with st.spinner("Saving and versioning with DVC..."):
            saved, log = datasets.save(name, frame.to_csv(index=False).encode(), schema, description)
        st.session_state["active_dataset"] = name
        st.success(f"Saved {saved['path']} and versioned it with DVC. It is now the active dataset.")
        with st.expander("DVC output"):
            st.code(log, language=None)
        st.page_link("pages/2_Explore.py", label="Next: explore it")
