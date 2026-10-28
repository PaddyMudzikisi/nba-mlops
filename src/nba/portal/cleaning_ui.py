"""Cleaning controls shared by the upload flow and the clean-a-registered-dataset tab."""
import pandas as pd
import streamlit as st

from nba import cleaning

FILL_TEXT = {"model": "leave to the model (fitted on training data only)", "median": "fill with the median",
             "mode": "fill with the most common value", "drop": "drop the rows"}


def controls(frame, schema, key):
    """Show the data quality profile and cleaning options; return the cleaned
    frame, its schema, the recipe and the log."""
    profile, duplicates = cleaning.quality(frame, schema)
    with st.expander(f"Data quality: {duplicates:,} duplicate rows · "
                     f"{int((profile.missing > 0).sum())} columns with missing values", expanded=True):
        st.dataframe(profile.style.format({"missing": "{:.1%}"}).background_gradient(subset=["missing"], cmap="Reds",
                                                                                     vmin=0, vmax=1),
                     hide_index=True, width="stretch")
        st.caption("Extra spellings: values that differ only by case or spaces, e.g. 'Mobile' and 'mobile '.")

    d = cleaning.DEFAULT
    recipe = {}
    c1, c2, c3 = st.columns(3)
    recipe["trim_text"] = c1.checkbox("Trim and standardise text", d["trim_text"], key=f"{key}-trim")
    recipe["drop_incomplete"] = c1.checkbox("Drop rows without a key, date, offer or outcome",
                                            d["drop_incomplete"], key=f"{key}-incomplete")
    recipe["drop_duplicates"] = c1.checkbox("Drop duplicate rows", d["drop_duplicates"], key=f"{key}-dups")
    recipe["drop_thin_offers"] = c1.checkbox("Drop offers with fewer than 10 acceptances or rejections",
                                             d["drop_thin_offers"], key=f"{key}-thin")

    dates = pd.to_datetime(frame[schema["columns"]["offer_date"]], format=schema.get("date_format"),
                           errors="coerce") if schema["columns"].get("offer_date") else pd.Series(dtype="datetime64[ns]")
    lo, hi = (dates.min(), dates.max()) if dates.notna().any() else (None, None)
    start = c2.date_input("Keep offers from", value=None, min_value=lo, max_value=hi, key=f"{key}-from",
                          help=f"Data runs {lo:%Y-%m-%d} to {hi:%Y-%m-%d}" if lo is not None else None)
    end = c2.date_input("Keep offers up to", value=None, min_value=lo, max_value=hi, key=f"{key}-to")
    recipe["date_from"] = start.isoformat() if start else None
    recipe["date_to"] = end.isoformat() if end else None
    sparse = c2.checkbox("Drop features with many missing values", key=f"{key}-sparse")
    share = c2.slider("... more than this share missing", 10, 90, 50, 5, format="%d%%", key=f"{key}-share",
                      disabled=not sparse)
    recipe["max_missing_share"] = share / 100 if sparse else None

    recipe["fill_numeric"] = c3.selectbox("Missing numeric features", cleaning.FILL_NUMERIC,
                                          format_func=FILL_TEXT.get, key=f"{key}-fillnum")
    recipe["fill_categorical"] = c3.selectbox("Missing categorical features", cleaning.FILL_CATEGORICAL,
                                              format_func=FILL_TEXT.get, key=f"{key}-fillcat")
    recipe["rare_category_min"] = int(c3.number_input("Group categories with fewer rows than this into 'other' "
                                                      "(0 = off)", 0, 10_000, 0, key=f"{key}-rare"))

    numeric = [c for c in schema["numeric"] if c in frame and pd.api.types.is_numeric_dtype(frame[c])
               and frame[c].nunique() > 2]
    c1, c2, c3 = st.columns(3)
    recipe["outliers"] = c1.radio("Outliers", cleaning.OUTLIERS, horizontal=True, key=f"{key}-out",
                                  format_func={"none": "keep", "cap": "cap", "drop": "drop rows"}.get)
    recipe["outlier_columns"] = c2.multiselect("in these numeric features", numeric,
                                               disabled=recipe["outliers"] == "none", key=f"{key}-outcols")
    tail = c3.slider("Tail beyond which a value is an outlier", 0.5, 5.0, 1.0, 0.5, format="%.1f%%",
                     disabled=recipe["outliers"] == "none", key=f"{key}-tail")
    recipe["outlier_quantiles"] = [round(tail / 100, 4), round(1 - tail / 100, 4)]
    if recipe["fill_numeric"] in ("median",) or recipe["fill_categorical"] == "mode" or recipe["outliers"] != "none":
        st.caption("Filling and outlier limits here are computed on the whole file, including the later offers "
                   "used for testing. Leaving missing values to the model avoids that, because its imputer is "
                   "fitted on the training data only.")

    cleaned, schema, log = cleaning.clean(frame, schema, recipe)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", f"{len(frame):,}")
    c2.metric("Rows after", f"{len(cleaned):,}", f"{len(cleaned) - len(frame):,}", delta_color="off")
    c3.metric("Features", len(schema["categorical"]) + len(schema["numeric"]))
    table = pd.DataFrame(log)
    if not table.empty:
        table["rows removed"] = table.rows_before - table.rows_after
        st.dataframe(table[["step", "rows removed", "rows_after", "detail"]].rename(columns={"rows_after": "rows left"}),
                     hide_index=True, width="stretch")
    return cleaned, schema, recipe, log
