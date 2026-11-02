"""Feature engineering: create features from the dataset's columns (feature
creation, log transform, binning, binarisation, date split), preview them, and
rank the features by relevance to help select them. The recipe is saved with
the dataset; every training run records it and the scoring API applies it."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter
from sklearn.feature_selection import mutual_info_classif

from nba import datasets, engineer
from nba.explore import load
from nba.portal import common
from nba.report import INK2

common.page("3 · Features")
name = common.active_dataset("features")
if name is None:
    st.stop()
schema = datasets.load_schema(name)
recipe = schema.get("engineered") or []
params, data = load(name)
numeric = [c for c in schema["numeric"] if c in data]


def save(new_schema, message):
    datasets.save_schema(name, new_schema)
    load.clear()
    st.toast(message)
    st.rerun()


def acceptance_chart(values, title):
    """Acceptance rate by value or band of a feature, with 95% binomial intervals."""
    g = data.accepted.groupby(values.astype(object).where(values.notna(), "missing"), observed=True).agg(
        ["mean", "size"])
    g = g[g["size"] >= 20]
    half = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(g) + 1)))
    ax.barh([str(i) for i in g.index], g["mean"], xerr=half, color=common.PALETTE[0], height=0.7,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
    ax.axvline(data.accepted.mean(), color=INK2, lw=1, ls="--")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    return fig


created, create, select, where = st.tabs(["Engineered features", "Create a feature", "Feature selection",
                                          "Where each technique is"])

with created:
    if not recipe:
        st.info("No engineered features yet for this dataset: create one in the next tab.")
    else:
        st.dataframe(pd.DataFrame([{"feature": s["name"], "technique": engineer.OPS[s["op"]].split(":")[0],
                                    "definition": engineer.describe(s), "type": engineer.kind(s),
                                    "missing": f"{data[s['name']].isna().mean():.1%}" if s["name"] in data else "-",
                                    "a model feature": "yes" if s["name"] in schema["categorical"] + schema["numeric"]
                                    else "no"} for s in recipe]), hide_index=True, width="stretch")
        st.caption("Engineered features are computed from the original columns every time the data is loaded, "
                   "for training, reproduction, evaluation and scoring, so they are never stored in the CSV.")
        c1, c2 = st.columns([2, 1])
        target = c1.selectbox("Remove a feature", [s["name"] for s in recipe], index=None, placeholder="choose")
        if target:
            used_by = engineer.dependents(target, recipe)
            if used_by:
                c1.warning(f"{target} is used by {', '.join(used_by)}; remove those first.")
            elif c2.button("Remove", type="primary"):
                new = {**schema, "engineered": [s for s in recipe if s["name"] != target],
                       "categorical": [c for c in schema["categorical"] if c != target],
                       "numeric": [c for c in schema["numeric"] if c != target]}
                save(new, f"Removed {target}")
        st.caption("Runs already trained keep the recipe they were trained with.")

with create:
    if not numeric:
        st.info("This dataset has no numeric features to engineer from.")
    op = st.selectbox("Technique", list(engineer.OPS), format_func=engineer.OPS.get)
    c1, c2, c3 = st.columns(3)
    step = {"op": op}
    if op == "combine":
        step["a"] = c1.selectbox("First feature", numeric, key="fe-a")
        step["how"] = c2.selectbox("Operation", list(engineer.HOW), format_func=lambda h: f"{h} ({engineer.HOW[h]})")
        step["b"] = c3.selectbox("Second feature", [c for c in numeric if c != step["a"]] or numeric, key="fe-b")
        default = f"{step['a']}_per_{step['b']}" if step["how"] == "divide" else f"{step['a']}_{step['how']}_{step['b']}"
        st.caption("Example: txn_value_30d ÷ txn_count_30d gives the average transaction value.")
    elif op == "date_part":
        step["part"] = c1.selectbox("Part of the offer date", engineer.DATE_PARTS, format_func=lambda p: p.replace("_", " "))
        default = f"offer_{step['part']}"
        st.caption("When scoring, the date parts are those of the day the recommendation is made.")
    else:
        step["column"] = c1.selectbox("Feature", numeric, key="fe-col")
        x = data[step["column"]]
        if op == "log":
            default = f"log_{step['column']}"
            st.caption(f"Skewness of {step['column']}: {x.skew():.2f} (above about 1 the log usually helps). "
                       "Values at or below -1 become missing.")
        elif op == "binarize":
            step["threshold"] = float(c2.number_input("Threshold", value=float(x.median()), format="%g"))
            default = f"{step['column']}_above_{step['threshold']:g}"
        else:
            how = c2.radio("Ranges", ["equal-sized", "my own cut points"], horizontal=True)
            if how == "equal-sized":
                n = c3.slider("Number of bins", 2, 10, 4)
                step["edges"] = engineer.quantile_edges(x, n)
            else:
                text = c3.text_input("Cut points, comma separated", ", ".join(f"{e:g}" for e in engineer.quantile_edges(x, 4)))
                try:
                    step["edges"] = sorted({float(v) for v in text.split(",") if v.strip()})
                except ValueError:
                    step["edges"] = []
                    st.error("Cut points must be numbers.")
            default = f"{step['column']}_band"
            st.caption("Cut points: " + ", ".join(f"{e:g}" for e in step["edges"])
                       + (" (chosen from the quantiles of the whole file)" if how == "equal-sized" else ""))
    step["name"] = st.text_input("Name of the new feature", re.sub(r"[^a-z0-9_]+", "_", default.lower())[:40])
    problems = engineer.check(step, data, recipe)
    for p in problems:
        st.error(p)
    if not problems:
        values = engineer.apply(data, [step])[step["name"]]
        c1, c2 = st.columns(2)
        if engineer.kind(step) == "numeric" and values.nunique() > 10:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(values.dropna(), bins=40, color=common.PALETTE[0], edgecolor="white", linewidth=0.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"Distribution of {step['name']}")
            c1.pyplot(fig)
            plt.close(fig)
            s = values.describe()
            c1.caption(f"mean {s['mean']:.4g} · median {s['50%']:.4g} · std {s['std']:.4g} · "
                       f"range {s['min']:.4g} to {s['max']:.4g} · skewness {values.skew():.2f} · "
                       f"missing {values.isna().mean():.1%}")
            bands = pd.qcut(values, 5, duplicates="drop")
            fig = acceptance_chart(bands, "Acceptance rate by fifth of the new feature")
        else:
            fig = acceptance_chart(values, f"Acceptance rate by {step['name']}")
            c1.write(values.value_counts(dropna=False).rename("offers").to_frame())
        c2.pyplot(fig)
        plt.close(fig)
        use = st.checkbox("Use it as a model feature", value=True)
        if st.button("Add the feature", type="primary"):
            kind = engineer.kind(step)
            new = {**schema, "engineered": recipe + [step]}
            if use:
                new[kind] = list(schema[kind]) + [step["name"]]
            save(new, f"Added {step['name']}")

with select:
    feats = [c for c in schema["categorical"] + schema["numeric"] if c in data]
    cutoff = data.offer_date.quantile(0.7)
    st.markdown("Relevance of each feature to acceptance: **mutual information**, how much knowing the feature "
                "reduces uncertainty about the outcome, computed within each offer and averaged. It uses only the "
                f"earliest 70% of offers (up to {cutoff:%Y-%m-%d}), so the later offers used for testing do not "
                "influence which features are chosen.")

    @st.cache_data(show_spinner="Computing relevance...")
    def relevance(name, feats, n_rows):
        train = data[data.offer_date <= cutoff]
        scores, weights = [], []
        for _, g in train.groupby("action"):
            X = pd.DataFrame({c: (g[c].cat.codes if isinstance(g[c].dtype, pd.CategoricalDtype)
                                  else g[c].fillna(g[c].median())) for c in feats})
            discrete = [isinstance(g[c].dtype, pd.CategoricalDtype) for c in feats]
            scores.append(mutual_info_classif(X, g.accepted, discrete_features=discrete, random_state=0))
            weights.append(len(g))
        return pd.Series(np.average(scores, axis=0, weights=weights), index=feats).sort_values()

    if feats:
        mi = relevance(name, tuple(feats), len(data))
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(mi) + 1))
        engineered = {s["name"] for s in recipe}
        ax.barh(mi.index, mi.values, height=0.7,
                color=[common.PALETTE[1] if c in engineered else common.PALETTE[0] for c in mi.index])
        for y, v in enumerate(mi.values):
            ax.annotate(f"{v:.4f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set_title("Mutual information with acceptance" + (" (engineered features in orange)" if engineered else ""))
        st.pyplot(fig)
        plt.close(fig)
        low = mi[mi < 0.1 * mi.max()].index.tolist()
        if low:
            st.caption(f"Weakest (under a tenth of the strongest): {', '.join(low)}. Candidates to leave out on the "
                       "Train page. Mutual information judges each feature alone, so a feature can still help in "
                       "combination with others: confirm with the feature importance on the Results page.")

with where:
    st.markdown("Where each feature engineering technique from the course is done in the portal. Techniques that "
                "learn from the data (imputation, scaling, encoding, PCA) are fitted inside the model pipeline on "
                "the training data only, so no information from the test offers leaks into the model.")
    st.table(pd.DataFrame([
        ("Feature creation", "3 · Features: combine two numbers (+, −, ×, ÷)"),
        ("Imputation", "1 · Data (clean: mean, median or mode) or 4 · Train (fitted on training data)"),
        ("Handling outliers", "1 · Data (clean: extreme quantiles or standard deviations; cap or drop)"),
        ("Log transform", "3 · Features"),
        ("Binning / discretisation", "3 · Features (equal-sized bins or own cut points); 1 · Data groups rare categories"),
        ("Feature split", "3 · Features (month, quarter, day of week, weekend of the offer date)"),
        ("Min-max scaling", "4 · Train (random forest, logistic regression)"),
        ("One-hot encoding", "4 · Train (automatic for categorical features; XGBoost uses native categories)"),
        ("Binarisation", "3 · Features"),
        ("Feature extraction (PCA)", "4 · Train (principal components of the numeric features)"),
        ("Feature selection", "3 · Features (relevance), 4 · Train (choose features), 5 · Results (importance)"),
    ], columns=["Technique", "Where"]))
