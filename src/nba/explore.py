"""Interactive EDA explorer: choose filters, variables and a view, and the chart,
table and tests are computed on demand from the versioned dataset.

  make explore   ->   http://localhost:8501

Runs locally in the job image; the data never leaves the machine.
"""
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from scipy import stats

from nba.config import REPO, load_params
from nba.datasets import params_for
from nba.eda import DIVERGING, association, ordered, pretty
from nba.features import load_data
from nba.report import BLUES, INK, INK2, NEUTRAL, SURFACE, label

# Validated categorical palette, fixed order (slots 1-8).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SAVE_DIR = REPO / "reports" / "eda" / "custom"
SEQ = LinearSegmentedColormap.from_list("blues", BLUES)


@st.cache_data
def load(dataset=None):
    """A registered dataset by name, or params.yaml's dataset."""
    params = params_for(dataset) if dataset else load_params()
    data = load_data(params)
    data["month"] = data.offer_date.dt.to_period("M").astype(str)
    return params, data


def name(col):
    return "offer" if col == "action" else pretty(col)


def value_name(col, v):
    return label(str(v)) if col == "action" else pretty(str(v))


def groups(df, col, numeric):
    """A grouping key: categories as they are, numerics with many values in quintiles."""
    if col in numeric and df[col].nunique() > 8:
        return pd.qcut(df[col], 5, duplicates="drop")
    return df[col]


def sort_index(col, series, numeric):
    if col in numeric or col.endswith("_band"):
        return ordered(col, series) if col.endswith("_band") else series.sort_index()
    return series.sort_values()


def heatmap(ax, table, fmt, cmap=SEQ, vmin=None, vmax=None, counts=None):
    im = ax.imshow(table.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    top = np.nanmax(np.abs(table.values)) if table.size else 1
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isnan(v):
                continue
            text = fmt(v) + (f"\nn={counts.values[i, j]:,}" if counts is not None else "")
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=SURFACE if abs(v) > top * 0.6 else INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def outputs(fig, table, slug):
    """Show the figure and table, with download and save buttons."""
    st.pyplot(fig)
    png = io.BytesIO()
    fig.savefig(png, format="png")
    csv = table.to_csv().encode()
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download PNG", png.getvalue(), f"{slug}.png", "image/png")
    c2.download_button("Download CSV", csv, f"{slug}.csv", "text/csv")
    if c3.button("Save to reports/eda/custom", key=f"save-{slug}"):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / f"{slug}.png").write_bytes(png.getvalue())
        (SAVE_DIR / f"{slug}.csv").write_bytes(csv)
        st.success(f"Saved reports/eda/custom/{slug}.png and .csv")
    st.dataframe(table)
    plt.close(fig)


def slugify(*parts):
    return re.sub(r"[^a-z0-9]+", "_", "_".join(str(p) for p in parts).lower()).strip("_")[:80]


def view_distribution(df, cats, nums):
    col = st.selectbox("Variable", cats + nums, format_func=name)
    split = st.checkbox("Split by accepted / not accepted")
    fig, ax = plt.subplots(figsize=(8, 4))
    if col in cats:
        if split:
            table = pd.crosstab(df[col], df.accepted, normalize="columns").rename(
                columns={0: "not accepted", 1: "accepted"})
            table = table.loc[sort_index(col, table["accepted"], nums).index]
            y, h = np.arange(len(table)), 0.4
            for i, c in enumerate(["not accepted", "accepted"]):
                ax.barh(y + (i - 0.5) * h, table[c], height=h - 0.04, color=PALETTE[i], label=c)
            ax.set_yticks(y, [value_name(col, v) for v in table.index])
            ax.legend(loc="lower right", fontsize=8.5)
        else:
            table = sort_index(col, df[col].value_counts(normalize=True), nums).to_frame("share")
            ax.barh([value_name(col, v) for v in table.index], table.share, color=PALETTE[0], height=0.7)
            for y, v in enumerate(table.share):
                ax.annotate(f"{v:.1%}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK2)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
    else:
        bins = np.arange(df[col].min(), df[col].max() + 2) - 0.5 if df[col].nunique() <= 25 else 30
        if split:
            for i, (v, c) in enumerate([(0, "not accepted"), (1, "accepted")]):
                ax.hist(df.loc[df.accepted == v, col], bins=bins, density=True, histtype="step", lw=2,
                        color=PALETTE[i], label=c)
            ax.legend(fontsize=8.5)
            ax.set_ylabel("density")
        else:
            ax.hist(df[col], bins=bins, color=PALETTE[0], edgecolor=SURFACE, linewidth=0.8)
            ax.set_ylabel("offers")
        ax.grid(axis="x", visible=False)
        table = df.groupby("accepted")[col].describe().rename(index={0: "not accepted", 1: "accepted"}) \
            if split else df[col].describe().to_frame(col)
    ax.set_title(f"Distribution of {name(col)}" + (" by outcome" if split else ""))
    outputs(fig, table, slugify("distribution", col, "split" if split else ""))


def view_acceptance(df, cats, nums):
    col = st.selectbox("Variable", cats + nums, format_func=name)
    split = st.selectbox("Split by (optional)", ["(none)"] + [c for c in cats if c != col], format_func=lambda c: c if c == "(none)" else name(c))
    key = groups(df, col, nums)
    overall = df.accepted.mean()
    if split == "(none)":
        g = df.groupby(key, observed=True).accepted.agg(["mean", "size"])
        g = g.loc[sort_index(col, g["mean"], nums).index]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(g) + 1)))
        ax.barh([value_name(col, v) for v in g.index], g["mean"], color=PALETTE[0], height=0.7)
        for y, (r, n) in enumerate(zip(g["mean"], g["size"])):
            ax.annotate(f"{r:.1%}  (n={n:,})", (r, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.axvline(overall, color=INK2, lw=1, ls="--")
        ax.set_xlim(0, g["mean"].max() * 1.35)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        table = g.rename(columns={"mean": "acceptance_rate", "size": "offers"})
    else:
        rate = df.pivot_table(index=key, columns=split, values="accepted", aggfunc="mean", observed=True)
        count = df.pivot_table(index=key, columns=split, values="accepted", aggfunc="size", observed=True)
        order = sort_index(col, rate.mean(axis=1), nums).index
        rate, count = rate.loc[order], count.loc[order]
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * rate.shape[1] + 2), max(3, 0.5 * len(rate) + 1)))
        im = heatmap(ax, rate, lambda v: f"{v:.0%}", vmin=0, counts=count)
        ax.set_xticks(range(rate.shape[1]), [value_name(split, v) for v in rate.columns], rotation=20, ha="right")
        ax.set_yticks(range(len(rate)), [value_name(col, v) for v in rate.index])
        fig.colorbar(im, ax=ax, fraction=0.03, format=PercentFormatter(1.0, decimals=0))
        table = rate
    ax.set_title(f"Acceptance rate by {name(col)}" + ("" if split == "(none)" else f" and {name(split)}")
                 + f" (overall {overall:.1%})")
    outputs(fig, table, slugify("acceptance", col, split))


def view_crosstab(df, cats, nums):
    c1, c2 = st.columns(2)
    a = c1.selectbox("Rows", cats + nums, format_func=name)
    b = c2.selectbox("Columns", [c for c in cats + nums if c != a], format_func=name)
    mode = st.radio("Show", ["row %", "column %", "counts"], horizontal=True)
    ka, kb = groups(df, a, nums), groups(df, b, nums)
    counts = pd.crosstab(ka, kb)
    table = {"counts": counts, "row %": counts.div(counts.sum(axis=1), axis=0),
             "column %": counts.div(counts.sum(axis=0), axis=1)}[mode]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * table.shape[1] + 2), max(3, 0.45 * len(table) + 1)))
    im = heatmap(ax, table.astype(float), (lambda v: f"{v:,.0f}") if mode == "counts" else (lambda v: f"{v:.0%}"))
    ax.set_xticks(range(table.shape[1]), [value_name(b, v) for v in table.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(table)), [value_name(a, v) for v in table.index])
    fig.colorbar(im, ax=ax, fraction=0.03)
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    v = np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1))) if min(counts.shape) > 1 else 0
    ax.set_title(f"{name(a)} by {name(b)} ({mode})")
    st.markdown(f"**Chi-square** χ²({dof}) = {chi2:,.1f}, p = {p:.3g}, Cramér's V = {v:.3f}")
    outputs(fig, table, slugify("crosstab", a, b, mode))


def view_correlation(df, nums):
    cols = st.multiselect("Numeric variables", nums + ["accepted"], default=nums + ["accepted"], format_func=name)
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    if len(cols) < 2:
        st.info("Choose at least two variables.")
        return
    corr = df[cols].corr(method=method)
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2.5, 0.6 * len(cols) + 1.8))
    im = heatmap(ax, corr, lambda v: f"{v:.2f}", cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [name(c) for c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), [name(c) for c in cols])
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(f"{method.title()} correlation")
    outputs(fig, corr, slugify("correlation", method, len(cols)))


def view_tests(df, cats, nums):
    chosen = st.multiselect("Variables to test against acceptance", cats + nums, default=cats + nums,
                            format_func=name)
    by_offer = st.checkbox("Test within each offer separately", value=True)
    if not chosen:
        return
    c, n = [v for v in chosen if v in cats and v != "action"], [v for v in chosen if v in nums]
    frames = {a: df[df.action == a] for a in sorted(df.action.unique())} if by_offer else {"all offers": df}
    rows = []
    for offer, frame in frames.items():
        for r in association(frame, c + (["action"] if "action" in chosen and not by_offer else []), n):
            rows.append({"offer": label(offer) if offer != "all offers" else offer, **r})
    if not rows:
        st.info("Not enough variation in the filtered data to run the tests.")
        return
    table = pd.DataFrame(rows)
    table["p_bh"] = stats.false_discovery_control(table.p, method="bh")
    table["significant"] = table.p_bh < 0.05
    table["variable"] = table.variable.map(name)
    table = table[["offer", "variable", "test", "statistic", "df", "p", "p_bh", "effect_name", "effect",
                   "significant"]].sort_values(["offer", "p"])
    st.markdown(f"{int(table.significant.sum())} of {len(table)} tests significant at 5% after "
                "Benjamini-Hochberg adjustment. Effect sizes: Cramér's V (categorical), rank-biserial r (numeric).")
    st.dataframe(table.round(4), hide_index=True)
    c1, c2 = st.columns(2)
    csv = table.to_csv(index=False).encode()
    c1.download_button("Download CSV", csv, "tests.csv", "text/csv")
    if c2.button("Save to reports/eda/custom", key="save-tests"):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / slugify("tests", "by_offer" if by_offer else "pooled")).with_suffix(".csv").write_bytes(csv)
        st.success("Saved to reports/eda/custom")


def view_trend(df, cats):
    split = st.selectbox("Split by (optional)", ["(none)"] + cats, format_func=lambda c: c if c == "(none)" else name(c))
    measure = st.radio("Measure", ["acceptance rate", "offers"], horizontal=True)
    agg = "mean" if measure == "acceptance rate" else "size"
    if split == "(none)":
        table = df.groupby("month").accepted.agg(agg).to_frame(measure)
    else:
        table = df.pivot_table(index="month", columns=split, values="accepted", aggfunc=agg, observed=True)
    if table.shape[1] > len(PALETTE):
        st.warning(f"{table.shape[1]} groups is too many to tell apart; showing the {len(PALETTE)} largest.")
        top = df[split].value_counts().index[:len(PALETTE)]
        table = table[[c for c in table.columns if c in top]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(table))
    for i, c in enumerate(table.columns):
        ax.plot(x, table[c], color=PALETTE[i], lw=2, marker="o", ms=4,
                label=value_name(split, c) if split != "(none)" else measure)
        ax.annotate(value_name(split, c) if split != "(none)" else "", (x[-1], table[c].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, table.index, rotation=45, ha="right")
    if measure == "acceptance rate":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if split != "(none)":
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.08, 1))
    ax.set_title(f"{measure.capitalize()} per month" + ("" if split == "(none)" else f" by {name(split)}"))
    outputs(fig, table, slugify("trend", measure, split))


def explorer(params, data, title="NBA data explorer"):
    ds = params["dataset"]
    cats = ["action"] + ds["categorical"] + [c for c in ds.get("protected", ["gender"]) if c in data]
    nums = ds["numeric"]

    st.sidebar.header("Filters")
    offers = st.sidebar.multiselect("Offer", sorted(data.action.unique()), default=sorted(data.action.unique()),
                                    format_func=label)
    df = data[data.action.isin(offers)]
    if "offer_channel" in data:
        channels = st.sidebar.multiselect("Channel", list(data.offer_channel.cat.categories),
                                          default=list(data.offer_channel.cat.categories))
        df = df[df.offer_channel.isin(channels)]
    start, end = st.sidebar.slider("Date range", data.offer_date.min().date(), data.offer_date.max().date(),
                                   (data.offer_date.min().date(), data.offer_date.max().date()))
    extra = st.sidebar.selectbox("Also filter by", ["(none)"] + [c for c in cats if c not in ("action", "offer_channel")],
                                 format_func=lambda c: c if c == "(none)" else name(c))
    df = df[df.offer_date.dt.date.between(start, end)]
    if extra != "(none)":
        levels = sorted(df[extra].dropna().unique(), key=str)
        keep = st.sidebar.multiselect(name(extra), levels, default=levels, format_func=lambda v: value_name(extra, v))
        df = df[df[extra].isin(keep)]

    st.title(title)
    st.caption(f"{ds['path']} · {len(df):,} of {len(data):,} offers selected · "
               f"acceptance {df.accepted.mean():.1%}" if len(df) else "No offers match the filters.")
    if len(df) < 20:
        st.warning("Too few offers selected for meaningful charts or tests.")
        return

    view = st.radio("View", ["Distribution", "Acceptance rate", "Cross-tab", "Correlation",
                             "Statistical tests", "Trend over time"], horizontal=True)
    if view == "Distribution":
        view_distribution(df, cats, nums)
    elif view == "Acceptance rate":
        view_acceptance(df, cats, nums)
    elif view == "Cross-tab":
        view_crosstab(df, cats, nums)
    elif view == "Correlation":
        view_correlation(df, nums)
    elif view == "Statistical tests":
        view_tests(df, cats, nums)
    else:
        view_trend(df, cats)


def main():
    st.set_page_config(page_title="NBA data explorer", layout="wide")
    explorer(*load())


if __name__ == "__main__":
    main()
