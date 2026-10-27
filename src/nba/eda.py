"""Exploratory data analysis for Chapter 3 (data understanding): data quality,
distributions, acceptance by segment, correlations and association tests.
Writes reports/eda/*.png, reports/eda/eda.json and reports/eda/eda.md.

The outcome is binary (offer accepted), so associations are tested with
chi-square (+ Cramer's V) for categorical variables and Mann-Whitney U for
numeric ones, overall and within each offer, with Benjamini-Hochberg
adjustment across the within-offer tests.
"""
import json
import re

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from nba.config import REPO, load_params
from nba.features import load_data
from nba.report import BLUES, GRID, INK, INK2, NEUTRAL, SERIES, SURFACE, label
from nba.train import provenance

OUT = REPO / "reports" / "eda"
DIVERGING = LinearSegmentedColormap.from_list("div", [SERIES[1], "#f0efec", SERIES[0]])
PII_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "phone": r"(?:\+?263|\b0)7\d{8}\b",
    "national_id": r"\b\d{2}-?\d{6,7}-?[A-Z]-?\d{2}\b",
    "long_digit_run": r"\b\d{10,}\b",
}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)


def pretty(col):
    return col.replace("_", " ")


def ordered(col, series):
    """Bands in their natural order (lowest at the bottom of a bar chart), others by value."""
    if not col.endswith("_band"):
        return series.sort_values()
    start = lambda s: 0 if s.lower().startswith("under") else int(re.findall(r"\d+", s)[0])
    return series.loc[sorted(series.index, key=lambda s: start(str(s)))]


def data_quality(raw, data, ds):
    numeric = ds["numeric"]
    q1, q3 = raw[numeric].quantile(0.25), raw[numeric].quantile(0.75)
    iqr = q3 - q1
    outliers = ((raw[numeric] < q1 - 1.5 * iqr) | (raw[numeric] > q3 + 1.5 * iqr)).sum()
    text = raw.select_dtypes("object").astype(str)
    pii = {name: int(sum(text[c].str.contains(p, regex=True).sum() for c in text)) for name, p in PII_PATTERNS.items()}
    ids = raw[ds["columns"]["customer_key"]].astype(str)
    return {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "customers": int(ids.nunique()),
        "offers": sorted(data.action.unique()),
        "date_range": [str(data.offer_date.min().date()), str(data.offer_date.max().date())],
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_offer_keys": int(data.offer_id.duplicated().sum()),
        "iqr_outliers": {c: int(v) for c, v in outliers.items()},
        "pii_pattern_matches": pii,
        "pseudonymous_ids": bool(ids.str.fullmatch(r"SYN\d+").all()),
        "overall_acceptance_rate": float(data.accepted.mean()),
    }


def fig_categorical_profile(data, cols):
    n = len(cols)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.3 * ((n + 1) // 2)))
    for ax, col in zip(axes.flat, cols):
        share = ordered(col, data[col].value_counts(normalize=True))
        ax.barh([label(str(i)) if col == "action" else pretty(str(i)) for i in share.index], share.values,
                color=SERIES[0], height=0.7)
        for y, v in enumerate(share.values):
            ax.annotate(f"{v:.0%}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set_xlim(0, share.max() * 1.25)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_title("offer" if col == "action" else pretty(col), fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Share of offers by category", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "categorical_profile.png")


def fig_by_month(data):
    monthly = data.groupby(data.offer_date.dt.to_period("M")).agg(offers=("accepted", "size"),
                                                                  rate=("accepted", "mean"))
    x = np.arange(len(monthly))
    labels = [str(p) for p in monthly.index]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    a1.bar(x, monthly.offers, color=SERIES[0], width=0.7)
    a1.set_title("Offers per month")
    a1.grid(axis="x", visible=False)
    a2.plot(x, monthly.rate, color=SERIES[0], lw=2, marker="o", ms=5)
    a2.axhline(data.accepted.mean(), color=NEUTRAL, lw=1.2, ls="--")
    a2.annotate(f"overall {data.accepted.mean():.1%}", (x[-1], data.accepted.mean()), xytext=(4, 4),
                textcoords="offset points", fontsize=8.5, color=INK2, ha="right")
    a2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a2.set_title("Acceptance rate per month")
    a2.set_xticks(x, labels, rotation=45, ha="right")
    fig.tight_layout()
    save(fig, "offers_by_month.png")


def fig_acceptance_by_segment(data, cols):
    overall = data.accepted.mean()
    n = len(cols)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(11, 2.3 * ((n + 1) // 2)))
    for ax, col in zip(axes.flat, cols):
        rate = ordered(col, data.groupby(col, observed=True).accepted.mean())
        ax.barh([label(str(i)) if col == "action" else pretty(str(i)) for i in rate.index], rate.values,
                color=SERIES[0], height=0.7)
        ax.axvline(overall, color=INK2, lw=1, ls="--")
        for y, v in enumerate(rate.values):
            ax.annotate(f"{v:.1%}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8, color=INK2)
        ax.set_xlim(0, rate.max() * 1.3)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_title("offer" if col == "action" else pretty(col), fontsize=10)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"Acceptance rate by category (dashed line: overall {overall:.1%})", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "acceptance_by_segment.png")


def fig_offer_channel(data):
    rate = data.pivot_table(index="action", columns="offer_channel", values="accepted", aggfunc="mean",
                            observed=True)
    count = data.pivot_table(index="action", columns="offer_channel", values="accepted", aggfunc="size",
                             observed=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(rate.values, cmap=LinearSegmentedColormap.from_list("blues", BLUES), aspect="auto", vmin=0)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.values[i, j]
            ax.text(j, i, f"{v:.0%}\nn={count.values[i, j]:,}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v > np.nanmax(rate.values) * 0.6 else INK)
    ax.set_xticks(range(rate.shape[1]), [pretty(c) for c in rate.columns])
    ax.set_yticks(range(rate.shape[0]), [label(a) for a in rate.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Acceptance rate by offer and channel")
    fig.colorbar(im, ax=ax, fraction=0.03, format=plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    save(fig, "acceptance_offer_channel.png")


def fig_numeric(data, cols):
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(11, 2.4 * ((n + 2) // 3)))
    for ax, col in zip(axes.flat, cols):
        values = data[col]
        bins = np.arange(values.min(), values.max() + 2) - 0.5 if values.nunique() <= 25 else 30
        ax.hist(values, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
        if set(values.unique()) <= {0, 1}:
            ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_title(pretty(col), fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Distributions of numeric features", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "numeric_distributions.png")


def fig_correlation(data, cols):
    corr = data[cols + ["accepted"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if abs(v) > 0.6 else INK)
    names = [pretty(c) for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Pearson correlation of numeric features and acceptance")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "correlation_matrix.png")
    return corr


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))) if min(table.shape) > 1 else 0.0


def association(frame, categorical, numeric):
    """One row per variable: test, statistic, p-value, effect size."""
    rows = []
    for col in categorical:
        table = pd.crosstab(frame[col], frame.accepted)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"variable": col, "test": "chi-square", "statistic": float(chi2), "df": int(dof),
                     "p": float(p), "effect": cramers_v(table), "effect_name": "Cramer's V"})
    for col in numeric:
        yes, no = frame.loc[frame.accepted == 1, col], frame.loc[frame.accepted == 0, col]
        u, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
        rows.append({"variable": col, "test": "Mann-Whitney U", "statistic": float(u), "df": None,
                     "p": float(p), "effect": float(2 * u / (len(yes) * len(no)) - 1),
                     "effect_name": "rank-biserial r", "mean_accepted": float(yes.mean()),
                     "mean_not_accepted": float(no.mean())})
    return rows


def fig_within_offer(within, variables, actions):
    """Heatmap of BH-adjusted significance and effect size, variables x offers."""
    eff = np.array([[within[a][v]["effect"] for a in actions] for v in variables])
    sig = np.array([[within[a][v]["p_bh"] < 0.05 for a in actions] for v in variables])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(variables) + 1.5))
    shown = np.where(sig, np.abs(eff), np.nan)
    ax.imshow(np.zeros_like(eff), cmap=LinearSegmentedColormap.from_list("bg", ["#f0efec", "#f0efec"]),
              aspect="auto")
    im = ax.imshow(shown, cmap=LinearSegmentedColormap.from_list("blues", BLUES), aspect="auto", vmin=0,
                   vmax=max(np.nanmax(shown) if sig.any() else 0.1, 0.1))
    for i in range(len(variables)):
        for j in range(len(actions)):
            ax.text(j, i, f"{abs(eff[i, j]):.2f}" if sig[i, j] else "n.s.", ha="center", va="center",
                    fontsize=8, color=INK if not sig[i, j] or abs(eff[i, j]) < np.nanmax(shown) * 0.6 else SURFACE)
    ax.set_xticks(range(len(actions)), [label(a) for a in actions], rotation=20, ha="right")
    ax.set_yticks(range(len(variables)), [pretty(v) for v in variables])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Association with acceptance within each offer\n(|effect size| where BH-adjusted p < 0.05)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    save(fig, "association_within_offer.png")


def write_md(quality, overall, within, corr, actions, variables, ds):
    q = quality
    lines = [
        "# Data understanding (Chapter 3)", "",
        f"Dataset `{ds['path']}`: {q['rows']:,} offers, {q['columns']} columns, {q['customers']:,} customers, "
        f"{len(q['offers'])} offers, {q['date_range'][0]} to {q['date_range'][1]}. "
        f"Overall acceptance rate {q['overall_acceptance_rate']:.1%}. Generated by `make eda`.", "",
        "## Data quality", "",
        "| Check | Result |", "|---|---|",
        f"| Missing values | {sum(q['missing_values'].values())} in total |",
        f"| Duplicate rows | {q['duplicate_rows']} |",
        f"| Duplicate customer + date + offer | {q['duplicate_offer_keys']} |",
        f"| Pseudonymous customer ids (SYN...) | {'yes, all' if q['pseudonymous_ids'] else 'NO'} |",
        "| PII pattern matches (e-mail, phone, national id, 10+ digit runs) | "
        + ", ".join(f"{k} {v}" for k, v in q["pii_pattern_matches"].items()) + " |",
        "", "IQR outliers (values beyond 1.5 x IQR; reported, not removed):", "",
        "| Feature | Outliers |", "|---|---|",
    ] + [f"| {pretty(c)} | {n:,} |" for c, n in q["iqr_outliers"].items()]
    lines += ["", "## Association with acceptance, all offers pooled", "",
              "| Variable | Test | Statistic | df | p-value | Effect size |", "|---|---|---|---|---|---|"]
    for r in sorted(overall, key=lambda r: r["p"]):
        df = "" if r["df"] is None else r["df"]
        lines.append(f"| {pretty(r['variable'])} | {r['test']} | {r['statistic']:.1f} | {df} | "
                     f"{r['p']:.3g} | {r['effect_name']} {r['effect']:.3f} |")
    lines += ["", "Pooling mixes offers with different acceptance rates, so the within-offer results below are "
              "the better guide to which factors matter.", "",
              "## Association with acceptance within each offer (Benjamini-Hochberg adjusted p)", "",
              "| Variable | " + " | ".join(label(a) for a in actions) + " |",
              "|---|" + "---|" * len(actions)]
    for v in variables:
        cells = [f"{within[a][v]['p_bh']:.3g}{' *' if within[a][v]['p_bh'] < 0.05 else ''}" for a in actions]
        lines.append(f"| {pretty(v)} | " + " | ".join(cells) + " |")
    lines += ["", "\\* significant at 5% after adjustment across all within-offer tests.", "",
              "## Strongest correlations between numeric features", "",
              "| Pair | Pearson r |", "|---|---|"]
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
    for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))[:8]:
        lines.append(f"| {pretty(a)} – {pretty(b)} | {r:.3f} |")
    lines += ["", "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in sorted(OUT.glob("*.png"))]
    (OUT / "eda.md").write_text("\n".join(lines) + "\n")


def main():
    params = load_params()
    ds = params["dataset"]
    raw = pd.read_csv(REPO / ds["path"])
    data = load_data(params)
    categorical = ds["categorical"] + ["gender"]
    numeric = ds["numeric"]
    actions = sorted(data.action.unique())

    quality = data_quality(raw, data, ds)
    fig_categorical_profile(data, ["action"] + categorical)
    fig_by_month(data)
    fig_acceptance_by_segment(data, ["action"] + categorical)
    fig_offer_channel(data)
    fig_numeric(data, numeric)
    corr = fig_correlation(data, numeric)

    overall = association(data, ["action"] + categorical, numeric)
    within = {a: {r["variable"]: r for r in association(data[data.action == a], categorical, numeric)}
              for a in actions}
    flat = [(a, v) for a in actions for v in within[a]]
    adjusted = stats.false_discovery_control([within[a][v]["p"] for a, v in flat], method="bh")
    for (a, v), p in zip(flat, adjusted):
        within[a][v]["p_bh"] = float(p)
    variables = categorical + numeric
    fig_within_offer(within, variables, actions)

    report = {"quality": quality, "association_overall": overall, "association_within_offer": within,
              "correlation": corr.round(4).to_dict()}
    (OUT / "eda.json").write_text(json.dumps(report, indent=2))
    write_md(quality, overall, within, corr, actions, variables, ds)

    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="eda"):
        mlflow.set_tags(provenance(params))
        mlflow.log_artifacts(str(OUT), artifact_path="eda")

    sig = [(a, v) for a, v in flat if within[a][v]["p_bh"] < 0.05]
    print(f"{quality['rows']:,} rows, {quality['customers']:,} customers; missing {sum(quality['missing_values'].values())}, "
          f"duplicates {quality['duplicate_rows']}, PII matches {sum(quality['pii_pattern_matches'].values())}")
    print(f"Significant within-offer associations (BH p < 0.05): {len(sig)} of {len(flat)}")
    for a, v in sig:
        r = within[a][v]
        print(f"  {label(a):16s} {pretty(v):22s} p_bh={r['p_bh']:.3g}  {r['effect_name']} {r['effect']:+.3f}")
    print(f"Wrote {len(list(OUT.glob('*.png')))} figures and {OUT / 'eda.md'}")


if __name__ == "__main__":
    main()
