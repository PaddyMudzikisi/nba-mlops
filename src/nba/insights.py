"""Which customer features predict acceptance, across all offers: importance
shares per offer, and the direction of each feature's effect from SHAP values
(which values make acceptance more or less likely)."""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


def importance_table(importances):
    """{offer: {feature: importance}} -> each feature's share (%) of each offer's
    total importance, the mean share across offers and the overall rank."""
    t = pd.DataFrame({a: pd.Series(v, dtype=float) for a, v in importances.items()}).fillna(0.0)
    t = 100 * t / t.sum().replace(0, np.nan)
    t["mean share"] = t.mean(axis=1)
    t = t.sort_values("mean share", ascending=False)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def shap_values(model, X):
    """Per-row SHAP contributions (log-odds) of an XGBoost model, one column per feature."""
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], index=X.index, columns=X.columns)


def directions(X, shap, categorical):
    """For each feature, which values raise acceptance and which lower it.
    Numeric: Spearman correlation between the value and its SHAP contribution.
    Categorical: the categories with the highest and lowest mean contribution."""
    rows = []
    for col in X.columns:
        s = shap[col]
        if col in categorical:
            means = s.groupby(X[col].astype(str), observed=True).mean()
            counts = X[col].astype(str).value_counts()
            means = means[counts.reindex(means.index).fillna(0) >= 20]
            if means.empty:
                continue
            rows.append({"feature": col, "type": "categorical", "raises acceptance": means.idxmax(),
                         "lowers acceptance": means.idxmin(), "strength": float(means.max() - means.min())})
        else:
            x = pd.to_numeric(X[col], errors="coerce")
            ok = x.notna()
            if ok.sum() < 20 or x[ok].nunique() < 2:
                continue
            rho = spearmanr(x[ok], s[ok]).statistic
            higher = "higher values" if rho > 0 else "lower values"
            rows.append({"feature": col, "type": "numeric", "raises acceptance": higher,
                         "lowers acceptance": "lower values" if rho > 0 else "higher values",
                         "strength": float(abs(rho)), "spearman_rho": float(rho)})
    return pd.DataFrame(rows)


def dependence(X, shap, col, categorical):
    """Mean SHAP contribution by category, or by fifth of a numeric feature."""
    key = X[col].astype(str) if col in categorical else pd.qcut(pd.to_numeric(X[col], errors="coerce"), 5,
                                                                 duplicates="drop").astype(str)
    g = shap[col].groupby(key, observed=True).agg(["mean", "size"])
    return g[g["size"] >= 20].rename(columns={"mean": "mean SHAP", "size": "offers"})


def summary(table, label=str, top=3):
    """A short written answer: the strongest predictors overall and for each offer."""
    overall = ", ".join(f"{f} ({v:.1f}%)" for f, v in table["mean share"].head(top).items())
    lines = [f"Across all offers the strongest predictors are {overall}."]
    for offer in [c for c in table.columns if c not in ("mean share", "rank")]:
        best = table[offer].sort_values(ascending=False).head(top)
        lines.append(f"{label(offer)}: " + ", ".join(f"{f} ({v:.1f}%)" for f, v in best.items()) + ".")
    weak = table.index[table["mean share"] < 2].tolist()
    if weak:
        lines.append(f"Least predictive (under 2% on average): {', '.join(weak)}.")
    return lines
