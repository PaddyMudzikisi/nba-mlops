"""Training results: test vs train metrics per offer with an overfitting /
underfitting verdict, confusion matrix, feature importance, tuning results,
model selection across model families, and side-by-side comparison of runs."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from nba.explore import load
from nba.portal import common
from nba.report import INK2, label

common.page("5 · Results")
name = common.active_dataset("results")
runs = common.train_runs(name)
if runs.empty:
    st.info("No training runs for this dataset yet: start one on the Train page.")
    st.stop()

st.subheader("Training runs")
show = runs.rename(columns={"mean_roc_auc": "ROC-AUC", "mean_pr_auc": "PR-AUC", "mean_f1": "F1",
                            "mean_train_roc_auc": "train ROC-AUC"})
st.dataframe(show.style.format({c: "{:.3f}" for c in ["ROC-AUC", "PR-AUC", "F1", "train ROC-AUC"] if c in show},
                               na_rep="-"), hide_index=True, width="stretch")
st.caption("Mean over the offers' models, on each run's held-out test set. Early runs did not record train or "
           "classification metrics.")



def fit(train_auc, test_auc):
    """Lecture definitions: overfitting = much better on training than on new data;
    underfitting = poor on both."""
    if not (np.isfinite(train_auc) and np.isfinite(test_auc)):
        return "-"
    if train_auc - test_auc > 0.05:
        return "overfitting"
    if train_auc < 0.65:
        return "underfitting"
    return "good fit"


detail, selection, compare = st.tabs(["Run detail", "Model selection", "Compare runs"])
with detail:
    run_id = common.pick_run(runs, "results-run")
    kids = common.child_runs(run_id)
    names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "precision": "Precision", "recall": "Recall",
             "accuracy": "Accuracy", "brier": "Brier"}
    rows = []
    for _, k in kids.iterrows():
        row = {"offer": label(k["tags.action"])}
        for m, nice in names.items():
            row[f"{nice} test"] = k.get(f"metrics.{m}", np.nan)
            if m in ("roc_auc", "pr_auc", "f1"):
                row[f"{nice} train"] = k.get(f"metrics.train_{m}", np.nan)
        row["ROC-AUC gap (train − test)"] = row["ROC-AUC train"] - row["ROC-AUC test"]
        row["fit"] = fit(row["ROC-AUC train"], row["ROC-AUC test"])
        row["baseline accuracy"] = k.get("metrics.baseline_accuracy", np.nan)
        row["threshold"] = k.get("metrics.threshold", np.nan)
        row["test n"] = k.get("metrics.n_test", np.nan)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("offer")
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns if c not in ("offer", "test n", "fit")} |
                                    {"test n": "{:,.0f}"}, na_rep="-"), hide_index=True, width="stretch")
    gap = table["ROC-AUC gap (train − test)"].max()
    if np.isfinite(gap):
        verdict = ("large: the model fits the training data much better than new data (overfitting)" if gap > 0.1
                   else "moderate: some overfitting" if gap > 0.05 else "small: no sign of strong overfitting")
        st.markdown(f"**Largest train − test ROC-AUC gap: {gap:.3f}**, {verdict}. Accuracy is shown next to the "
                    "baseline of always predicting the majority outcome; with ~20% acceptance, high accuracy alone "
                    "means little. Precision, recall and F1 use a threshold chosen on the training data.")
    st.caption("Fit: overfitting when training ROC-AUC exceeds test ROC-AUC by more than 0.05; underfitting when "
               "even the training ROC-AUC is below 0.65 (the model learns little from the features); otherwise a "
               "good fit.")

    st.markdown("**Feature importance**")
    offer = st.selectbox("Offer", sorted(kids["tags.action"]), format_func=label)
    child = kids[kids["tags.action"] == offer].iloc[0]
    st.markdown("**Confusion matrix** (test offers, at the run's threshold)")
    preds = common.load_artifact_csv(child.run_id, "test_predictions.csv")
    if preds is not None and "accepted" not in preds:  # runs recorded before labels were saved
        preds = preds.merge(load(name)[1][["offer_id", "accepted"]], on="offer_id", how="inner")
    threshold = child.get("metrics.threshold", np.nan)
    if preds is None or not np.isfinite(threshold):
        st.caption("This run did not record test predictions with a threshold.")
    else:
        pred = (preds.proba >= threshold).astype(int)
        cm = pd.crosstab(preds.accepted.map({0: "not accepted", 1: "accepted"}),
                         pred.map({0: "not accepted", 1: "accepted"})).reindex(
            index=["accepted", "not accepted"], columns=["accepted", "not accepted"], fill_value=0)
        c1, c2 = st.columns([1, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.imshow(cm.to_numpy(), cmap="Blues")
        for (i, j), v in np.ndenumerate(cm.to_numpy()):
            ax.text(j, i, f"{v:,}\n{['TP', 'FN', 'FP', 'TN'][2 * i + j]}", ha="center", va="center",
                    color="white" if v > cm.to_numpy().max() / 2 else INK2, fontsize=10)
        ax.set_xticks([0, 1], ["accepted", "not accepted"])
        ax.set_yticks([0, 1], ["accepted", "not accepted"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.grid(False)
        ax.set_title(f"{label(offer)} · threshold {threshold:.3f}", fontsize=10)
        c1.pyplot(fig)
        plt.close(fig)
        tp, fn, fp, tn = cm.to_numpy().ravel()
        c2.markdown(f"- **Accuracy** (TP + TN) / all = {(tp + tn) / cm.to_numpy().sum():.3f}\n"
                    f"- **Precision** TP / (TP + FP) = {tp / max(tp + fp, 1):.3f}: of the offers predicted to be "
                    "accepted, the share that were\n"
                    f"- **Recall** TP / (TP + FN) = {tp / max(tp + fn, 1):.3f}: of the offers accepted, the share "
                    "the model found\n"
                    f"- **F1** = {2 * tp / max(2 * tp + fp + fn, 1):.3f}: balance of precision and recall")
    st.markdown("**Feature importance**")
    imp = common.load_artifact_json(child.run_id, "importance.json")
    if imp is None:
        values = common.load_artifact_json(child.run_id, "shap_mean_abs.json") or {}
        imp = {"method": "mean |SHAP|", "importance": values}
    top = pd.Series(imp["importance"]).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1))
    ax.barh(top.index, top.values, color=common.PALETTE[0], height=0.7)
    for y, v in enumerate(top.values):
        ax.annotate(f"{v:.3f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{label(offer)}: {imp['method']}")
    st.pyplot(fig)
    plt.close(fig)
    tuning = common.load_artifact_json(child.run_id, "tuning.json")
    if tuning:
        st.markdown(f"**Tuning** ({tuning['scoring']}, cross-validated on training data): best "
                    f"{tuning['best_params']} with score {tuning['best_cv_score']:.3f}")
    params = common.load_artifact_yaml(run_id, "params.yaml")
    with st.expander("Recorded parameter set"):
        st.json(params or {}, expanded=False)

with selection:
    st.markdown("The latest finished run of each model family on this dataset, compared on the same metrics. "
                "Start all three at once with **Model selection** on the Train page so they share the split and "
                "features.")
    finished = runs[runs.status == "FINISHED"]
    latest = finished.groupby("model", sort=False).head(1)
    if len(latest) < 2:
        st.info("Train at least two model families on this dataset to compare them.")
    else:
        cols = {"mean_roc_auc": "ROC-AUC", "mean_pr_auc": "PR-AUC", "mean_f1": "F1", "mean_train_roc_auc": "train ROC-AUC"}
        sel = latest[["model", "split", "started", "run_id"] + [c for c in cols if c in latest]].rename(columns=cols)
        if "train ROC-AUC" in sel:
            sel["gap"] = sel["train ROC-AUC"] - sel["ROC-AUC"]
            sel["fit"] = [fit(a, b) for a, b in zip(sel["train ROC-AUC"], sel["ROC-AUC"])]
        sel = sel.sort_values("PR-AUC", ascending=False)
        st.dataframe(sel.style.format({c: "{:.3f}" for c in ["ROC-AUC", "PR-AUC", "F1", "train ROC-AUC", "gap"] if c in sel},
                                      na_rep="-").highlight_max(subset=["PR-AUC"], color="#d6e6f9"),
                     hide_index=True, width="stretch")
        best = sel.iloc[0]
        st.markdown(f"**Best by PR-AUC: {best['model']}** (run `{best['run_id'][:8]}`). PR-AUC is used because "
                    "acceptances are rare, so it rewards finding them more than ROC-AUC does.")
        if sel.split.nunique() > 1:
            st.warning("These runs use different splits, so their scores are not directly comparable.")
        st.caption("Choosing the model on the test offers and then reporting the same test score is slightly "
                   "optimistic. For a strict choice, compare the cross-validated scores from tuning, which use the "
                   "training data only.")

with compare:
    chosen = st.multiselect("Runs to compare (up to 4)", runs.run_id.tolist(), max_selections=4,
                            format_func=lambda i: common.run_label(runs.set_index("run_id").loc[i].to_dict()
                                                                   | {"run_id": i}))
    metric = st.selectbox("Metric", ["roc_auc", "pr_auc", "f1", "brier"], format_func=lambda m: m.upper().replace("_", "-"))
    if chosen:
        frames = []
        for i, rid in enumerate(chosen):
            kids = common.child_runs(rid)
            if f"metrics.{metric}" not in kids:
                continue
            frames.append(kids.set_index("tags.action")[f"metrics.{metric}"].rename(f"{rid[:8]}"))
        if frames:
            table = pd.concat(frames, axis=1).sort_index()
            fig, ax = plt.subplots(figsize=(8, 0.6 * len(table) + 1.2))
            h = 0.8 / table.shape[1]
            y = np.arange(len(table))
            for i, col in enumerate(table.columns):
                ax.barh(y + (i - (table.shape[1] - 1) / 2) * h, table[col], height=h - 0.04,
                        color=common.PALETTE[i], label=col)
            ax.set_yticks(y, [label(a) for a in table.index])
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=table.shape[1])
            ax.set_title(f"Test {metric.upper().replace('_', '-')} by offer")
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(table.style.format("{:.3f}"), width="stretch")
        else:
            st.info(f"None of the chosen runs recorded {metric}.")
