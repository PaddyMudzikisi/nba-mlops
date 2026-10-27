"""Figures and results tables for Chapter 4, built from reports/evaluation.json,
reports/drift.json, reports/retrain_after_drift.json and the training run they
reference in MLflow. Writes reports/figures/*.png and reports/results.md."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mlflow.tracking import MlflowClient

from nba.config import REPO

REPORTS = REPO / "reports"
FIGURES = REPORTS / "figures"

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3, fixed order
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABEL = {
    "Debit_Card_Activation": "Debit card",
    "Fixed_Deposit_Offer": "Fixed deposit",
    "Mobile_Banking_Activation": "Mobile banking",
    "Personal_Loan_Offer": "Personal loan",
    "Savings_Account_Upsell": "Savings upsell",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
})


def label(action):
    return LABEL.get(action, action.replace("_", " "))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def training_results(run_id):
    client = MlflowClient()
    parent = client.get_run(run_id)
    children = client.search_runs([parent.info.experiment_id],
                                  filter_string=f"tags.mlflow.parentRunId = '{run_id}'")
    out = {}
    for child in children:
        shap = json.loads(open(client.download_artifacts(child.info.run_id, "shap_mean_abs.json")).read())
        out[child.data.tags["action"]] = {**child.data.metrics, "shap": shap}
    return dict(sorted(out.items())), parent.data.tags


def bar_labels(ax, bars, fmt):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt(w), (w, b.get_y() + b.get_height() / 2), xytext=(4 if w >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if w >= 0 else "right",
                    fontsize=8.5, color=INK2)


def fig_model_performance(train):
    actions = list(train)
    y = np.arange(len(actions))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    a1.axvline(0.5, color=NEUTRAL, lw=1.5, ls="--")
    a1.annotate("chance", (0.5, len(actions) - 0.45), xytext=(4, 0), textcoords="offset points",
                fontsize=8.5, color=INK2)
    a1.scatter([train[a]["roc_auc"] for a in actions], y, s=70, color=SERIES[0], zorder=3)
    for yi, a in zip(y, actions):
        a1.annotate(f"{train[a]['roc_auc']:.3f}", (train[a]["roc_auc"], yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a1.set_xlim(0.45, 0.82)
    a1.set_title("ROC-AUC, out-of-time test (Jan-Jun 2026)")
    for yi, a in zip(y, actions):
        a2.plot([train[a]["pos_rate_test"], train[a]["pr_auc"]], [yi, yi], color=GRID, lw=2, zorder=2)
    a2.scatter([train[a]["pos_rate_test"] for a in actions], y, s=70, color=NEUTRAL, zorder=3,
               label="Base rate (random ranking)")
    a2.scatter([train[a]["pr_auc"] for a in actions], y, s=70, color=SERIES[0], zorder=3, label="PR-AUC")
    a2.set_xlim(0, 0.5)
    a2.set_title("PR-AUC vs acceptance base rate")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    a1.set_yticks(y, [label(a) for a in actions])
    a1.invert_yaxis()
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    save(fig, "model_performance.png")


def fig_policy_value(ev):
    names = ["NBA (expected value)", "Best single offer", "Most offered", "Random"]
    short = {"NBA (expected value)": "NBA policy", "Best single offer": "Always personal loan",
             "Most offered": "Always savings upsell", "Random": "Random offer"}
    panels = [("policy_value_per_offer_usd", "Value per offer (USD)",
               ev["policy_value_per_offer_usd"]["Logged (bank's actual offers)"]["observed"], "${:.2f}"),
              ("policy_acceptance_rate", "Acceptance rate",
               ev["logged_acceptance_rate"], "{:.1%}")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
    y = np.arange(len(names))
    for ax, (key, title, logged, fmt) in zip(axes, panels):
        est = [ev[key][n]["DR"] for n in names]
        colors = [SERIES[0] if n.startswith("NBA") else NEUTRAL for n in names]
        lo = [e["value"] - e["ci_low"] for e in est]
        hi = [e["ci_high"] - e["value"] for e in est]
        ax.errorbar([e["value"] for e in est], y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2,
                    capsize=3, zorder=2)
        ax.scatter([e["value"] for e in est], y, s=70, color=colors, zorder=3)
        for yi, e in zip(y, est):
            ax.annotate(fmt.format(e["value"]), (e["ci_high"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.axvline(logged, color=INK2, lw=1.2, ls="--")
        ax.annotate(f"bank's actual offers {fmt.format(logged)}", (logged, len(names) - 0.4), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=INK2)
        ax.set_title(title)
        ax.set_xlim(0, max(e["ci_high"] for e in est) * 1.3)
        ax.grid(axis="y", visible=False)
        if key == "policy_acceptance_rate":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_yticks(y, [short[n] for n in names])
    axes[0].set_ylim(len(names) - 0.2, -0.6)
    fig.text(0, -0.04, "Doubly robust off-policy estimates, test period Jan-Jun 2026 (4,120 logged offers). "
             "Whiskers: 95% bootstrap intervals.",
             fontsize=8.5, color=INK2)
    save(fig, "policy_value.png")


def grouped_share(ax, groups, actions, title):
    """Horizontal grouped bars of share per action; groups is [(name, {action: share})]."""
    h = 0.8 / len(groups)
    y = np.arange(len(actions))
    for i, (name, share) in enumerate(groups):
        bars = ax.barh(y + (i - (len(groups) - 1) / 2) * h, [share.get(a, 0) for a in actions], height=h - 0.04,
                       color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1)
        bar_labels(ax, bars, lambda v: f"{v:.0%}")
    ax.set_yticks(y, [label(a) for a in actions])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8.5)


def fig_recommendation_mix(ev):
    actions = sorted(ev["recommendation_share"]["Logged"])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    grouped_share(ax, [("Bank's actual offers", ev["recommendation_share"]["Logged"]),
                       ("NBA policy", ev["recommendation_share"]["NBA"])], actions,
                  "Offer mix: bank's actual offers vs NBA policy (test period)")
    ax.set_xlim(0, 0.6)
    save(fig, "recommendation_mix.png")


def fig_fairness(ev):
    fair = ev["fairness_by_gender"]
    actions = sorted(next(iter(fair.values()))["recommendation_share"])
    groups = [(f"{g.replace('_', ' ')} (n={f['n']:,})", f["recommendation_share"]) for g, f in fair.items()]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    grouped_share(ax, groups, actions, "NBA recommendations by gender (gender is not a model input)")
    ax.set_xlim(0, 0.62)
    save(fig, "fairness_gender.png")


def fig_drift(drift):
    psi = drift["psi"]
    names = list(psi)[::-1]
    vals = [psi[n]["value"] for n in names]
    cap = 0.3
    fig, ax = plt.subplots(figsize=(7, 5.2))
    colors = [NEUTRAL if n in drift["expected_trend"] else SERIES[0] for n in names]
    pretty = [f"score: {label(n[6:])}" if n.startswith("score:") else n for n in names]
    ax.barh(pretty, [min(v, cap) for v in vals], color=colors, height=0.7, edgecolor=SURFACE, linewidth=1)
    for i, (n, v) in enumerate(zip(names, vals)):
        text = f"{v:.2f}, expected trend (cumulative count)" if n in drift["expected_trend"] else f"{v:.3f}"
        ax.annotate(text, (min(v, cap), i), xytext=(-4 if v > cap else 4, 0), textcoords="offset points",
                    va="center", ha="right" if v > cap else "left", fontsize=8.5,
                    color=INK if v > cap else INK2)
    for x, name in [(drift["thresholds"]["warn"], "warn 0.10"), (drift["thresholds"]["alert"], "alert 0.25")]:
        ax.axvline(x, color=INK2, lw=1, ls="--")
        ax.annotate(name, (x, len(names) - 0.3), xytext=(3, 0), textcoords="offset points", fontsize=8.5,
                    color=INK2)
    ax.set_xlim(0, cap)
    ax.set_ylim(-0.6, len(names) + 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Data drift: PSI, 2025 training window vs Jan-Jun 2026")
    ax.set_xlabel("Population Stability Index (axis capped at 0.30)")
    save(fig, "drift_psi.png")


def fig_outcome_drift(retrain):
    res = retrain["results"]
    actions = list(res)
    y = np.arange(len(actions))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, (key, name) in enumerate([("champion", f"Trained to {retrain['champion_train_end']}"),
                                     ("challenger", f"Retrained to {retrain['challenger_train_end']}")]):
        bars = ax.barh(y + (i - 0.5) * h, [res[a][key]["relative_gap"] for a in actions], height=h - 0.04,
                       color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=1)
        bar_labels(ax, bars, lambda v: f"{v:+.0%}")
    ax.axvline(0, color=INK2, lw=1)
    for x in (-0.25, 0.25):
        ax.axvline(x, color=INK2, lw=1, ls="--")
    ax.annotate("alert ±25%", (0.25, -0.75), xytext=(3, 0), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [label(a) for a in actions])
    ax.set_ylim(len(actions) - 0.4, -1.0)
    ax.set_xlim(-0.35, 0.75)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Outcome drift: observed vs predicted acceptance, offers after {retrain['challenger_train_end']}")
    ax.set_xlabel("Observed / predicted acceptance - 1 (positive = model under-predicts)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    save(fig, "outcome_drift_retrain.png")


def fig_shap(train):
    actions = list(train)
    features = sorted({f for a in actions for f in train[a]["shap"]},
                      key=lambda f: -np.mean([train[a]["shap"].get(f, 0) / sum(train[a]["shap"].values())
                                              for a in actions]))
    share = np.array([[train[a]["shap"].get(f, 0) / sum(train[a]["shap"].values()) for a in actions]
                      for f in features])
    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(share, cmap=LinearSegmentedColormap.from_list("blues", BLUES), aspect="auto", vmin=0)
    for i in range(len(features)):
        for j in range(len(actions)):
            v = share[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v > share.max() * 0.55 else INK)
    ax.set_xticks(range(len(actions)), [label(a) for a in actions], rotation=20, ha="right")
    ax.set_yticks(range(len(features)), features)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Feature importance: share of mean |SHAP| per offer model")
    fig.colorbar(im, ax=ax, fraction=0.03, format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    save(fig, "shap_importance.png")


def results_md(train, tags, ev, drift, retrain):
    pv, acc = ev["policy_value_per_offer_usd"], ev["policy_acceptance_rate"]
    ci = lambda e, f: f"{f.format(e['value'])} [{f.format(e['ci_low'])}, {f.format(e['ci_high'])}]"
    lines = [
        "# Chapter 4 results",
        "",
        f"Training run `{ev['train_run_id']}`, git commit `{tags['git_commit'][:7]}`, "
        f"data sha256 `{tags['data_sha256'][:12]}`, image `{tags['image_id'][7:19]}`.",
        f"Out-of-time test period {ev['test_period'][0]} to {ev['test_period'][1]}, "
        f"{ev['n_test_offers']:,} logged offers. Generated by `make report`.",
        "",
        "## Acceptance models",
        "",
        "| Offer | Train n | Test n | Test acceptance | ROC-AUC | PR-AUC | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for a, m in train.items():
        lines.append(f"| {label(a)} | {m['n_train']:.0f} | {m['n_test']:.0f} | {m['pos_rate_test']:.1%} | "
                     f"{m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {ev['calibration'][a]['brier']:.4f} |")
    lines += ["", "## Policy evaluation (off-policy, test period)", "",
              "| Policy | Value per offer, USD, DR [95% CI] | SNIPS | IPS | Acceptance rate, DR [95% CI] | Matched offers |",
              "|---|---|---|---|---|---|"]
    for n in ["NBA (expected value)", "Best single offer", "Most offered", "Random"]:
        lines.append(f"| {n} | {ci(pv[n]['DR'], '{:.2f}')} | {pv[n]['SNIPS']['value']:.2f} | "
                     f"{pv[n]['IPS']['value']:.2f} | {ci(acc[n]['DR'], '{:.3f}')} | {pv[n]['matched_offers']:,} |")
    lg = pv["Logged (bank's actual offers)"]
    lines += [f"| Bank's actual offers (observed) | {lg['observed']:.2f} [{lg['ci_low']:.2f}, {lg['ci_high']:.2f}] "
              f"| | | {ev['logged_acceptance_rate']:.3f} | {ev['n_test_offers']:,} |",
              "",
              f"Propensity check: mean importance weight under a uniform policy = "
              f"{ev['propensity_check_mean_uniform_weight']:.3f} (1.0 if the logged propensities are proper "
              "probabilities over the five offers), so SNIPS and DR are preferred over IPS.",
              "", "## Calibration (test period, offers actually made)", "",
              "| Offer | Observed acceptance | Mean predicted | Brier |", "|---|---|---|---|"]
    for a, c in ev["calibration"].items():
        lines.append(f"| {label(a)} | {c['observed_rate']:.3f} | {c['mean_calibrated_proba']:.3f} | {c['brier']:.4f} |")
    lines += ["", "## Fairness by gender (gender is not a model feature)", "",
              "| Group | n | Acceptance | AUC | Share recommended personal loan |", "|---|---|---|---|---|"]
    for g, f in ev["fairness_by_gender"].items():
        lines.append(f"| {g.replace('_', ' ')} | {f['n']:,} | {f['acceptance_rate']:.3f} | "
                     f"{f['auc_logged_action']:.3f} | {f['recommendation_share'].get('Personal_Loan_Offer', 0):.1%} |")
    lines += ["", "## Drift (2025 training window vs Jan-Jun 2026)", "",
              "| Signal | Value | Status |", "|---|---|---|"]
    for n, v in drift["psi"].items():
        if v["status"] != "ok":
            note = " (expected trend, not used for retrain)" if n in drift["expected_trend"] else ""
            lines.append(f"| PSI {n} | {v['value']:.3f} | {v['status']}{note} |")
    lines.append(f"| All other features and model scores | PSI < {max(v['value'] for n, v in drift['psi'].items() if v['status'] == 'ok'):.3f} | ok |")
    for a, v in drift["outcome_drift"].items():
        lines.append(f"| Calibration gap {label(a)} | {v['relative_gap']:+.0%} | {v['status']} |")
    lines += [f"", f"Retrain recommended: **{drift['retrain_recommended']}** ({', '.join(drift['retrain_reasons'])}).",
              "", f"## Retraining (scored on offers after {retrain['challenger_train_end']})", "",
              "| Offer | AUC trained to " + retrain["champion_train_end"] + " | AUC retrained | Gap before | Gap after |",
              "|---|---|---|---|---|"]
    for a, r in retrain["results"].items():
        lines.append(f"| {label(a)} | {r['champion']['auc']:.3f} | {r['challenger']['auc']:.3f} | "
                     f"{r['champion']['relative_gap']:+.0%} | {r['challenger']['relative_gap']:+.0%} |")
    lines += ["", "## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted(FIGURES.glob("*.png"))]
    (REPORTS / "results.md").write_text("\n".join(lines) + "\n")


def main():
    ev = json.loads((REPORTS / "evaluation.json").read_text())
    drift = json.loads((REPORTS / "drift.json").read_text())
    retrain = json.loads((REPORTS / "retrain_after_drift.json").read_text())
    train, tags = training_results(ev["train_run_id"])
    fig_model_performance(train)
    fig_policy_value(ev)
    fig_recommendation_mix(ev)
    fig_fairness(ev)
    fig_drift(drift)
    fig_outcome_drift(retrain)
    fig_shap(train)
    results_md(train, tags, ev, drift, retrain)
    print(f"Wrote {len(list(FIGURES.glob('*.png')))} figures and {REPORTS / 'results.md'}")


if __name__ == "__main__":
    main()
