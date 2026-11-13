"""All plots used in the report and dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import FIG_DIR, INPUTS, TARGETS
from src.evaluate import predict_model
from src.train import MODEL_NAMES

LABEL = {"temperature_c": "Temperature (C)", "pressure_bar": "Pressure (bar)",
         "flow_rate_m3h": "Flow rate (m3/h)", "energy_kw": "Energy (kW)"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / name, dpi=130); plt.close(fig)


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(df[INPUTS + TARGETS].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap (inputs and targets)"); _save(fig, "01_correlation_heatmap.png")


def eda_overview(df):
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), INPUTS + TARGETS):
        sns.histplot(df[c], bins=40, ax=ax, color="steelblue"); ax.set_title(c, fontsize=9)
    fig.suptitle("Distributions of cleaned sensor variables (synthetic data)"); _save(fig, "02_distributions.png")
    seg = df.iloc[2000:2576]   # 2 days
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, t in zip(axes, TARGETS):
        ax.plot(seg["timestamp"], seg[t], lw=0.8); ax.set_ylabel(LABEL[t], fontsize=8)
    fig.suptitle("Two days of process outputs (setpoint changes every few hours)"); _save(fig, "03_timeseries.png")


def actual_vs_predicted(bundle, test_df, name=None):
    name = name or bundle["primary"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, t in zip(axes.ravel(), TARGETS):
        p = predict_model(bundle, test_df, t, name); y = test_df[t].values
        ax.scatter(y, p, s=3, alpha=0.3); lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.set_title(f"{LABEL[t]}  (R2={r2:.3f})", fontsize=10); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    fig.suptitle(f"Actual vs predicted on held-out test set - {name}"); _save(fig, "04_actual_vs_predicted.png")


def residual_distribution(bundle, test_df, name=None):
    name = name or bundle["primary"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, t in zip(axes.ravel(), TARGETS):
        r = test_df[t].values - predict_model(bundle, test_df, t, name)
        sns.histplot(r, bins=60, kde=True, ax=ax); ax.axvline(0, color="r", lw=1)
        ax.set_title(f"{LABEL[t]} residuals (mean={r.mean():.3f}, std={r.std():.3f})", fontsize=9)
    fig.suptitle(f"Residual distribution (actual - predicted) - {name}"); _save(fig, "05_residuals.png")


def model_comparison(metrics):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, t in zip(axes, TARGETS):
        m = metrics[metrics.target == t].set_index("model").loc[MODEL_NAMES]
        ax.bar(range(len(m)), m["RMSE"], color=["#bbb", "#999", "#4c9", "#37c"])
        ax.set_xticks(range(len(m))); ax.set_xticklabels(["Lin raw", "Lin eng", "RF", "XGB"], fontsize=8)
        for i, (v, r2) in enumerate(zip(m["RMSE"], m["R2"])):
            ax.text(i, v, f"R2={r2:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"{LABEL[t]} - RMSE (lower is better)", fontsize=9)
    _save(fig, "06_model_comparison.png")


def importance_plot(imp, name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, t in zip(axes.ravel(), TARGETS):
        s = imp[t].sort_values().tail(8); ax.barh(s.index, s.values, color="steelblue"); ax.set_title(LABEL[t], fontsize=10)
        ax.set_xlabel("Permutation importance (drop in R2)")
    fig.suptitle(f"Feature importance on test data - {name}"); _save(fig, "07_feature_importance.png")


def whatif_plot(sweeps):
    """sweeps: list of (variable, x, pred_df, sim_df, targets_to_show)"""
    fig, axes = plt.subplots(1, len(sweeps), figsize=(7 * len(sweeps), 4.5))
    for ax, (var, x, pred, sim, shown) in zip(np.atleast_1d(axes), sweeps):
        for t in shown:
            l, = ax.plot(x, sim[t], lw=2, alpha=0.5, label=f"{LABEL[t]} - simulator")
            ax.plot(x, pred[t], "--", color=l.get_color(), label=f"{LABEL[t]} - model")
        ax.set_xlabel(var); ax.legend(fontsize=7); ax.set_title(f"What-if: sweep {var}", fontsize=10)
    _save(fig, "08_what_if.png")
