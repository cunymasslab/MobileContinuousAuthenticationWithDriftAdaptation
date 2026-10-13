# Publication-rendering reconstruction, 2026-10-04; not recovered historical code.
"""
generate_confound_figures.py — Publication figures for the TIFS confound experiment.

Reads results_confound/results_confound_summary.csv and produces six figures
in results_confound/figures/, each saved as both PNG (300 dpi) and PDF.

Run from the repo root:
    python generate_confound_figures.py

Reuses the seed-averaging and cluster-bootstrap conventions from
analyze_confound.py. Uses the pair_category column from the CSV; does not
re-derive categorization.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from _figure_params import (
    AXIS_LABEL_SIZE,
    FIG_HEIGHT,
    FIG_WIDTH,
    LEGEND_SIZE,
    SUBFIG_AXIS_LABEL_SIZE,
    SUBFIG_HEIGHT,
    SUBFIG_TICK_LABEL_SIZE,
    SUBFIG_WIDTH,
    TICK_LABEL_SIZE,
)

# ── Paths ──────────────────────────────────────────────────────────────────
SUMMARY_CSV = "./artifacts/reference/matched_stream_m9/results_confound/results_confound_summary.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CATEGORIES = ["temporal", "cross-activity", "cross-posture", "cross-both"]

# Grayscale publication presentation. Scientific category order is unchanged.
CAT_COLORS = {
    "temporal": "0.85",
    "cross-activity": "0.70",
    "cross-posture": "0.55",
    "cross-both": "0.40",
}
CAT_MARKERS = {
    "temporal": "o",
    "cross-activity": "^",
    "cross-posture": "s",
    "cross-both": "D",
}
CAT_FACE_COLORS = {
    "temporal": "none",
    "cross-activity": "0.80",
    "cross-posture": "none",
    "cross-both": "0.40",
}
STREAM_COLORS = {"no-drift": "white", "drift": "0.60"}
MODEL_COLORS = {"M1 (frozen)": "0.80", "M4 (adapter)": "0.40"}

N_BOOT = 10000
RNG = np.random.default_rng(42)


# ── Helpers ────────────────────────────────────────────────────────────────


def save_both(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)


def require_cols(df, cols, fig_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  [WARN] {fig_name}: missing columns {missing}, skipping.")
        return False
    return True


def cluster_bootstrap_point_ci(values_by_user, n_boot=N_BOOT, alpha=0.05):
    """User-weighted cluster bootstrap (matches analyze_confound.py).
    Each user contributes one number per resample (mean of their values).
    Returns (point_estimate, ci_lo, ci_hi)."""
    users = list(values_by_user.keys())
    if len(users) < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean([np.mean(values_by_user[u]) for u in users]))
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(users, size=len(users), replace=True)
        means[i] = float(np.mean([np.mean(values_by_user[u]) for u in sample]))
    return (
        point,
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def per_category_bootstrap(df, value_col):
    """Returns dict: category -> (point, lo, hi)."""
    out = {}
    for cat in CATEGORIES:
        sub = df[df.pair_category == cat]
        vals_by_user = {}
        for u in sub.target_user.unique():
            v = sub.loc[sub.target_user == u, value_col].dropna().to_numpy()
            if len(v) > 0:
                vals_by_user[u] = v
        out[cat] = (
            cluster_bootstrap_point_ci(vals_by_user)
            if vals_by_user
            else (float("nan"), float("nan"), float("nan"))
        )
    return out


def seed_averaged(df):
    """Collapse 3 seeds per (user, train, stream) into one row.
    Mirrors analyze_confound.py §2's groupby. pair_category is preserved
    (deterministic function of (train, stream) so doesn't split groups)."""
    return (
        df.groupby(
            ["target_user", "train_condition", "stream_condition", "pair_category"]
        )
        .agg(
            benefit_diff=("benefit_diff", "mean"),
            ks_mean=("ks_mean", "first"),
            classifier_accuracy=("classifier_accuracy", "first"),
            mmd_statistic=("mmd_statistic", "first"),
        )
        .reset_index()
    )


def err_lohi(per_cat):
    """Convert per-category (point, lo, hi) dict to matplotlib yerr [[low_err], [high_err]]."""
    return [
        [per_cat[c][0] - per_cat[c][1] for c in CATEGORIES],
        [per_cat[c][2] - per_cat[c][0] for c in CATEGORIES],
    ]


def points(per_cat):
    return [per_cat[c][0] for c in CATEGORIES]


# ── Figure generators ──────────────────────────────────────────────────────


def fig1_paired_bar_v2(df):
    name = "paired_bar_v2"
    if not require_cols(
        df,
        [
            "pair_category",
            "target_user",
            "adaptation_benefit_nodrift",
            "adaptation_benefit_drift",
        ],
        name,
    ):
        return
    print(f"  [{name}] computing per-category bootstrap CIs...")
    nd = per_category_bootstrap(df, "adaptation_benefit_nodrift")
    dr = per_category_bootstrap(df, "adaptation_benefit_drift")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(CATEGORIES))
    w = 0.35

    ax.bar(
        x - w / 2,
        points(nd),
        w,
        yerr=err_lohi(nd),
        capsize=4,
        color=STREAM_COLORS["no-drift"],
        hatch="///",
        label="Control",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        points(dr),
        w,
        yerr=err_lohi(dr),
        capsize=4,
        color=STREAM_COLORS["drift"],
        hatch="xx",
        label="Drift",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=15, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Adaptation benefit\n(EER reduction)", fontsize=AXIS_LABEL_SIZE)
    if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
        ax.set_title(
            "Adaptation Benefit by Drift Category and Stream Type",
            fontsize=AXIS_LABEL_SIZE,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_both(fig, name)


def _scatter_with_fit(ax, df, x_col, x_label, title):
    """Color scatter by category + linear regression line + Pearson r in legend."""
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for cat in CATEGORIES:
        sub = df[df.pair_category == cat]
        ax.scatter(
            sub[x_col],
            sub.benefit_diff,
            s=10,
            alpha=0.45,
            marker=CAT_MARKERS[cat],
            facecolors=CAT_FACE_COLORS[cat],
            label=cat,
            edgecolors="0.25",
            linewidths=0.4,
        )
    x = df[x_col].to_numpy()
    y = df.benefit_diff.to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        r, _ = pearsonr(x[mask], y[mask])
        xs = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(
            xs,
            slope * xs + intercept,
            color="black",
            linewidth=1.8,
            label=f"linear fit: slope={slope:+.3f}, r={r:+.2f}",
        )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(
        "Benefit difference\n(drift - control, EER)", fontsize=AXIS_LABEL_SIZE
    )
    if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
        ax.set_title(title, fontsize=AXIS_LABEL_SIZE)
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)


def fig2_benefit_vs_mmd(df):
    name = "benefit_diff_vs_mmd_v2"
    if not require_cols(
        df,
        [
            "pair_category",
            "target_user",
            "train_condition",
            "stream_condition",
            "benefit_diff",
            "mmd_statistic",
        ],
        name,
    ):
        return
    print(f"  [{name}] seed-averaging and plotting...")
    sa = seed_averaged(df)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    _scatter_with_fit(
        ax,
        sa,
        "mmd_statistic",
        "MMD statistic",
        "Adaptation Benefit Scales with Drift Magnitude (MMD)",
    )
    fig.tight_layout()
    save_both(fig, name)


def fig3_benefit_vs_ks(df):
    name = "benefit_diff_vs_ks_v2"
    if not require_cols(
        df,
        [
            "pair_category",
            "target_user",
            "train_condition",
            "stream_condition",
            "benefit_diff",
            "ks_mean",
        ],
        name,
    ):
        return
    print(f"  [{name}] seed-averaging and plotting...")
    sa = seed_averaged(df)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    _scatter_with_fit(
        ax,
        sa,
        "ks_mean",
        "Mean KS statistic",
        "Adaptation Benefit Scales with Drift Magnitude (KS)",
    )
    fig.tight_layout()
    save_both(fig, name)


def fig4_drift_by_category(df):
    name = "drift_magnitude_by_category"
    if not require_cols(
        df, ["pair_category", "mmd_statistic", "ks_mean", "classifier_accuracy"], name
    ):
        return
    print(f"  [{name}] building box plots...")
    metrics = [
        ("mmd_statistic", "MMD Statistic"),
        ("ks_mean", "Mean KS Statistic"),
        ("classifier_accuracy", "Classifier Accuracy"),
    ]
    sub_fig_fn_suffix = "a"
    for col, label in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(SUBFIG_WIDTH, SUBFIG_HEIGHT))
        data = [
            df.loc[df.pair_category == cat, col].dropna().to_numpy()
            for cat in CATEGORIES
        ]
        bp = ax.boxplot(
            data, labels=CATEGORIES, patch_artist=True, widths=0.6, showfliers=False
        )
        for patch, cat in zip(bp["boxes"], CATEGORIES):
            patch.set_facecolor(CAT_COLORS[cat])
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
        for med in bp["medians"]:
            med.set_color("black")
            med.set_linewidth(1.5)
        ax.set_ylabel(label, fontsize=SUBFIG_AXIS_LABEL_SIZE)
        if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
            ax.set_title(label, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", rotation=15, labelsize=SUBFIG_TICK_LABEL_SIZE)
        # fig.suptitle('Drift Magnitude by Category — Three Independent Metrics', y=0.98)
        fig.tight_layout()
        save_both(fig, f"{name}_{sub_fig_fn_suffix}")
        sub_fig_fn_suffix = chr(ord(sub_fig_fn_suffix) + 1)  # next letter


def fig5_eer_comparison(df):
    name = "per_category_eer_comparison"
    if not require_cols(
        df, ["pair_category", "target_user", "M1_drift_eer", "M4_drift_eer"], name
    ):
        return
    print(f"  [{name}] computing per-category bootstrap CIs...")
    m1 = per_category_bootstrap(df, "M1_drift_eer")
    m4 = per_category_bootstrap(df, "M4_drift_eer")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    x = np.arange(len(CATEGORIES))
    w = 0.35

    ax.bar(
        x - w / 2,
        points(m1),
        w,
        yerr=err_lohi(m1),
        capsize=4,
        color=MODEL_COLORS["M1 (frozen)"],
        label="M1 (frozen)",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        points(m4),
        w,
        yerr=err_lohi(m4),
        capsize=4,
        color=MODEL_COLORS["M4 (adapter)"],
        label="M4 (adapter)",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=15, fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Equal Error Rate (drift stream)", fontsize=AXIS_LABEL_SIZE)
    if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
        ax.set_title("Frozen vs Adapter EER on the Drift Stream by Category")
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_both(fig, name)


def fig6_forgetting(df):
    name = "forgetting_distribution"
    if not require_cols(df, ["M1_drift_eer", "M4_drift_eer"], name):
        return
    print(f"  [{name}] building histogram...")
    diff = (df.M4_drift_eer - df.M1_drift_eer).dropna().to_numpy()
    threshold = 0.05
    pct_forgetting = float((diff > threshold).mean() * 100)
    pct_helped = float((diff < 0).mean() * 100)
    median_diff = float(np.median(diff))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.hist(diff, bins=80, color="0.65", alpha=0.78, edgecolor="black", linewidth=0.3)
    ax.axvline(
        0,
        color="black",
        linewidth=0.7,
        label=f"no change (median observed: {median_diff:+.3f})",
    )
    ax.axvline(
        threshold,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label=f"forgetting threshold (+{threshold})",
    )
    ax.set_xlabel(
        "M4 EER − M1 EER on drift stream\n(negative = adaptation helped)",
        fontsize=AXIS_LABEL_SIZE,
    )
    ax.set_ylabel("Number of experiments", fontsize=AXIS_LABEL_SIZE)
    if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
        ax.set_title(
            f"Distribution of (M4 − M1) Drift-Stream EER\n"
            f"{pct_helped:.1f}% helped (<0), "
            f"{pct_forgetting:.1f}% above forgetting threshold (>+{threshold})"
        )
    ax.legend(loc="upper right", frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_both(fig, name)


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"ERROR: {SUMMARY_CSV} not found. Run from the repo root.")
        sys.exit(1)
    df = pd.read_csv(SUMMARY_CSV)
    df["benefit_diff"] = (
        df["adaptation_benefit_drift"] - df["adaptation_benefit_nodrift"]
    )

    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")
    print(f"Output dir : {FIG_DIR}/")
    print(f"Bootstrap  : {N_BOOT} resamples (user-level)")
    print()

    fig1_paired_bar_v2(df)
    fig2_benefit_vs_mmd(df)
    fig3_benefit_vs_ks(df)
    fig4_drift_by_category(df)
    fig5_eer_comparison(df)
    fig6_forgetting(df)

    print()
    print("Done. Figures (PNG + PDF):")
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith((".png", ".pdf")):
            print(f"  {os.path.join(FIG_DIR, f)}")


if __name__ == "__main__":
    main()
