# Publication-rendering reconstruction, 2026-10-04; not recovered historical code.
"""
generate_forgetting_histogram.py — TIFS figure: distribution of the adaptation
forgetting metric (M2 minus M1 end-of-stream EER) from the main factorial
summary CSV.

Usage:
    python generate_forgetting_histogram.py \
        m7_repaired_20260901/results_stage4_merged_summary_m7_repaired_20260901.csv

Outputs (figures/ in cwd, created if missing):
    figures/forgetting_histogram_m2.png   (dpi=300)
    figures/forgetting_histogram_m2.pdf
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from _figure_params import (
    AXIS_LABEL_SIZE,
    FIG_HEIGHT,
    FIG_WIDTH,
    LEGEND_SIZE,
    TICK_LABEL_SIZE,
)

FIG_DIR = "figures"
FORGETTING_THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "csv_path", help="Path to results_stage4_merged_summary_with_m7.csv"
    )
    args = ap.parse_args()

    # ── Load ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: CSV not found at: {args.csv_path}", file=sys.stderr)
        print("Check the path; expected an existing file.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"ERROR loading {args.csv_path}: {type(e).__name__}: {e}", file=sys.stderr
        )
        sys.exit(1)

    # ── Step 1 — row count ─────────────────────────────────────────────────
    print(f"data rows: {len(df)}")
    print()

    # ── Step 2 — compute delta (M2_end_eer - M1_end_eer) per experiment ───
    needed = ["M1_end_eer", "M2_end_eer"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(
            f"ERROR: missing columns {missing}; cannot compute delta.", file=sys.stderr
        )
        sys.exit(1)

    # Coerce non-numeric (failed-experiment blanks) to NaN before subtraction.
    m1 = pd.to_numeric(df["M1_end_eer"], errors="coerce")
    m2 = pd.to_numeric(df["M2_end_eer"], errors="coerce")
    delta = (m2 - m1).dropna().to_numpy()

    # ── Step 3 — verification numbers ──────────────────────────────────────
    n_valid = int(delta.size)
    if n_valid == 0:
        print(
            "ERROR: no rows had both M1_end_eer and M2_end_eer defined.",
            file=sys.stderr,
        )
        sys.exit(1)
    n_helped = int((delta < 0).sum())
    n_forget = int((delta > FORGETTING_THRESHOLD).sum())
    pct_helped = n_helped / n_valid * 100
    pct_forget = n_forget / n_valid * 100
    median_d = float(np.median(delta))

    print(
        f"Valid (M1_end_eer & M2_end_eer both present): {n_valid:,} " f"of {len(df):,}"
    )
    print(
        f"  delta < 0  (adaptation helped)               : "
        f"{n_helped:,}  ({pct_helped:5.1f}%)"
    )
    print(f"  median delta                                  : " f"{median_d:+.4f}")
    print(
        f"  delta > {FORGETTING_THRESHOLD} (forgetting threshold)        : "
        f"{n_forget:,}  ({pct_forget:5.1f}%)"
    )
    print()

    # ── Step 4 — histogram ─────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.hist(delta, bins=50, color="0.65", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0, label="No change")
    ax.axvline(
        FORGETTING_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Forgetting threshold (+{FORGETTING_THRESHOLD})",
    )
    ax.set_xlabel("End-of-stream EER difference (M2 - M1)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Number of configurations", fontsize=AXIS_LABEL_SIZE)
    if os.getenv("HAS_FIG_TITLE", "No").lower() == "yes":
        ax.set_title(
            "Histogram of End-of-Stream EER Differences (M2 - M1)",
            fontsize=AXIS_LABEL_SIZE,
        )
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.set_xlabel("End-of-stream EER difference (M2 - M1)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Number of configurations", fontsize=AXIS_LABEL_SIZE)
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, "forgetting_histogram_m2.png")
    pdf_path = os.path.join(FIG_DIR, "forgetting_histogram_m2.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Wrote figures:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


if __name__ == "__main__":
    main()
