"""
generate_forgetting_histogram.py — TIFS figure: distribution of the adaptation
forgetting metric (M2 minus M1 end-of-stream EER) from the main factorial
summary CSV.

Usage:
    python generate_forgetting_histogram.py results_stage4_merged_summary_with_m7.csv

Outputs (figures/ in cwd, created if missing):
    figures/forgetting_histogram_m2.png   (dpi=150)
    figures/forgetting_histogram_m2.pdf
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIG_DIR              = 'figures'
FORGETTING_THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('csv_path',
                    help='Path to results_stage4_merged_summary_with_m7.csv')
    args = ap.parse_args()

    # ── Load ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_path, low_memory=False)
    except FileNotFoundError:
        print(f'ERROR: CSV not found at: {args.csv_path}', file=sys.stderr)
        print('Check the path; expected an existing file.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR loading {args.csv_path}: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(1)

    # ── Step 1 — row count ─────────────────────────────────────────────────
    print(f'data rows: {len(df)}')
    print()

    # ── Step 2 — compute delta (M2_end_eer - M1_end_eer) per experiment ───
    needed = ['M1_end_eer', 'M2_end_eer']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f'ERROR: missing columns {missing}; cannot compute delta.',
              file=sys.stderr)
        sys.exit(1)

    # Coerce non-numeric (failed-experiment blanks) to NaN before subtraction.
    m1 = pd.to_numeric(df['M1_end_eer'], errors='coerce')
    m2 = pd.to_numeric(df['M2_end_eer'], errors='coerce')
    delta = (m2 - m1).dropna().to_numpy()

    # ── Step 3 — verification numbers ──────────────────────────────────────
    n_valid    = int(delta.size)
    if n_valid == 0:
        print('ERROR: no rows had both M1_end_eer and M2_end_eer defined.',
              file=sys.stderr)
        sys.exit(1)
    n_helped   = int((delta < 0).sum())
    n_forget   = int((delta > FORGETTING_THRESHOLD).sum())
    pct_helped = n_helped / n_valid * 100
    pct_forget = n_forget / n_valid * 100
    median_d   = float(np.median(delta))

    print(f'Valid (M1_end_eer & M2_end_eer both present): {n_valid:,} '
          f'of {len(df):,}')
    print(f'  delta < 0  (adaptation helped)               : '
          f'{n_helped:,}  ({pct_helped:5.1f}%)')
    print(f'  median delta                                  : '
          f'{median_d:+.4f}')
    print(f'  delta > {FORGETTING_THRESHOLD} (forgetting threshold)        : '
          f'{n_forget:,}  ({pct_forget:5.1f}%)')
    print(f'    (Results-text claim: ~9.4%)')
    print()

    # ── Step 4 — histogram ─────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta, bins=50, color='#4c72b0', edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax.axvline(0.0, color='black', linewidth=1.0,
               label='no change')
    ax.axvline(FORGETTING_THRESHOLD, color='red', linestyle='--',
               linewidth=1.5,
               label=f'forgetting threshold (+{FORGETTING_THRESHOLD})')
    ax.set_xlabel('End-of-stream EER change from adaptation (M2 minus M1)')
    ax.set_ylabel('Number of experiments')
    ax.set_title('Distribution of Adaptation Effect (Main Factorial, M2 vs M1)')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, 'forgetting_histogram_m2.png')
    pdf_path = os.path.join(FIG_DIR, 'forgetting_histogram_m2.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    print('Wrote figures:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')


if __name__ == '__main__':
    main()