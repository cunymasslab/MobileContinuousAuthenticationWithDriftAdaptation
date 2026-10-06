# Publication-rendering reconstruction, 2026-10-04; not recovered historical code.
"""
generate_drift_gap_figure.py — TIFS figure: distribution of drift gaps across
user × condition combinations from the cross-validation drift verification
(run_stage4b_exp1.py output).

Usage:
    python generate_drift_gap_figure.py results_stage4b_exp1/results_stage4b_cv_summary.csv

Outputs (figures/ in cwd, created if missing):
    figures/drift_gap_distribution.png   (dpi=300)
    figures/drift_gap_distribution.pdf
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EXPECTED_ROWS = 50
EXPECTED_MEDIANS = {
    'within_eer':   0.069,
    'session4_eer': 0.416,
    'drift_gap':    0.319,
}
FIG_DIR = 'figures'
CATEGORY_ORDER = ['temporal', 'cross-posture', 'cross-activity', 'cross-both']


def classify_pair(train_condition, stream_condition):
    """Derive drift category from train vs stream condition labels.

    Conditions are 'ACTIVITY+POSTURE' (e.g. READING+WALKING).
    Returns one of: temporal, cross-posture, cross-activity, cross-both.
    """
    if train_condition == stream_condition:
        return 'temporal'
    t_act, t_pos = train_condition.split('+')
    s_act, s_pos = stream_condition.split('+')
    if t_act == s_act:
        return 'cross-posture'
    if t_pos == s_pos:
        return 'cross-activity'
    return 'cross-both'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('csv_path',
                    help='Path to results_stage4b_cv_summary.csv')
    args = ap.parse_args()

    # ── Load ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f'ERROR: CSV not found at: {args.csv_path}', file=sys.stderr)
        print('Check the path; expected an existing file.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR loading {args.csv_path}: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(1)

    # ── Step 1 — print row count and columns ───────────────────────────────
    print(f'data rows: {len(df)}')
    if len(df) != EXPECTED_ROWS:
        print(f'WARNING: expected {EXPECTED_ROWS} rows, got {len(df)}.')
    print(f'columns ({len(df.columns)}):')
    for c in df.columns:
        print(f'  - {c}')
    print()

    # ── Step 2 — median sanity check ───────────────────────────────────────
    print('Median values (vs expected):')
    for col, expected in EXPECTED_MEDIANS.items():
        if col not in df.columns:
            print(f'  {col:14s}  MISSING from CSV')
            continue
        actual = float(df[col].median())
        diff = abs(actual - expected)
        flag = '  OK' if diff < 0.01 else '  MISMATCH (check the input file)'
        print(f'  {col:14s}  actual={actual:.3f}  expected={expected:.3f}'
              f'  diff={diff:+.3f}{flag}')
    print()

    # ── Step 3 — tail percentages vs draft claims ──────────────────────────
    if 'drift_gap' not in df.columns:
        print('ERROR: drift_gap column missing; cannot continue.',
              file=sys.stderr)
        sys.exit(1)

    gap = df['drift_gap'].dropna()
    pct_high = float((gap > 0.50).mean() * 100)
    pct_low  = float((gap < 0.10).mean() * 100)
    print('Drift gap range checks:')
    print(f'  drift_gap > 0.50    : {pct_high:5.1f}%   (draft claims ~38%)')
    print(f'  drift_gap < 0.10    : {pct_low:5.1f}%   (draft claims ~28%)')
    print()

    # ── Step 4 — derive drift category per row ─────────────────────────────
    required = ['train_condition', 'stream_condition']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f'ERROR: missing columns {missing}; cannot classify pairs.',
              file=sys.stderr)
        sys.exit(1)

    df['pair_category'] = df.apply(
        lambda r: classify_pair(r['train_condition'], r['stream_condition']),
        axis=1,
    )

    print('Per-category counts and median drift_gap:')
    g = df.groupby('pair_category')['drift_gap'].agg(['count', 'median'])
    g = g.reindex(CATEGORY_ORDER)
    for cat in CATEGORY_ORDER:
        row = g.loc[cat]
        n = int(row['count']) if pd.notna(row['count']) else 0
        if pd.notna(row['median']):
            med_str = f'{row["median"]:.3f}'
        else:
            med_str = 'NaN (no rows in category)'
        print(f'  {cat:16s} n={n:3d}   median drift_gap={med_str}')
    print()

    # ── Step 5 — plot histogram ────────────────────────────────────────────
    median_gap = float(gap.median())

    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gap, bins=20, color='0.65', edgecolor='black',
            linewidth=0.5, alpha=0.85)
    ax.axvline(median_gap, color='black', linestyle='--', linewidth=1.5,
               label=f'median = {median_gap:.3f}')
    ax.set_xlabel('Drift gap (session-4 EER minus within-distribution EER)')
    ax.set_ylabel('Number of combinations')
    ax.set_title('Distribution of Drift Gaps Across User-Condition Combinations')
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, 'drift_gap_distribution.png')
    pdf_path = os.path.join(FIG_DIR, 'drift_gap_distribution.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print('Wrote figures:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')


if __name__ == '__main__':
    main()