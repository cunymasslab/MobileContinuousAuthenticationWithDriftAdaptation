"""
generate_drift_detection_figure.py — TIFS figure: drift magnitude by category
from the cross-validation subset's drift-detection CSV
(run_drift_detection.py output).

Usage:
    python generate_drift_detection_figure.py results_stage4b_exp1/results_drift_detection.csv

Outputs (figures/ in cwd, created if missing):
    figures/drift_detection_boxplots.png   (dpi=150)
    figures/drift_detection_boxplots.pdf
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


EXPECTED_ROWS  = 50
FIG_DIR        = 'figures'
# Categories that appear in this CV subset (cross-posture is intentionally absent).
CATEGORY_ORDER = ['temporal', 'cross-activity', 'cross-both']
ALL_CATEGORIES = ['temporal', 'cross-posture', 'cross-activity', 'cross-both']


def classify_pair(train_condition, stream_condition):
    """Derive drift category from train vs stream condition labels.

    Conditions are 'ACTIVITY+POSTURE' (e.g. READING+WALKING).
    Returns: temporal, cross-posture, cross-activity, or cross-both.
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
                    help='Path to results_drift_detection.csv')
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

    # ── Step 1 — row count + columns ───────────────────────────────────────
    print(f'data rows: {len(df)}')
    if len(df) != EXPECTED_ROWS:
        print(f'WARNING: expected {EXPECTED_ROWS} rows, got {len(df)}.')
    print(f'columns ({len(df.columns)}):')
    for c in df.columns:
        print(f'  - {c}')
    print()

    # ── Step 2 — derive drift category per row ─────────────────────────────
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

    cat_counts = df['pair_category'].value_counts()
    print('Per-category counts:')
    for cat in ALL_CATEGORIES:
        n = int(cat_counts.get(cat, 0))
        note = ''
        if cat == 'cross-posture' and n == 0:
            note = '  (expected absent from this subset)'
        print(f'  {cat:16s} n={n:3d}{note}')
    print()

    # ── Step 3 — per-category medians ──────────────────────────────────────
    needed = ['mean_ks_stat', 'classifier_accuracy', 'n_significant_ks']
    missing3 = [c for c in needed if c not in df.columns]
    if missing3:
        print(f'ERROR: missing columns {missing3}; cannot compute medians.',
              file=sys.stderr)
        sys.exit(1)

    print('Per-category medians:')
    g = df.groupby('pair_category').agg(
        mean_ks_stat=('mean_ks_stat',        'median'),
        classifier_accuracy=('classifier_accuracy', 'median'),
        n_significant_ks=('n_significant_ks',     'median'),
    )
    for cat in ALL_CATEGORIES:
        if cat not in g.index:
            continue
        row = g.loc[cat]
        print(f'  {cat:16s}  '
              f'mean_ks={row["mean_ks_stat"]:.3f}  '
              f'clf_acc={row["classifier_accuracy"]:.3f}  '
              f'n_sig_ks={row["n_significant_ks"]:.1f}')
    print()

    # ── Step 4 — overall summaries vs draft claims ─────────────────────────
    ks_min = float(df['mean_ks_stat'].min())
    ks_max = float(df['mean_ks_stat'].max())
    pct_hi = float((df['classifier_accuracy'] > 0.95).mean() * 100)
    print('Overall summary:')
    print(f'  mean_ks_stat range            : [{ks_min:.3f}, {ks_max:.3f}]')
    print(f'    (draft claims 0.21 to 0.77)')
    print(f'  classifier_accuracy > 0.95    : {pct_hi:5.1f}% of rows')
    print(f'    (draft claims "exceeds 0.95 for most users")')
    print()

    # ── Step 5 — three-panel box plot figure ───────────────────────────────
    plot_cats = [c for c in CATEGORY_ORDER if cat_counts.get(c, 0) > 0]
    if len(plot_cats) == 0:
        print('ERROR: no plottable categories.', file=sys.stderr)
        sys.exit(1)
    if len(plot_cats) != len(CATEGORY_ORDER):
        omitted = set(CATEGORY_ORDER) - set(plot_cats)
        print(f'NOTE: omitting empty categories from plot: {sorted(omitted)}')

    panels = [
        ('mean_ks_stat',        'Mean KS Statistic'),
        ('max_ks_stat',         'Max KS Statistic'),
        ('classifier_accuracy', 'Classifier Accuracy'),
    ]

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (col, title) in zip(axes, panels):
        if col not in df.columns:
            ax.text(0.5, 0.5, f'column "{col}" missing',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.set_axis_off()
            continue
        data = [df.loc[df['pair_category'] == cat, col].dropna().to_numpy()
                for cat in plot_cats]
        bp = ax.boxplot(data, labels=plot_cats, patch_artist=True,
                        widths=0.6, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#4c72b0')
            patch.set_alpha(0.65)
            patch.set_edgecolor('black')
        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(1.5)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        'Drift Magnitude by Category (50-Combination Cross-Validation Subset)',
        y=1.02,
    )
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, 'drift_detection_boxplots.png')
    pdf_path = os.path.join(FIG_DIR, 'drift_detection_boxplots.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    print('Wrote figures:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')


if __name__ == '__main__':
    main()