"""
generate_m9_sweep_figure.py — TIFS figure: M9 (persistent-state RTRL) mean EER
across online learning rates, compared against the M2 per-window baseline.

Usage:
    python generate_m9_sweep_figure.py results_m9_lr_sweep/comparison.csv

Outputs (figures/ in cwd, created if missing):
    figures/m9_lr_sweep.png   (dpi=150)
    figures/m9_lr_sweep.pdf
"""
import os
import sys
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FIG_DIR = 'figures'
M9_LRS  = [0.0001, 0.0005, 0.001, 0.002]
M2_LR   = 0.002

# Values claimed in the V-E draft text, for side-by-side reporting only.
DRAFT_M9_MEANS = {0.0001: 0.321, 0.0005: 0.258, 0.001: 0.210, 0.002: 0.160}
DRAFT_M2_MEAN  = 0.151


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('csv_path',
                    help='Path to results_m9_lr_sweep/comparison.csv')
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

    needed = ['model', 'lr_online', 'mean_overall_eer', 'n_with_eer']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f'ERROR: missing columns {missing}.', file=sys.stderr)
        sys.exit(1)

    # ── Step 1 — print every row ───────────────────────────────────────────
    print(f'data rows: {len(df)}')
    print('All rows in CSV:')
    print(f'  {"model":<6s}  {"lr_online":>10s}  {"mean_eer":>10s}  '
          f'{"n_with_eer":>10s}')
    for _, r in df.iterrows():
        print(f'  {str(r["model"]):<6s}  {float(r["lr_online"]):10.4f}  '
              f'{float(r["mean_overall_eer"]):10.4f}  '
              f'{int(r["n_with_eer"]):10d}')
    print()

    # ── Step 2 — lookup M2 baseline + 4 M9 entries ─────────────────────────
    def _lookup(model_name, lr):
        sub = df[(df['model'] == model_name) &
                 (abs(df['lr_online'] - lr) < 1e-9)]
        if len(sub) == 0:
            return None
        return float(sub.iloc[0]['mean_overall_eer'])

    m2_baseline = _lookup('M2', M2_LR)
    m9_means    = {lr: _lookup('M9', lr) for lr in M9_LRS}

    print('Extracted values vs draft claims:')
    m2_str = f'{m2_baseline:.4f}' if m2_baseline is not None else 'MISSING'
    print(f'  M2 baseline @ lr={M2_LR}    : actual={m2_str}   '
          f'(draft claims {DRAFT_M2_MEAN:.3f})')
    print()
    print(f'  {"lr_online":>10s}  {"M9 actual":>12s}  {"M9 draft":>10s}')
    for lr in M9_LRS:
        actual = m9_means[lr]
        actual_str = f'{actual:.4f}' if actual is not None else 'MISSING'
        print(f'  {lr:10.4f}  {actual_str:>12s}  '
              f'{DRAFT_M9_MEANS[lr]:10.3f}')
    print()

    # Defensive bail-out before plotting
    if m2_baseline is None:
        print('ERROR: M2 baseline row missing; cannot draw reference line.',
              file=sys.stderr)
        sys.exit(1)
    missing_lrs = [lr for lr, v in m9_means.items() if v is None]
    if missing_lrs:
        print(f'ERROR: M9 row(s) missing for lr={missing_lrs}.',
              file=sys.stderr)
        sys.exit(1)

    # ── Step 3 — bar chart with reference line ─────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos  = list(range(len(M9_LRS)))
    values = [m9_means[lr] for lr in M9_LRS]
    ax.bar(x_pos, values, width=0.6, color='#4c72b0',
           edgecolor='black', linewidth=0.5, alpha=0.85)

    # Annotate each M9 bar with its numeric value
    for x, v in zip(x_pos, values):
        ax.text(x, v + 0.005, f'{v:.3f}', ha='center', va='bottom',
                fontsize=10)

    # M2 baseline reference line (dashed)
    ax.axhline(m2_baseline, color='red', linestyle='--', linewidth=1.5,
               label=f'M2 per-window baseline ({m2_baseline:.3f})')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{lr:g}' for lr in M9_LRS])
    ax.set_xlabel('Online learning rate')
    ax.set_ylabel('Mean EER (cross-both subset)')
    ax.set_title(
        'Persistent-State RTRL (M9) vs Per-Window M2 Across Learning Rates'
    )
    # Y-axis headroom so the annotation text isn't clipped at the top
    ymax = max(max(values), m2_baseline) * 1.18
    ax.set_ylim(0, ymax)
    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, 'm9_lr_sweep.png')
    pdf_path = os.path.join(FIG_DIR, 'm9_lr_sweep.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    print('Wrote figures:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')


if __name__ == '__main__':
    main()