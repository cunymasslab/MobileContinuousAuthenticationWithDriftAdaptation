#!/usr/bin/env python3
"""
Statistical + per-user analysis of run_touchalytics_raw_adaptation results.

Two analyses:
  1) Wilcoxon signed-rank tests on per-user (frozen vs adapting) EER pairs
     for each drift type. Reports n, mean delta, median delta, W, p-value.
  2) Per-user breakdown: win rate, big-win / big-loss fractions, best /
     worst delta, plus scatter plots (frozen on x, adapting on y; points
     below diagonal = adaptation helps).

The script averages each user's EER across seeds first (so each (user,
drift) contributes one paired observation, avoiding within-user correlation).

Usage:
    python analyze_touchalytics_raw_results.py
    python analyze_touchalytics_raw_results.py --no-plots
    python analyze_touchalytics_raw_results.py --csv path/to/other.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy import stats


CSV_PATH = 'results_touchalytics_raw/touchalytics_raw_adaptation_results.csv'
FIG_DIR = 'results_touchalytics_raw/figures'

MODEL_IDS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
DRIFT_ORDER = ['temporal', 'cross_activity', 'cross_both']

# Headline frozen -> adapting comparisons
PAIRS = [
    ('M1', 'M2', 'VanillaRNN: M1 (frozen) vs M2 (RTRL)'),
    ('M1', 'M4', 'VanillaRNN: M1 (frozen) vs M4 (BPTT-finetune)'),
    ('M6', 'M7', 'GRU: M6 (frozen) vs M7 (RTRL)'),
    ('M6', 'M8', 'GRU: M6 (frozen) vs M8 (BPTT-finetune)'),
]


# ── CSV loading ──────────────────────────────────────────────────────────────

def load_per_user(csv_path):
    """Load CSV and average each model's EER over seeds per (user, drift)."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                for m in MODEL_IDS:
                    r[f'{m}_overall_eer'] = float(r[f'{m}_overall_eer'])
            except (ValueError, KeyError, TypeError):
                continue
            rows.append(r)

    by_user_drift = {}
    for r in rows:
        key = (r['user_id'], r['drift_type'])
        by_user_drift.setdefault(key, []).append(
            {m: r[f'{m}_overall_eer'] for m in MODEL_IDS})

    per_user = {}
    for (uid, drift), seed_rows in by_user_drift.items():
        per_user[(uid, drift)] = {
            m: float(np.mean([s[m] for s in seed_rows]))
            for m in MODEL_IDS
        }
    return per_user


def collect_pair(per_user, drift, m_frozen, m_adapt):
    """Return (frozen_eers, adapt_eers, deltas=frozen-adapt) for one drift."""
    frozen, adapt = [], []
    for (uid, d), ms in per_user.items():
        if d == drift:
            frozen.append(ms[m_frozen])
            adapt.append(ms[m_adapt])
    frozen = np.array(frozen)
    adapt = np.array(adapt)
    return frozen, adapt, frozen - adapt


# ── Analysis 1: Wilcoxon ─────────────────────────────────────────────────────

def wilcoxon_analysis(per_user):
    print('\n' + '=' * 92)
    print('WILCOXON SIGNED-RANK TESTS (paired per-user EER, frozen vs adapting)')
    print('=' * 92)
    print('Positive delta = adapting model has LOWER EER (adaptation helps).')
    print('Two-sided test. Significance: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.')
    print()
    print(f"{'Drift':<16} {'Pair':<10} {'n':>4} "
          f"{'mean Δ':>10} {'median Δ':>10} {'W':>10} {'p':>10}  sig")
    print('-' * 92)

    for drift in DRIFT_ORDER:
        for m_f, m_a, _ in PAIRS:
            frozen, adapt, delta = collect_pair(per_user, drift, m_f, m_a)
            n = len(delta)
            if n == 0:
                continue
            mean_d = float(np.mean(delta))
            median_d = float(np.median(delta))
            if np.allclose(delta, 0):
                W_str, p_str, sig = '   n/a', '   n/a', 'eq '
            else:
                W, p = stats.wilcoxon(frozen, adapt, zero_method='wilcox',
                                      alternative='two-sided')
                W_str = f'{W:>10.2f}'
                p_str = f'{p:>10.4f}'
                sig = ('***' if p < 0.001 else
                       '** ' if p < 0.01 else
                       '*  ' if p < 0.05 else
                       'ns ')
            label = f'{m_f}->{m_a}'
            print(f'{drift:<16} {label:<10} {n:>4} '
                  f'{mean_d:>+10.4f} {median_d:>+10.4f} '
                  f'{W_str} {p_str}  {sig}')
        print()
    print('=' * 92)


# ── Analysis 2: per-user breakdown + scatter plots ───────────────────────────

def per_user_analysis(per_user, make_plots=True):
    print('\n' + '=' * 92)
    print('PER-USER BREAKDOWN')
    print('=' * 92)
    print('wins        = fraction of users where adapting gives LOWER EER than frozen')
    print('big wins    = fraction with Δ > +0.02   (>= 2 EER-point improvement)')
    print('big losses  = fraction with Δ < -0.02   (>= 2 EER-point degradation)')
    print()
    print(f"{'Drift':<16} {'Pair':<10} {'n':>4} "
          f"{'wins':>8} {'big wins':>10} {'big losses':>12} "
          f"{'best Δ':>10} {'worst Δ':>10}")
    print('-' * 92)

    for drift in DRIFT_ORDER:
        for m_f, m_a, _ in PAIRS:
            frozen, adapt, delta = collect_pair(per_user, drift, m_f, m_a)
            n = len(delta)
            if n == 0:
                continue
            wins = 100.0 * float(np.mean(delta > 0))
            big_wins = 100.0 * float(np.mean(delta > 0.02))
            big_losses = 100.0 * float(np.mean(delta < -0.02))
            best = float(np.max(delta))
            worst = float(np.min(delta))
            label = f'{m_f}->{m_a}'
            print(f'{drift:<16} {label:<10} {n:>4} '
                  f'{wins:>7.1f}% {big_wins:>9.1f}% {big_losses:>11.1f}% '
                  f'{best:>+10.4f} {worst:>+10.4f}')
        print()
    print('=' * 92)

    if not make_plots:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[matplotlib not installed — skipping scatter plots]')
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'\nWriting scatter plots to {FIG_DIR}/ ...')

    for m_f, m_a, label in PAIRS:
        fig, axes = plt.subplots(1, len(DRIFT_ORDER), figsize=(13, 4.4),
                                 sharex=True, sharey=True)
        for ax, drift in zip(axes, DRIFT_ORDER):
            frozen, adapt, delta = collect_pair(per_user, drift, m_f, m_a)
            if len(frozen) == 0:
                ax.set_title(f'{drift}: no data')
                ax.set_xlabel(f'{m_f} EER (frozen)')
                ax.set_ylabel(f'{m_a} EER (adapting)')
                continue

            hi = max(float(frozen.max()), float(adapt.max())) * 1.05
            lo = 0.0
            ax.plot([lo, hi], [lo, hi], 'k--', lw=0.7, alpha=0.5)
            # Colour users where adaptation helps (delta > 0) green, hurts red
            colors = ['#3a9d3a' if d > 0 else '#cc3333' for d in delta]
            ax.scatter(frozen, adapt, c=colors, s=45, alpha=0.75,
                       edgecolor='black', linewidth=0.4)

            wins_pct = 100.0 * float(np.mean(delta > 0))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f'{m_f} EER (frozen)')
            ax.set_ylabel(f'{m_a} EER (adapting)')
            ax.set_title(f'{drift}  n={len(frozen)}  wins={wins_pct:.0f}%')
            ax.grid(alpha=0.3)

        fig.suptitle(label + '  —  green = adaptation helps, red = hurts, '
                              'dashed line = no change',
                     fontsize=11)
        fig.tight_layout()
        fname = os.path.join(FIG_DIR, f'scatter_{m_f}_vs_{m_a}.png')
        fig.savefig(fname, dpi=140, bbox_inches='tight')
        plt.close(fig)
        print(f'  {fname}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', default=CSV_PATH,
                        help=f'Path to results CSV (default: {CSV_PATH})')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip scatter-plot generation')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f'ERROR: {args.csv} not found.')

    per_user = load_per_user(args.csv)
    if not per_user:
        sys.exit('ERROR: no parsable rows in CSV.')

    print(f'Loaded {len(per_user)} (user, drift) pairs from {args.csv}')

    wilcoxon_analysis(per_user)
    per_user_analysis(per_user, make_plots=not args.no_plots)


if __name__ == '__main__':
    main()