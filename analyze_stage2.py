"""
analyze_stage2.py
Analyzes Stage 2 results to select the best LR_online.
Decision: Lowest mean end-of-stream EER across all users, conditions, and seeds.
"""

import csv
import numpy as np
from collections import defaultdict

INPUT_FILE = 'results_stage2_online_hparams.csv'


def main():
    # Load results
    rows = []
    with open(INPUT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f'Loaded {len(rows)} results from {INPUT_FILE}\n')

    # Group by lr_online
    by_lr = defaultdict(list)
    for row in rows:
        lr = row['lr_online']
        end_eer = row['adapt_end_eer']
        if end_eer and end_eer != 'None' and end_eer != '':
            by_lr[lr].append(float(end_eer))

    # Print summary table
    print(f'{"LR_online":<12} {"N":>5} {"Mean End EER":>14} {"Std":>10} {"Median":>10} {"Min":>10} {"Max":>10}')
    print('-' * 75)

    best_lr = None
    best_mean = float('inf')

    for lr in sorted(by_lr.keys(), key=float):
        values = np.array(by_lr[lr])
        mean_eer = np.mean(values)
        std_eer = np.std(values)
        median_eer = np.median(values)
        min_eer = np.min(values)
        max_eer = np.max(values)

        print(f'{lr:<12} {len(values):>5} {mean_eer:>14.4f} {std_eer:>10.4f} {median_eer:>10.4f} {min_eer:>10.4f} {max_eer:>10.4f}')

        if mean_eer < best_mean:
            best_mean = mean_eer
            best_lr = lr

    print()

    # Also show frozen baseline for reference
    frozen_eers = []
    for row in rows:
        end_eer = row['frozen_end_eer']
        if end_eer and end_eer != 'None' and end_eer != '':
            frozen_eers.append(float(end_eer))

    if frozen_eers:
        frozen_arr = np.array(frozen_eers)
        print(f'Frozen baseline: Mean End EER = {np.mean(frozen_arr):.4f} (std={np.std(frozen_arr):.4f})')
        print()

    # Show recovery stats per LR
    print(f'\n{"LR_online":<12} {"Mean Recovery":>14} {"Win Rate":>10}')
    print('-' * 40)

    by_lr_recovery = defaultdict(list)
    by_lr_wins = defaultdict(lambda: [0, 0])  # [wins, total]
    for row in rows:
        lr = row['lr_online']
        adapt_end = row['adapt_end_eer']
        frozen_end = row['frozen_end_eer']
        recovery = row['adapt_recovery']

        if recovery and recovery != 'None' and recovery != '':
            by_lr_recovery[lr].append(float(recovery))

        if (adapt_end and adapt_end != 'None' and adapt_end != '' and
            frozen_end and frozen_end != 'None' and frozen_end != ''):
            by_lr_wins[lr][1] += 1
            if float(adapt_end) < float(frozen_end):
                by_lr_wins[lr][0] += 1

    for lr in sorted(by_lr_recovery.keys(), key=float):
        rec = np.array(by_lr_recovery[lr])
        wins, total = by_lr_wins[lr]
        win_rate = wins / total * 100 if total > 0 else 0
        print(f'{lr:<12} {np.mean(rec):>14.4f} {win_rate:>9.1f}%')

    print(f'\n{"=" * 50}')
    print(f'WINNER: LR_online = {best_lr} (mean end EER = {best_mean:.4f})')
    print(f'Lock this value into LEARNING_RATE_ONLINE for Stage 4.')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
