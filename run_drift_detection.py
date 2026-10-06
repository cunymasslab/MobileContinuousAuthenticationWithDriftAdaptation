"""
run_drift_detection.py — Distribution-shift detection on raw sensor data.

Independent confirmation that the raw sensor data itself shifts between
sessions 1-3 and session 4, complementing the frozen-model EER degradation
observed in Experiment 2.

For each (user, train_condition, stream_condition) combination:
  1. Load legit-user windows only (no impostors) for sessions 1-3 of the
     train condition and session 4 of the stream condition.
  2. Compute per-window feature means (100 timesteps -> 1 vector of 12 features).
  3. KS test: per-feature two-sample KS test between the two time periods.
  4. Classifier-based detector: logistic regression 5-fold stratified CV
     distinguishing sessions 1-3 vectors (label 0) from session 4 vectors (label 1).

Usage:
    # Single combination
    python run_drift_detection.py \\
        --user 525584 \\
        --train-condition READING+WALKING \\
        --stream-condition WRITING+SITTING \\
        --output-dir results_stage4b_exp1

    # All 50 combinations from exp1_50_users.txt
    python run_drift_detection.py \\
        --all \\
        --output-dir results_stage4b_exp1
"""

import os
import sys
import csv
import argparse
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from dotenv import load_dotenv

from data_loader import get_sessions_by_condition
from run_experiment import load_user_windows
from run_streaming import SENSORS
from run_stage1 import WINDOW_SIZE, STEP_SIZE

load_dotenv()

OUTPUT_FIELDS = [
    'target_user',
    'train_condition',
    'stream_condition',
    'n_train_windows',
    'n_session4_windows',
    'n_significant_ks',
    'mean_ks_stat',
    'max_ks_stat',
    'classifier_accuracy',
    'drift_gap',
]


def load_completed(csv_path):
    """Read existing CSV; return set of completed (user, train, stream) keys."""
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['target_user'], row['train_condition'],
                   row['stream_condition'])
            completed.add(key)
    return completed


def append_row(csv_path, row):
    """Append a single row to the CSV, writing header if the file is new."""
    is_new = not os.path.exists(csv_path)
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def read_drift_gap(exp2_csv_path, target_user, train_condition, stream_condition):
    """Look up drift_gap from the Experiment 2 results CSV."""
    if not os.path.exists(exp2_csv_path):
        return ''
    with open(exp2_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['target_user'] == target_user
                    and row['train_condition'] == train_condition
                    and row['stream_condition'] == stream_condition):
                return row.get('drift_gap', '')
    return ''


def run_one_drift(target_user, train_condition, stream_condition):
    """Run drift detection for a single (user, train, stream) combination."""
    train_activity, train_posture = train_condition.split('+')
    stream_activity, stream_posture = stream_condition.split('+')

    # Sessions 1-3 of train condition
    train_session_numbers = get_sessions_by_condition(
        target_user, train_activity, train_posture,
    )
    if len(train_session_numbers) < 3:
        print(f'  Skip: user {target_user} has <3 sessions for {train_condition}')
        return None
    train_sessions = train_session_numbers[:3]

    # Session 4 of stream condition
    stream_session_numbers = get_sessions_by_condition(
        target_user, stream_activity, stream_posture,
    )
    if len(stream_session_numbers) < 4:
        print(f'  Skip: user {target_user} has <4 sessions for {stream_condition}')
        return None
    session4 = stream_session_numbers[3]

    # Load legit-user windows only (no impostors)
    train_windows = load_user_windows(
        target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE,
    )
    s4_windows = load_user_windows(
        target_user, [session4], SENSORS, WINDOW_SIZE, STEP_SIZE,
    )

    if len(train_windows) == 0 or len(s4_windows) == 0:
        print(f'  Skip: no windows (train={len(train_windows)}, s4={len(s4_windows)})')
        return None

    # Per-window feature means: (n_windows, 100, 12) -> (n_windows, 12)
    train_means = np.mean(train_windows, axis=1)
    s4_means = np.mean(s4_windows, axis=1)

    # Algorithm 1: KS test per feature
    ks_stats = []
    ks_pvalues = []
    for feat_idx in range(train_means.shape[1]):
        stat, pval = ks_2samp(train_means[:, feat_idx], s4_means[:, feat_idx])
        ks_stats.append(stat)
        ks_pvalues.append(pval)

    n_significant_ks = sum(1 for p in ks_pvalues if p < 0.05)
    mean_ks_stat = float(np.mean(ks_stats))
    max_ks_stat = float(np.max(ks_stats))

    # Algorithm 2: Classifier-based detector
    X = np.vstack([train_means, s4_means])
    y = np.array([0] * len(train_means) + [1] * len(s4_means))

    # Deterministic shuffle with seed 42
    rng = np.random.RandomState(42)
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    classifier_accuracy = float(np.mean(scores))

    return {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'n_train_windows': len(train_windows),
        'n_session4_windows': len(s4_windows),
        'n_significant_ks': n_significant_ks,
        'mean_ks_stat': round(mean_ks_stat, 6),
        'max_ks_stat': round(max_ks_stat, 6),
        'classifier_accuracy': round(classifier_accuracy, 4),
        'drift_gap': '',  # filled in by caller
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--user',
                        help='Target user id (e.g. 525584)')
    parser.add_argument('--train-condition',
                        help='Train condition (e.g. READING+WALKING)')
    parser.add_argument('--stream-condition',
                        help='Stream condition (e.g. WRITING+SITTING)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for results_drift_detection.csv')
    parser.add_argument('--all', action='store_true',
                        help='Run all 50 combinations from exp1_50_users.txt')
    args = parser.parse_args()

    if not args.all and not (args.user and args.train_condition and args.stream_condition):
        parser.error('--user, --train-condition, and --stream-condition are required '
                     'unless --all is specified')

    csv_path = os.path.join(args.output_dir, 'results_drift_detection.csv')
    exp2_csv_path = os.path.join(args.output_dir, 'results_stage4b_cv_summary.csv')

    # Build list of combinations to run
    if args.all:
        combos = []
        txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'exp1_50_users.txt')
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                combos.append((parts[0], parts[1], parts[2]))
    else:
        combos = [(args.user, args.train_condition, args.stream_condition)]

    # Resume logic
    completed = load_completed(csv_path)
    n_total = len(combos)
    n_skipped = 0

    for i, (user, train_cond, stream_cond) in enumerate(combos, 1):
        key = (user, train_cond, stream_cond)
        if key in completed:
            print(f'[{i}/{n_total}] Already completed: {key}. Skipping.')
            n_skipped += 1
            continue

        print(f'[{i}/{n_total}] user={user} train={train_cond} stream={stream_cond}')
        row = run_one_drift(user, train_cond, stream_cond)
        if row is None:
            print(f'  Skipped (insufficient data).')
            continue

        # Look up drift_gap from Experiment 2 results
        drift_gap = read_drift_gap(exp2_csv_path, user, train_cond, stream_cond)
        row['drift_gap'] = drift_gap

        append_row(csv_path, row)

        print(f'  n_train_windows      = {row["n_train_windows"]}')
        print(f'  n_session4_windows   = {row["n_session4_windows"]}')
        print(f'  n_significant_ks     = {row["n_significant_ks"]}/12')
        print(f'  mean_ks_stat         = {row["mean_ks_stat"]}')
        print(f'  max_ks_stat          = {row["max_ks_stat"]}')
        print(f'  classifier_accuracy  = {row["classifier_accuracy"]}')
        print(f'  drift_gap            = {row["drift_gap"]}')
        print()

    print('=' * 60)
    print(f'Done. {n_total - n_skipped} processed, {n_skipped} skipped (already completed).')
    print(f'Results: {csv_path}')


if __name__ == '__main__':
    main()