#!/usr/bin/env python3
"""
Touchalytics Drift Detection Script
For cross-dataset validation in TIFS paper on online adaptation
for continuous mobile authentication under behavioral drift.

Detects three types of drift per user:
  - Temporal: docs 1-3 vs doc 6 (same task, different day)
  - Cross-activity: docs 1-3 vs docs 4-5 (different task, same day)
  - Cross-both: docs 1-3 vs doc 7 (different task AND different day)

Three detection methods per comparison:
  - Per-feature KS test (Bonferroni-corrected)
  - Logistic regression classifier (5-fold stratified CV)
  - MMD with RBF kernel (median heuristic bandwidth)

Usage:
    python touchalytics_drift_check.py /path/to/data/directory
    (expects featMat.csv in that directory)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

COLUMN_HEADERS = [
    'user_id', 'doc_id', 'inter_stroke_time', 'stroke_duration',
    'start_x', 'start_y', 'stop_x', 'stop_y',
    'direct_end_to_end_distance', 'mean_resultant_length',
    'up_down_left_right_flag', 'direction_of_end_to_end_line', 'phone_id',
    '20pct_pairwise_velocity', '50pct_pairwise_velocity', '80pct_pairwise_velocity',
    '20pct_pairwise_acc', '50pct_pairwise_acc', '80pct_pairwise_acc',
    'median_velocity_last_3pts', 'largest_deviation_from_end_to_end',
    '20pct_dev_from_end_to_end', '50pct_dev_from_end_to_end',
    '80pct_dev_from_end_to_end', 'average_direction', 'length_of_trajectory',
    'ratio_end_to_end_dist_and_length', 'average_velocity',
    'median_acc_first_5pts', 'mid_stroke_pressure', 'mid_stroke_area_covered',
    'mid_stroke_finger_orientation', 'change_of_finger_orientation',
    'phone_orientation',
]

META_COLS = ['user_id', 'doc_id', 'phone_id', 'phone_orientation']

DRIFT_CONFIGS = {
    'temporal':       {'train_docs': [1, 2, 3], 'test_docs': [6]},
    'cross_activity': {'train_docs': [1, 2, 3], 'test_docs': [4, 5]},
    'cross_both':     {'train_docs': [1, 2, 3], 'test_docs': [7]},
}

MIN_STROKES = 20


def load_features(data_dir):
    path = os.path.join(data_dir, 'featMat.csv')
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    feat = pd.read_csv(path, header=None, names=COLUMN_HEADERS)
    feature_cols = [c for c in COLUMN_HEADERS if c not in META_COLS]
    # Drop rows with inf or NaN in any feature column
    before = len(feat)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=feature_cols)
    after = len(feat)
    print(f"Loaded {before} strokes, dropped {before - after} with inf/NaN, kept {after}.")
    return feat, feature_cols


def compute_mmd_rbf(X, Y, max_samples=500):
    """MMD^2 with RBF kernel using median heuristic for bandwidth."""
    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]

    XY = np.vstack([X, Y])
    # Median heuristic: sigma = median of pairwise distances
    dists = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=2)
    median_dist = np.median(dists[dists > 0])
    if median_dist == 0:
        median_dist = 1.0
    gamma = 1.0 / median_dist

    Kxx = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    Kyy = np.exp(-gamma * np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
    Kxy = np.exp(-gamma * np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

    nx, ny = len(X), len(Y)
    # Unbiased MMD^2 estimator
    mmd2 = (Kxx.sum() - np.trace(Kxx)) / (nx * (nx - 1)) \
         + (Kyy.sum() - np.trace(Kyy)) / (ny * (ny - 1)) \
         - 2 * Kxy.sum() / (nx * ny)
    return float(mmd2)


def run_ks_tests(X_train, X_test, feature_cols):
    """Per-feature KS test with Bonferroni correction."""
    n_features = X_train.shape[1]
    ks_stats = []
    ks_pvals = []
    for j in range(n_features):
        stat, pval = stats.ks_2samp(X_train[:, j], X_test[:, j])
        ks_stats.append(stat)
        ks_pvals.append(pval)
    ks_stats = np.array(ks_stats)
    ks_pvals = np.array(ks_pvals)
    bonf_threshold = 0.05 / n_features
    n_significant = int(np.sum(ks_pvals < bonf_threshold))
    return float(ks_stats.mean()), float(ks_stats.max()), n_significant


def run_classifier(X_train, X_test):
    """Logistic regression 5-fold stratified CV to distinguish partitions."""
    X = np.vstack([X_train, X_test])
    y = np.array([0] * len(X_train) + [1] * len(X_test))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    return float(scores.mean())


def analyze_user_drift(user_data, feature_cols, drift_type, config):
    train_mask = user_data['doc_id'].isin(config['train_docs'])
    test_mask = user_data['doc_id'].isin(config['test_docs'])
    train_df = user_data[train_mask]
    test_df = user_data[test_mask]

    if len(train_df) < MIN_STROKES or len(test_df) < MIN_STROKES:
        return None

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    ks_mean, ks_max, n_sig = run_ks_tests(X_train, X_test, feature_cols)
    clf_acc = run_classifier(X_train, X_test)
    mmd = compute_mmd_rbf(X_train, X_test)

    return {
        'n_strokes_train': len(X_train),
        'n_strokes_test': len(X_test),
        'ks_mean': ks_mean,
        'ks_max': ks_max,
        'n_significant_ks_bonf': n_sig,
        'classifier_accuracy': clf_acc,
        'mmd_statistic': mmd,
    }


def main():
    parser = argparse.ArgumentParser(description="Touchalytics drift detection")
    parser.add_argument('data_dir', help="Directory containing featMat.csv")
    args = parser.parse_args()

    feat, feature_cols = load_features(args.data_dir)
    # Remove meta cols from feature list for analysis
    analysis_cols = [c for c in feature_cols if c not in META_COLS]

    all_results = []

    for drift_type, config in DRIFT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"DRIFT TYPE: {drift_type}")
        print(f"  Train: docs {config['train_docs']}  vs  Test: docs {config['test_docs']}")
        print(f"{'='*60}")

        n_eligible = 0
        for uid in sorted(feat['user_id'].unique()):
            user_data = feat[feat['user_id'] == uid]
            result = analyze_user_drift(user_data, analysis_cols, drift_type, config)
            if result is None:
                continue
            n_eligible += 1
            result['user_id'] = int(uid)
            result['drift_type'] = drift_type
            all_results.append(result)
            print(f"  User {uid:>3d}: n_train={result['n_strokes_train']:>4d}, "
                  f"n_test={result['n_strokes_test']:>4d}, "
                  f"KS_mean={result['ks_mean']:.3f}, "
                  f"KS_max={result['ks_max']:.3f}, "
                  f"n_sig={result['n_significant_ks_bonf']:>2d}, "
                  f"clf_acc={result['classifier_accuracy']:.3f}, "
                  f"MMD={result['mmd_statistic']:.4f}")

        print(f"  -> {n_eligible} users eligible")

    # Save per-user results CSV
    results_df = pd.DataFrame(all_results)
    col_order = ['user_id', 'drift_type', 'n_strokes_train', 'n_strokes_test',
                 'ks_mean', 'ks_max', 'n_significant_ks_bonf',
                 'classifier_accuracy', 'mmd_statistic']
    results_df = results_df[col_order]
    out_path = os.path.join(args.data_dir, 'touchalytics_drift_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nPer-user results saved to: {out_path}")

    # Console summary
    print(f"\n{'='*60}")
    print("SUMMARY: Mean (std) across users per drift type")
    print(f"{'='*60}")
    metrics = ['ks_mean', 'ks_max', 'n_significant_ks_bonf', 'classifier_accuracy', 'mmd_statistic']
    for drift_type in DRIFT_CONFIGS:
        subset = results_df[results_df['drift_type'] == drift_type]
        if len(subset) == 0:
            print(f"\n{drift_type}: no eligible users")
            continue
        print(f"\n{drift_type} (n={len(subset)} users):")
        for m in metrics:
            mean_val = subset[m].mean()
            std_val = subset[m].std()
            print(f"  {m:<30s}: {mean_val:.4f} +/- {std_val:.4f}")

        mean_acc = subset['classifier_accuracy'].mean()
        mean_mmd = subset['mmd_statistic'].mean()
        detected = mean_acc > 0.70 and mean_mmd > 0
        verdict = "DRIFT DETECTED" if detected else "No significant drift"
        print(f"  VERDICT: {verdict} (mean_acc={mean_acc:.3f}, mean_mmd={mean_mmd:.4f})")

    # Comparison table template
    print(f"\n{'='*60}")
    print("COMPARISON TABLE (Touchalytics values filled, HMOG blank)")
    print(f"{'='*60}")
    header = f"{'Metric':<30s} | {'HMOG Temp':>10s} | {'HMOG XAct':>10s} | {'HMOG XBoth':>10s} | {'Touch Temp':>10s} | {'Touch XAct':>10s} | {'Touch XBoth':>10s}"
    print(header)
    print("-" * len(header))
    for m in ['ks_mean', 'classifier_accuracy', 'mmd_statistic']:
        vals = {}
        for dt in ['temporal', 'cross_activity', 'cross_both']:
            subset = results_df[results_df['drift_type'] == dt]
            vals[dt] = f"{subset[m].mean():.4f}" if len(subset) > 0 else "N/A"
        print(f"{m:<30s} | {'':>10s} | {'':>10s} | {'':>10s} | {vals['temporal']:>10s} | {vals['cross_activity']:>10s} | {vals['cross_both']:>10s}")


if __name__ == '__main__':
    main()
