#!/usr/bin/env python3
"""
Touchalytics Data Loader for RNN Experiments

Converts precomputed stroke features into fixed-length windows
suitable for VanillaRNN / RTRL training.

Each window = N consecutive strokes from the same user+document.
Output shape: (n_windows, N, 30) with corresponding user labels.

Usage:
    python touchalytics_loader.py /path/to/data/directory [--window_size 10]

    Or import as a module:
        from touchalytics_loader import load_windows, split_by_document
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

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
FEATURE_COLS = [c for c in COLUMN_HEADERS if c not in META_COLS]


def load_features(data_dir):
    """Load and clean the precomputed features CSV."""
    path = os.path.join(data_dir, 'featMat.csv')
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found.")
    feat = pd.read_csv(path, header=None, names=COLUMN_HEADERS)
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=FEATURE_COLS)
    return feat


def load_windows(data_dir, window_size=10, doc_ids=None, user_ids=None):
    """
    Build fixed-length windows from stroke features.

    Parameters
    ----------
    data_dir : str
        Directory containing featMat.csv.
    window_size : int
        Number of consecutive strokes per window (default 10).
    doc_ids : list of int or None
        If provided, only include strokes from these document IDs.
    user_ids : list of int or None
        If provided, only include these users.

    Returns
    -------
    windows : np.ndarray of shape (n_windows, window_size, 30)
    labels : np.ndarray of shape (n_windows,) with user IDs
    doc_labels : np.ndarray of shape (n_windows,) with doc IDs
    """
    feat = load_features(data_dir)

    if doc_ids is not None:
        feat = feat[feat['doc_id'].isin(doc_ids)]
    if user_ids is not None:
        feat = feat[feat['user_id'].isin(user_ids)]

    windows = []
    labels = []
    doc_labels = []

    for (uid, did), group in feat.groupby(['user_id', 'doc_id']):
        X = group[FEATURE_COLS].values
        n = len(X)
        for start in range(0, n - window_size + 1, window_size):
            windows.append(X[start:start + window_size])
            labels.append(uid)
            doc_labels.append(did)

    if not windows:
        return np.empty((0, window_size, len(FEATURE_COLS))), np.empty(0), np.empty(0)

    return np.array(windows), np.array(labels), np.array(doc_labels)


def split_by_document(data_dir, train_docs, test_docs, window_size=10, user_ids=None):
    """
    Split into train/test sets by document ID.

    Parameters
    ----------
    data_dir : str
        Directory containing featMat.csv.
    train_docs : list of int
        Document IDs for the training partition.
    test_docs : list of int
        Document IDs for the test partition.
    window_size : int
        Strokes per window.
    user_ids : list of int or None
        If provided, only include these users.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarrays
        X arrays have shape (n_windows, window_size, 30).
        y arrays have shape (n_windows,) with user IDs.
    """
    X_train, y_train, _ = load_windows(data_dir, window_size, doc_ids=train_docs, user_ids=user_ids)
    X_test, y_test, _ = load_windows(data_dir, window_size, doc_ids=test_docs, user_ids=user_ids)
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Touchalytics window-based data loader")
    parser.add_argument('data_dir', help="Directory containing featMat.csv")
    parser.add_argument('--window_size', type=int, default=10, help="Strokes per window (default 10)")
    args = parser.parse_args()

    feat = load_features(args.data_dir)
    print(f"Loaded {len(feat)} clean strokes, {feat['user_id'].nunique()} users, "
          f"{feat['doc_id'].nunique()} documents")
    print(f"Feature columns: {len(FEATURE_COLS)}")
    print(f"Window size: {args.window_size}")

    windows, labels, doc_labels = load_windows(args.data_dir, args.window_size)
    print(f"\nAll data: {windows.shape[0]} windows of shape ({args.window_size}, {len(FEATURE_COLS)})")
    print(f"  Users represented: {len(np.unique(labels))}")

    # Demo: temporal split (docs 1-3 train, doc 6 test)
    X_tr, y_tr, X_te, y_te = split_by_document(
        args.data_dir, train_docs=[1, 2, 3], test_docs=[6], window_size=args.window_size
    )
    print(f"\nTemporal split (docs 1-3 vs doc 6):")
    print(f"  Train: {X_tr.shape[0]} windows, {len(np.unique(y_tr))} users")
    print(f"  Test:  {X_te.shape[0]} windows, {len(np.unique(y_te))} users")

    # Demo: cross-activity split (docs 1-3 train, docs 4-5 test)
    X_tr, y_tr, X_te, y_te = split_by_document(
        args.data_dir, train_docs=[1, 2, 3], test_docs=[4, 5], window_size=args.window_size
    )
    print(f"\nCross-activity split (docs 1-3 vs docs 4-5):")
    print(f"  Train: {X_tr.shape[0]} windows, {len(np.unique(y_tr))} users")
    print(f"  Test:  {X_te.shape[0]} windows, {len(np.unique(y_te))} users")

    # Demo: cross-both split (docs 1-3 train, doc 7 test)
    X_tr, y_tr, X_te, y_te = split_by_document(
        args.data_dir, train_docs=[1, 2, 3], test_docs=[7], window_size=args.window_size
    )
    print(f"\nCross-both split (docs 1-3 vs doc 7):")
    print(f"  Train: {X_tr.shape[0]} windows, {len(np.unique(y_tr))} users")
    print(f"  Test:  {X_te.shape[0]} windows, {len(np.unique(y_te))} users")


if __name__ == '__main__':
    main()
