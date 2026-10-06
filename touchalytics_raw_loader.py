#!/usr/bin/env python3
"""
Touchalytics raw-data loader.

Loads raw touch-event data and converts each (user, document) trace into
fixed-length windows of consecutive move events. Supports several feature
sets, optionally including on-the-fly dynamics (dx, dy, velocity).

Document IDs are 0-indexed (as they appear in data.csv).

Output shape: (n_windows, W, D) where D depends on the feature_set.

Usage:
    export TOUCHALYTICS_DATA=/path/to/dir/with/data.csv
    python touchalytics_raw_loader.py                       # diagnostics
    python touchalytics_raw_loader.py --window_size 50 --feature_set raw_dyn6
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    'phone_id', 'user_id', 'doc_id', 'time', 'action',
    'phone_orientation', 'x', 'y', 'pressure', 'area', 'finger_orientation',
]
BASE_FEATURE_COLS = ['x', 'y', 'pressure', 'area', 'finger_orientation']
ACTION_MOVE = 2

# Available feature sets. Dynamics (dx, dy, velocity) are computed within
# each (user, doc) group from time-sorted events. First event = 0.
FEATURE_SETS = {
    'raw5':     ['x', 'y', 'pressure', 'area', 'finger_orientation'],
    'raw3':     ['x', 'y', 'pressure'],
    'raw2':     ['x', 'y'],
    'raw_dyn8': ['x', 'y', 'pressure', 'area', 'finger_orientation',
                 'dx', 'dy', 'velocity'],
    'raw_dyn6': ['x', 'y', 'pressure', 'dx', 'dy', 'velocity'],
    'raw_dyn4': ['x', 'y', 'dx', 'dy'],
}
_DYNAMIC_COLS = {'dx', 'dy', 'velocity'}

_EVENTS_CACHE = {}


def _resolve_data_dir(data_dir):
    if data_dir is None:
        data_dir = os.getenv('TOUCHALYTICS_DATA')
    if not data_dir:
        sys.exit('Error: TOUCHALYTICS_DATA env var not set and no data_dir given.')
    return data_dir


def load_events(data_dir=None):
    """Load and cache the raw event log from data.csv."""
    data_dir = _resolve_data_dir(data_dir)
    if data_dir in _EVENTS_CACHE:
        return _EVENTS_CACHE[data_dir]
    path = os.path.join(data_dir, 'data.csv')
    if not os.path.exists(path):
        sys.exit(f'Error: {path} not found.')
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    _EVENTS_CACHE[data_dir] = df
    return df


def _build_feature_array(group, cols):
    """Materialize the requested feature columns for one (user, doc) group.

    Dynamics columns are computed from the time-sorted events. First event's
    dx, dy, velocity are 0.
    """
    x = group['x'].values.astype(float)
    y = group['y'].values.astype(float)
    t = group['time'].values.astype(float)

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]

    velocity = np.zeros_like(x)
    dt_sec = np.zeros_like(t)
    dt_sec[1:] = (t[1:] - t[:-1]) / 1000.0   # ms -> seconds
    ok = dt_sec > 0
    velocity[ok] = np.sqrt(dx[ok] ** 2 + dy[ok] ** 2) / dt_sec[ok]

    col_map = {
        'x': x,
        'y': y,
        'pressure': group['pressure'].values.astype(float),
        'area': group['area'].values.astype(float),
        'finger_orientation': group['finger_orientation'].values.astype(float),
        'dx': dx,
        'dy': dy,
        'velocity': velocity,
    }
    return np.column_stack([col_map[c] for c in cols])


def load_windows(data_dir=None, window_size=50, doc_ids=None, user_ids=None,
                 feature_set='raw5'):
    """
    Non-overlapping windows of consecutive move events. Windows never cross
    document boundaries. Trailing events shorter than window_size are dropped.

    feature_set selects which columns to return; see FEATURE_SETS.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f'Unknown feature_set {feature_set!r}. '
                         f'Options: {sorted(FEATURE_SETS)}')
    cols = FEATURE_SETS[feature_set]
    D = len(cols)

    data_dir = _resolve_data_dir(data_dir)
    events = load_events(data_dir)

    events = events[events['action'] == ACTION_MOVE]
    events = events.replace([np.inf, -np.inf], np.nan)\
                   .dropna(subset=BASE_FEATURE_COLS)

    if doc_ids is not None:
        events = events[events['doc_id'].isin(doc_ids)]
    if user_ids is not None:
        events = events[events['user_id'].isin(user_ids)]

    events = events.sort_values(['user_id', 'doc_id', 'time'])

    windows = []
    user_labels = []
    doc_labels = []
    for (uid, did), group in events.groupby(['user_id', 'doc_id']):
        X = _build_feature_array(group, cols)
        n = len(X)
        for start in range(0, n - window_size + 1, window_size):
            windows.append(X[start:start + window_size])
            user_labels.append(uid)
            doc_labels.append(did)

    if not windows:
        return (np.empty((0, window_size, D)),
                np.empty(0, dtype=int), np.empty(0, dtype=int))

    return (np.array(windows),
            np.array(user_labels, dtype=int),
            np.array(doc_labels, dtype=int))


def split_by_document(data_dir=None, train_docs=None, test_docs=None,
                      window_size=50, min_windows=5, feature_set='raw5'):
    """Train/test split by document. Returns (train_dict, test_dict)."""
    data_dir = _resolve_data_dir(data_dir)
    X_tr, y_tr, _ = load_windows(data_dir, window_size,
                                 doc_ids=train_docs, feature_set=feature_set)
    X_te, y_te, _ = load_windows(data_dir, window_size,
                                 doc_ids=test_docs, feature_set=feature_set)

    train_per_user = {int(u): X_tr[y_tr == u] for u in np.unique(y_tr)}
    test_per_user = {int(u): X_te[y_te == u] for u in np.unique(y_te)}

    keep = {u for u in train_per_user
            if u in test_per_user
            and len(train_per_user[u]) >= min_windows
            and len(test_per_user[u]) >= min_windows}

    return ({u: train_per_user[u] for u in keep},
            {u: test_per_user[u] for u in keep})


def get_eligible_users(data_dir=None, train_docs=None, test_docs=None,
                       window_size=50, min_windows=5, feature_set='raw5'):
    train_dict, _ = split_by_document(
        data_dir, train_docs, test_docs, window_size, min_windows, feature_set)
    return sorted(train_dict.keys())


def report_window_distribution(data_dir=None, window_sizes=(25, 50, 100)):
    data_dir = _resolve_data_dir(data_dir)
    events = load_events(data_dir)
    events = events[events['action'] == ACTION_MOVE]
    events = events.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE_FEATURE_COLS)

    n_users = events['user_id'].nunique()
    docs = sorted(events['doc_id'].unique().astype(int).tolist())
    print(f'Loaded {len(events):,} clean move events '
          f'from {n_users} users; docs present: {docs}')

    counts = events.groupby(['user_id', 'doc_id']).size().reset_index(name='n_events')

    print(f'\nMove events per (user, doc):')
    print(f'  median={counts["n_events"].median():.0f}  '
          f'mean={counts["n_events"].mean():.1f}  '
          f'min={counts["n_events"].min()}  max={counts["n_events"].max()}')

    print('\nWindow counts at different W (non-overlapping):')
    print(f"  {'W':>4}  {'total':>8}  {'median/(u,d)':>13}  "
          f"{'(u,d) w/0':>10}  {'(u,d) w/>=5':>12}")
    for W in window_sizes:
        per_group = counts['n_events'] // W
        print(f'  {W:>4}  {per_group.sum():>8,d}  {per_group.median():>13.0f}  '
              f'{(per_group == 0).sum():>10d}  {(per_group >= 5).sum():>12d}')


def main():
    parser = argparse.ArgumentParser(description='Touchalytics raw-data window loader')
    parser.add_argument('data_dir', nargs='?', default=None)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--feature_set', default='raw5', choices=list(FEATURE_SETS))
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    report_window_distribution(data_dir, window_sizes=(25, 50, 100))

    windows, users, docs = load_windows(
        data_dir, args.window_size, feature_set=args.feature_set)
    cols = FEATURE_SETS[args.feature_set]
    print(f'\nload_windows(W={args.window_size}, feature_set={args.feature_set}): '
          f'shape={windows.shape}  columns={cols}')

    # Per-drift eligibility preview (0-indexed doc IDs)
    drift_configs = {
        'temporal':       ([0, 1, 2], [5]),
        'cross_activity': ([0, 1, 2], [3, 4]),
        'cross_both':     ([0, 1, 2], [6]),
    }
    print(f'\nEligibility per drift type '
          f'(W={args.window_size}, feature_set={args.feature_set}, >=5 windows):')
    for dt, (tr, te) in drift_configs.items():
        elig = get_eligible_users(data_dir, tr, te, args.window_size,
                                  min_windows=5, feature_set=args.feature_set)
        print(f'  {dt:<16} train={tr} test={te} -> {len(elig)} eligible users')


if __name__ == '__main__':
    main()