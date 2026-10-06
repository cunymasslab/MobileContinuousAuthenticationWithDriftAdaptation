#!/usr/bin/env python3
"""
Hyperparameter sweep for the Touchalytics raw adaptation experiment.

Sequential 4-round elimination on the cross_activity drift (largest user
pool). Each round runs M1 (frozen VanillaRNN) and M4 (BPTT fine-tuning)
on all canonical users with seed 42, and picks the configuration with
the largest statistically significant adaptation benefit (Wilcoxon
p < 0.05). If no config is significant in a round, the one with the
lowest mean M4 EER is kept.

  Round 1: window size W in {25, 50, 100}
  Round 2: online learning rate in {0.0005, 0.001, 0.002, 0.005, 0.01}
  Round 3: feature set in {raw5, raw3, raw2, raw_dyn8, raw_dyn6, raw_dyn4}
  Round 4: hidden dimension H in {16, 32, 64}

The eligible "canonical" user set is determined once at the default config
(W=50, raw5) and locked across rounds, so comparisons are over the same
users wherever they still have >=5 windows under the round's config.

Usage:
    export TOUCHALYTICS_DATA=/path/to/dir/with/data.csv
    python touchalytics_hyperparam_sweep.py
"""

import os
import random
import sys
import time

import numpy as np
from scipy import stats

from preprocessor import fit_normalizer, normalize, create_dataset
from run_experiment import compute_eer
from run_streaming import clone_rnn, MAX_NORM
from run_stage1 import (
    AdamState, train_with_early_stopping,
    MAX_EPOCHS, PATIENCE, ES_DELTA,
)
from touchalytics_raw_loader import split_by_document, FEATURE_SETS


# ── Fixed sweep settings ─────────────────────────────────────────────────────
DRIFT_TYPE = 'cross_activity'
TRAIN_DOCS = [0, 1, 2]
TEST_DOCS = [3, 4]
SEED = 42
MIN_WINDOWS = 5
N_IMPOSTORS = 3
LR_TRAIN = 0.01

DEFAULTS = {
    'W': 50,
    'lr_online': 0.002,
    'feature_set': 'raw5',
    'H': 32,
}

ROUNDS = [
    ('window_size',   'W',           [25, 50, 100]),
    ('learning_rate', 'lr_online',   [0.0005, 0.001, 0.002, 0.005, 0.01]),
    ('feature_set',   'feature_set', ['raw5', 'raw3', 'raw2',
                                      'raw_dyn8', 'raw_dyn6', 'raw_dyn4']),
    ('hidden_dim',    'H',           [16, 32, 64]),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def pick_impostors(target_user, canonical_pool, n=N_IMPOSTORS):
    """Deterministic pick from the canonical pool, seeded by digit-sum of user."""
    candidates = sorted(int(u) for u in canonical_pool if int(u) != int(target_user))
    seed = sum(ord(c) for c in str(int(target_user)))
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def stream_frozen(rnn, X_stream):
    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        _, h_cache = rnn.forward(X_stream[w])
        h_final = h_cache[len(X_stream[w]) - 1]
        scores[w] = rnn.output(h_final)[0]
    return scores


def stream_bptt_finetune(rnn, X_stream, y_stream, lr_online):
    param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape,
                    rnn.b_h.shape, rnn.b_y.shape]
    opt = AdamState(param_shapes, lr=lr_online)
    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        a_cache, h_cache = rnn.forward(X_stream[w])
        h_final = h_cache[len(X_stream[w]) - 1]
        pred = rnn.output(h_final)
        scores[w] = pred[0]
        grads = rnn.backward(y_stream[w], pred, h_cache, a_cache, X_stream[w])
        grads = rnn.clip_gradients(grads, MAX_NORM)
        opt.step(rnn, grads)
    return scores


# ── Single (user, config) experiment ─────────────────────────────────────────

def run_user(target_user, impostors, train_dict, test_dict, config):
    """Return (m1_eer, m4_eer) for this user, or None if skipped."""
    input_dim = train_dict[target_user].shape[2]

    legit_train = train_dict[target_user]
    imp_chunks = [train_dict[u] for u in impostors if u in train_dict]
    if not imp_chunks:
        return None
    impostor_train = np.concatenate(imp_chunks, axis=0)

    # 1:1 balance
    ds_rng = np.random.RandomState(SEED)
    if len(impostor_train) > len(legit_train):
        idx = ds_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[idx]
    elif len(legit_train) > len(impostor_train):
        idx = ds_rng.choice(len(legit_train), size=len(impostor_train), replace=False)
        legit_train = legit_train[idx]

    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train_n = normalize(legit_train, mean, std)
    impostor_train_n = normalize(impostor_train, mean, std)

    X_all, y_all = create_dataset(
        legit_train_n, impostor_train_n, rng=np.random.RandomState(SEED))

    split_rng = np.random.RandomState(SEED)
    n = len(X_all)
    perm = split_rng.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[perm[:split]], y_all[perm[:split]]
    X_val, y_val = X_all[perm[split:]], y_all[perm[split:]]
    if len(X_train) == 0 or len(X_val) == 0:
        return None

    np.random.seed(SEED)
    rnn_offline, _, _, _ = train_with_early_stopping(
        X_train, y_train, X_val, y_val, input_dim,
        optimizer_name='adam', lr=LR_TRAIN,
        hidden_dim=config['H'], max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
    )

    # Streaming data
    legit_stream = test_dict[target_user]
    imp_chunks = [test_dict[u] for u in impostors if u in test_dict]
    if not imp_chunks:
        return None
    impostor_stream = np.concatenate(imp_chunks, axis=0)
    if len(legit_stream) < MIN_WINDOWS or len(impostor_stream) < MIN_WINDOWS:
        return None

    user_seed = int(target_user) * 1000 + SEED
    ds = np.random.RandomState(user_seed)
    if len(impostor_stream) > len(legit_stream):
        idx = ds.choice(len(impostor_stream), size=len(legit_stream), replace=False)
        impostor_stream = impostor_stream[idx]
    elif len(legit_stream) > len(impostor_stream):
        idx = ds.choice(len(legit_stream), size=len(impostor_stream), replace=False)
        legit_stream = legit_stream[idx]

    legit_stream = normalize(legit_stream, mean, std)
    impostor_stream = normalize(impostor_stream, mean, std)

    n_legit = len(legit_stream)
    n_total = n_legit + len(impostor_stream)
    inter = np.random.RandomState(user_seed + 1)
    legit_pos = np.sort(inter.choice(n_total, size=n_legit, replace=False))
    imp_pos = np.sort(np.setdiff1d(np.arange(n_total), legit_pos))

    X_stream = np.empty((n_total, config['W'], input_dim))
    y_stream = np.empty(n_total)
    X_stream[legit_pos] = legit_stream
    y_stream[legit_pos] = 1.0
    X_stream[imp_pos] = impostor_stream
    y_stream[imp_pos] = 0.0

    m1 = clone_rnn(rnn_offline)
    m1_eer = float(compute_eer(y_stream, stream_frozen(m1, X_stream)))

    m4 = clone_rnn(rnn_offline)
    m4_eer = float(compute_eer(
        y_stream, stream_bptt_finetune(m4, X_stream, y_stream, config['lr_online'])))

    return m1_eer, m4_eer


# ── Per-config aggregation ───────────────────────────────────────────────────

def run_config(config, canonical_users, data_dir):
    train_dict, test_dict = split_by_document(
        data_dir, TRAIN_DOCS, TEST_DOCS,
        window_size=config['W'], min_windows=MIN_WINDOWS,
        feature_set=config['feature_set'],
    )

    users_to_run = [u for u in canonical_users
                    if u in train_dict and u in test_dict]

    m1_eers, m4_eers = [], []
    for user in users_to_run:
        impostors = pick_impostors(user, canonical_users, n=N_IMPOSTORS)
        impostors = [u for u in impostors if u in train_dict and u in test_dict]
        if len(impostors) == 0:
            continue
        result = run_user(user, impostors, train_dict, test_dict, config)
        if result is None:
            continue
        m1_eers.append(result[0])
        m4_eers.append(result[1])

    if not m1_eers:
        return None

    m1_arr = np.array(m1_eers)
    m4_arr = np.array(m4_eers)
    delta = m1_arr - m4_arr

    if np.allclose(delta, 0):
        W_stat, p = float('nan'), 1.0
    else:
        W_stat, p = stats.wilcoxon(m1_arr, m4_arr, zero_method='wilcox',
                                   alternative='two-sided')

    return {
        'n': len(m1_eers),
        'm1_mean': float(m1_arr.mean()),
        'm4_mean': float(m4_arr.mean()),
        'benefit_mean': float(delta.mean()),
        'p': float(p),
        'W_stat': float(W_stat),
    }


def select_winner(results):
    """Largest significant positive benefit, else lowest M4 EER."""
    sig_pos = {k: r for k, r in results.items()
               if r is not None and r['p'] < 0.05 and r['benefit_mean'] > 0}
    if sig_pos:
        return max(sig_pos, key=lambda k: sig_pos[k]['benefit_mean'])
    valid = {k: r for k, r in results.items() if r is not None}
    if not valid:
        return None
    return min(valid, key=lambda k: valid[k]['m4_mean'])


def print_round_table(round_name, key, results, winner):
    print(f'\n{"="*92}')
    print(f'ROUND: {round_name}  (sweeping {key})')
    print('=' * 92)
    print(f"| {key:<12} | {'n':>3} | {'M1 EER':>7} | {'M4 EER':>7} | "
          f"{'Benefit':>9} | {'Wilcoxon p':>11} | sig | Winner |")
    print(f"|{'-'*14}|{'-'*5}|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*13}|{'-'*5}|{'-'*8}|")
    for k, r in results.items():
        if r is None:
            print(f"| {str(k):<12} | {'--':>3} | {'--':>7} | {'--':>7} | "
                  f"{'--':>9} | {'--':>11} | --  |        |")
            continue
        sig = ('***' if r['p'] < 0.001 else
               '** ' if r['p'] < 0.01 else
               '*  ' if r['p'] < 0.05 else
               'ns ')
        win_mark = ' YES ' if k == winner else '     '
        print(f"| {str(k):<12} | {r['n']:>3} | {r['m1_mean']:>7.4f} | "
              f"{r['m4_mean']:>7.4f} | {r['benefit_mean']:>+9.4f} | "
              f"{r['p']:>11.4f} | {sig} | {win_mark}|")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.getenv('TOUCHALYTICS_DATA')
    if not data_dir:
        sys.exit('ERROR: TOUCHALYTICS_DATA not set.')

    print('Touchalytics raw — hyperparameter sweep')
    print(f'  Drift: {DRIFT_TYPE} (train docs {TRAIN_DOCS}, test docs {TEST_DOCS})')
    print(f'  Models: M1 (frozen), M4 (BPTT fine-tune)')
    print(f'  Seed: {SEED}  |  Min windows: {MIN_WINDOWS}')
    print(f'  Starting config: {DEFAULTS}')

    # Lock the canonical user set using the default config
    print('\nDetermining canonical eligible user set (at defaults) ...')
    canonical_train, _ = split_by_document(
        data_dir, TRAIN_DOCS, TEST_DOCS,
        window_size=DEFAULTS['W'], min_windows=MIN_WINDOWS,
        feature_set=DEFAULTS['feature_set'],
    )
    canonical_users = sorted(canonical_train.keys())
    print(f'  -> {len(canonical_users)} canonical users')

    config = dict(DEFAULTS)
    global_t0 = time.time()

    for round_name, key, values in ROUNDS:
        print(f'\n{"#"*92}\n#  Round: {round_name}  (varying {key})')
        print(f'#  Frozen config: {[(k, v) for k, v in config.items() if k != key]}')
        print('#' * 92)

        results = {}
        for v in values:
            cfg = dict(config)
            cfg[key] = v
            t0 = time.time()
            print(f'\n--- {key}={v} ---')
            r = run_config(cfg, canonical_users, data_dir)
            elapsed = time.time() - t0
            if r is None:
                print(f'    no eligible users  ({elapsed:.0f}s)')
            else:
                print(f'    n={r["n"]:>2}  M1={r["m1_mean"]:.4f}  '
                      f'M4={r["m4_mean"]:.4f}  '
                      f'Δ={r["benefit_mean"]:+.4f}  '
                      f'p={r["p"]:.4f}  ({elapsed:.0f}s)')
            results[v] = r

        winner = select_winner(results)
        print_round_table(round_name, key, results, winner)
        if winner is None:
            print(f'\n>>> No usable winner — keeping {key}={config[key]}')
        else:
            config[key] = winner
            print(f'\n>>> Winner: {key}={winner}')
            print(f'>>> Current config: {config}')

    elapsed = time.time() - global_t0
    print(f'\n{"="*92}\nFINAL BEST CONFIG  (total runtime: {elapsed/60:.1f} min)')
    print('=' * 92)
    for k, v in config.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()