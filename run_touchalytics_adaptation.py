#!/usr/bin/env python3
"""
run_touchalytics_adaptation.py
Cross-dataset validation for the TIFS online-adaptation paper.

Compares all 8 HMOG model roles on the Touchalytics touch-dynamics dataset
under three drift conditions:

    M1 — VanillaRNN-Frozen           (NumPy BPTT, frozen)
    M2 — VanillaRNN-RTRL             (NumPy, per-window RTRL update)
    M3 — VanillaRNN-RTRL-LegitOnly   (NumPy, RTRL update on legit windows only)
    M4 — VanillaRNN-BPTTFinetune     (NumPy, per-window BPTT update)
    M5 — LSTM-Frozen                 (PyTorch trained, frozen)
    M6 — GRU-Frozen                  (PyTorch -> NumPy GRU, frozen)
    M7 — GRU-RTRL                    (PyTorch -> NumPy GRU, per-window RTRL update)
    M8 — GRU-BPTTFinetune            (PyTorch -> NumPy GRU, per-window BPTT update)

RTRL/BPTT models reset their internal state (sensitivity matrices for RTRL,
hidden state for BPTT) at every window boundary. Adaptation lr = 0.002 (Adam).

Drift configurations (per Frank et al. 2013 document numbering):
    temporal:        train docs 1-3 (reading), stream doc 6  (reading, weeks later)
    cross_activity:  train docs 1-3 (reading), stream docs 4-5 (gaming, same day)
    cross_both:      train docs 1-3 (reading), stream doc 7  (gaming, weeks later)

Hyperparameters match HMOG: H=32, Adam, lr_train=0.01, lr_online=0.002,
gradient clip = 1.0, early stopping (patience=3, delta=1e-3), max 30 epochs.

Resume-safe: skips (user_id, drift_type, seed) combinations already in the CSV.

Usage:
    export TOUCHALYTICS_DATA=/path/to/dir/with/featMat.csv
    python run_touchalytics_adaptation.py
"""

import csv
import os
import random
import sys
import time

import numpy as np
import torch

from preprocessor import fit_normalizer, normalize, create_dataset
from run_experiment import compute_eer
from run_streaming import clone_rnn, MAX_NORM
from run_stage1 import (
    AdamState, train_with_early_stopping,
    MAX_EPOCHS, PATIENCE, ES_DELTA,
)
from pytorch_models import train_pytorch_model
from gru_model import GRU_NumPy
from touchalytics_raw_loader import load_windows 


# ── Hyperparameters (mirrors HMOG protocol) ──────────────────────────────────
HIDDEN_DIM = 32
INPUT_DIM = 5        # 30 stroke features
WINDOW_SIZE = 50      # 10 strokes per RNN window
LR_TRAIN = 0.01
LR_ONLINE = 0.002
SEEDS = [42, 123, 456]
N_IMPOSTORS = 3
MIN_STREAM_WINDOWS = 5

# Match HMOG GRU parameter ordering (run_stage4.py M7 wiring)
GRU_PARAM_NAMES = [
    'W_xr', 'W_hr', 'b_r', 'W_xz', 'W_hz', 'b_z',
    'W_xn', 'W_hn', 'b_n_input', 'b_n_hidden', 'W_hy', 'b_y',
]

DRIFT_CONFIGS = {
    'temporal':       {'train_docs': [1, 2, 3], 'test_docs': [6]},
    'cross_activity': {'train_docs': [1, 2, 3], 'test_docs': [4, 5]},
    'cross_both':     {'train_docs': [1, 2, 3], 'test_docs': [7]},
}

OUTPUT_DIR = 'results_touchalytics'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'touchalytics_adaptation_raw_results.csv')

MODEL_IDS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
EER_FIELDS = [f'{m}_overall_eer' for m in MODEL_IDS]
FIELDS = (
    ['user_id', 'drift_type', 'seed', 'n_train_windows', 'n_test_windows']
    + EER_FIELDS
    + ['adaptation_benefit']
)


# ── Window-array helpers ─────────────────────────────────────────────────────

def select_user_doc_windows(all_X, all_y, all_doc, user_id, doc_ids):
    """Slice all-windows array for one user across a set of doc_ids."""
    mask = (all_y == user_id) & np.isin(all_doc, doc_ids)
    return all_X[mask]


def gather_impostor_windows(all_X, all_y, all_doc, impostor_ids, doc_ids):
    """Concatenate windows from multiple impostors for given doc_ids."""
    chunks = []
    for u in impostor_ids:
        chunk = select_user_doc_windows(all_X, all_y, all_doc, u, doc_ids)
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return np.empty((0, WINDOW_SIZE, INPUT_DIM))
    return np.concatenate(chunks, axis=0)


def pick_impostors(target_user, eligible_pool, n=N_IMPOSTORS, seed=None):
    """Deterministic impostor selection from a sorted eligibility pool."""
    candidates = sorted(int(u) for u in eligible_pool if int(u) != int(target_user))
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def clone_numpy_gru_from_pytorch(gru_pytorch, input_dim, hidden_dim):
    """Fresh NumPy GRU loaded from a trained PyTorch GRUClassifier."""
    gru_np = GRU_NumPy(input_dim=input_dim, hidden_dim=hidden_dim)
    gru_np.load_from_pytorch(gru_pytorch)
    return gru_np


# ── Streaming primitives ─────────────────────────────────────────────────────

def stream_score_frozen_rnn(rnn, X_stream):
    """M1: VanillaRNN, BPTT-trained, frozen."""
    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        _, h_cache = rnn.forward(X_stream[w])
        h_final = h_cache[len(X_stream[w]) - 1]
        pred = rnn.output(h_final)
        scores[w] = pred[0]
    return scores


def stream_score_rtrl_rnn(rnn, X_stream, y_stream, lr_online, legit_only=False):
    """M2 / M3: VanillaRNN RTRL fine-tune. Sensitivity resets each window."""
    param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape,
                    rnn.b_h.shape, rnn.b_y.shape]
    opt = AdamState(param_shapes, lr=lr_online)

    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(X_stream[w])
        pred = rnn.output(h_final)
        scores[w] = pred[0]
        if legit_only and y_stream[w] != 1:
            continue
        grads = rnn.rtrl_compute_gradients(
            y_stream[w], pred, h_final, S_xh, S_hh, S_bh)
        grads = rnn.clip_gradients(grads, MAX_NORM)
        opt.step(rnn, grads)
    return scores


def stream_score_bptt_finetune_rnn(rnn, X_stream, y_stream, lr_online):
    """M4: VanillaRNN BPTT fine-tune."""
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


def stream_score_frozen_pytorch(model, X_stream):
    """M5: PyTorch model (LSTM here), frozen inference per window."""
    device = next(model.parameters()).device
    scores = np.empty(len(X_stream))
    model.eval()
    with torch.no_grad():
        for w in range(len(X_stream)):
            x_t = torch.tensor(
                X_stream[w], dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(x_t)
            scores[w] = pred.item()
    return scores


def stream_score_frozen_gru(gru, X_stream):
    """M6: NumPy GRU (PyTorch-transferred weights), frozen."""
    scores = np.empty(len(X_stream))
    T = X_stream.shape[1]
    for w in range(len(X_stream)):
        cache = gru.forward(X_stream[w])
        h_final = cache['h'][T - 1]
        pred = gru.output(h_final)
        scores[w] = pred[0]
    return scores


def stream_score_rtrl_gru(gru, X_stream, y_stream, lr_online):
    """M7: NumPy GRU RTRL fine-tune. Sensitivity resets each window."""
    param_shapes = [getattr(gru, k).shape for k in GRU_PARAM_NAMES]
    opt = AdamState(param_shapes, lr=lr_online)

    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        h_final, sensitivities = gru.rtrl_forward(X_stream[w])
        pred = gru.output(h_final)
        scores[w] = pred[0]
        grads = gru.rtrl_compute_gradients(
            y_stream[w], pred, h_final, sensitivities)
        grads = gru.clip_gradients(grads, MAX_NORM)
        opt.step_named(gru, grads, GRU_PARAM_NAMES)
    return scores


def stream_score_bptt_finetune_gru(gru, X_stream, y_stream, lr_online):
    """M8: NumPy GRU BPTT fine-tune."""
    param_shapes = [getattr(gru, k).shape for k in GRU_PARAM_NAMES]
    opt = AdamState(param_shapes, lr=lr_online)

    scores = np.empty(len(X_stream))
    T = X_stream.shape[1]
    for w in range(len(X_stream)):
        cache = gru.forward(X_stream[w])
        h_final = cache['h'][T - 1]
        pred = gru.output(h_final)
        scores[w] = pred[0]
        grads = gru.backward(y_stream[w], pred, cache, X_stream[w])
        grads = gru.clip_gradients(grads, MAX_NORM)
        opt.step_named(gru, grads, GRU_PARAM_NAMES)
    return scores


# ── Single experiment ────────────────────────────────────────────────────────

def run_one(target_user, impostors, drift_cfg, seed, all_X, all_y, all_doc):
    """Train offline, stream M1..M8. Returns metrics dict or None if skipped."""
    train_docs = drift_cfg['train_docs']
    test_docs = drift_cfg['test_docs']

    # ── Training data ────────────────────────────────────────────────────────
    legit_train = select_user_doc_windows(all_X, all_y, all_doc, target_user, train_docs)
    if len(legit_train) == 0:
        return None
    impostor_train = gather_impostor_windows(all_X, all_y, all_doc, impostors, train_docs)
    if len(impostor_train) == 0:
        return None

    # 1:1 balance
    ds_rng = np.random.RandomState(seed)
    if len(impostor_train) > len(legit_train):
        idx = ds_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[idx]
    elif len(legit_train) > len(impostor_train):
        idx = ds_rng.choice(len(legit_train), size=len(impostor_train), replace=False)
        legit_train = legit_train[idx]

    # Normalize using training-only stats
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train_n = normalize(legit_train, mean, std)
    impostor_train_n = normalize(impostor_train, mean, std)

    X_all, y_all = create_dataset(
        legit_train_n, impostor_train_n, rng=np.random.RandomState(seed))

    # 80/20 train/val split for early stopping (Stage 1 protocol)
    split_rng = np.random.RandomState(seed)
    n = len(X_all)
    perm = split_rng.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[perm[:split]], y_all[perm[:split]]
    X_val, y_val = X_all[perm[split:]], y_all[perm[split:]]
    if len(X_train) == 0 or len(X_val) == 0:
        return None

    # ── Offline training: VanillaRNN (NumPy BPTT) for M1-M4 ──────────────────
    np.random.seed(seed)
    rnn_offline, _, _, _ = train_with_early_stopping(
        X_train, y_train, X_val, y_val, INPUT_DIM,
        optimizer_name='adam', lr=LR_TRAIN,
        hidden_dim=HIDDEN_DIM, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
    )

    # ── Offline training: LSTM (PyTorch) for M5 ──────────────────────────────
    lstm_pytorch, _, _, _ = train_pytorch_model(
        X_train, y_train, X_val, y_val,
        arch='lstm', input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
        lr=LR_TRAIN, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, seed=seed,
    )

    # ── Offline training: GRU (PyTorch) for M6-M8 ────────────────────────────
    gru_pytorch, _, _, _ = train_pytorch_model(
        X_train, y_train, X_val, y_val,
        arch='gru', input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
        lr=LR_TRAIN, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, seed=seed,
    )

    # ── Streaming data ───────────────────────────────────────────────────────
    legit_stream = select_user_doc_windows(all_X, all_y, all_doc, target_user, test_docs)
    impostor_stream = gather_impostor_windows(all_X, all_y, all_doc, impostors, test_docs)
    if len(legit_stream) < MIN_STREAM_WINDOWS or len(impostor_stream) < MIN_STREAM_WINDOWS:
        return None

    # 1:1 balance for the stream — per-(user, seed) RNG
    user_seed = int(target_user) * 1000 + seed
    stream_ds_rng = np.random.RandomState(user_seed)
    if len(impostor_stream) > len(legit_stream):
        idx = stream_ds_rng.choice(len(impostor_stream), size=len(legit_stream), replace=False)
        impostor_stream = impostor_stream[idx]
    elif len(legit_stream) > len(impostor_stream):
        idx = stream_ds_rng.choice(len(legit_stream), size=len(impostor_stream), replace=False)
        legit_stream = legit_stream[idx]

    legit_stream = normalize(legit_stream, mean, std)
    impostor_stream = normalize(impostor_stream, mean, std)

    # Interleave: legit at random positions, impostors fill remainder
    n_legit = len(legit_stream)
    n_total = n_legit + len(impostor_stream)
    inter_rng = np.random.RandomState(user_seed + 1)
    legit_pos = np.sort(inter_rng.choice(n_total, size=n_legit, replace=False))
    impostor_pos = np.sort(np.setdiff1d(np.arange(n_total), legit_pos))

    X_stream = np.empty((n_total, WINDOW_SIZE, INPUT_DIM))
    y_stream = np.empty(n_total)
    X_stream[legit_pos] = legit_stream
    y_stream[legit_pos] = 1.0
    X_stream[impostor_pos] = impostor_stream
    y_stream[impostor_pos] = 0.0

    # ── Stream all 8 models ──────────────────────────────────────────────────
    eers = {}

    # M1: frozen VanillaRNN
    m1 = clone_rnn(rnn_offline)
    eers['M1'] = float(compute_eer(y_stream, stream_score_frozen_rnn(m1, X_stream)))

    # M2: VanillaRNN-RTRL (update every window)
    m2 = clone_rnn(rnn_offline)
    eers['M2'] = float(compute_eer(
        y_stream,
        stream_score_rtrl_rnn(m2, X_stream, y_stream, LR_ONLINE, legit_only=False)))

    # M3: VanillaRNN-RTRL-LegitOnly
    m3 = clone_rnn(rnn_offline)
    eers['M3'] = float(compute_eer(
        y_stream,
        stream_score_rtrl_rnn(m3, X_stream, y_stream, LR_ONLINE, legit_only=True)))

    # M4: VanillaRNN BPTT fine-tune
    m4 = clone_rnn(rnn_offline)
    eers['M4'] = float(compute_eer(
        y_stream,
        stream_score_bptt_finetune_rnn(m4, X_stream, y_stream, LR_ONLINE)))

    # M5: LSTM-Frozen (PyTorch)
    eers['M5'] = float(compute_eer(
        y_stream, stream_score_frozen_pytorch(lstm_pytorch, X_stream)))

    # M6: GRU-Frozen (NumPy w/ PyTorch weights)
    m6 = clone_numpy_gru_from_pytorch(gru_pytorch, INPUT_DIM, HIDDEN_DIM)
    eers['M6'] = float(compute_eer(y_stream, stream_score_frozen_gru(m6, X_stream)))

    # M7: GRU-RTRL
    m7 = clone_numpy_gru_from_pytorch(gru_pytorch, INPUT_DIM, HIDDEN_DIM)
    eers['M7'] = float(compute_eer(
        y_stream,
        stream_score_rtrl_gru(m7, X_stream, y_stream, LR_ONLINE)))

    # M8: GRU BPTT fine-tune
    m8 = clone_numpy_gru_from_pytorch(gru_pytorch, INPUT_DIM, HIDDEN_DIM)
    eers['M8'] = float(compute_eer(
        y_stream,
        stream_score_bptt_finetune_gru(m8, X_stream, y_stream, LR_ONLINE)))

    return {
        'n_train_windows': int(len(X_all)),
        'n_test_windows': int(n_total),
        **{f'{m}_overall_eer': eers[m] for m in MODEL_IDS},
        'adaptation_benefit': eers['M1'] - eers['M4'],
    }


# ── Driver helpers ───────────────────────────────────────────────────────────

def find_eligible_users(all_y, all_doc, all_users, drift_cfg):
    """Eligible = has any train-doc windows AND >= MIN_STREAM_WINDOWS test-doc windows."""
    elig = []
    for u in all_users:
        n_train = int(((all_y == u) & np.isin(all_doc, drift_cfg['train_docs'])).sum())
        n_test = int(((all_y == u) & np.isin(all_doc, drift_cfg['test_docs'])).sum())
        if n_train > 0 and n_test >= MIN_STREAM_WINDOWS:
            elig.append(u)
    return elig


def load_completed():
    completed = set()
    if not os.path.exists(OUTPUT_FILE):
        return completed
    with open(OUTPUT_FILE, 'r') as f:
        for row in csv.DictReader(f):
            completed.add((row['user_id'], row['drift_type'], row['seed']))
    return completed


def write_summary():
    if not os.path.exists(OUTPUT_FILE):
        print('No results to summarize.')
        return

    rows = []
    with open(OUTPUT_FILE, 'r') as f:
        for r in csv.DictReader(f):
            try:
                for m in MODEL_IDS:
                    r[f'{m}_overall_eer'] = float(r[f'{m}_overall_eer'])
            except (ValueError, TypeError, KeyError):
                continue
            rows.append(r)

    if not rows:
        print('No usable rows in results CSV.')
        return

    # Average across seeds per (user, drift_type)
    by_user_drift = {}
    for r in rows:
        key = (r['user_id'], r['drift_type'])
        by_user_drift.setdefault(key, []).append(
            tuple(r[f'{m}_overall_eer'] for m in MODEL_IDS))

    by_drift = {}
    for (user, drift), tuples in by_user_drift.items():
        means = tuple(float(np.mean([t[i] for t in tuples])) for i in range(len(MODEL_IDS)))
        by_drift.setdefault(drift, []).append(means)

    drift_order = ['temporal', 'cross_activity', 'cross_both']

    # ── Table 1: mean EER per model per drift ───────────────────────────────
    print('\n' + '=' * 96)
    print('Mean EER per model per drift type (averaged across seeds, then across users)')
    print('-' * 96)
    header = f"{'Drift Type':<18} {'n':>4}  " + '  '.join(f'{m:>6}' for m in MODEL_IDS)
    print(header)
    print('-' * 96)
    for drift in drift_order:
        if drift not in by_drift:
            continue
        data = by_drift[drift]
        n = len(data)
        means = [float(np.mean([d[i] for d in data])) for i in range(len(MODEL_IDS))]
        cells = '  '.join(f'{v:>6.4f}' for v in means)
        print(f"{drift:<18} {n:>4}  {cells}")
    print('=' * 96)

    # ── Table 2: adaptation gains (frozen − adapting), positive = better ────
    print('\n' + '=' * 96)
    print('Adaptation gains (Frozen − Adapting EER); positive = adaptation helps')
    print('-' * 96)
    pair_specs = [
        ('M1-M2', 'M1', 'M2'), ('M1-M3', 'M1', 'M3'), ('M1-M4', 'M1', 'M4'),
        ('M6-M7', 'M6', 'M7'), ('M6-M8', 'M6', 'M8'),
    ]
    pair_header = '  '.join(f'{lbl:>8}' for lbl, _, _ in pair_specs)
    print(f"{'Drift Type':<18} {'n':>4}  {pair_header}")
    print('-' * 96)
    idx = {m: i for i, m in enumerate(MODEL_IDS)}
    for drift in drift_order:
        if drift not in by_drift:
            continue
        data = by_drift[drift]
        n = len(data)
        gains = []
        for _, frozen, adapt in pair_specs:
            g = float(np.mean([d[idx[frozen]] - d[idx[adapt]] for d in data]))
            gains.append(g)
        cells = '  '.join(f'{v:>+8.4f}' for v in gains)
        print(f"{drift:<18} {n:>4}  {cells}")
    print('=' * 96)

    # ── Table 3: M2≡M4 and M7≡M8 RTRL/BPTT equivalence check ────────────────
    print('\n' + '=' * 96)
    print('RTRL ≡ BPTT equivalence (mean |delta| across users; expect ~0)')
    print('-' * 96)
    print(f"{'Drift Type':<18} {'n':>4}  {'|M2-M4|':>10}  {'|M7-M8|':>10}")
    print('-' * 96)
    for drift in drift_order:
        if drift not in by_drift:
            continue
        data = by_drift[drift]
        n = len(data)
        rnn_eq = float(np.mean([abs(d[idx['M2']] - d[idx['M4']]) for d in data]))
        gru_eq = float(np.mean([abs(d[idx['M7']] - d[idx['M8']]) for d in data]))
        print(f"{drift:<18} {n:>4}  {rnn_eq:>10.4f}  {gru_eq:>10.4f}")
    print('=' * 96)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = os.getenv('TOUCHALYTICS_DATA')
    if not data_dir:
        sys.exit('ERROR: TOUCHALYTICS_DATA environment variable not set.')
    if not os.path.isdir(data_dir):
        sys.exit(f'ERROR: TOUCHALYTICS_DATA={data_dir} is not a directory.')
    if not os.path.exists(os.path.join(data_dir, 'featMat.csv')):
        sys.exit(f'ERROR: featMat.csv not found in {data_dir}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If an older CSV exists with a different schema, refuse to append silently.
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            header = f.readline()
        missing = [m for m in MODEL_IDS if f'{m}_overall_eer' not in header]
        if missing:
            sys.exit(
                f'ERROR: {OUTPUT_FILE} exists but is missing columns for: '
                f'{", ".join(missing)}. Delete or rename it, then re-run.'
            )

    print(f'Loading Touchalytics windows from {data_dir} ...')
    t0 = time.time()
    all_X, all_y, all_doc = load_windows(data_dir, window_size=WINDOW_SIZE)
    print(f'  Loaded {len(all_X)} windows of shape {all_X.shape[1:]} '
          f'in {time.time() - t0:.1f}s')

    if all_X.shape[2] != INPUT_DIM:
        sys.exit(f'ERROR: expected {INPUT_DIM} features per stroke, got {all_X.shape[2]}')

    all_users = sorted(int(u) for u in np.unique(all_y))
    print(f'  Users: {len(all_users)}, '
          f'Docs: {sorted(int(d) for d in np.unique(all_doc))}')

    # Find eligible users per drift type
    eligible = {}
    for drift_type, cfg in DRIFT_CONFIGS.items():
        eligible[drift_type] = find_eligible_users(all_y, all_doc, all_users, cfg)
        print(f'  {drift_type}: {len(eligible[drift_type])} eligible users')

    # Initialize output CSV (header only) and load completed keys for resume
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    completed = load_completed()
    print(f'Resume: {len(completed)} (user, drift, seed) entries already done\n')

    total_jobs = sum(len(eligible[dt]) for dt in DRIFT_CONFIGS) * len(SEEDS)
    print(f'Total experiments: {total_jobs}\n')

    count = 0
    skipped = 0
    failed = 0

    for drift_type, cfg in DRIFT_CONFIGS.items():
        users = eligible[drift_type]
        for user in users:
            # Impostors picked from the same eligible pool so they have
            # train-doc and test-doc data with high probability.
            impostors = pick_impostors(user, users, n=N_IMPOSTORS, seed=user)
            for seed in SEEDS:
                count += 1
                key = (str(user), drift_type, str(seed))
                if key in completed:
                    print(f'[{count}/{total_jobs}] SKIP user={user} {drift_type} '
                          f'seed={seed} (already done)')
                    continue

                print(f'[{count}/{total_jobs}] user={user} {drift_type} seed={seed}',
                      flush=True)
                t_start = time.time()
                try:
                    result = run_one(user, impostors, cfg, seed,
                                     all_X, all_y, all_doc)
                except Exception as e:
                    failed += 1
                    print(f'  FAILED in {time.time() - t_start:.1f}s: {e}')
                    continue

                elapsed = time.time() - t_start
                if result is None:
                    skipped += 1
                    print(f'  SKIP (insufficient data) in {elapsed:.1f}s')
                    continue

                row = {
                    'user_id': user,
                    'drift_type': drift_type,
                    'seed': seed,
                    **result,
                }
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                eer_str = '  '.join(
                    f'{m}={result[f"{m}_overall_eer"]:.4f}' for m in MODEL_IDS)
                print(f'  {eer_str}  ({elapsed:.1f}s)')

    print(f'\nFinished: {count} jobs, {skipped} skipped, {failed} failed')
    write_summary()


if __name__ == '__main__':
    main()