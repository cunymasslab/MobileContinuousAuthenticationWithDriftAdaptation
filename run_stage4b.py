"""
run_stage4b.py
Stage 4B — Advisor-Requested Follow-Up Experiments

Experiment 1: Does more training data eliminate drift?
    Train on sessions 1-3 + first X% of session 4, stream on remainder.
    Frozen models only (M1, M5, M6). Splits: 0.25, 0.50, 0.75.

Experiment 2: Per-session cross-validation (drift verification)
    Train on 2/3 of sessions 1-3, test on held-out 1/3 vs session 4.
    Frozen models only (M1, M5, M6).

Experiment 4: Semi-supervised adaptation (pseudo-labels)
    RTRL adaptation only when model confidence exceeds threshold.
    Thresholds: 0.70, 0.80, 0.90, 0.95.

Experiment 5: Sliding-window retrain baseline
    Periodically retrain with BPTT on recent windows.
    Intervals: 5, 10, 25, 50 windows.

Experiment 6: Threshold-tuning baseline
    Use first N windows to find optimal decision threshold, apply to rest.
    No weight updates — tests if simple threshold shift suffices.

Prerequisites: Stage 4 complete. run_stage4.py unchanged.
"""

import numpy as np
import csv
import os
import sys
import time
import argparse
import torch
from dotenv import load_dotenv

from data_loader import get_sessions_by_condition
from preprocessor import fit_normalizer, normalize, create_dataset
from model import VanillaRNN
from run_experiment import load_user_windows, compute_eer, evaluate
from run_streaming import (
    get_impostors, clone_rnn, compute_rolling_eer, classify_pair, get_pairs,
    SENSORS, HIDDEN_DIM, MAX_NORM, ROLLING_WINDOW,
    LEARNING_RATE_TRAIN as LR_TRAIN, LEARNING_RATE_ONLINE as LR_ONLINE,
)
from run_stage1 import (
    AdamState, train_with_early_stopping,
    WINDOW_SIZE, STEP_SIZE, MAX_EPOCHS, PATIENCE, ES_DELTA,
)
from pytorch_models import (
    train_pytorch_model, evaluate_pytorch_model, create_model as create_pytorch_model,
)

# noqa: F401 — needed for Experiments 4/5
from gru_model import GRU_NumPy

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

SEEDS = [42, 123, 456]
CONDITIONS = [
    'READING+SITTING', 'READING+WALKING',
    'WRITING+SITTING', 'WRITING+WALKING',
    'MAP+SITTING',     'MAP+WALKING',
]

def train_numpy_rnn(X_train, y_train, X_val, y_val, input_dim, seed):
    """Train VanillaRNN with early stopping (Stage 1 protocol)."""
    np.random.seed(seed)
    rnn, convergence_epoch, final_loss, curves = train_with_early_stopping(
        X_train, y_train, X_val, y_val, input_dim,
        optimizer_name='adam', lr=LR_TRAIN,
        hidden_dim=HIDDEN_DIM, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
    )
    return rnn, convergence_epoch, final_loss

def train_pytorch(X_train, y_train, X_val, y_val, input_dim, arch, seed):
    """Train PyTorch LSTM or GRU with early stopping."""
    model, convergence_epoch, final_loss, curves = train_pytorch_model(
        X_train, y_train, X_val, y_val,
        arch=arch, input_dim=input_dim, hidden_dim=HIDDEN_DIM,
        lr=LR_TRAIN, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, seed=seed,
    )
    return model, convergence_epoch, final_loss

def pytorch_infer(model, window):
    """Run a single window through a PyTorch model (frozen)."""
    device = next(model.parameters()).device
    x_t = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(x_t)
    return pred.item()

# ── Shared data loading ──────────────────────────────────────────────────────

def load_training_data(target_user, impostor_users, train_condition, seed):
    """
    Load and prepare training data from sessions 1-3 for a user/condition.

    Returns:
        legit_train, impostor_train (normalized), mean, std, input_dim, train_sessions
        or None if insufficient data.
    """
    train_activity, train_posture = train_condition.split('+')
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)
    if len(train_session_numbers) < 3:
        return None
    train_sessions = train_session_numbers[:3]

    legit_train = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_train) == 0:
        return None

    impostor_train = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_train) == 0:
        return None
    impostor_train = np.concatenate(impostor_train)

    # Balanced downsampling
    ds_rng = np.random.RandomState(seed)
    if len(impostor_train) > len(legit_train):
        indices = ds_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[indices]

    # Normalize (fit on training data)
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)

    input_dim = legit_train.shape[2]
    return legit_train, impostor_train, mean, std, input_dim, train_sessions


def make_train_val_split(legit_train, impostor_train, seed):
    """Create shuffled dataset and 80/20 train/val split."""
    X_all, y_all = create_dataset(legit_train, impostor_train, rng=np.random.RandomState(seed))
    split_rng = np.random.RandomState(seed + 500)  # Decoupled from create_dataset seed
    n = len(X_all)
    indices = split_rng.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[indices[:split]], y_all[indices[:split]]
    X_val, y_val = X_all[indices[split:]], y_all[indices[split:]]
    return X_train, y_train, X_val, y_val

# ── Streaming data loading ───────────────────────────────────────────────────

def load_streaming_data(target_user, impostor_users, stream_condition, train_sessions,
                        train_condition, mean, std, seed):
    """
    Load session 4 streaming data, normalize, and interleave.

    Returns:
        X_stream, y_stream (interleaved) or None if insufficient data.
    """
    stream_activity, stream_posture = stream_condition.split('+')

    # Load ALL session 4 legit windows from STREAM condition (before normalization).
    # For cross-condition pairs, this is intentional: the session 4 data comes from
    # the stream condition (e.g., WALKING), not the train condition (e.g., SITTING).
    # This matches Stage 4 protocol and tests adaptation to condition shift.
    stream_session_numbers = get_sessions_by_condition(target_user, stream_activity, stream_posture)

    if len(stream_session_numbers) < 4:
        return None, None

    stream_test_session = stream_session_numbers[3]

    # Temporal overlap guard
    if train_condition == stream_condition:
        overlap = set(train_sessions) & {stream_test_session}
        if overlap:
            raise ValueError(
                f'Temporal session overlap for {target_user} on {train_condition}: '
                f'train={train_sessions}, stream={stream_test_session}')

    legit_stream = load_user_windows(target_user, [stream_test_session], SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_stream) == 0:
        return None, None

    impostor_stream = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, stream_activity, stream_posture)
        if len(sessions) >= 4:
            impostor_stream.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_stream) == 0:
        return None, None
    impostor_stream = np.concatenate(impostor_stream)

    # Balanced downsampling
    user_seed = sum(ord(c) for c in target_user) + seed
    stream_ds_rng = np.random.RandomState(user_seed)
    if len(impostor_stream) > len(legit_stream):
        indices = stream_ds_rng.choice(len(impostor_stream), size=len(legit_stream), replace=False)
        impostor_stream = impostor_stream[indices]

    # Normalize with training stats
    legit_stream = normalize(legit_stream, mean, std)
    impostor_stream = normalize(impostor_stream, mean, std)

    # Interleave
    n_legit = len(legit_stream)
    n_total = n_legit + len(impostor_stream)
    interleave_rng = np.random.RandomState(user_seed + 1)
    legit_positions = np.sort(interleave_rng.choice(n_total, size=n_legit, replace=False))
    impostor_positions = np.sort(np.setdiff1d(np.arange(n_total), legit_positions))

    X_stream = np.empty_like(np.concatenate([legit_stream, impostor_stream]))
    y_stream = np.empty(n_total)
    X_stream[legit_positions] = legit_stream
    y_stream[legit_positions] = 1.0
    X_stream[impostor_positions] = impostor_stream
    y_stream[impostor_positions] = 0.0

    return X_stream, y_stream

# ── Frozen evaluation ────────────────────────────────────────────────────────

def evaluate_frozen(model_id, model, X_stream, y_stream, is_pytorch=False):
    """
    Run a frozen model over a stream, compute EER and metrics.

    Returns dict with overall_eer, initial_eer, end_eer, recovery, accuracy,
    precision, recall, f1, mcc, auc_roc.
    """
    labels = []
    scores = []
    for w in range(len(X_stream)):
        window = X_stream[w]
        if is_pytorch:
            score = pytorch_infer(model, window)
        else:
            a_cache, h_cache = model.forward(window)
            h_final = h_cache[len(window) - 1]
            pred = model.output(h_final)
            score = pred[0]
        labels.append(y_stream[w])
        scores.append(score)

    labels_np = np.array(labels)
    scores_np = np.array(scores)

    result = {}
    overall_eer = compute_eer(labels_np, scores_np) if len(np.unique(labels_np)) >= 2 else None
    init_n = min(ROLLING_WINDOW, len(labels_np))
    has_both_init = len(np.unique(labels_np[:init_n])) >= 2
    initial_eer = compute_eer(labels_np[:init_n], scores_np[:init_n]) if has_both_init else None
    has_both_end = len(labels_np) >= ROLLING_WINDOW and len(np.unique(labels_np[-ROLLING_WINDOW:])) >= 2
    end_eer = compute_eer(labels_np[-ROLLING_WINDOW:], scores_np[-ROLLING_WINDOW:]) if has_both_end else None
    recovery = None
    if initial_eer is not None and end_eer is not None and initial_eer > 0:
        recovery = (initial_eer - end_eer) / initial_eer

    result[f'{model_id}_overall_eer'] = overall_eer
    result[f'{model_id}_initial_eer'] = initial_eer
    result[f'{model_id}_end_eer'] = end_eer
    result[f'{model_id}_recovery'] = recovery

    accuracy, precision, recall, f1, mcc = evaluate(labels_np, scores_np)
    result[f'{model_id}_accuracy'] = accuracy
    result[f'{model_id}_precision'] = precision
    result[f'{model_id}_recall'] = recall
    result[f'{model_id}_f1'] = f1
    result[f'{model_id}_mcc'] = mcc

    if len(np.unique(labels_np)) >= 2:
        from sklearn.metrics import roc_auc_score
        result[f'{model_id}_auc_roc'] = roc_auc_score(labels_np, scores_np)
    else:
        result[f'{model_id}_auc_roc'] = None

    return result, scores_np


def score_frozen_eer(model, X, y, is_pytorch=False):
    """Score all windows with a frozen model, return EER only."""
    scores = []
    for w in range(len(X)):
        if is_pytorch:
            score = pytorch_infer(model, X[w])
        else:
            a_cache, h_cache = model.forward(X[w])
            h_final = h_cache[len(X[w]) - 1]
            pred = model.output(h_final)
            score = pred[0]
        scores.append(score)
    if len(np.unique(y)) >= 2:
        return compute_eer(y, np.array(scores))
    return None



# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Does more training data eliminate drift?
# ══════════════════════════════════════════════════════════════════════════════

EXP1_SUMMARY_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category',
    'seed', 'split_fraction', 'train_sessions', 'stream_session',
    'legit_train_windows', 'legit_stream_windows',
    'M1_overall_eer', 'M1_initial_eer', 'M1_end_eer', 'M1_recovery',
    'M1_accuracy', 'M1_precision', 'M1_recall', 'M1_f1', 'M1_mcc', 'M1_auc_roc',
    'M5_overall_eer', 'M5_initial_eer', 'M5_end_eer', 'M5_recovery',
    'M5_accuracy', 'M5_precision', 'M5_recall', 'M5_f1', 'M5_mcc', 'M5_auc_roc',
    'M6_overall_eer', 'M6_initial_eer', 'M6_end_eer', 'M6_recovery',
    'M6_accuracy', 'M6_precision', 'M6_recall', 'M6_f1', 'M6_mcc', 'M6_auc_roc',
]

def run_exp1(target_user, impostor_users, train_condition, stream_condition,
             seed, split_fraction):
    """
    Experiment 1: Train on sessions 1-3 + first split_fraction of session 4.
    Stream on remaining portion of session 4. Frozen models only.
    """
    train_activity, train_posture = train_condition.split('+')
    stream_activity, stream_posture = stream_condition.split('+')

    # Find training sessions (lightweight — no data loading yet)
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)
    if len(train_session_numbers) < 3:
        return None
    train_sessions = train_session_numbers[:3]


    # Load ALL session 4 legit windows from STREAM condition (before normalization).
    # For cross-condition pairs, session 4 data comes from the stream condition.
    # The first split_fraction of these windows augments training data — this gives
    # the frozen model exposure to the target distribution (strongest test for RTRL).

    stream_session_numbers = get_sessions_by_condition(target_user, stream_activity, stream_posture)
    if len(stream_session_numbers) < 4:
        return None
    stream_test_session = stream_session_numbers[3]

    # Temporal overlap guard
    if train_condition == stream_condition:
        if stream_test_session in train_sessions:
            raise ValueError(f'Session overlap for {target_user}')

    legit_stream_raw = load_user_windows(target_user, [stream_test_session], SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_stream_raw) == 0:
        return None

    # Split session 4 legit windows.
    # NOTE: At high split fractions (e.g., 0.75), the streaming portion is small
    # (25% of session 4), which increases EER variance. This is expected and
    # should be reported alongside sample sizes in the paper.
    n_s4 = len(legit_stream_raw)
    n_to_train = int(n_s4 * split_fraction)
    s4_for_training = legit_stream_raw[:n_to_train]  # first portion → training
    s4_for_streaming = legit_stream_raw[n_to_train:]  # remainder → streaming


    if len(s4_for_streaming) == 0:
        return None

    # Augment training data with session 4 portion
    legit_train_raw = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)

    if len(legit_train_raw) == 0:
        return None

    if n_to_train > 0:
        legit_train_augmented = np.concatenate([legit_train_raw, s4_for_training])
    else:
        legit_train_augmented = legit_train_raw

    # Re-load impostor training data from TRAIN condition (raw, before normalization).
    # Impostors always come from the same condition as the target's training data.

    impostor_train_raw = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train_raw.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_train_raw) == 0:
        return None
    impostor_train_raw = np.concatenate(impostor_train_raw)

    # Balanced downsampling against augmented legit
    ds_rng = np.random.RandomState(seed)
    if len(impostor_train_raw) > len(legit_train_augmented):
        indices = ds_rng.choice(len(impostor_train_raw), size=len(legit_train_augmented), replace=False)
        impostor_train_raw = impostor_train_raw[indices]

    # Refit normalizer on augmented training data
    mean, std = fit_normalizer(np.concatenate([legit_train_augmented, impostor_train_raw]))
    legit_train_norm = normalize(legit_train_augmented, mean, std)
    impostor_train_norm = normalize(impostor_train_raw, mean, std)

    input_dim = legit_train_norm.shape[2]

    # Train/val split and train models from scratch
    X_train, y_train, X_val, y_val = make_train_val_split(legit_train_norm, impostor_train_norm, seed)


    print(f'    Exp1 split={split_fraction}: {len(legit_train_augmented)} legit train, '
          f'{len(s4_for_streaming)} legit stream')

    # Train frozen models
    rnn, _, _ = train_numpy_rnn(X_train, y_train, X_val, y_val, input_dim, seed)
    lstm_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'lstm', seed)
    gru_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'gru', seed)


    # Prepare streaming data: impostor windows from STREAM condition session 4.
    # For cross-condition pairs, stream impostors come from the stream condition.

    impostor_stream = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, stream_activity, stream_posture)
        if len(sessions) >= 4:
            impostor_stream.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_stream) == 0:
        return None
    impostor_stream = np.concatenate(impostor_stream)

    # Downsample impostors to match remaining legit stream
    user_seed = sum(ord(c) for c in target_user) + seed
    stream_ds_rng = np.random.RandomState(user_seed)
    if len(impostor_stream) > len(s4_for_streaming):
        indices = stream_ds_rng.choice(len(impostor_stream), size=len(s4_for_streaming), replace=False)
        impostor_stream = impostor_stream[indices]

    # Normalize streaming data with augmented training stats
    s4_stream_norm = normalize(s4_for_streaming, mean, std)
    impostor_stream_norm = normalize(impostor_stream, mean, std)

    # Interleave
    n_legit = len(s4_stream_norm)
    n_total = n_legit + len(impostor_stream_norm)
    interleave_rng = np.random.RandomState(user_seed + 1)
    legit_positions = np.sort(interleave_rng.choice(n_total, size=n_legit, replace=False))
    impostor_positions = np.sort(np.setdiff1d(np.arange(n_total), legit_positions))

    X_stream = np.empty((n_total, s4_stream_norm.shape[1], s4_stream_norm.shape[2]))
    y_stream = np.empty(n_total)
    X_stream[legit_positions] = s4_stream_norm
    y_stream[legit_positions] = 1.0
    X_stream[impostor_positions] = impostor_stream_norm
    y_stream[impostor_positions] = 0.0

    # Evaluate frozen models
    m1_results, _ = evaluate_frozen('M1', rnn, X_stream, y_stream, is_pytorch=False)
    m5_results, _ = evaluate_frozen('M5', lstm_model, X_stream, y_stream, is_pytorch=True)
    m6_results, _ = evaluate_frozen('M6', gru_model, X_stream, y_stream, is_pytorch=True)


    summary = {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'pair_category': classify_pair(train_condition, stream_condition),
        'seed': seed,
        'split_fraction': split_fraction,
        'train_sessions': ','.join(str(s) for s in train_sessions),
        'stream_session': str(stream_test_session),
        'legit_train_windows': len(legit_train_augmented),
        'legit_stream_windows': len(s4_for_streaming),
    }
    summary.update(m1_results)
    summary.update(m5_results)
    summary.update(m6_results)
    return summary



# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Per-session cross-validation (drift verification)
# ══════════════════════════════════════════════════════════════════════════════

EXP2_SUMMARY_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category', 'seed',
    'legit_train_windows', 'legit_within_test_windows', 'legit_session4_windows',
    'M1_within_eer', 'M1_session4_eer', 'M1_drift_gap',
    'M5_within_eer', 'M5_session4_eer', 'M5_drift_gap',
    'M6_within_eer', 'M6_session4_eer', 'M6_drift_gap',
]

def run_exp2(target_user, impostor_users, train_condition, stream_condition, seed):
    """
    Experiment 2: Drift verification via cross-validation.
    Train on 2/3 of sessions 1-3 windows, test on held-out 1/3 (within-distribution).
    Compare within-distribution EER vs session 4 EER (drift).
    Large gap proves drift is real.
    """
    train_activity, train_posture = train_condition.split('+')
    stream_activity, stream_posture = stream_condition.split('+')

    # Load ALL legit windows from sessions 1-3 (train condition)
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)
    if len(train_session_numbers) < 3:
        return None
    train_sessions = train_session_numbers[:3]

    all_legit = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(all_legit) == 0:
        return None

    # Split legit: 2/3 train, 1/3 within-test (random with seed)
    split_rng = np.random.RandomState(seed)
    n_legit = len(all_legit)
    legit_indices = split_rng.permutation(n_legit)
    legit_split = int(n_legit * 2 / 3)
    legit_train = all_legit[legit_indices[:legit_split]]
    legit_within_test = all_legit[legit_indices[legit_split:]]

    # Load ALL impostor windows from sessions 1-3 (train condition)
    impostor_all = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_all.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_all) == 0:
        return None
    impostor_all = np.concatenate(impostor_all)

    # Split impostor: 2/3 train, 1/3 within-test
    imp_indices = split_rng.permutation(len(impostor_all))
    imp_split = int(len(impostor_all) * 2 / 3)
    impostor_train = impostor_all[imp_indices[:imp_split]]
    impostor_within_test = impostor_all[imp_indices[imp_split:]]

    # Balance training set
    if len(impostor_train) > len(legit_train):
        idx = split_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[idx]

    # Normalize (fit on training portion only)
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)
    legit_within_test = normalize(legit_within_test, mean, std)
    impostor_within_test = normalize(impostor_within_test, mean, std)

    input_dim = legit_train.shape[2]

    # Create training dataset and 80/20 train/val split
    X_all, y_all = create_dataset(legit_train, impostor_train, rng=np.random.RandomState(seed))
    tv_rng = np.random.RandomState(seed + 1000)
    n = len(X_all)
    tv_indices = tv_rng.permutation(n)
    tv_split = int(0.8 * n)
    X_train, y_train = X_all[tv_indices[:tv_split]], y_all[tv_indices[:tv_split]]
    X_val, y_val = X_all[tv_indices[tv_split:]], y_all[tv_indices[tv_split:]]

    print(f'    Exp2: {len(legit_train)} legit train, {len(legit_within_test)} legit within-test')

    # Train frozen models
    rnn, _, _ = train_numpy_rnn(X_train, y_train, X_val, y_val, input_dim, seed)
    lstm_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'lstm', seed)
    gru_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'gru', seed)

    # ── Within-distribution test ─────────────────────────────────────────
    # Balance within-test impostor to match within-test legit
    bal_rng = np.random.RandomState(seed + 2000)
    imp_within = impostor_within_test
    if len(imp_within) > len(legit_within_test):
        idx = bal_rng.choice(len(imp_within), size=len(legit_within_test), replace=False)
        imp_within = imp_within[idx]

    # Interleave within-test set
    user_seed = sum(ord(c) for c in target_user) + seed
    n_legit_w = len(legit_within_test)
    n_total_w = n_legit_w + len(imp_within)
    within_rng = np.random.RandomState(user_seed + 100)
    legit_pos_w = np.sort(within_rng.choice(n_total_w, size=n_legit_w, replace=False))
    imp_pos_w = np.sort(np.setdiff1d(np.arange(n_total_w), legit_pos_w))

    X_within = np.empty((n_total_w, legit_within_test.shape[1], legit_within_test.shape[2]))
    y_within = np.empty(n_total_w)
    X_within[legit_pos_w] = legit_within_test
    y_within[legit_pos_w] = 1.0
    X_within[imp_pos_w] = imp_within
    y_within[imp_pos_w] = 0.0

    m1_within_eer = score_frozen_eer(rnn, X_within, y_within, is_pytorch=False)
    m5_within_eer = score_frozen_eer(lstm_model, X_within, y_within, is_pytorch=True)
    m6_within_eer = score_frozen_eer(gru_model, X_within, y_within, is_pytorch=True)

    # ── Session 4 test (drift) ───────────────────────────────────────────
    result = load_streaming_data(target_user, impostor_users, stream_condition,
                                 train_sessions, train_condition, mean, std, seed)
    if result[0] is None:
        return None
    X_s4, y_s4 = result

    m1_s4_eer = score_frozen_eer(rnn, X_s4, y_s4, is_pytorch=False)
    m5_s4_eer = score_frozen_eer(lstm_model, X_s4, y_s4, is_pytorch=True)
    m6_s4_eer = score_frozen_eer(gru_model, X_s4, y_s4, is_pytorch=True)

    n_legit_s4 = int(np.sum(y_s4 == 1))

    summary = {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'pair_category': classify_pair(train_condition, stream_condition),
        'seed': seed,
        'legit_train_windows': len(legit_train),
        'legit_within_test_windows': len(legit_within_test),
        'legit_session4_windows': n_legit_s4,
        'M1_within_eer': m1_within_eer,
        'M1_session4_eer': m1_s4_eer,
        'M1_drift_gap': m1_s4_eer - m1_within_eer if m1_s4_eer is not None and m1_within_eer is not None else None,
        'M5_within_eer': m5_within_eer,
        'M5_session4_eer': m5_s4_eer,
        'M5_drift_gap': m5_s4_eer - m5_within_eer if m5_s4_eer is not None and m5_within_eer is not None else None,
        'M6_within_eer': m6_within_eer,
        'M6_session4_eer': m6_s4_eer,
        'M6_drift_gap': m6_s4_eer - m6_within_eer if m6_s4_eer is not None and m6_within_eer is not None else None,
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Threshold-tuning baseline
# ══════════════════════════════════════════════════════════════════════════════

EXP6_SUMMARY_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category',
    'seed', 'calibration_windows',
    'optimal_threshold_M1', 'optimal_threshold_M5', 'optimal_threshold_M6',
    'M1_eer',
    'M1_default_accuracy', 'M1_default_precision', 'M1_default_recall', 'M1_default_f1', 'M1_default_mcc',
    'M1_tuned_accuracy', 'M1_tuned_precision', 'M1_tuned_recall', 'M1_tuned_f1', 'M1_tuned_mcc',
    'M5_eer',
    'M5_default_accuracy', 'M5_default_precision', 'M5_default_recall', 'M5_default_f1', 'M5_default_mcc',
    'M5_tuned_accuracy', 'M5_tuned_precision', 'M5_tuned_recall', 'M5_tuned_f1', 'M5_tuned_mcc',
    'M6_eer',
    'M6_default_accuracy', 'M6_default_precision', 'M6_default_recall', 'M6_default_f1', 'M6_default_mcc',
    'M6_tuned_accuracy', 'M6_tuned_precision', 'M6_tuned_recall', 'M6_tuned_f1', 'M6_tuned_mcc',
]



def find_optimal_threshold(labels, scores):
    """Sweep 101 thresholds, return the one that minimizes EER."""
    best_eer = float('inf')
    best_thresh = 0.5
    for t in np.linspace(0, 1, 101):
        preds = (np.array(scores) >= t).astype(float)
        tp = np.sum((preds == 1) & (np.array(labels) == 1))
        fp = np.sum((preds == 1) & (np.array(labels) == 0))
        tn = np.sum((preds == 0) & (np.array(labels) == 0))
        fn = np.sum((preds == 0) & (np.array(labels) == 1))
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        eer = (far + frr) / 2
        if eer < best_eer:
            best_eer = eer
            best_thresh = t
    return best_thresh

def run_exp6(target_user, impostor_users, train_condition, stream_condition,
             seed, calibration_windows=10):
    """
    Experiment 6: Threshold-tuning baseline.
    Train frozen models normally. Use first N windows to find optimal threshold.
    Evaluate on remaining windows with tuned threshold vs default 0.5.
    """
    # Load training data and train models (standard protocol)
    data = load_training_data(target_user, impostor_users, train_condition, seed)
    if data is None:
        return None
    legit_train, impostor_train, mean, std, input_dim, train_sessions = data

    X_train, y_train, X_val, y_val = make_train_val_split(legit_train, impostor_train, seed)

    rnn, _, _ = train_numpy_rnn(X_train, y_train, X_val, y_val, input_dim, seed)
    lstm_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'lstm', seed)
    gru_model, _, _ = train_pytorch(X_train, y_train, X_val, y_val, input_dim, 'gru', seed)

    # Load streaming data (standard protocol — all of session 4)
    result = load_streaming_data(target_user, impostor_users, stream_condition,
                                 train_sessions, train_condition, mean, std, seed)
    if result[0] is None:
        return None
    X_stream, y_stream = result

    if len(X_stream) <= calibration_windows:
        return None

    # Get all scores from frozen models
    _, m1_scores = evaluate_frozen('M1', rnn, X_stream, y_stream, is_pytorch=False)
    _, m5_scores = evaluate_frozen('M5', lstm_model, X_stream, y_stream, is_pytorch=True)
    _, m6_scores = evaluate_frozen('M6', gru_model, X_stream, y_stream, is_pytorch=True)

    labels_all = y_stream

    # Split into calibration and evaluation
    cal_labels = labels_all[:calibration_windows]
    cal_m1_scores = m1_scores[:calibration_windows]
    cal_m5_scores = m5_scores[:calibration_windows]
    cal_m6_scores = m6_scores[:calibration_windows]

    eval_labels = labels_all[calibration_windows:]
    eval_m1_scores = m1_scores[calibration_windows:]
    eval_m5_scores = m5_scores[calibration_windows:]
    eval_m6_scores = m6_scores[calibration_windows:]

    # Find optimal thresholds on calibration set.
    # NOTE: With only N calibration windows, class balance depends on interleaving.
    # Extreme imbalance (e.g., 9 legit / 1 impostor) may bias threshold estimation.
    # The paper should report the mean class ratio in calibration sets.
    if len(np.unique(cal_labels)) < 2:
        return None

    thresh_m1 = find_optimal_threshold(cal_labels, cal_m1_scores)
    thresh_m5 = find_optimal_threshold(cal_labels, cal_m5_scores)
    thresh_m6 = find_optimal_threshold(cal_labels, cal_m6_scores)

    # Compute EER (threshold-independent — same regardless of calibration)
    if len(np.unique(eval_labels)) < 2:
        return None

    m1_eer = compute_eer(eval_labels, eval_m1_scores)
    m5_eer = compute_eer(eval_labels, eval_m5_scores)
    m6_eer = compute_eer(eval_labels, eval_m6_scores)

    # Compute fixed-threshold metrics: default (0.5) vs calibrated threshold
    # This is where threshold tuning can actually help
    m1_def_acc, m1_def_prec, m1_def_rec, m1_def_f1, m1_def_mcc = evaluate(eval_labels, eval_m1_scores, threshold=0.5)
    m5_def_acc, m5_def_prec, m5_def_rec, m5_def_f1, m5_def_mcc = evaluate(eval_labels, eval_m5_scores, threshold=0.5)
    m6_def_acc, m6_def_prec, m6_def_rec, m6_def_f1, m6_def_mcc = evaluate(eval_labels, eval_m6_scores, threshold=0.5)

    m1_tun_acc, m1_tun_prec, m1_tun_rec, m1_tun_f1, m1_tun_mcc = evaluate(eval_labels, eval_m1_scores, threshold=thresh_m1)
    m5_tun_acc, m5_tun_prec, m5_tun_rec, m5_tun_f1, m5_tun_mcc = evaluate(eval_labels, eval_m5_scores, threshold=thresh_m5)
    m6_tun_acc, m6_tun_prec, m6_tun_rec, m6_tun_f1, m6_tun_mcc = evaluate(eval_labels, eval_m6_scores, threshold=thresh_m6)

    summary = {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'pair_category': classify_pair(train_condition, stream_condition),
        'seed': seed,
        'calibration_windows': calibration_windows,
        'optimal_threshold_M1': thresh_m1,
        'optimal_threshold_M5': thresh_m5,
        'optimal_threshold_M6': thresh_m6,
        'M1_eer': m1_eer,
        'M1_default_accuracy': m1_def_acc, 'M1_default_precision': m1_def_prec,
        'M1_default_recall': m1_def_rec, 'M1_default_f1': m1_def_f1, 'M1_default_mcc': m1_def_mcc,
        'M1_tuned_accuracy': m1_tun_acc, 'M1_tuned_precision': m1_tun_prec,
        'M1_tuned_recall': m1_tun_rec, 'M1_tuned_f1': m1_tun_f1, 'M1_tuned_mcc': m1_tun_mcc,
        'M5_eer': m5_eer,
        'M5_default_accuracy': m5_def_acc, 'M5_default_precision': m5_def_prec,
        'M5_default_recall': m5_def_rec, 'M5_default_f1': m5_def_f1, 'M5_default_mcc': m5_def_mcc,
        'M5_tuned_accuracy': m5_tun_acc, 'M5_tuned_precision': m5_tun_prec,
        'M5_tuned_recall': m5_tun_rec, 'M5_tuned_f1': m5_tun_f1, 'M5_tuned_mcc': m5_tun_mcc,
        'M6_eer': m6_eer,
        'M6_default_accuracy': m6_def_acc, 'M6_default_precision': m6_def_prec,
        'M6_default_recall': m6_def_rec, 'M6_default_f1': m6_def_f1, 'M6_default_mcc': m6_def_mcc,
        'M6_tuned_accuracy': m6_tun_acc, 'M6_tuned_precision': m6_tun_prec,
        'M6_tuned_recall': m6_tun_rec, 'M6_tuned_f1': m6_tun_f1, 'M6_tuned_mcc': m6_tun_mcc,
    }
    return summary



# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 4B: Advisor-Requested Experiments')
    parser.add_argument('--experiment', type=int, required=True, choices=[1, 2, 4, 5, 6],
                        help='Which experiment to run')
    parser.add_argument('--users-file', type=str, default='test_users.txt')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456)')
    parser.add_argument('--chunk', type=int, default=None)
    parser.add_argument('--num-chunks', type=int, default=12)
    parser.add_argument('--pairs', type=str, default='all',
                        choices=['all', 'cross-posture', 'cross-activity', 'cross-both', 'temporal'])
    # Experiment-specific
    parser.add_argument('--split', type=float, default=0.25,
                        help='Exp 1: fraction of session 4 to add to training (0.25, 0.50, 0.75)')
    parser.add_argument('--calibration-windows', type=int, default=10,
                        help='Exp 6: number of windows for threshold calibration')
    args = parser.parse_args()

    seeds = SEEDS
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # Load users
    if not os.path.exists(args.users_file):
        print(f'ERROR: {args.users_file} not found.')
        sys.exit(1)
    with open(args.users_file) as f:
        all_test_users = sorted([line.strip() for line in f if line.strip()])

    # Chunk splitting
    if args.chunk is not None:
        chunk_idx = args.chunk - 1
        chunk_size = len(all_test_users) // args.num_chunks
        remainder = len(all_test_users) % args.num_chunks
        start = chunk_idx * chunk_size + min(chunk_idx, remainder)
        end = start + chunk_size + (1 if chunk_idx < remainder else 0)
        users_to_run = all_test_users[start:end]
        chunk_tag = f'_chunk{args.chunk}'
        print(f'Chunk {args.chunk}/{args.num_chunks}: {len(users_to_run)} users '
              f'(indices {start}-{end-1})')
    else:
        users_to_run = all_test_users
        chunk_tag = ''

    # Output directory
    os.makedirs('results_stage4b', exist_ok=True)

    # Select experiment
    if args.experiment == 1:
        split_tag = f'_split{int(args.split * 100)}'
        output_csv = f'results_stage4b/exp1{split_tag}{chunk_tag}_summary.csv'
        summary_fields = EXP1_SUMMARY_FIELDS
    elif args.experiment == 2:
        output_csv = f'results_stage4b/exp2{chunk_tag}_summary.csv'
        summary_fields = EXP2_SUMMARY_FIELDS
    elif args.experiment == 6:
        output_csv = f'results_stage4b/exp6{chunk_tag}_summary.csv'
        summary_fields = EXP6_SUMMARY_FIELDS
    else:
        print(f'Experiment {args.experiment} not yet implemented.')
        sys.exit(1)


    pairs = get_pairs(args.pairs)

    # Resume support
    completed = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('target_user'), row.get('train_condition'),
                       row.get('stream_condition'), row.get('seed'))
                completed.add(key)
        print(f'Resuming: {len(completed)} experiments already completed')
    else:
        with open(output_csv, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    # Get all users for impostor selection
    all_entries = os.listdir(dataset_path)
    all_users = []
    for d in sorted(all_entries):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
        if len(session_dirs) > 0:
            all_users.append(d)

    total = len(users_to_run) * len(pairs) * len(seeds)
    count = 0
    skipped = 0
    failed = 0

    print(f'Stage 4B — Experiment {args.experiment}')
    print(f'  Users: {len(users_to_run)} | Pairs: {len(pairs)} | Seeds: {seeds}')

    if args.experiment == 1:
        print(f'  Split fraction: {args.split}')
    elif args.experiment == 2:
        print(f'  Mode: 2/3 train, 1/3 within-test, session 4 drift test')
    elif args.experiment == 6:
        print(f'  Calibration windows: {args.calibration_windows}')

    print(f'  Output: {output_csv}')

    for user in users_to_run:
        impostors = get_impostors(user, all_users, n=3, seed=sum(ord(c) for c in user))

        for seed_val in seeds:
            for train_cond, stream_cond in pairs:
                count += 1
                key = (user, train_cond, stream_cond, str(seed_val))
                if key in completed:
                    continue

                pair_cat = classify_pair(train_cond, stream_cond)
                print(f'\n[{count}/{total}] User={user} seed={seed_val} | '
                      f'{train_cond} -> {stream_cond} ({pair_cat})')

                start_time = time.time()
                try:
                    if args.experiment == 1:
                        summary = run_exp1(user, impostors, train_cond, stream_cond,
                                           seed_val, args.split)
                    elif args.experiment == 2:
                        summary = run_exp2(user, impostors, train_cond, stream_cond,
                                           seed_val)
                    elif args.experiment == 6:
                        summary = run_exp6(user, impostors, train_cond, stream_cond,
                                           seed_val, args.calibration_windows)


                    elapsed = time.time() - start_time

                    if summary is None:
                        skipped += 1
                        print(f'    SKIPPED in {elapsed:.1f}s')
                        continue

                    eer_str = ''
                    for m in ['M1', 'M5', 'M6']:
                        
                        eer = next((v for k in [f'{m}_overall_eer', f'{m}_within_eer', f'{m}_eer'] if (v := summary.get(k)) is not None), None)


                        if eer is not None:
                            eer_str += f' {m}={eer:.4f}'
                    print(f'    Done in {elapsed:.1f}s |{eer_str}')

                    with open(output_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=summary_fields)
                        writer.writerow({k: summary.get(k, '') for k in summary_fields})

                except Exception as e:
                    elapsed = time.time() - start_time
                    failed += 1
                    print(f'    FAILED in {elapsed:.1f}s: {e}')
                    import traceback
                    traceback.print_exc()

    print(f'\n=== Experiment {args.experiment} complete ===')
    print(f'  Total: {count} | Skipped: {skipped} | Failed: {failed}')
    print(f'  Output: {output_csv}')
