"""
run_stage4.py
Stage 4 — Full Main Experiment
Goal: Primary results for the paper.
Users: 90 test users (from test_users.txt)
Drift categories: temporal, cross-posture, cross-activity, cross-both (36 condition pairs)
Seeds: 3 per user per condition per model (42, 123, 456)
Hyperparameters: All locked values from Stages 0-3

Models implemented:
    M1 — VanillaRNN-Frozen      (BPTT trained, frozen during streaming)
    M2 — VanillaRNN-RTRL        (BPTT trained, RTRL adaptation on all windows)
    M3 — VanillaRNN-RTRL-LegitOnly (BPTT trained, RTRL adaptation on legit windows only)
    M4 — VanillaRNN-BPTTFinetune (BPTT trained, per-window BPTT fine-tuning during streaming)
    M5 — LSTM-Frozen            (PyTorch trained, frozen during streaming)
    M6 — GRU-Frozen             (PyTorch trained, frozen during streaming)
    M7 — GRU-RTRL               (PyTorch trained, RTRL adaptation on all windows)
    M8 — GRU-BPTTFinetune       (PyTorch trained, per-window BPTT fine-tuning during streaming)


Prerequisites: WINDOW_SIZE, STEP_SIZE, OPTIMIZER, LR_TRAIN, LR_ONLINE locked from Stages 0-2.
"""

import numpy as np
import csv
import os
import sys
import time
import random
import argparse
import torch
import torch.nn as nn
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
from gru_model import GRU_NumPy





load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# ── Locked hyperparameters from Stages 0-2 ──────────────────────────────────
# WINDOW_SIZE=100, STEP_SIZE=100 (imported from run_stage1)
# OPTIMIZER=Adam, LR_TRAIN=0.01 (locked from Stage 1)

SEEDS = [42, 123, 456]

# All 6 HMOG conditions → 36 condition pairs
CONDITIONS = [
    'READING+SITTING', 'READING+WALKING',
    'WRITING+SITTING', 'WRITING+WALKING',
    'MAP+SITTING',     'MAP+WALKING',
]

# ── Model registry ──────────────────────────────────────────────────────────
ALL_MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']

MODEL_NAMES = {
    'M1': 'VanillaRNN-Frozen',
    'M2': 'VanillaRNN-RTRL',
    'M3': 'VanillaRNN-RTRL-LegitOnly',
    'M4': 'VanillaRNN-BPTTFinetune',
    'M5': 'LSTM-Frozen',
    'M6': 'GRU-Frozen',
    'M7': 'GRU-RTRL',
    'M8': 'GRU-BPTTFinetune',
    'M9': 'VanillaRNN-RTRL-Persistent',
}


# ── Output CSV fields ───────────────────────────────────────────────────────
SUMMARY_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category',
    'seed', 'train_sessions', 'stream_session',
    'stream_legit_count', 'stream_impostor_count', 'total_windows',
]
for m in ALL_MODELS:
    SUMMARY_FIELDS += [
        f'{m}_overall_eer', f'{m}_initial_eer', f'{m}_end_eer', f'{m}_recovery',
        f'{m}_accuracy', f'{m}_precision', f'{m}_recall', f'{m}_f1', f'{m}_mcc',
        f'{m}_auc_roc',
    ]
SUMMARY_FIELDS.append('forgetting_flag')


DETAIL_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category', 'seed',
    'window_index', 'true_label',
]
for m in ALL_MODELS:
    DETAIL_FIELDS += [f'{m}_score', f'{m}_rolling_eer', f'{m}_cumulative_eer']

TRAINING_CURVES_FIELDS = [
    'target_user', 'train_condition', 'model_type', 'hidden_dim',
    'lr_train', 'optimizer', 'window_size', 'step_size', 'seed',
    'epoch', 'train_loss', 'train_accuracy', 'val_loss',
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _train_numpy_rnn(X_train, y_train, X_val, y_val, input_dim, seed):
    """Train VanillaRNN with early stopping (Stage 1 protocol)."""
    np.random.seed(seed)
    rnn, convergence_epoch, final_loss, curves = train_with_early_stopping(
        X_train, y_train, X_val, y_val, input_dim,
        optimizer_name='adam', lr=LR_TRAIN,
        hidden_dim=HIDDEN_DIM, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
    )
    return rnn, convergence_epoch, final_loss, curves


def _train_pytorch(X_train, y_train, X_val, y_val, input_dim, arch, seed):
    """Train PyTorch LSTM or GRU with early stopping."""
    model, convergence_epoch, final_loss, curves = train_pytorch_model(
        X_train, y_train, X_val, y_val,
        arch=arch, input_dim=input_dim, hidden_dim=HIDDEN_DIM,
        lr=LR_TRAIN, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, seed=seed,
    )
    return model, convergence_epoch, final_loss, curves


def _pytorch_infer(model, window):
    """Run a single window through a PyTorch model (frozen)."""
    device = next(model.parameters()).device
    x_t = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(x_t)
    return pred.item()

def _transfer_gru_weights(pytorch_gru_model, input_dim):
    """Transfer PyTorch GRU weights to NumPy GRU for RTRL streaming."""
    gru_np = GRU_NumPy(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    gru_np.load_from_pytorch(pytorch_gru_model)
    return gru_np




# ── Core experiment function ─────────────────────────────────────────────────

def run_stage4_experiment(target_user, impostor_users, train_condition, stream_condition,
                          seed, models_to_run=None):
    """
    Run one Stage 4 experiment for a single user, condition pair, and seed.
    Trains all needed architectures and streams all requested models.

    Returns:
        summary: dict with summary metrics (or None if skipped)
        detail_log: list of per-window detail dicts
        training_curves: list of per-epoch training curve dicts
    """
    if models_to_run is None:
        models_to_run = ALL_MODELS

    train_activity, train_posture = train_condition.split('+')
    stream_activity, stream_posture = stream_condition.split('+')

    # ── Load training data ───────────────────────────────────────────────────
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)
    if len(train_session_numbers) < 3:
        return None, None, []
    train_sessions = train_session_numbers[:3]

    legit_train = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_train) == 0:
        return None, None, []

    impostor_train = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_train) == 0:
        return None, None, []
    impostor_train = np.concatenate(impostor_train)

    # Balanced downsampling — use experiment seed
    ds_rng = np.random.RandomState(seed)
    if len(impostor_train) > len(legit_train):
        indices = ds_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[indices]

    # Normalize (fit on training data)
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)

    X_all, y_all = create_dataset(legit_train, impostor_train, rng=np.random.RandomState(seed))

    input_dim = X_all.shape[2]

    # Train/val split (80/20) — matches Stage 1 protocol
    split_rng = np.random.RandomState(seed)
    n = len(X_all)
    indices = split_rng.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[indices[:split]], y_all[indices[:split]]
    X_val, y_val = X_all[indices[split:]], y_all[indices[split:]]

    # ── Train models ─────────────────────────────────────────────────────────
    all_training_curves = []
    curve_base = {
        'target_user': target_user,
        'train_condition': train_condition,
        'hidden_dim': HIDDEN_DIM,
        'lr_train': LR_TRAIN,
        'optimizer': 'Adam',
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'seed': seed,
    }

    # Need VanillaRNN for M1-M4?
    rnn_bptt = None
    need_vanilla = any(m in models_to_run for m in ['M1', 'M2', 'M3', 'M4', 'M9'])

    if need_vanilla:
        print(f'    Training VanillaRNN (BPTT, early stopping)...')
        rnn_bptt, conv_epoch, final_loss, curves = _train_numpy_rnn(
            X_train, y_train, X_val, y_val, input_dim, seed)
        for row in curves:
            all_training_curves.append({**curve_base, 'model_type': 'VanillaRNN', **row})
        print(f'    VanillaRNN converged at epoch {conv_epoch}, val_loss={final_loss:.4f}')

    # Need LSTM for M5?
    lstm_model = None
    if 'M5' in models_to_run:
        print(f'    Training LSTM (PyTorch, early stopping)...')
        lstm_model, conv_epoch, final_loss, curves = _train_pytorch(
            X_train, y_train, X_val, y_val, input_dim, 'lstm', seed)
        for row in curves:
            all_training_curves.append({**curve_base, 'model_type': 'LSTM', **row})
        print(f'    LSTM converged at epoch {conv_epoch}, val_loss={final_loss:.4f}')

    # Need GRU for M6, M7, M8?
    gru_model = None
    need_gru = any(m in models_to_run for m in ['M6', 'M7', 'M8'])
    if need_gru:
        print(f'    Training GRU (PyTorch, early stopping)...')
        gru_model, conv_epoch, final_loss, curves = _train_pytorch(
            X_train, y_train, X_val, y_val, input_dim, 'gru', seed)
        for row in curves:
            all_training_curves.append({**curve_base, 'model_type': 'GRU', **row})
        print(f'    GRU converged at epoch {conv_epoch}, val_loss={final_loss:.4f}')




    # ── Load streaming data ──────────────────────────────────────────────────
    stream_session_numbers = get_sessions_by_condition(target_user, stream_activity, stream_posture)
    if len(stream_session_numbers) < 4:
        return None, None, []
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
        return None, None, []

    impostor_stream = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, stream_activity, stream_posture)
        if len(sessions) >= 4:
            impostor_stream.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_stream) == 0:
        return None, None, []
    impostor_stream = np.concatenate(impostor_stream)

    # Balanced downsampling of stream impostors — use per-user seed
    user_seed = sum(ord(c) for c in target_user) + seed
    stream_ds_rng = np.random.RandomState(user_seed)
    if len(impostor_stream) > len(legit_stream):
        indices = stream_ds_rng.choice(len(impostor_stream), size=len(legit_stream), replace=False)
        impostor_stream = impostor_stream[indices]

    # Normalize stream with training stats
    legit_stream = normalize(legit_stream, mean, std)
    impostor_stream = normalize(impostor_stream, mean, std)

    # Interleave: within-class temporal order preserved, positions randomized
    # Use per-user seed (not fixed 42) so each user gets different interleaving
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

    print(f'    Stream: {n_total} windows ({n_legit} legit, {len(impostor_stream)} impostor)')

    # ── Initialize streaming models ──────────────────────────────────────────
    active_models = {}

    if 'M1' in models_to_run and rnn_bptt is not None:
        active_models['M1'] = {
            'rnn': clone_rnn(rnn_bptt), 'type': 'frozen_numpy',
            'labels': [], 'scores': [],
        }

    if 'M2' in models_to_run and rnn_bptt is not None:
        rnn_m2 = clone_rnn(rnn_bptt)
        param_shapes = [rnn_m2.W_xh.shape, rnn_m2.W_hh.shape, rnn_m2.W_hy.shape,
                        rnn_m2.b_h.shape, rnn_m2.b_y.shape]
        active_models['M2'] = {
            'rnn': rnn_m2, 'type': 'rtrl_adapt_all',
            'opt': AdamState(param_shapes, lr=LR_ONLINE),
            'labels': [], 'scores': [],
        }

    if 'M3' in models_to_run and rnn_bptt is not None:
        rnn_m3 = clone_rnn(rnn_bptt)
        param_shapes = [rnn_m3.W_xh.shape, rnn_m3.W_hh.shape, rnn_m3.W_hy.shape,
                        rnn_m3.b_h.shape, rnn_m3.b_y.shape]
        active_models['M3'] = {
            'rnn': rnn_m3, 'type': 'rtrl_adapt_legit',
            'opt': AdamState(param_shapes, lr=LR_ONLINE),
            'labels': [], 'scores': [],
        }

    if 'M4' in models_to_run and rnn_bptt is not None:
        rnn_m4 = clone_rnn(rnn_bptt)
        param_shapes = [rnn_m4.W_xh.shape, rnn_m4.W_hh.shape, rnn_m4.W_hy.shape,
                        rnn_m4.b_h.shape, rnn_m4.b_y.shape]
        active_models['M4'] = {
            'rnn': rnn_m4, 'type': 'bptt_finetune_numpy',
            'opt': AdamState(param_shapes, lr=LR_ONLINE),
            'labels': [], 'scores': [],
        }

    if 'M5' in models_to_run and lstm_model is not None:
        active_models['M5'] = {
            'model': lstm_model, 'type': 'frozen_pytorch',
            'labels': [], 'scores': [],
        }

    if 'M6' in models_to_run and gru_model is not None:
        active_models['M6'] = {
            'model': gru_model, 'type': 'frozen_pytorch',
            'labels': [], 'scores': [],
        }

    if 'M7' in models_to_run and gru_model is not None:
        gru_m7 = GRU_NumPy(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
        gru_m7.load_from_pytorch(gru_model)
        param_names = ['W_xr', 'W_hr', 'b_r', 'W_xz', 'W_hz', 'b_z',
                        'W_xn', 'W_hn', 'b_n_input', 'b_n_hidden', 'W_hy', 'b_y']
        param_shapes = [getattr(gru_m7, k).shape for k in param_names]
        active_models['M7'] = {
            'rnn': gru_m7, 'type': 'gru_rtrl_adapt',
            'opt': AdamState(param_shapes, lr=LR_ONLINE),
            'param_names': param_names,
            'labels': [], 'scores': [],
        }


    if 'M8' in models_to_run and gru_model is not None:
        import copy
        gru_m8 = copy.deepcopy(gru_model)
        device = next(gru_m8.parameters()).device
        active_models['M8'] = {
            'model': gru_m8, 'type': 'bptt_finetune_pytorch',
            'optimizer': torch.optim.Adam(gru_m8.parameters(), lr=LR_ONLINE),
            'criterion': nn.BCELoss(),
            'device': device,
            'labels': [], 'scores': [],
        }

    if 'M9' in models_to_run and rnn_bptt is not None:
        rnn_m9 = clone_rnn(rnn_bptt)
        param_shapes = [rnn_m9.W_xh.shape, rnn_m9.W_hh.shape, rnn_m9.W_hy.shape,
                        rnn_m9.b_h.shape, rnn_m9.b_y.shape]
        H_m9 = rnn_m9.b_h.shape[0]
        D_m9 = rnn_m9.W_xh.shape[1]
        active_models['M9'] = {
            'rnn': rnn_m9, 'type': 'rtrl_adapt_persistent',
            'opt': AdamState(param_shapes, lr=LR_ONLINE),
            'S_xh': np.zeros((H_m9, H_m9, D_m9)),
            'S_hh': np.zeros((H_m9, H_m9, H_m9)),
            'S_bh': np.zeros((H_m9, H_m9)),
            'h': np.zeros(H_m9),
            'labels': [], 'scores': [],
        }

    if len(active_models) == 0:
        return None, None, []

    # ── Stream and adapt ─────────────────────────────────────────────────────
    detail_log = []

    for w in range(len(X_stream)):
        window = X_stream[w]
        true_label = y_stream[w]

        row = {'window_index': w, 'true_label': true_label}

        for model_id, m in active_models.items():
            mtype = m['type']

            if mtype == 'frozen_numpy':
                a_cache, h_cache = m['rnn'].forward(window)
                h_final = h_cache[len(window) - 1]
                pred = m['rnn'].output(h_final)
                score = pred[0]

            elif mtype == 'rtrl_adapt_all':
                rnn = m['rnn']
                h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(window)
                pred = rnn.output(h_final)
                score = pred[0]
                # Always update
                grads = rnn.rtrl_compute_gradients(true_label, pred, h_final, S_xh, S_hh, S_bh)
                grads = rnn.clip_gradients(grads, MAX_NORM)
                m['opt'].step(rnn, grads)

            elif mtype == 'rtrl_adapt_persistent':
                rnn = m['rnn']
                h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward_persistent(
                    window, m['S_xh'], m['S_hh'], m['S_bh'], m['h']
                )
                pred = rnn.output(h_final)
                score = pred[0]
                grads = rnn.rtrl_compute_gradients(true_label, pred, h_final, S_xh, S_hh, S_bh)
                grads = rnn.clip_gradients(grads, MAX_NORM)
                m['opt'].step(rnn, grads)
                # Persist sensitivities and hidden state for the next window
                m['S_xh'] = S_xh
                m['S_hh'] = S_hh
                m['S_bh'] = S_bh
                m['h'] = h_final

            elif mtype == 'rtrl_adapt_legit':
                rnn = m['rnn']
                h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(window)
                pred = rnn.output(h_final)
                score = pred[0]
                # Only update on legitimate windows
                if true_label == 1:
                    grads = rnn.rtrl_compute_gradients(true_label, pred, h_final, S_xh, S_hh, S_bh)
                    grads = rnn.clip_gradients(grads, MAX_NORM)
                    m['opt'].step(rnn, grads)

            elif mtype == 'bptt_finetune_numpy':
                rnn = m['rnn']
                # Forward pass for score (before update)
                a_cache, h_cache = rnn.forward(window)
                h_final = h_cache[len(window) - 1]
                pred = rnn.output(h_final)
                score = pred[0]
                # BPTT fine-tune update
                grads = rnn.backward(true_label, pred, h_cache, a_cache, window)
                grads = rnn.clip_gradients(grads, MAX_NORM)
                m['opt'].step(rnn, grads)

            elif mtype == 'frozen_pytorch':
                score = _pytorch_infer(m['model'], window)

            elif mtype == 'bptt_finetune_pytorch':
                model = m['model']
                device = m['device']
                x_t = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
                # Single forward pass: get score BEFORE update
                model.train()
                m['optimizer'].zero_grad()
                pred_t = model(x_t)
                score = pred_t.item()
                # Backward + update
                y_t = torch.tensor([[true_label]], dtype=torch.float32, device=device)
                loss = m['criterion'](pred_t, y_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                m['optimizer'].step()

            elif mtype == 'gru_rtrl_adapt':
                rnn = m['rnn']
                h_final, sensitivities = rnn.rtrl_forward(window)
                pred = rnn.output(h_final)
                score = pred[0]
                grads = rnn.rtrl_compute_gradients(true_label, pred, h_final, sensitivities)
                grads = rnn.clip_gradients(grads, MAX_NORM)
                m['opt'].step_named(rnn, grads, m['param_names'])




            else:
                raise ValueError(f'Unknown model type: {mtype}')

            m['labels'].append(true_label)
            m['scores'].append(score)

            # Cumulative EER (all predictions up to this window)
            cum_labels = np.array(m['labels'])
            cum_scores = np.array(m['scores'])
            cum_eer = compute_eer(cum_labels, cum_scores) if len(np.unique(cum_labels)) >= 2 else None
            row[f'{model_id}_cumulative_eer'] = cum_eer


            rolling = compute_rolling_eer(m['labels'], m['scores'])
            row[f'{model_id}_score'] = score
            row[f'{model_id}_rolling_eer'] = rolling

        detail_log.append(row)

        # Progress every 50 windows
        if (w + 1) % 50 == 0:
            m2_eer = row.get('M2_rolling_eer')
            m1_eer = row.get('M1_rolling_eer')
            m2_s = f'{m2_eer:.4f}' if m2_eer is not None else 'N/A'
            m1_s = f'{m1_eer:.4f}' if m1_eer is not None else 'N/A'
            print(f'      Window {w+1}/{len(X_stream)} | M1(frozen)={m1_s} | M2(rtrl)={m2_s}')

    # ── Compute summary metrics ──────────────────────────────────────────────
    summary = {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'pair_category': classify_pair(train_condition, stream_condition),
        'seed': seed,
        'train_sessions': ','.join(str(s) for s in train_sessions),
        'stream_session': str(stream_test_session),
        'stream_legit_count': int(np.sum(y_stream == 1)),
        'stream_impostor_count': int(np.sum(y_stream == 0)),
        'total_windows': len(X_stream),
    }

    for model_id, m in active_models.items():
        labels_np = np.array(m['labels'])
        scores_np = np.array(m['scores'])

        overall_eer = compute_eer(labels_np, scores_np) if len(np.unique(labels_np)) >= 2 else None

        init_n = min(ROLLING_WINDOW, len(labels_np))
        has_both_init = len(np.unique(labels_np[:init_n])) >= 2
        initial_eer = compute_eer(labels_np[:init_n], scores_np[:init_n]) if has_both_init else None

        has_both_end = len(labels_np) >= ROLLING_WINDOW and len(np.unique(labels_np[-ROLLING_WINDOW:])) >= 2
        end_eer = compute_eer(labels_np[-ROLLING_WINDOW:], scores_np[-ROLLING_WINDOW:]) if has_both_end else None

        recovery = None
        if initial_eer is not None and end_eer is not None and initial_eer > 0:
            recovery = (initial_eer - end_eer) / initial_eer

        summary[f'{model_id}_overall_eer'] = overall_eer
        summary[f'{model_id}_initial_eer'] = initial_eer
        summary[f'{model_id}_end_eer'] = end_eer
        summary[f'{model_id}_recovery'] = recovery

        # Blueprint Section 10: Accuracy, Precision, Recall, F1, MCC, AUC-ROC
        accuracy, precision, recall, f1, mcc = evaluate(labels_np, scores_np)
        summary[f'{model_id}_accuracy'] = accuracy
        summary[f'{model_id}_precision'] = precision
        summary[f'{model_id}_recall'] = recall
        summary[f'{model_id}_f1'] = f1
        summary[f'{model_id}_mcc'] = mcc

        # AUC-ROC (requires both classes)
        if len(np.unique(labels_np)) >= 2:
            from sklearn.metrics import roc_auc_score
            summary[f'{model_id}_auc_roc'] = roc_auc_score(labels_np, scores_np)
        else:
            summary[f'{model_id}_auc_roc'] = None

    # Forgetting flag: any adapting model's end_eer > frozen end_eer + 0.05
    forgetting = False
    baseline_map = {'M2': 'M1', 'M3': 'M1', 'M4': 'M1', 'M7': 'M6', 'M8': 'M6'}

    for adapt_id, frozen_id in baseline_map.items():
        adapt_end = summary.get(f'{adapt_id}_end_eer')
        frozen_end = summary.get(f'{frozen_id}_end_eer')
        if frozen_end is not None and adapt_end is not None:
            if adapt_end > frozen_end + 0.05:
                forgetting = True
                break

    summary['forgetting_flag'] = forgetting

    return summary, detail_log, all_training_curves


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 4: Full Main Experiment')
    parser.add_argument('--users-file', type=str, default='test_users.txt',
                        help='File with user IDs to run (one per line)')
    parser.add_argument('--pairs', type=str, default='all',
                        choices=['all', 'cross-posture', 'cross-activity', 'cross-both', 'temporal'],
                        help='Which condition pairs to run (default: all)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model IDs to run (e.g. M1,M2,M5). Default: all')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (default: 42,123,456)')
    parser.add_argument('--chunk', type=int, default=None,
                        help='Chunk number (1-4) for parallel runs')
    parser.add_argument('--num-chunks', type=int, default=4,
                        help='Total number of chunks (default: 4)')
    parser.add_argument('--summary', type=str, default=None,
                        help='Output summary CSV filename')
    parser.add_argument('--detail', type=str, default=None,
                        help='Output detail CSV filename')
    args = parser.parse_args()

    # Parse models
    models_to_run = ALL_MODELS
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(',')]
        for m in models_to_run:
            if m not in ALL_MODELS:
                print(f'ERROR: Unknown model {m}. Available: {ALL_MODELS}')
                sys.exit(1)

    # Parse seeds
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
        chunk_idx = args.chunk - 1  # 0-indexed
        chunk_size = len(all_test_users) // args.num_chunks
        remainder = len(all_test_users) % args.num_chunks
        # Distribute remainder across first chunks
        start = chunk_idx * chunk_size + min(chunk_idx, remainder)
        end = start + chunk_size + (1 if chunk_idx < remainder else 0)
        users_to_run = all_test_users[start:end]
        chunk_tag = f'_chunk{args.chunk}'
        print(f'Chunk {args.chunk}/{args.num_chunks}: {len(users_to_run)} users '
              f'(indices {start}-{end-1})')
    else:
        users_to_run = all_test_users
        chunk_tag = ''

    # Output filenames
    output_summary = args.summary or f'results_stage4{chunk_tag}_summary.csv'
    output_detail = args.detail or f'results_stage4{chunk_tag}_detail.csv'
    output_curves = f'results_stage4{chunk_tag}_training_curves.csv'

    # Condition pairs
    pairs = get_pairs(args.pairs)

    print(f'Stage 4 Main Experiment')
    print(f'  Users: {len(users_to_run)} | Pairs: {len(pairs)} | Seeds: {seeds}')
    print(f'  Models: {models_to_run}')
    print(f'  Summary: {output_summary}')
    print(f'  Detail: {output_detail}')

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

    # Resume support — read completed experiment keys
    completed = set()
    if os.path.exists(output_summary):
        with open(output_summary, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('target_user'), row.get('train_condition'),
                       row.get('stream_condition'), row.get('seed'))
                completed.add(key)
        print(f'  Resuming: {len(completed)} experiments already completed')
    else:
        with open(output_summary, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()
        with open(output_detail, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=DETAIL_FIELDS).writeheader()
        with open(output_curves, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=TRAINING_CURVES_FIELDS).writeheader()

    total = len(users_to_run) * len(pairs) * len(seeds)
    count = 0
    skipped = 0
    failed = 0

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
                    summary, detail_log, training_curves = run_stage4_experiment(
                        target_user=user,
                        impostor_users=impostors,
                        train_condition=train_cond,
                        stream_condition=stream_cond,
                        seed=seed_val,
                        models_to_run=models_to_run,
                    )

                    elapsed = time.time() - start_time

                    if summary is None:
                        skipped += 1
                        print(f'    SKIPPED in {elapsed:.1f}s')
                        continue

                    # Print key results
                    m2_eer = summary.get('M2_overall_eer')
                    m1_eer = summary.get('M1_overall_eer')
                    m2_s = f'{m2_eer:.4f}' if m2_eer is not None else 'N/A'
                    m1_s = f'{m1_eer:.4f}' if m1_eer is not None else 'N/A'
                    flag = ' [FORGETTING]' if summary.get('forgetting_flag') else ''
                    print(f'    Done in {elapsed:.1f}s | M1={m1_s} M2={m2_s}{flag}')

                    # Write detail FIRST (before summary, since summary is the resume marker)
                    with open(output_detail, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
                        for row in detail_log:
                            row['target_user'] = user
                            row['train_condition'] = train_cond
                            row['stream_condition'] = stream_cond
                            row['pair_category'] = pair_cat
                            row['seed'] = seed_val
                            writer.writerow({k: row.get(k, '') for k in DETAIL_FIELDS})

                    # Write training curves
                    if training_curves:
                        with open(output_curves, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVES_FIELDS)
                            for row in training_curves:
                                writer.writerow({k: row.get(k, '') for k in TRAINING_CURVES_FIELDS})

                    # Write summary LAST (this marks the experiment as complete for resume)
                    with open(output_summary, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                        writer.writerow({k: summary.get(k, '') for k in SUMMARY_FIELDS})


                except Exception as e:
                    elapsed = time.time() - start_time
                    failed += 1
                    print(f'    FAILED in {elapsed:.1f}s: {e}')
                    with open(output_summary, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow({
                            'target_user': user,
                            'train_condition': train_cond,
                            'stream_condition': stream_cond,
                            'pair_category': classify_pair(train_cond, stream_cond),
                            'seed': seed_val,
                        })

    print(f'\n=== Stage 4 complete ===')
    print(f'  Total: {count} | Skipped: {skipped} | Failed: {failed}')
    print(f'  Summary: {output_summary}')
    print(f'  Detail: {output_detail}')
    print(f'  Training curves: {output_curves}')