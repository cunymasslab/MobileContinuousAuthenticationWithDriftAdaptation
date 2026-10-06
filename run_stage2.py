"""
run_stage2.py
Stage 2 — Online Adaptation Hyperparameter Search
Goal: Select online learning rate for RTRL streaming adaptation.
Grid: LR_online {0.0001, 0.0005, 0.001, 0.002} = 4 values
Subset: 10 validation users, 2 conditions (READING+SITTING, WRITING+WALKING)
Models: Frozen BPTT baseline + RTRL adapting (temporal pairs only)
Seeds: 3 per combination (42, 123, 456)

Decision: Lowest mean end-of-stream EER across all users, conditions, and seeds.
Lock in: LR_ONLINE permanently after Stage 2.

Prerequisite: WINDOW_SIZE, STEP_SIZE, OPTIMIZER, LR_TRAIN locked from Stages 0-1.
"""

import numpy as np
import csv
import os
import time
import random
from dotenv import load_dotenv
from data_loader import get_sessions_by_condition
from preprocessor import fit_normalizer, normalize, create_dataset
from model import VanillaRNN
from run_experiment import load_user_windows, compute_eer
from run_streaming import (
    get_impostors, train_model_bptt, clone_rnn, compute_rolling_eer,
    SENSORS, HIDDEN_DIM, LEARNING_RATE_TRAIN, EPOCHS, MAX_NORM,
    WINDOW_SIZE, STEP_SIZE, ROLLING_WINDOW,
)

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# ── Locked from Stages 0-1 ──────────────────────────────────────────────────
# WINDOW_SIZE, STEP_SIZE, LEARNING_RATE_TRAIN, OPTIMIZER (Adam) imported above

# ── Stage 2 search grid ─────────────────────────────────────────────────────
LR_ONLINES = [0.0001, 0.0005, 0.001, 0.002]
SEEDS = [42, 123, 456]
CONDITIONS = ['READING+SITTING', 'WRITING+WALKING']

OUTPUT_FILE = 'results_stage2_online_hparams.csv'

FIELDS = [
    'lr_online', 'seed', 'target_user', 'condition',
    'frozen_overall_eer', 'frozen_initial_eer', 'frozen_end_eer',
    'adapt_overall_eer', 'adapt_initial_eer', 'adapt_end_eer', 'adapt_recovery',
    'total_windows', 'stream_legit_count', 'stream_impostor_count',
]


def _train_and_prepare_stream(target_user, impostor_users, condition, seed):
    """Train BPTT model and prepare stream data. Returns reusable state dict or None."""
    activity, posture = condition.split('+')

    # ---- Training data (sessions 1-3) ----
    train_session_numbers = get_sessions_by_condition(target_user, activity, posture)
    if len(train_session_numbers) < 3:
        return None
    train_sessions = train_session_numbers[:3]

    legit_train = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_train) == 0:
        return None

    impostor_train = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_train) == 0:
        return None
    impostor_train = np.concatenate(impostor_train)

    # Balanced downsampling
    if len(impostor_train) > len(legit_train):
        ds_rng = np.random.RandomState(seed)
        indices = ds_rng.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[indices]

    # Normalize (fit on training data)
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)

    X_train, y_train = create_dataset(legit_train, impostor_train, rng=np.random.RandomState(seed))


    input_dim = X_train.shape[2]

    # ---- Train BPTT model ----
    # Train/val split (80/20) — matches Stage 1 and Stage 4 protocol
    split_rng = np.random.RandomState(seed)
    n = len(X_train)
    split_indices = split_rng.permutation(n)
    split_pt = int(0.8 * n)
    X_tr, y_tr = X_train[split_indices[:split_pt]], y_train[split_indices[:split_pt]]
    X_vl, y_vl = X_train[split_indices[split_pt:]], y_train[split_indices[split_pt:]]

    from run_stage1 import train_with_early_stopping, MAX_EPOCHS, PATIENCE, ES_DELTA
    np.random.seed(seed)
    rnn_bptt, _, _, _ = train_with_early_stopping(
        X_tr, y_tr, X_vl, y_vl, input_dim,
        optimizer_name='adam', lr=LEARNING_RATE_TRAIN,
        hidden_dim=HIDDEN_DIM, max_epochs=MAX_EPOCHS,
        patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
    )


    # ---- Streaming data (session 4, temporal pair) ----
    stream_session_numbers = get_sessions_by_condition(target_user, activity, posture)
    if len(stream_session_numbers) < 4:
        return None
    stream_session = stream_session_numbers[3]

    # Temporal overlap guard
    overlap = set(train_sessions) & {stream_session}
    if overlap:
        raise ValueError(f'Temporal session overlap: train={train_sessions}, stream={stream_session}')

    legit_stream = load_user_windows(target_user, [stream_session], SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_stream) == 0:
        return None

    impostor_stream = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)
        if len(sessions) >= 4:
            impostor_stream.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_stream) == 0:
        return None
    impostor_stream = np.concatenate(impostor_stream)

    # Balanced downsampling of stream impostors
    if len(impostor_stream) > len(legit_stream):
        ds_rng = np.random.RandomState(seed)
        indices = ds_rng.choice(len(impostor_stream), size=len(legit_stream), replace=False)
        impostor_stream = impostor_stream[indices]

    # Normalize stream with training stats
    legit_stream = normalize(legit_stream, mean, std)
    impostor_stream = normalize(impostor_stream, mean, std)

    # Interleave: within-class temporal order preserved
    n_legit = len(legit_stream)
    n_total = n_legit + len(impostor_stream)
    
    user_seed = sum(ord(c) for c in target_user) + seed
    stream_rng = np.random.RandomState(user_seed + 1)

    legit_positions = np.sort(stream_rng.choice(n_total, size=n_legit, replace=False))
    impostor_positions = np.sort(np.setdiff1d(np.arange(n_total), legit_positions))

    X_stream = np.empty_like(np.concatenate([legit_stream, impostor_stream]))
    y_stream = np.empty(n_total)
    X_stream[legit_positions] = legit_stream
    y_stream[legit_positions] = 1.0
    X_stream[impostor_positions] = impostor_stream
    y_stream[impostor_positions] = 0.0

    return {
        'rnn_bptt': rnn_bptt,
        'X_stream': X_stream,
        'y_stream': y_stream,
    }


def _stream_with_lr(trained, lr_online):
    """Run streaming with a specific LR_online. Clones the model fresh each time."""
    from run_stage1 import AdamState

    rnn_bptt = trained['rnn_bptt']
    X_stream = trained['X_stream']
    y_stream = trained['y_stream']

    rnn_frozen = clone_rnn(rnn_bptt)
    rnn_adapt = clone_rnn(rnn_bptt)

    param_shapes = [rnn_adapt.W_xh.shape, rnn_adapt.W_hh.shape, rnn_adapt.W_hy.shape,
                    rnn_adapt.b_h.shape, rnn_adapt.b_y.shape]
    opt = AdamState(param_shapes, lr=lr_online)

    frozen_scores = []
    adapt_scores = []

    for w in range(len(X_stream)):
        window = X_stream[w]
        true_label = y_stream[w]

        # Frozen model
        a_cache, h_cache = rnn_frozen.forward(window)
        h_final = h_cache[len(window) - 1]
        pred_frozen = rnn_frozen.output(h_final)
        frozen_scores.append(pred_frozen[0])

        # Adapting model (RTRL)
        h_final, S_xh, S_hh, S_bh = rnn_adapt.rtrl_forward(window)
        pred_adapt = rnn_adapt.output(h_final)
        adapt_scores.append(pred_adapt[0])

        # Update weights
        grads = rnn_adapt.rtrl_compute_gradients(true_label, pred_adapt, h_final, S_xh, S_hh, S_bh)
        grads = rnn_adapt.clip_gradients(grads, MAX_NORM)
        opt.step(rnn_adapt, grads)

    # ---- Compute metrics ----
    labels_np = y_stream
    frozen_scores_np = np.array(frozen_scores)
    adapt_scores_np = np.array(adapt_scores)

    def compute_eer_safe(labels, scores):
        if len(np.unique(labels)) < 2:
            return None
        return compute_eer(labels, scores)

    frozen_overall = compute_eer_safe(labels_np, frozen_scores_np)
    adapt_overall = compute_eer_safe(labels_np, adapt_scores_np)

    init_n = min(ROLLING_WINDOW, len(labels_np))
    has_both_init = len(np.unique(labels_np[:init_n])) >= 2
    frozen_initial = compute_eer(labels_np[:init_n], frozen_scores_np[:init_n]) if has_both_init else None
    adapt_initial = compute_eer(labels_np[:init_n], adapt_scores_np[:init_n]) if has_both_init else None

    has_both_end = len(labels_np) >= ROLLING_WINDOW and len(np.unique(labels_np[-ROLLING_WINDOW:])) >= 2
    frozen_end = compute_eer(labels_np[-ROLLING_WINDOW:], frozen_scores_np[-ROLLING_WINDOW:]) if has_both_end else None
    adapt_end = compute_eer(labels_np[-ROLLING_WINDOW:], adapt_scores_np[-ROLLING_WINDOW:]) if has_both_end else None

    recovery = None
    if adapt_initial is not None and adapt_end is not None and adapt_initial > 0:
        recovery = (adapt_initial - adapt_end) / adapt_initial

    return {
        'frozen_overall_eer': frozen_overall,
        'frozen_initial_eer': frozen_initial,
        'frozen_end_eer': frozen_end,
        'adapt_overall_eer': adapt_overall,
        'adapt_initial_eer': adapt_initial,
        'adapt_end_eer': adapt_end,
        'adapt_recovery': recovery,
        'total_windows': len(X_stream),
        'stream_legit_count': int(np.sum(y_stream == 1)),
        'stream_impostor_count': int(np.sum(y_stream == 0)),
    }


if __name__ == '__main__':
    # Load validation users
    if not os.path.exists('validation_users.txt'):
        print('ERROR: validation_users.txt not found. Run select_validation_users.py first.')
        exit(1)

    with open('validation_users.txt') as f:
        val_users = [line.strip() for line in f if line.strip()]
    print(f'Validation users: {len(val_users)}')

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

    # Write CSV header
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # Load completed experiments for resume support
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['lr_online'], row['seed'], row['target_user'], row['condition'])
                completed.add(key)

    total = len(SEEDS) * len(val_users) * len(CONDITIONS) * len(LR_ONLINES)
    count = 0

    for seed in SEEDS:
        for user in val_users:
            impostors = get_impostors(user, all_users, n=3,
                                      seed=sum(ord(c) for c in user))
            for cond in CONDITIONS:
                # Check which LR values still need running
                lrs_to_run = []
                for lr_online in LR_ONLINES:
                    key = (str(lr_online), str(seed), user, cond)
                    if key not in completed:
                        lrs_to_run.append(lr_online)

                if len(lrs_to_run) == 0:
                    count += len(LR_ONLINES)
                    print(f'  SKIP all LRs for seed={seed} user={user} {cond} (already done)')
                    continue

                # Train once for this (seed, user, condition)
                print(f'\n  Training: seed={seed} user={user} {cond}')
                train_start = time.time()

                try:
                    trained = _train_and_prepare_stream(user, impostors, cond, seed)
                except Exception as e:
                    train_elapsed = time.time() - train_start
                    print(f'  FAILED training in {train_elapsed:.1f}s: {e}')
                    count += len(LR_ONLINES)
                    continue

                if trained is None:
                    train_elapsed = time.time() - train_start
                    print(f'  SKIPPED training in {train_elapsed:.1f}s')
                    count += len(LR_ONLINES)
                    continue

                train_elapsed = time.time() - train_start
                print(f'  Trained in {train_elapsed:.1f}s')

                # Stream once per LR value
                for lr_online in LR_ONLINES:
                    count += 1
                    key = (str(lr_online), str(seed), user, cond)
                    if key in completed:
                        print(f'[{count}/{total}] SKIP (already done): lr={lr_online} seed={seed} user={user} {cond}')
                        continue

                    print(f'[{count}/{total}] lr={lr_online} seed={seed} user={user} {cond}')
                    stream_start = time.time()

                    try:
                        result = _stream_with_lr(trained, lr_online)
                        elapsed = time.time() - stream_start

                        row = {
                            'lr_online': lr_online,
                            'seed': seed,
                            'target_user': user,
                            'condition': cond,
                            **result,
                        }

                        with open(OUTPUT_FILE, 'a', newline='') as f:
                            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

                        adapt_end = result['adapt_end_eer']
                        adapt_end_s = f'{adapt_end:.4f}' if adapt_end is not None else 'N/A'
                        rec = result['adapt_recovery']
                        rec_s = f'{rec:.3f}' if rec is not None else 'N/A'
                        print(f'  End EER={adapt_end_s} Recovery={rec_s} in {elapsed:.1f}s')

                    except Exception as e:
                        elapsed = time.time() - stream_start
                        print(f'  FAILED in {elapsed:.1f}s: {e}')


    print(f'\nStage 2 complete. Results in {OUTPUT_FILE}')
