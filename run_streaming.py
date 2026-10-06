'''
    RTRL ONLINE ADAPTATION EXPERIMENT

    Protocol:
        1. Train model(s) on condition A
        2. Clone into frozen and RTRL-adapting variants
        3. Stream test data one window at a time
        4. Log per-window predictions and rolling EER

    Modes:
        Full (default):      BPTT + RTRL training, 4 models, 36 condition pairs
        BPTT-only (--bptt-only): BPTT training only, 2 models, 36 condition pairs


    Usage:
        python run_streaming.py 100                        # full run, 100 users
        python run_streaming.py 100 --bptt-only            # bptt-only, 100 users
        python run_streaming.py 5 --bptt-only --temporal   # bptt-only + temporal, 5 users
'''

import numpy as np
import csv
import os
import time
import sys
import random
import argparse
from dotenv import load_dotenv
from data_loader import load_session_sensors, get_sessions_by_condition
from preprocessor import compute_magnitude, segment_into_windows, fit_normalizer, normalize, create_dataset
from model import VanillaRNN
from run_experiment import load_user_windows, compute_eer, evaluate

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# Hyperparameters
WINDOW_SIZE = 100
STEP_SIZE = 100
HIDDEN_DIM = 32
LEARNING_RATE_TRAIN = 0.01
LEARNING_RATE_ONLINE = 0.002
EPOCHS = 20
MAX_NORM = 1.0
ROLLING_WINDOW = 50
SENSORS = ['Accelerometer', 'Gyroscope', 'Magnetometer']

# All 6 HMOG conditions
CONDITIONS = [
    'READING+SITTING', 'READING+WALKING',
    'WRITING+SITTING', 'WRITING+WALKING',
    'MAP+SITTING',     'MAP+WALKING',
]

def classify_pair(train_cond, stream_cond):
    '''Classify a condition pair into a category.'''
    if train_cond == stream_cond:
        return 'temporal'
    t_act, t_pos = train_cond.split('+')
    s_act, s_pos = stream_cond.split('+')
    if t_act == s_act:
        return 'cross-posture'
    if t_pos == s_pos:
        return 'cross-activity'
    return 'cross-both'

def get_pairs(pair_filter='all'):
    '''Generate condition pairs based on filter.'''
    all_pairs = [(a, b) for a in CONDITIONS for b in CONDITIONS]
    if pair_filter == 'all':
        return all_pairs
    if pair_filter == 'cross-condition':
        return [(a, b) for a, b in all_pairs if a != b]
    return [(a, b) for a, b in all_pairs if classify_pair(a, b) == pair_filter]




def get_impostors(target_user, all_users, n=3, seed=None):
    candidates = sorted(u for u in all_users if u != target_user)
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def clone_rnn(rnn):
    new_rnn = VanillaRNN(input_dim=rnn.W_xh.shape[1], hidden_dim=rnn.W_xh.shape[0])
    new_rnn.W_xh = rnn.W_xh.copy()
    new_rnn.W_hh = rnn.W_hh.copy()
    new_rnn.W_hy = rnn.W_hy.copy()
    new_rnn.b_h = rnn.b_h.copy()
    new_rnn.b_y = rnn.b_y.copy()
    return new_rnn

def train_model_bptt(X_train, y_train, input_dim, hidden_dim=HIDDEN_DIM,
                     learning_rate=LEARNING_RATE_TRAIN, epochs=EPOCHS, max_norm=MAX_NORM):
    from run_stage1 import AdamState
    rnn = VanillaRNN(input_dim=input_dim, hidden_dim=hidden_dim)
    param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape, rnn.b_h.shape, rnn.b_y.shape]
    opt = AdamState(param_shapes, lr=learning_rate)

    
    training_curves = []



    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        perm = np.random.permutation(len(X_train))
        for i in perm:
            a_cache, h_cache = rnn.forward(X_train[i])
            h_final = h_cache[len(X_train[i]) - 1]
            y_pred = rnn.output(h_final)
            loss = rnn.compute_loss(y_train[i], y_pred)
            total_loss += loss[0]

            predicted_label = 1 if y_pred[0] > 0.5 else 0
            if predicted_label == y_train[i]:
                correct += 1

            grads = rnn.backward(y_train[i], y_pred, h_cache, a_cache, X_train[i])
            grads = rnn.clip_gradients(grads, max_norm)
            opt.step(rnn, grads)


        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        training_curves.append({'epoch': epoch + 1, 'train_loss': avg_loss, 'train_accuracy': accuracy})

        print(f'  [BPTT] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return rnn, training_curves

def train_model_rtrl(X_train, y_train, input_dim, hidden_dim=HIDDEN_DIM,
                     learning_rate=LEARNING_RATE_TRAIN, epochs=EPOCHS, max_norm=MAX_NORM):
    from run_stage1 import AdamState
    rnn = VanillaRNN(input_dim=input_dim, hidden_dim=hidden_dim)
    param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape, rnn.b_h.shape, rnn.b_y.shape]
    opt = AdamState(param_shapes, lr=learning_rate)
    training_curves = []




    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        perm = np.random.permutation(len(X_train))
        for i in perm:
            h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(X_train[i])
            y_pred = rnn.output(h_final)
            loss = rnn.compute_loss(y_train[i], y_pred)
            total_loss += loss[0]

            predicted_label = 1 if y_pred[0] > 0.5 else 0
            if predicted_label == y_train[i]:
                correct += 1

            grads = rnn.rtrl_compute_gradients(y_train[i], y_pred, h_final, S_xh, S_hh, S_bh)
            grads = rnn.clip_gradients(grads, max_norm)
            opt.step(rnn, grads)


        avg_loss = total_loss / len(X_train)
        accuracy = correct / len(X_train)
        training_curves.append({'epoch': epoch + 1, 'train_loss': avg_loss, 'train_accuracy': accuracy})
        print(f'  [RTRL] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return rnn, training_curves


def compute_rolling_eer(labels, scores, window_size=ROLLING_WINDOW):
    if len(labels) < window_size:
        return None
    recent_labels = np.array(labels[-window_size:])
    recent_scores = np.array(scores[-window_size:])
    # Blueprint requires minimum 5 per class
    if np.sum(recent_labels == 1) < 5 or np.sum(recent_labels == 0) < 5:
        return None
    return compute_eer(recent_labels, recent_scores)



def run_level3_experiment(target_user, impostor_users, train_condition, stream_condition, bptt_only=False, lr_online=LEARNING_RATE_ONLINE, hidden_dim=HIDDEN_DIM, legit_only_adapt=False):

    '''
    Run a single Level 3 experiment for one user and one condition pair.

    Args:
        bptt_only: if True, train BPTT only and stream 2 models (frozen + adapt).
                   if False, train BPTT + RTRL and stream 4 models.
    '''
    train_activity, train_posture = train_condition.split('+')
    stream_activity, stream_posture = stream_condition.split('+')

    # ---- PHASE 1: Load training data ----
    train_session_numbers = get_sessions_by_condition(target_user, train_activity, train_posture)
    if len(train_session_numbers) < 3:
        print(f'  SKIPPED: only {len(train_session_numbers)} training sessions for {train_condition}')
        return None, None, []
    train_sessions = train_session_numbers[:3]

    legit_train_windows = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)

    if len(legit_train_windows) == 0:
        print(f'  SKIPPED: no legitimate training windows for {target_user} on {train_condition}')
        return None, None, []



    impostor_train_windows = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, train_activity, train_posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train_windows.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))

    if len(impostor_train_windows) == 0:
        print(f'  SKIPPED: no impostor training data for {train_condition}')
        return None, None, []
    impostor_train_windows = np.concatenate(impostor_train_windows)

    # Balanced downsampling
    if len(impostor_train_windows) > len(legit_train_windows):
        ds_rng = np.random.RandomState(42)
        indices = ds_rng.choice(len(impostor_train_windows), size=len(legit_train_windows), replace=False)
        impostor_train_windows = impostor_train_windows[indices]


    mean, std = fit_normalizer(np.concatenate([legit_train_windows, impostor_train_windows]))
    legit_train_windows = normalize(legit_train_windows, mean, std)
    impostor_train_windows = normalize(impostor_train_windows, mean, std)

    X_train, y_train = create_dataset(legit_train_windows, impostor_train_windows)
    input_dim = X_train.shape[2]

    # ---- Train ----
    print(f'  Training on {train_condition} ({len(X_train)} windows)...')

    np.random.seed(42)
    rnn_bptt, bptt_curves = train_model_bptt(X_train, y_train, input_dim, hidden_dim=hidden_dim)

    rtrl_curves = []
    if not bptt_only:
        np.random.seed(42)
        rnn_rtrl, rtrl_curves = train_model_rtrl(X_train, y_train, input_dim, hidden_dim=hidden_dim)




    # ---- PHASE 2: Load streaming data ----
    stream_session_numbers = get_sessions_by_condition(target_user, stream_activity, stream_posture)
    if len(stream_session_numbers) < 4:
        print(f'  SKIPPED: only {len(stream_session_numbers)} sessions for {stream_condition}')
        return None, None, []
    stream_test_session = stream_session_numbers[3]
    # Temporal drift validation: train and stream sessions must not overlap
    if train_condition == stream_condition:
        overlap = set(train_sessions) & {stream_test_session}
        if overlap:
            raise ValueError(
                f'Temporal pair session overlap for {target_user} on {train_condition}: '
                f'train_sessions={train_sessions}, stream_session={stream_test_session}, overlap={overlap}'
            )


    legit_stream_windows = load_user_windows(target_user, [stream_test_session], SENSORS, WINDOW_SIZE, STEP_SIZE)

    if len(legit_stream_windows) == 0:
        print(f'  SKIPPED: no legitimate streaming windows for {target_user} on {stream_condition}')
        return None, None, []



    impostor_stream_windows = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, stream_activity, stream_posture)
        if len(sessions) >= 4:
            impostor_stream_windows.append(
                load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE)
            )

    if len(impostor_stream_windows) == 0:
        print(f'  SKIPPED: no impostor streaming data for {stream_condition}')
        return None, None, []
    impostor_stream_windows = np.concatenate(impostor_stream_windows)

    

    # Balanced downsampling of stream impostors
    if len(impostor_stream_windows) > len(legit_stream_windows):
        ds_rng = np.random.RandomState(42)
        indices = ds_rng.choice(len(impostor_stream_windows), size=len(legit_stream_windows), replace=False) # Legacy — use run_stage4.py for seeded runs
        impostor_stream_windows = impostor_stream_windows[indices]


    # Normalizing
    legit_stream_windows = normalize(legit_stream_windows, mean, std)
    impostor_stream_windows = normalize(impostor_stream_windows, mean, std)

    # Interleave legit/impostor: within-class temporal order preserved, cross-class positions randomised
    n_legit = len(legit_stream_windows)
    n_total = n_legit + len(impostor_stream_windows)
    stream_rng = np.random.RandomState(42)  # Legacy — use run_stage4.py for per-user seeded interleaving
    legit_positions = np.sort(stream_rng.choice(n_total, size=n_legit, replace=False))
    impostor_positions = np.sort(np.setdiff1d(np.arange(n_total), legit_positions))

    X_stream = np.empty_like(np.concatenate([legit_stream_windows, impostor_stream_windows]))
    y_stream = np.empty(n_total)
    X_stream[legit_positions] = legit_stream_windows
    y_stream[legit_positions] = 1.0
    X_stream[impostor_positions] = impostor_stream_windows
    y_stream[impostor_positions] = 0.0


    print(f'  Streaming {stream_condition}: {len(X_stream)} windows '
          f'({len(legit_stream_windows)} legit, {len(impostor_stream_windows)} impostor)')

    from run_stage1 import AdamState

    def _make_model(rnn_src, adapt):
        rnn = clone_rnn(rnn_src)
        opt = None
        if adapt:
            param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape, rnn.b_h.shape, rnn.b_y.shape]
            opt = AdamState(param_shapes, lr=lr_online)
        return {'rnn': rnn, 'adapt': adapt, 'opt': opt, 'labels': [], 'scores': []}

    if bptt_only:
        models = {
            'frozen': _make_model(rnn_bptt, False),
            'adapt':  _make_model(rnn_bptt, True),
        }
    else:
        models = {
            'bptt_frozen': _make_model(rnn_bptt, False),
            'bptt_adapt':  _make_model(rnn_bptt, True),
            'rtrl_frozen': _make_model(rnn_rtrl, False),
            'rtrl_adapt':  _make_model(rnn_rtrl, True),
        }

        if legit_only_adapt:
            models['rtrl_legit_only'] = _make_model(rnn_rtrl, 'legit_only')

    # ---- Stream and adapt ----
    detail_log = []

    for w in range(len(X_stream)):
        window = X_stream[w]
        true_label = y_stream[w]

        row = {
            'window_index': w,
            'true_label': true_label,
        }

        for model_name, m in models.items():
            rnn = m['rnn']

            if m['adapt']:
                h_final, S_xh, S_hh, S_bh = rnn.rtrl_forward(window)
                pred = rnn.output(h_final)

                m['labels'].append(true_label)
                m['scores'].append(pred[0])

                # Only update weights if this is full adaptation, or legit-only and label is legitimate
                if m['adapt'] == True or (m['adapt'] == 'legit_only' and true_label == 1):
                    grads = rnn.rtrl_compute_gradients(true_label, pred, h_final, S_xh, S_hh, S_bh)
                    grads = rnn.clip_gradients(grads, MAX_NORM)
                    m['opt'].step(rnn, grads)


            else:
                a_cache, h_cache = rnn.forward(window)
                h_final = h_cache[len(window) - 1]
                pred = rnn.output(h_final)

                m['labels'].append(true_label)
                m['scores'].append(pred[0])

            rolling = compute_rolling_eer(m['labels'], m['scores'])
            row[f'{model_name}_score'] = pred[0]
            row[f'{model_name}_rolling_eer'] = rolling

        detail_log.append(row)

        # Progress printing
        if (w + 1) % 20 == 0:
            if bptt_only:
                f_eer = row.get('frozen_rolling_eer')
                a_eer = row.get('adapt_rolling_eer')
                f_s = f'{f_eer:.4f}' if f_eer is not None else 'N/A'
                a_s = f'{a_eer:.4f}' if a_eer is not None else 'N/A'
                print(f'    Window {w+1}/{len(X_stream)} | Frozen: {f_s} | Adapt: {a_s}')
            else:
                bf = row.get('bptt_frozen_rolling_eer')
                ba = row.get('bptt_adapt_rolling_eer')
                rf = row.get('rtrl_frozen_rolling_eer')
                ra = row.get('rtrl_adapt_rolling_eer')
                bf_s = f'{bf:.4f}' if bf is not None else 'N/A'
                ba_s = f'{ba:.4f}' if ba is not None else 'N/A'
                rf_s = f'{rf:.4f}' if rf is not None else 'N/A'
                ra_s = f'{ra:.4f}' if ra is not None else 'N/A'
                print(f'    Window {w+1}/{len(X_stream)} | '
                      f'BPTT frozen: {bf_s} | BPTT adapt: {ba_s} | '
                      f'RTRL frozen: {rf_s} | RTRL adapt: {ra_s}')

    # ---- Summary metrics ----
    first_model = list(models.keys())[0]
    labels_np = np.array(models[first_model]['labels'])

    summary = {
        'target_user': target_user,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'pair_category': classify_pair(train_condition, stream_condition),
        'train_sessions': ','.join(str(s) for s in train_sessions),
        'stream_session': str(stream_test_session),
        'stream_legit_count': int(np.sum(y_stream == 1)),
        'stream_impostor_count': int(np.sum(y_stream == 0)),
        'total_windows': len(X_stream),

    }



    for model_name, m in models.items():
        scores_np = np.array(m['scores'])
        overall_eer = compute_eer(labels_np, scores_np)

        init_n = min(ROLLING_WINDOW, len(labels_np))
        has_both_init = len(np.unique(labels_np[:init_n])) >= 2
        initial_eer = compute_eer(labels_np[:init_n], scores_np[:init_n]) if has_both_init else None

        has_both_end = len(labels_np) >= ROLLING_WINDOW and len(np.unique(labels_np[-ROLLING_WINDOW:])) >= 2
        end_eer = compute_eer(labels_np[-ROLLING_WINDOW:], scores_np[-ROLLING_WINDOW:]) if has_both_end else None

        recovery = None
        if initial_eer is not None and end_eer is not None and initial_eer > 0:
            recovery = (initial_eer - end_eer) / initial_eer

        summary[f'{model_name}_overall_eer'] = overall_eer
        summary[f'{model_name}_initial_eer'] = initial_eer
        summary[f'{model_name}_end_eer'] = end_eer
        summary[f'{model_name}_recovery'] = recovery

    # Collect training curves for CSV output
    all_training_curves = []
    curve_base = {
        'target_user': target_user,
        'train_condition': train_condition,
        'hidden_dim': hidden_dim,
        'lr_train': LEARNING_RATE_TRAIN,
        'optimizer': 'Adam',
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'seed': 42, # Legacy hardcoded — Stage 4 passes actual experiment seed


    }
    for row in bptt_curves:
        all_training_curves.append({**curve_base, 'algorithm': 'bptt', **row})
    for row in rtrl_curves:
        all_training_curves.append({**curve_base, 'algorithm': 'rtrl', **row})

    return summary, detail_log, all_training_curves



def get_csv_fields(bptt_only, legit_only_adapt=False):
    if bptt_only:
        summary_fields = [
            'target_user', 'train_condition', 'stream_condition', 'pair_category',
            'train_sessions', 'stream_session',
            'stream_legit_count', 'stream_impostor_count', 'total_windows',
            'frozen_overall_eer', 'frozen_initial_eer', 'frozen_end_eer', 'frozen_recovery',
            'adapt_overall_eer', 'adapt_initial_eer', 'adapt_end_eer', 'adapt_recovery',
        ]
        detail_fields = [
            'target_user', 'train_condition', 'stream_condition','pair_category',
            'window_index', 'true_label',
            'frozen_score', 'frozen_rolling_eer',
            'adapt_score', 'adapt_rolling_eer',
        ]
    else:
        summary_fields = [
            'target_user', 'train_condition', 'stream_condition', 'pair_category',
            'train_sessions', 'stream_session',
            'stream_legit_count', 'stream_impostor_count', 'total_windows',

            'bptt_frozen_overall_eer', 'bptt_frozen_initial_eer', 'bptt_frozen_end_eer', 'bptt_frozen_recovery',
            'bptt_adapt_overall_eer', 'bptt_adapt_initial_eer', 'bptt_adapt_end_eer', 'bptt_adapt_recovery',
            'rtrl_frozen_overall_eer', 'rtrl_frozen_initial_eer', 'rtrl_frozen_end_eer', 'rtrl_frozen_recovery',
            'rtrl_adapt_overall_eer', 'rtrl_adapt_initial_eer', 'rtrl_adapt_end_eer', 'rtrl_adapt_recovery',
        ]
        detail_fields = [
            'target_user', 'train_condition', 'stream_condition', 'pair_category',

            'window_index', 'true_label',
            'bptt_frozen_score', 'bptt_frozen_rolling_eer',
            'bptt_adapt_score', 'bptt_adapt_rolling_eer',
            'rtrl_frozen_score', 'rtrl_frozen_rolling_eer',
            'rtrl_adapt_score', 'rtrl_adapt_rolling_eer',
        ]



    if legit_only_adapt:
        summary_fields += [
            'rtrl_legit_only_overall_eer', 'rtrl_legit_only_initial_eer',
            'rtrl_legit_only_end_eer', 'rtrl_legit_only_recovery',
        ]
        detail_fields += [
            'rtrl_legit_only_score', 'rtrl_legit_only_rolling_eer',
        ]
    return summary_fields, detail_fields




# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Level 3: RTRL Online Adaptation Experiment')
    parser.add_argument('num_users', type=int, nargs='?', default=5,
                        help='Number of users to run (default: 5)')
    parser.add_argument('--bptt-only', action='store_true',
                        help='BPTT-only mode: skip RTRL training, 2 models, cross-condition pairs only')
    parser.add_argument('--pairs', type=str, default='all',
                        choices=['all', 'cross-posture', 'cross-activity', 'cross-both', 'temporal', 'cross-condition'],
                        help='Which condition pairs to run (default: all)')

    
    parser.add_argument('--lr-online', type=float, default=0.002,
                        help='Online learning rate for RTRL adaptation (default: 0.002)')
    parser.add_argument('--hidden-dim', type=int, default=32,
                        help='Hidden dimension for RNN (default: 32)')

    parser.add_argument('--legit-only-adapt', action='store_true',
                        help='Add a legit-only adaptation model that only updates on legitimate windows')


    parser.add_argument('--summary', type=str, default=None,
                        help='Output summary CSV filename')
    parser.add_argument('--detail', type=str, default=None,
                        help='Output detail CSV filename')
    parser.add_argument('--rerun-users', type=str, default=None,
                        help='Comma-separated list of user IDs to rerun (appends to existing CSV)')
    args = parser.parse_args()

    # Set defaults based on mode
    mode_label = "bptt_only" if args.bptt_only else "full"
    pairs = get_pairs(args.pairs)


    lr_tag = f'_lr{args.lr_online}' if args.lr_online != 0.002 else ''
    hd_tag = f'_hd{args.hidden_dim}' if args.hidden_dim != 32 else ''


    output_summary = args.summary or f'results_level3_{mode_label}{lr_tag}{hd_tag}_summary.csv'
    output_detail = args.detail or f'results_level3_{mode_label}{lr_tag}{hd_tag}_detail.csv'




    print(f'Mode: {mode_label} | Users: {args.num_users} | Pairs: {len(pairs)} | '
          f'Models per experiment: {"2" if args.bptt_only else "4"}')
    print(f'Total experiments: {args.num_users * len(pairs)}')

    np.random.seed(42)
    random.seed(42)

    all_entries = os.listdir(dataset_path)
    valid_users = []
    for d in sorted(all_entries):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
        if len(session_dirs) > 0:
            valid_users.append(d)

    print(f'Found {len(valid_users)} valid users, running {args.num_users}')
    if args.rerun_users:
        users_to_run = [u for u in args.rerun_users.split(',') if u in valid_users]
        print(f'Rerun mode: {len(users_to_run)} users specified')
    else:
        users_to_run = valid_users[:args.num_users]


    summary_fields, detail_fields = get_csv_fields(args.bptt_only, args.legit_only_adapt)

    output_training_curves = f'results_training_curves{lr_tag}{hd_tag}.csv'
    training_curves_fields = [
        'target_user', 'train_condition', 'algorithm', 'hidden_dim',
        'lr_train', 'optimizer', 'window_size', 'step_size', 'seed',
        'epoch', 'train_loss', 'train_accuracy',
    ]

    if args.rerun_users and os.path.exists(output_summary):
        pass  # append to existing files
    else:
        with open(output_summary, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()
        with open(output_detail, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=detail_fields).writeheader()
        with open(output_training_curves, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=training_curves_fields).writeheader()



    total_experiments = len(users_to_run) * len(pairs)
    exp_count = 0

    for user in users_to_run:
        # deterministic hash for impostor seeding
        impostors = get_impostors(user, valid_users, n=3, seed=sum(ord(c) for c in user))

        for train_cond, stream_cond in pairs:
            exp_count += 1
            print(f'\n=== Experiment {exp_count}/{total_experiments}: User {user} | '
                  f'{train_cond} -> {stream_cond} ===')

            start_time = time.time()

            try:
                summary, detail_log , training_curves = run_level3_experiment(
                    target_user=user,
                    impostor_users=impostors,
                    train_condition=train_cond,
                    stream_condition=stream_cond,
                    bptt_only=args.bptt_only,
                    lr_online=args.lr_online,
                    hidden_dim=args.hidden_dim,
                    legit_only_adapt=args.legit_only_adapt,
                )




                if summary is None:
                    elapsed = time.time() - start_time
                    print(f'  Skipped in {elapsed:.1f}s')
                    continue

                elapsed = time.time() - start_time

                if args.bptt_only:
                    rec = f'{summary["adapt_recovery"]:.3f}' if summary["adapt_recovery"] is not None else 'N/A'
                    print(f'  Done in {elapsed:.1f}s')
                    print(f'    Frozen EER: {summary["frozen_overall_eer"]:.4f} | '
                          f'Adapt EER: {summary["adapt_overall_eer"]:.4f} | Recovery: {rec}')
                else:
                    ba_rec = f'{summary["bptt_adapt_recovery"]:.3f}' if summary["bptt_adapt_recovery"] is not None else 'N/A'
                    ra_rec = f'{summary["rtrl_adapt_recovery"]:.3f}' if summary["rtrl_adapt_recovery"] is not None else 'N/A'
                    print(f'  Done in {elapsed:.1f}s')
                    print(f'    BPTT frozen EER: {summary["bptt_frozen_overall_eer"]:.4f} | '
                          f'BPTT adapt EER: {summary["bptt_adapt_overall_eer"]:.4f} | Recovery: {ba_rec}')
                    print(f'    RTRL frozen EER: {summary["rtrl_frozen_overall_eer"]:.4f} | '
                          f'RTRL adapt EER: {summary["rtrl_adapt_overall_eer"]:.4f} | Recovery: {ra_rec}')

                with open(output_summary, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow(summary)

                with open(output_detail, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=detail_fields)
                    for row in detail_log:
                        row['target_user'] = user
                        row['train_condition'] = train_cond
                        row['stream_condition'] = stream_cond
                        row['pair_category'] = classify_pair(train_cond, stream_cond)
                        writer.writerow(row)

                if training_curves:
                    with open(output_training_curves, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=training_curves_fields)
                        for row in training_curves:
                            writer.writerow(row)

            except Exception as e:

                elapsed = time.time() - start_time
                print(f'  FAILED after {elapsed:.1f}s: {e}')
                with open(output_summary, 'a', newline='') as f:
                    csv.DictWriter(f, fieldnames=summary_fields).writerow({
                        'target_user': user,
                        'train_condition': train_cond,
                        'stream_condition': stream_cond,
                        'pair_category': classify_pair(train_cond, stream_cond),
                    })


    print(f'\n=== All done. {exp_count} experiments completed. ===')
    print(f'Summary: {output_summary}')
    print(f'Detail:  {output_detail}')
    print(f'Training curves: {output_training_curves}')
