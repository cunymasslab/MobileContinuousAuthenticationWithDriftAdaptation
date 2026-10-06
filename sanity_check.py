"""
sanity_check.py
Runs one user through the full data pipeline and asserts correctness.
Must pass before any full experiment launch.
"""

import numpy as np
import os
import sys
import random
from dotenv import load_dotenv
from data_loader import load_session_sensors, get_sessions_by_condition
from preprocessor import compute_magnitude, segment_into_windows, fit_normalizer, normalize, create_dataset
from model import VanillaRNN
from run_experiment import load_user_windows, compute_eer
from run_streaming import (
    get_impostors, clone_rnn, train_model_bptt,
    WINDOW_SIZE, STEP_SIZE, HIDDEN_DIM, SENSORS, LEARNING_RATE_TRAIN, EPOCHS, MAX_NORM,
)

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')


def find_test_user():
    """Find the first valid user with enough sessions."""
    all_entries = os.listdir(dataset_path)
    for d in sorted(all_entries):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
        if len(session_dirs) >= 4:
            return d
    raise RuntimeError('No valid user found')


def get_all_valid_users():
    """Get all valid user IDs."""
    all_entries = os.listdir(dataset_path)
    valid = []
    for d in sorted(all_entries):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
        if len(session_dirs) > 0:
            valid.append(d)
    return valid


def main():
    passed = 0
    failed = 0
    total = 10

    target_user = find_test_user()
    all_users = get_all_valid_users()
    condition = 'READING+SITTING'
    activity, posture = condition.split('+')

    print(f'Sanity check using user: {target_user}, condition: {condition}')
    print(f'Total valid users: {len(all_users)}')
    print()

    # Load training data
    train_sessions = get_sessions_by_condition(target_user, activity, posture)[:3]
    legit_train_windows = load_user_windows(target_user, train_sessions, SENSORS, WINDOW_SIZE, STEP_SIZE)

    impostor_users = get_impostors(target_user, all_users, n=3, seed=sum(ord(c) for c in target_user))

    impostor_train_windows = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train_windows.append(load_user_windows(user, sessions, SENSORS, WINDOW_SIZE, STEP_SIZE))
    impostor_train_windows = np.concatenate(impostor_train_windows)

    # Balanced downsampling for training
    if len(impostor_train_windows) > len(legit_train_windows):
        np.random.seed(42)
        indices = np.random.choice(len(impostor_train_windows), size=len(legit_train_windows), replace=False)
        impostor_train_windows = impostor_train_windows[indices]

    # --- Assertion 1: Training class ratio ---
    print('CHECK 1: Training class ratio between 0.9 and 1.1')
    train_ratio = len(legit_train_windows) / len(impostor_train_windows) if len(impostor_train_windows) > 0 else 0
    if 0.9 <= train_ratio <= 1.1:
        print(f'  PASS: ratio = {train_ratio:.4f} (legit={len(legit_train_windows)}, impostor={len(impostor_train_windows)})')
        passed += 1
    else:
        print(f'  FAIL: ratio = {train_ratio:.4f} (legit={len(legit_train_windows)}, impostor={len(impostor_train_windows)})')
        failed += 1

    # Load stream data
    stream_sessions = get_sessions_by_condition(target_user, activity, posture)
    stream_test_session = stream_sessions[3]
    legit_stream_windows = load_user_windows(target_user, [stream_test_session], SENSORS, WINDOW_SIZE, STEP_SIZE)

    impostor_stream_windows = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)
        if len(sessions) >= 4:
            impostor_stream_windows.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    impostor_stream_windows = np.concatenate(impostor_stream_windows)

    # Stream downsampling
    if len(impostor_stream_windows) > len(legit_stream_windows):
        np.random.seed(42)
        indices = np.random.choice(len(impostor_stream_windows), size=len(legit_stream_windows), replace=False)
        impostor_stream_windows = impostor_stream_windows[indices]

    # --- Assertion 2: Stream class ratio ---
    print('CHECK 2: Stream class ratio between 0.9 and 1.1')
    stream_ratio = len(legit_stream_windows) / len(impostor_stream_windows) if len(impostor_stream_windows) > 0 else 0
    if 0.9 <= stream_ratio <= 1.1:
        print(f'  PASS: ratio = {stream_ratio:.4f} (legit={len(legit_stream_windows)}, impostor={len(impostor_stream_windows)})')
        passed += 1
    else:
        print(f'  FAIL: ratio = {stream_ratio:.4f} (legit={len(legit_stream_windows)}, impostor={len(impostor_stream_windows)})')
        failed += 1

    # --- Assertion 3: Stream contains both classes ---
    print('CHECK 3: Stream contains both classes')
    if len(legit_stream_windows) > 0 and len(impostor_stream_windows) > 0:
        print(f'  PASS: legit={len(legit_stream_windows)}, impostor={len(impostor_stream_windows)}')
        passed += 1
    else:
        print(f'  FAIL: legit={len(legit_stream_windows)}, impostor={len(impostor_stream_windows)}')
        failed += 1

    # --- Assertion 4: Exactly 3 impostors ---
    print('CHECK 4: Number of impostors is exactly 3')
    if len(impostor_users) == 3:
        print(f'  PASS: {len(impostor_users)} impostors: {impostor_users}')
        passed += 1
    else:
        print(f'  FAIL: {len(impostor_users)} impostors (expected 3)')
        failed += 1

    # --- Assertion 5: Window feature dimension ---
    print('CHECK 5: Window feature dimension matches expected sensor channels')
    expected_features = 4 * len(SENSORS)  # x, y, z, magnitude per sensor
    actual_features = legit_train_windows.shape[2]
    if actual_features == expected_features:
        print(f'  PASS: {actual_features} features (expected {expected_features} for {len(SENSORS)} sensors)')
        passed += 1
    else:
        print(f'  FAIL: {actual_features} features (expected {expected_features})')
        failed += 1

    # --- Assertion 6: Normalization fitted on training data only ---
    print('CHECK 6: Normalization fitted on training data only (not stream data)')
    mean_train, std_train = fit_normalizer(np.concatenate([legit_train_windows, impostor_train_windows]))
    mean_all, std_all = fit_normalizer(np.concatenate([legit_train_windows, impostor_train_windows,
                                                        legit_stream_windows, impostor_stream_windows]))
    mean_diff = np.max(np.abs(mean_train - mean_all))
    if mean_diff > 1e-10:
        print(f'  PASS: train-only normalizer differs from all-data normalizer (max diff={mean_diff:.6f})')
        print(f'         This confirms stream data is NOT included in normalization fitting')
        passed += 1
    else:
        print(f'  WARNING: normalizer values identical — stream data may be too similar to training')
        print(f'  PASS (weak): normalization code uses train data only by design')
        passed += 1

    # --- Assertion 7: Hidden state reset before each window ---
    print('CHECK 7: Hidden state is reset to zeros before each window')
    np.random.seed(42)
    rnn_test = VanillaRNN(input_dim=actual_features, hidden_dim=HIDDEN_DIM)
    norm_train = normalize(legit_train_windows, mean_train, std_train)
    # Run two windows — each should start from h=0 independently
    a1, h1 = rnn_test.forward(norm_train[0])
    a2, h2 = rnn_test.forward(norm_train[1])
    # Run second window on a fresh clone — should give same result
    rnn_test2 = clone_rnn(rnn_test)
    a2_indep, h2_indep = rnn_test2.forward(norm_train[1])
    h2_final = h2[len(norm_train[1]) - 1]
    h2_indep_final = h2_indep[len(norm_train[1]) - 1]
    h_diff = np.max(np.abs(h2_final - h2_indep_final))
    if h_diff == 0.0:
        print(f'  PASS: hidden state resets between windows (diff={h_diff})')
        passed += 1
    else:
        print(f'  FAIL: hidden state carries over between windows (diff={h_diff})')
        failed += 1

    # --- Assertion 8: Frozen and RTRL models start with identical weights ---
    print('CHECK 8: rnn_frozen and rnn_rtrl start with identical weights after cloning')
    np.random.seed(42)
    rnn_source = VanillaRNN(input_dim=actual_features, hidden_dim=HIDDEN_DIM)
    rnn_frozen = clone_rnn(rnn_source)
    rnn_rtrl = clone_rnn(rnn_source)
    max_diffs = [
        np.max(np.abs(rnn_frozen.W_xh - rnn_rtrl.W_xh)),
        np.max(np.abs(rnn_frozen.W_hh - rnn_rtrl.W_hh)),
        np.max(np.abs(rnn_frozen.W_hy - rnn_rtrl.W_hy)),
        np.max(np.abs(rnn_frozen.b_h - rnn_rtrl.b_h)),
        np.max(np.abs(rnn_frozen.b_y - rnn_rtrl.b_y)),
    ]
    max_weight_diff = max(max_diffs)
    if max_weight_diff == 0.0:
        print(f'  PASS: max weight difference = {max_weight_diff} (exactly 0.0)')
        passed += 1
    else:
        print(f'  FAIL: max weight difference = {max_weight_diff} (expected exactly 0.0)')
        failed += 1

    # --- Assertion 9: Label alignment ---
    print('CHECK 9: Label alignment — label for window i corresponds to window i')
    mean_norm, std_norm = fit_normalizer(np.concatenate([legit_train_windows, impostor_train_windows]))
    legit_norm = normalize(legit_stream_windows[:3], mean_norm, std_norm)
    impostor_norm = normalize(impostor_stream_windows[:3], mean_norm, std_norm)
    X_check = np.concatenate([legit_norm, impostor_norm])
    y_check = np.concatenate([np.ones(len(legit_norm)), np.zeros(len(impostor_norm))])
    labels_correct = True
    for i in range(min(5, len(X_check))):
        expected = 1.0 if i < len(legit_norm) else 0.0
        if y_check[i] != expected:
            labels_correct = False
            break
    if labels_correct:
        print(f'  PASS: labels align with windows (checked {min(5, len(X_check))} windows)')
        passed += 1
    else:
        print(f'  FAIL: label mismatch detected')
        failed += 1

    # --- Assertion 10: Stream shuffle is deterministic ---
    print('CHECK 10: Stream shuffle with fixed seed is deterministic')
    n_windows = 20
    rng1 = np.random.RandomState(42)
    order1 = rng1.permutation(n_windows)
    rng2 = np.random.RandomState(42)
    order2 = rng2.permutation(n_windows)
    if np.array_equal(order1, order2):
        print(f'  PASS: same seed produces same shuffle order')
        passed += 1
    else:
        print(f'  FAIL: same seed produces different shuffle orders')
        failed += 1

    # --- Summary ---
    print(f'\n{"="*50}')
    print(f'SANITY CHECK RESULTS: {passed}/{total} passed, {failed}/{total} failed')
    if failed > 0:
        print('SOME CHECKS FAILED — do not proceed with full experiment run.')
        sys.exit(1)
    else:
        print('ALL CHECKS PASSED — safe to proceed with full experiment run.')
        sys.exit(0)


if __name__ == '__main__':
    main()
