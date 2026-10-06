"""
run_stage0.py
Stage 0 — Preprocessing Search
Goal: Select window_size and step_size.
Grid: window_size {100, 200, 400} x step_size {100, 200} = 6 combinations
Subset: 10 validation users, 2 conditions (READING+SITTING, WRITING+WALKING)
Model: Frozen BPTT VanillaRNN only
Seeds: 3 per combination (42, 123, 456)
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
from run_experiment import load_user_windows, compute_eer, evaluate
from run_streaming import get_impostors, train_model_bptt, SENSORS, HIDDEN_DIM, LEARNING_RATE_TRAIN, EPOCHS, MAX_NORM

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# Stage 0 grid
WINDOW_SIZES = [100, 200, 400]
STEP_SIZES = [100, 200]
SEEDS = [42, 123, 456]
CONDITIONS = ['READING+SITTING', 'WRITING+WALKING']
OUTPUT_FILE = 'results_stage0_preprocessing.csv'

FIELDS = [
    'window_size', 'step_size', 'seed', 'target_user', 'condition',
    'eer', 'accuracy', 'precision', 'recall', 'f1',
]


def run_single(target_user, impostor_users, condition, window_size, step_size, seed):
    """Run one frozen BPTT experiment with given preprocessing params."""
    activity, posture = condition.split('+')

    train_sessions = get_sessions_by_condition(target_user, activity, posture)[:3]
    if len(train_sessions) < 3:
        return None

    test_session_numbers = get_sessions_by_condition(target_user, activity, posture)
    if len(test_session_numbers) < 4:
        return None
    test_session = test_session_numbers[3]

    # Load and window data
    legit_train = load_user_windows(target_user, train_sessions, SENSORS, window_size, step_size)
    if len(legit_train) == 0:
        return None

    impostor_train = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)[:3]
        if len(sessions) == 0:
            continue
        impostor_train.append(load_user_windows(user, sessions, SENSORS, window_size, step_size))
    if len(impostor_train) == 0:
        return None
    impostor_train = np.concatenate(impostor_train)

    # Balanced downsampling
    np.random.seed(seed)
    if len(impostor_train) > len(legit_train):
        indices = np.random.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[indices]

    # Normalize
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)

    X_train, y_train = create_dataset(legit_train, impostor_train)
    input_dim = X_train.shape[2]

    # Train
    np.random.seed(seed)
    rnn, _ = train_model_bptt(X_train, y_train, input_dim, hidden_dim=HIDDEN_DIM,
                               learning_rate=LEARNING_RATE_TRAIN, epochs=EPOCHS, max_norm=MAX_NORM)

    # Test (frozen — no adaptation)
    legit_test = load_user_windows(target_user, [test_session], SENSORS, window_size, step_size)
    if len(legit_test) == 0:
        return None

    impostor_test = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)
        if len(sessions) >= 4:
            impostor_test.append(load_user_windows(user, [sessions[3]], SENSORS, window_size, step_size))
    if len(impostor_test) == 0:
        return None
    impostor_test = np.concatenate(impostor_test)

    legit_test = normalize(legit_test, mean, std)
    impostor_test = normalize(impostor_test, mean, std)
    X_test, y_test = create_dataset(legit_test, impostor_test)

    # Evaluate
    y_scores = []
    for i in range(len(X_test)):
        a_cache, h_cache = rnn.forward(X_test[i])
        h_final = h_cache[len(X_test[i]) - 1]
        pred = rnn.output(h_final)
        y_scores.append(pred[0])
    y_scores = np.array(y_scores)

    eer = compute_eer(y_test, y_scores)
    accuracy, precision, recall, f1, mcc = evaluate(y_test, y_scores)

    return {
        'eer': eer,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
                key = (row['window_size'], row['step_size'], row['seed'],
                       row['target_user'], row['condition'])
                completed.add(key)

    total = len(WINDOW_SIZES) * len(STEP_SIZES) * len(SEEDS) * len(val_users) * len(CONDITIONS)
    count = 0

    for ws in WINDOW_SIZES:
        for ss in STEP_SIZES:
            for seed in SEEDS:
                for user in val_users:
                    impostors = get_impostors(user, all_users, n=3,
                                              seed=sum(ord(c) for c in user))
                    for cond in CONDITIONS:
                        count += 1
                        key = (str(ws), str(ss), str(seed), user, cond)
                        if key in completed:
                            print(f'[{count}/{total}] SKIP (already done): ws={ws} ss={ss} seed={seed} user={user} {cond}')
                            continue

                        print(f'[{count}/{total}] ws={ws} ss={ss} seed={seed} user={user} {cond}')
                        start = time.time()

                        try:
                            result = run_single(user, impostors, cond, ws, ss, seed)
                            elapsed = time.time() - start

                            if result is None:
                                print(f'  SKIPPED in {elapsed:.1f}s')
                                continue

                            row = {
                                'window_size': ws,
                                'step_size': ss,
                                'seed': seed,
                                'target_user': user,
                                'condition': cond,
                                **result,
                            }

                            with open(OUTPUT_FILE, 'a', newline='') as f:
                                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

                            print(f'  EER={result["eer"]:.4f} in {elapsed:.1f}s')

                        except Exception as e:
                            elapsed = time.time() - start
                            print(f'  FAILED in {elapsed:.1f}s: {e}')

    print(f'\nStage 0 complete. Results in {OUTPUT_FILE}')
