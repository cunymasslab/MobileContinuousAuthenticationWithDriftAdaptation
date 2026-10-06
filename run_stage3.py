"""
run_stage3.py
Stage 3 — Architecture and Hidden Size Search
Goal: Show larger frozen models cannot match small adapting models.
Grid: architectures {vanilla_rnn, lstm, gru} x hidden_dim {32, 64, 128} = 9 combinations
Subset: 10 validation users, 2 conditions (READING+SITTING, WRITING+WALKING)
Seeds: 3 per combination (42, 123, 456)
All models frozen — no RTRL, no online adaptation.

Prerequisite: WINDOW_SIZE, STEP_SIZE, OPTIMIZER, LR_TRAIN locked from Stages 0-1.
"""

import numpy as np
import csv
import os
import time
from dotenv import load_dotenv
from data_loader import get_sessions_by_condition
from preprocessor import fit_normalizer, normalize, create_dataset
from model import VanillaRNN
from run_experiment import load_user_windows, compute_eer, evaluate
from run_streaming import get_impostors, SENSORS, LEARNING_RATE_TRAIN, MAX_NORM
from run_stage1 import train_with_early_stopping, WINDOW_SIZE, STEP_SIZE, MAX_EPOCHS, PATIENCE, ES_DELTA
from pytorch_models import train_pytorch_model, evaluate_pytorch_model

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# ── Locked from Stages 0-1 ──────────────────────────────────────────────────
# WINDOW_SIZE=100, STEP_SIZE=100 (imported from run_stage1)
# OPTIMIZER=Adam, LR_TRAIN=0.01 (imported from run_streaming)

# ── Stage 3 search grid ─────────────────────────────────────────────────────
ARCHITECTURES = ['vanilla_rnn', 'lstm', 'gru']
HIDDEN_DIMS = [32, 64, 128]
SEEDS = [42, 123, 456]
CONDITIONS = ['READING+SITTING', 'WRITING+WALKING']

OUTPUT_FILE = 'results_stage3_architecture_search.csv'

FIELDS = [
    'architecture', 'hidden_dim', 'seed', 'target_user', 'condition',
    'eer', 'accuracy', 'precision', 'recall', 'f1', 'mcc',
    'convergence_epoch', 'final_loss',
]


def run_single(target_user, impostor_users, condition, arch, hidden_dim, seed):
    """Run one frozen experiment with given architecture and hidden size."""
    activity, posture = condition.split('+')

    train_sessions = get_sessions_by_condition(target_user, activity, posture)[:3]
    if len(train_sessions) < 3:
        return None

    test_session_numbers = get_sessions_by_condition(target_user, activity, posture)
    if len(test_session_numbers) < 4:
        return None
    test_session = test_session_numbers[3]

    # Load and window data
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
    np.random.seed(seed)
    if len(impostor_train) > len(legit_train):
        indices = np.random.choice(len(impostor_train), size=len(legit_train), replace=False)
        impostor_train = impostor_train[indices]

    # Normalize
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train = normalize(legit_train, mean, std)
    impostor_train = normalize(impostor_train, mean, std)

    X_all, y_all = create_dataset(legit_train, impostor_train, rng=np.random.RandomState(seed))

    input_dim = X_all.shape[2]

    # Train/val split (80/20) — same as Stage 1
    np.random.seed(seed)
    n = len(X_all)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[indices[:split]], y_all[indices[:split]]
    X_val, y_val = X_all[indices[split:]], y_all[indices[split:]]

    # Load test data
    legit_test = load_user_windows(target_user, [test_session], SENSORS, WINDOW_SIZE, STEP_SIZE)
    if len(legit_test) == 0:
        return None

    impostor_test = []
    for user in impostor_users:
        sessions = get_sessions_by_condition(user, activity, posture)
        if len(sessions) >= 4:
            impostor_test.append(load_user_windows(user, [sessions[3]], SENSORS, WINDOW_SIZE, STEP_SIZE))
    if len(impostor_test) == 0:
        return None
    impostor_test = np.concatenate(impostor_test)

    legit_test = normalize(legit_test, mean, std)
    impostor_test = normalize(impostor_test, mean, std)
    X_test, y_test = create_dataset(legit_test, impostor_test, rng=np.random.RandomState(seed))



    # ---- Train and evaluate ----
    if arch == 'vanilla_rnn':
        # Use existing NumPy BPTT training with early stopping
        np.random.seed(seed)
        rnn, convergence_epoch, final_loss, _ = train_with_early_stopping(
            X_train, y_train, X_val, y_val, input_dim,
            optimizer_name='adam', lr=LEARNING_RATE_TRAIN,
            hidden_dim=hidden_dim, max_epochs=MAX_EPOCHS,
            patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM,
        )

        # Evaluate frozen
        y_scores = []
        for i in range(len(X_test)):
            a_cache, h_cache = rnn.forward(X_test[i])
            h_final = h_cache[len(X_test[i]) - 1]
            pred = rnn.output(h_final)
            y_scores.append(pred[0])
        y_scores = np.array(y_scores)

    else:
        # PyTorch LSTM or GRU
        model, convergence_epoch, final_loss, _ = train_pytorch_model(
            X_train, y_train, X_val, y_val,
            arch=arch, input_dim=input_dim, hidden_dim=hidden_dim,
            lr=LEARNING_RATE_TRAIN, max_epochs=MAX_EPOCHS,
            patience=PATIENCE, es_delta=ES_DELTA, seed=seed,
        )

        y_scores = evaluate_pytorch_model(model, X_test, y_test)

    # Metrics
    eer = compute_eer(y_test, y_scores)
    accuracy, precision, recall, f1, mcc = evaluate(y_test, y_scores)

    return {
        'eer': eer,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'convergence_epoch': convergence_epoch,
        'final_loss': final_loss,
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

    # Resume support
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['architecture'], row['hidden_dim'], row['seed'],
                       row['target_user'], row['condition'])
                completed.add(key)

    total = len(ARCHITECTURES) * len(HIDDEN_DIMS) * len(SEEDS) * len(val_users) * len(CONDITIONS)
    count = 0

    for arch in ARCHITECTURES:
        for hd in HIDDEN_DIMS:
            for seed in SEEDS:
                for user in val_users:
                    impostors = get_impostors(user, all_users, n=3,
                                              seed=sum(ord(c) for c in user))
                    for cond in CONDITIONS:
                        count += 1
                        key = (arch, str(hd), str(seed), user, cond)
                        if key in completed:
                            print(f'[{count}/{total}] SKIP (already done): {arch} H={hd} seed={seed} user={user} {cond}')
                            continue

                        print(f'[{count}/{total}] {arch} H={hd} seed={seed} user={user} {cond}')
                        start = time.time()

                        try:
                            result = run_single(user, impostors, cond, arch, hd, seed)
                            elapsed = time.time() - start

                            if result is None:
                                print(f'  SKIPPED in {elapsed:.1f}s')
                                continue

                            row = {
                                'architecture': arch,
                                'hidden_dim': hd,
                                'seed': seed,
                                'target_user': user,
                                'condition': cond,
                                **result,
                            }

                            with open(OUTPUT_FILE, 'a', newline='') as f:
                                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

                            print(f'  EER={result["eer"]:.4f} converged@{result["convergence_epoch"]} in {elapsed:.1f}s')

                        except Exception as e:
                            elapsed = time.time() - start
                            print(f'  FAILED in {elapsed:.1f}s: {e}')

    print(f'\nStage 3 complete. Results in {OUTPUT_FILE}')
