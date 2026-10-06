"""
run_stage1.py
Stage 1 — Offline Training Hyperparameter Search
Goal: Select optimizer and training learning rate.
Grid: optimizers {Adam, SGD+momentum} x lr_train {0.0001, 0.001, 0.01} = 6 combinations
Subset: 10 validation users, 2 conditions (READING+SITTING, WRITING+WALKING)
Model: Frozen BPTT VanillaRNN only
Seeds: 3 per combination (42, 123, 456)
Early stopping: patience=3, delta=1e-3, max_epochs=30

Lock in: OPTIMIZER and LR_TRAIN permanently after Stage 1.

Prerequisite: WINDOW_SIZE and STEP_SIZE locked from Stage 0.
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
from run_streaming import get_impostors, SENSORS, HIDDEN_DIM, MAX_NORM

load_dotenv()
dataset_path = os.getenv('HMOG_ROOT')

# ── Locked from Stage 0 ──────────────────────────────────────────────────────
WINDOW_SIZE = 100    
STEP_SIZE = 100      

# ── Stage 1 search grid ──────────────────────────────────────────────────────
OPTIMIZERS = ['adam', 'sgd_momentum']
LR_TRAINS = [0.0001, 0.001, 0.01]
SEEDS = [42, 123, 456]
CONDITIONS = ['READING+SITTING', 'WRITING+WALKING']

# Early stopping
MAX_EPOCHS = 30
PATIENCE = 3
ES_DELTA = 1e-3

OUTPUT_FILE = 'results_stage1_training_hparams.csv'

FIELDS = [
    'optimizer', 'lr_train', 'seed', 'target_user', 'condition',
    'eer', 'accuracy', 'precision', 'recall', 'f1', 'mcc',
    'convergence_epoch', 'final_loss',
]



# ── Adam optimizer state ─────────────────────────────────────────────────────
class AdamState:
    """Maintains Adam optimizer state across updates."""
    def __init__(self, param_shapes, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros(s) for s in param_shapes]
        self.v = [np.zeros(s) for s in param_shapes]

    def step(self, rnn, grads):
        self.t += 1
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads
        grad_list = [dW_xh, dW_hh, dW_hy, db_h, db_y]
        param_names = ['W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y']

        for i, (name, g) in enumerate(zip(param_names, grad_list)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(rnn, name, getattr(rnn, name) - update)

    def step_named(self, model, grads_dict, param_names):
        """Adam step for models with named parameters (e.g., GRU)."""
        self.t += 1
        for i, name in enumerate(param_names):
            g = grads_dict[name]
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(model, name, getattr(model, name) - update)


# ── SGD with momentum optimizer state ────────────────────────────────────────
class SGDMomentumState:
    """Maintains SGD+momentum state across updates."""
    def __init__(self, param_shapes, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros(s) for s in param_shapes]

    def step(self, rnn, grads):
        dW_xh, dW_hh, dW_hy, db_h, db_y = grads
        grad_list = [dW_xh, dW_hh, dW_hy, db_h, db_y]
        param_names = ['W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y']

        for i, (name, g) in enumerate(zip(param_names, grad_list)):
            self.velocity[i] = self.momentum * self.velocity[i] + g
            update = self.lr * self.velocity[i]
            setattr(rnn, name, getattr(rnn, name) - update)


def create_optimizer(optimizer_name, rnn, lr):
    """Create optimizer with state for given RNN."""
    param_shapes = [
        rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape,
        rnn.b_h.shape, rnn.b_y.shape,
    ]
    if optimizer_name == 'adam':
        return AdamState(param_shapes, lr=lr)
    elif optimizer_name == 'sgd_momentum':
        return SGDMomentumState(param_shapes, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


def train_with_early_stopping(X_train, y_train, X_val, y_val, input_dim, optimizer_name, lr,
                               hidden_dim=HIDDEN_DIM, max_epochs=MAX_EPOCHS,
                               patience=PATIENCE, es_delta=ES_DELTA, max_norm=MAX_NORM):
    rnn = VanillaRNN(input_dim=input_dim, hidden_dim=hidden_dim)
    opt = create_optimizer(optimizer_name, rnn, lr)

    training_curves = []
    best_val_loss = float('inf')
    best_weights = None
    wait = 0
    convergence_epoch = max_epochs

    for epoch in range(max_epochs):
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

        avg_train_loss = total_loss / len(X_train)
        train_accuracy = correct / len(X_train)

        # Compute validation loss
        val_loss = 0
        for i in range(len(X_val)):
            a_cache, h_cache = rnn.forward(X_val[i])
            h_final = h_cache[len(X_val[i]) - 1]
            y_pred = rnn.output(h_final)
            val_loss += rnn.compute_loss(y_val[i], y_pred)[0]
        avg_val_loss = val_loss / len(X_val)

        training_curves.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
        })

        print(f'    [{optimizer_name}] Epoch {epoch + 1}/{max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Acc: {train_accuracy:.4f}')

        # Early stopping on validation loss + best model checkpoint
        if avg_val_loss < best_val_loss - es_delta:
            best_val_loss = avg_val_loss
            best_weights = {
                'W_xh': rnn.W_xh.copy(),
                'W_hh': rnn.W_hh.copy(),
                'W_hy': rnn.W_hy.copy(),
                'b_h': rnn.b_h.copy(),
                'b_y': rnn.b_y.copy(),
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                convergence_epoch = epoch + 1
                print(f'    Early stopping at epoch {convergence_epoch} (patience={patience})')
                break
    else:
        convergence_epoch = max_epochs

    # Restore best weights
    if best_weights is not None:
        for name, w in best_weights.items():
            setattr(rnn, name, w)

    final_loss = best_val_loss
    return rnn, convergence_epoch, final_loss, training_curves


def run_single(target_user, impostor_users, condition, optimizer_name, lr, seed):
    """Run one frozen BPTT experiment with given optimizer and learning rate."""
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

    # Train/val split (80/20)
    np.random.seed(seed)
    n = len(X_all)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[indices[:split]], y_all[indices[:split]]
    X_val, y_val = X_all[indices[split:]], y_all[indices[split:]]

    # Train with early stopping on validation loss
    np.random.seed(seed)
    rnn, convergence_epoch, final_loss, _ = train_with_early_stopping(
        X_train, y_train, X_val, y_val, input_dim,
        optimizer_name=optimizer_name, lr=lr,
    )


    # Test (frozen — no adaptation)
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

    # Load completed experiments for resume support
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['optimizer'], row['lr_train'], row['seed'],
                       row['target_user'], row['condition'])
                completed.add(key)

    total = len(OPTIMIZERS) * len(LR_TRAINS) * len(SEEDS) * len(val_users) * len(CONDITIONS)
    count = 0

    for optimizer_name in OPTIMIZERS:
        for lr in LR_TRAINS:
            for seed in SEEDS:
                for user in val_users:
                    impostors = get_impostors(user, all_users, n=3,
                                              seed=sum(ord(c) for c in user))
                    for cond in CONDITIONS:
                        count += 1
                        key = (optimizer_name, str(lr), str(seed), user, cond)
                        if key in completed:
                            print(f'[{count}/{total}] SKIP (already done): {optimizer_name} lr={lr} seed={seed} user={user} {cond}')
                            continue

                        print(f'[{count}/{total}] {optimizer_name} lr={lr} seed={seed} user={user} {cond}')
                        start = time.time()

                        try:
                            result = run_single(user, impostors, cond, optimizer_name, lr, seed)
                            elapsed = time.time() - start

                            if result is None:
                                print(f'  SKIPPED in {elapsed:.1f}s')
                                continue

                            row = {
                                'optimizer': optimizer_name,
                                'lr_train': lr,
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

    print(f'\nStage 1 complete. Results in {OUTPUT_FILE}')
