"""
run_stage4b_exp1.py — Drift verification with per-session 2/3 split.

Tests Advisor's hypothesis: is the EER degradation on session 4 a real
distribution-shift problem, or just an artifact of insufficient training data?

For each (user, train_condition, stream_condition, seed):
    - For each of the first 3 sessions matching train_condition:
        - Load legit windows from target_user; shuffle deterministically (seeded);
          take 2/3 for training, 1/3 for within-distribution holdout.
        - Same per-session split for impostor windows from each of 3 impostors.
        - Class strata are preserved by splitting legit and impostor INDEPENDENTLY
          within each session (matches the existing pipeline's class handling).
    - Concatenate per-session train pieces -> training set.
    - Concatenate per-session holdout pieces -> within-distribution test set.
    - Balance training and holdout 1:1 (downsample impostor to match legit).
    - Normalize using stats fit on the training set only.
    - Train a frozen VanillaRNN using the existing Stage 1 protocol
      (train_numpy_rnn imported from run_stage4b — verified identical to run_stage4).
    - Compute within_distribution_eer on the holdout.
    - Compute session4_eer on full session 4 under stream_condition.
    - drift_gap = session4_eer - within_distribution_eer.

Why per-session (and not pool-then-split): forces holdout windows to come from
the SAME sessions the model trained on, just different windows of those sessions.
This is a stricter within-distribution control than pooling sessions 1-3 then
random-sampling 2/3 (which is what run_stage4b.run_exp2 does). The advisor
specifically asked for the per-session version.

Frozen VanillaRNN only. No adaptation. No RTRL. No modifications to
run_stage4.py, run_stage4b.py, model.py, or any other existing file.

Usage:
    python run_stage4b_exp1.py \\
        --user 151985 \\
        --train-condition READING+SITTING \\
        --stream-condition WRITING+WALKING \\
        --seed 42 \\
        --output-dir results_stage4b_exp1

NAMING NOTE — read this before assuming the file name implies the experiment number:
    This file is named `run_stage4b_exp1.py`, but it implements Experiment 2
    (Per-Session Cross-Validation / Drift Verification) from the canonical
    document RTRL_Stage4B_Advisor_Experiments.md. The "_exp1" suffix is a
    session-history artifact, kept because substantial work was already
    committed under this name.

    The function `run_exp2()` inside the existing `run_stage4b.py` performs a
    SIMILAR but DIFFERENT experiment — it pools sessions 1-3 then random-splits
    2/3, whereas THIS file does a strict per-session 2/3 split. The two are
    NOT interchangeable. Use this file for the strict-per-session protocol the
    advisor asked for.

    Output CSV is `results_stage4b_cv_summary.csv` (matches doc), with 6
    canonical columns from the doc plus 8 supplementary diagnostic columns
    (the train/stream split, window counts, and impostor-drop accounting).
    See SUMMARY_FIELDS for details.

"""
import os
import sys
import csv
import argparse
import numpy as np
from dotenv import load_dotenv

from data_loader import get_sessions_by_condition
from preprocessor import fit_normalizer, normalize, create_dataset
from run_experiment import load_user_windows
from run_streaming import get_impostors, SENSORS
from run_stage1 import WINDOW_SIZE, STEP_SIZE

# Read-only imports from existing Stage 4B pipeline.
# These are NOT modified by this script.
from run_stage4b import (
    train_numpy_rnn,
    score_frozen_eer,
    load_streaming_data,
)

load_dotenv()

# Schema reconciled with RTRL_Stage4B_Advisor_Experiments.md (Experiment 2).
#
# Canonical 6 columns from the doc:
#     target_user, condition, seed, within_eer, session4_eer, drift_gap
#
# We keep `train_condition` and `stream_condition` as separate columns in
# addition to `condition`, because the script accepts cross-condition inputs
# and the doc's single `condition` column would lose that information.
# Convention: `condition = train_condition` when train == stream;
# otherwise `condition = f"{train}->{stream}"`.
#
# Supplementary diagnostic columns (n_*, impostor_pairs_*) are retained per
# the prior audit-approved patch — they are needed to verify training-set
# size and silent impostor drops.
SUMMARY_FIELDS = [
    'target_user',
    'condition',
    'train_condition', 'stream_condition',
    'seed',
    'n_train_windows', 'n_val_windows',
    'n_heldout_windows', 'n_session4_windows',
    'impostor_pairs_attempted', 'impostor_pairs_used',
    'within_eer', 'session4_eer', 'drift_gap',
]


def _list_all_users():
    """List all users with at least one session directory in HMOG_ROOT.
    Mirrors the construction in run_stage4.py:691-699 / run_stage4b.py:855-862.
    """
    dataset_path = os.getenv('HMOG_ROOT')
    if not dataset_path:
        raise RuntimeError('HMOG_ROOT environment variable is not set')
    all_users = []
    for d in sorted(os.listdir(dataset_path)):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path)
                        if s.startswith(d + '_session_')]
        if session_dirs:
            all_users.append(d)
    return all_users



def per_session_split(target_user, impostor_users, train_sessions,
                      train_activity, train_posture, seed):
    """
    Per-session 2/3 train, 1/3 holdout split (Advisor's protocol).

    Within EACH session independently:
        - Load that session's legit and impostor windows.
        - Deterministically (seeded RNG, distinct per session AND per seed)
          shuffle within the session — NO cross-session shuffling.
        - Take first 2/3 as train, last 1/3 as holdout, for legit and impostor
          INDEPENDENTLY (preserves class strata).

    Then concatenate per-session train pieces -> training set,
    and  concatenate per-session holdout pieces -> holdout set.

    Tracks per-session impostor accounting so silent drops are visible
    in both the console log and the output CSV.

    Returns a dict with arrays + accounting, or None on insufficient data.
    """
    legit_train_pieces = []
    legit_holdout_pieces = []
    impostor_train_pieces = []
    impostor_holdout_pieces = []

    impostor_pairs_attempted = 0
    impostor_pairs_used = 0

    # Offset by *10000 so RNG streams from different seeds don't overlap.
    rng_counter = int(seed) * 10000

    for session in train_sessions:
        legit_s = load_user_windows(
            target_user, [session], SENSORS, WINDOW_SIZE, STEP_SIZE,
        )
        if len(legit_s) == 0:
            print(f'    Skip session {session}: no legit windows')
            return None
        if len(legit_s) < 2:
            print(f'    Skip session {session}: only 1 legit window (cannot split)')
            return None

        legit_rng = np.random.RandomState(rng_counter); rng_counter += 1
        idx = legit_rng.permutation(len(legit_s))
        split = max(1, int(len(legit_s) * 2 / 3))
        legit_train_pieces.append(legit_s[idx[:split]])
        legit_holdout_pieces.append(legit_s[idx[split:]])

        session_index = train_sessions.index(session)
        for impostor_user in impostor_users:
            impostor_pairs_attempted += 1
            imp_sessions = get_sessions_by_condition(
                impostor_user, train_activity, train_posture,
            )
            if len(imp_sessions) <= session_index:
                print(f'    Drop impostor {impostor_user} for session#{session_index+1}: '
                      f'only has {len(imp_sessions)} sessions matching '
                      f'{train_activity}+{train_posture}')
                continue
            imp_session = imp_sessions[session_index]
            imp_s = load_user_windows(
                impostor_user, [imp_session], SENSORS, WINDOW_SIZE, STEP_SIZE,
            )
            if len(imp_s) < 2:
                print(f'    Drop impostor {impostor_user} session {imp_session}: '
                      f'only {len(imp_s)} window(s)')
                continue

            imp_rng = np.random.RandomState(rng_counter); rng_counter += 1
            iidx = imp_rng.permutation(len(imp_s))
            isplit = max(1, int(len(imp_s) * 2 / 3))
            impostor_train_pieces.append(imp_s[iidx[:isplit]])
            impostor_holdout_pieces.append(imp_s[iidx[isplit:]])
            impostor_pairs_used += 1

    if not legit_train_pieces or not impostor_train_pieces:
        return None

    legit_holdout = (np.concatenate(legit_holdout_pieces)
                     if legit_holdout_pieces else legit_train_pieces[0][:0])
    impostor_holdout = (np.concatenate(impostor_holdout_pieces)
                        if impostor_holdout_pieces else impostor_train_pieces[0][:0])

    return {
        'legit_train': np.concatenate(legit_train_pieces),
        'impostor_train': np.concatenate(impostor_train_pieces),
        'legit_holdout': legit_holdout,
        'impostor_holdout': impostor_holdout,
        'impostor_pairs_attempted': impostor_pairs_attempted,
        'impostor_pairs_used': impostor_pairs_used,
    }


def balance_one_to_one(legit, impostor, rng):
    """Downsample the larger class to match the smaller. Same as run_exp2."""
    n = min(len(legit), len(impostor))
    if n == 0:
        return legit[:0], impostor[:0]
    if len(impostor) > n:
        idx = rng.choice(len(impostor), size=n, replace=False)
        impostor = impostor[idx]
    if len(legit) > n:
        idx = rng.choice(len(legit), size=n, replace=False)
        legit = legit[idx]
    return legit, impostor


def interleave_for_eval(legit, impostor, rng):
    """Interleave legit (label 1) and impostor (label 0) into one eval array."""
    n_legit = len(legit)
    n_total = n_legit + len(impostor)
    legit_pos = np.sort(rng.choice(n_total, size=n_legit, replace=False))
    imp_pos = np.sort(np.setdiff1d(np.arange(n_total), legit_pos))

    X = np.empty((n_total, legit.shape[1], legit.shape[2]))
    y = np.empty(n_total)
    X[legit_pos] = legit; y[legit_pos] = 1.0
    X[imp_pos] = impostor; y[imp_pos] = 0.0
    return X, y


def run_one(target_user, train_condition, stream_condition, seed):
    """Execute the experiment for a single (user, train, stream, seed) tuple."""
    train_activity, train_posture = train_condition.split('+')

    # First 3 sessions matching train condition (matches existing pipeline).
    train_session_numbers = get_sessions_by_condition(
        target_user, train_activity, train_posture,
    )
    if len(train_session_numbers) < 3:
        print(f'  Skip: user {target_user} has <3 sessions for {train_condition}')
        return None
    train_sessions = train_session_numbers[:3]

    # Same impostor selection as existing pipeline.
    # Impostor selection — matches existing pipeline at run_stage4.py:725 and
    # run_stage4b.py:882. The seed is USER-derived (not the experiment seed)
    # so the same 3 impostors are chosen for a given target_user across all
    # (condition, seed) combinations. This isolates the experiment seed's
    # effect to splits/training, not impostor identity.
    all_users = _list_all_users()
    impostors_seed = sum(ord(c) for c in target_user)
    impostor_users = get_impostors(target_user, all_users,
                                   n=3, seed=impostors_seed)


    # Per-session 2/3 / 1/3 split (the experiment's central protocol).
    split_result = per_session_split(
        target_user, impostor_users, train_sessions,
        train_activity, train_posture, seed,
    )
    if split_result is None:
        print(f'  Skip: insufficient per-session data')
        return None

    legit_train = split_result['legit_train']
    impostor_train = split_result['impostor_train']
    legit_holdout = split_result['legit_holdout']
    impostor_holdout = split_result['impostor_holdout']
    impostor_pairs_attempted = split_result['impostor_pairs_attempted']
    impostor_pairs_used = split_result['impostor_pairs_used']

    print(f'  Impostor accounting: {impostor_pairs_used}/{impostor_pairs_attempted} '
          f'session-impostor pairs contributed')

    # Balance both training and holdout 1:1 (same as run_exp2's within-test logic).
    train_bal_rng = np.random.RandomState(seed + 7777)
    legit_train, impostor_train = balance_one_to_one(
        legit_train, impostor_train, train_bal_rng,
    )
    holdout_bal_rng = np.random.RandomState(seed + 8888)
    legit_holdout, impostor_holdout = balance_one_to_one(
        legit_holdout, impostor_holdout, holdout_bal_rng,
    )

    if len(legit_train) == 0 or len(legit_holdout) == 0:
        print(f'  Skip: empty train or holdout after balancing')
        return None

    # Normalize: fit on training portion only.
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train_n = normalize(legit_train, mean, std)
    impostor_train_n = normalize(impostor_train, mean, std)
    legit_holdout_n = normalize(legit_holdout, mean, std)
    impostor_holdout_n = normalize(impostor_holdout, mean, std)

    input_dim = legit_train_n.shape[2]

    # 80/20 train/val split for early stopping (same as run_exp2).
    X_all, y_all = create_dataset(
        legit_train_n, impostor_train_n, rng=np.random.RandomState(seed),
    )
    tv_rng = np.random.RandomState(seed + 1000)
    n = len(X_all)
    tv_idx = tv_rng.permutation(n)
    tv_split = int(0.8 * n)
    X_tr, y_tr = X_all[tv_idx[:tv_split]], y_all[tv_idx[:tv_split]]
    X_va, y_va = X_all[tv_idx[tv_split:]], y_all[tv_idx[tv_split:]]

    # Report the gradients-only training count (X_tr), not X_all. The val split
    # is for early stopping, not for gradient updates.
    n_train_windows = len(X_tr)
    n_val_windows = len(X_va)
    n_heldout_windows = len(legit_holdout_n) + len(impostor_holdout_n)

    print(f'  Per-session split: '
          f'legit_train={len(legit_train)} impostor_train={len(impostor_train)} '
          f'legit_holdout={len(legit_holdout)} impostor_holdout={len(impostor_holdout)}')

    # Train frozen VanillaRNN — existing Stage 1 protocol, unchanged.
    print(f'  Training VanillaRNN (BPTT, early stopping)...')
    rnn, conv_epoch, final_loss = train_numpy_rnn(
        X_tr, y_tr, X_va, y_va, input_dim, seed,
    )
    print(f'  Converged at epoch {conv_epoch}, val_loss={final_loss:.4f}')

    # ── Within-distribution evaluation ─────────────────────────────────────
    interleave_rng = np.random.RandomState(seed + 9999)
    X_within, y_within = interleave_for_eval(
        legit_holdout_n, impostor_holdout_n, interleave_rng,
    )
    within_eer = score_frozen_eer(rnn, X_within, y_within, is_pytorch=False)

    # ── Session 4 evaluation (drift) ───────────────────────────────────────
    # Cross-condition is intentional: train_condition and stream_condition
    # are independent inputs. When they differ, we test drift across both
    # the temporal axis (session 1-3 -> session 4) AND the condition axis
    # (e.g., READING+SITTING -> WRITING+WALKING). When they match, we test
    # drift across the temporal axis only.
    X_s4, y_s4 = load_streaming_data(
        target_user, impostor_users, stream_condition,
        train_sessions, train_condition, mean, std, seed,
    )
    if X_s4 is None:
        print(f'  Skip: no session 4 data for {stream_condition}')
        return None
    s4_eer = score_frozen_eer(rnn, X_s4, y_s4, is_pytorch=False)

    if within_eer is None or s4_eer is None:
        print(f'  Skip: EER undefined (single-class slice)')
        return None

    drift_gap = s4_eer - within_eer

    condition = (train_condition if train_condition == stream_condition
                 else f'{train_condition}->{stream_condition}')

    return {
        'target_user': target_user,
        'condition': condition,
        'train_condition': train_condition,
        'stream_condition': stream_condition,
        'seed': seed,
        'n_train_windows': n_train_windows,
        'n_val_windows': n_val_windows,
        'n_heldout_windows': n_heldout_windows,
        'n_session4_windows': len(X_s4),
        'impostor_pairs_attempted': impostor_pairs_attempted,
        'impostor_pairs_used': impostor_pairs_used,
        'within_eer': within_eer,
        'session4_eer': s4_eer,
        'drift_gap': drift_gap,
    }




def load_completed(csv_path):
    """Read existing CSV; return set of completed (user, train, stream, seed) keys."""
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['target_user'], row['train_condition'],
                   row['stream_condition'], int(row['seed']))
            completed.add(key)
    return completed


def append_row(csv_path, row):
    """Append a single row to the CSV, writing header if the file is new."""
    is_new = not os.path.exists(csv_path)
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--user', required=True,
                        help='Target user id (e.g. 151985)')
    parser.add_argument('--train-condition', required=True,
                        help='Train condition (e.g. READING+SITTING)')
    parser.add_argument('--stream-condition', required=True,
                        help='Stream condition (e.g. WRITING+WALKING)')
    parser.add_argument('--seed', type=int, required=True,
                        help='Seed (e.g. 42)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for stage4b_exp1_summary.csv')
    args = parser.parse_args()

    csv_path = os.path.join(args.output_dir, 'results_stage4b_cv_summary.csv')


    key = (args.user, args.train_condition, args.stream_condition, args.seed)

    completed = load_completed(csv_path)
    if key in completed:
        print(f'Already completed: {key}. Skipping.')
        return

    print(f'Running: user={args.user} train={args.train_condition} '
          f'stream={args.stream_condition} seed={args.seed}')
    row = run_one(args.user, args.train_condition,
                  args.stream_condition, args.seed)
    if row is None:
        print('Experiment skipped (insufficient data or undefined EER).')
        sys.exit(2)

    append_row(csv_path, row)
    print()
    print('=' * 70)
    print(f'  within_eer               = {row["within_eer"]:.4f}')


    print(f'  session4_eer             = {row["session4_eer"]:.4f}')
    print(f'  drift_gap                = {row["drift_gap"]:+.4f}')
    print(f'  n_train_windows          = {row["n_train_windows"]}')
    print(f'  n_val_windows            = {row["n_val_windows"]}')
    print(f'  n_heldout_windows        = {row["n_heldout_windows"]}')
    print(f'  n_session4_windows       = {row["n_session4_windows"]}')
    print(f'  impostor_pairs_used      = {row["impostor_pairs_used"]}'
          f'/{row["impostor_pairs_attempted"]}')
    print('=' * 70)
    print(f'Appended to {csv_path}')



if __name__ == '__main__':
    main()
