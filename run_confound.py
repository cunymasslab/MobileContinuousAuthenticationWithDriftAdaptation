"""
run_confound.py — Confounding-variable test: same #windows, drift vs no-drift.

Goal: rule out the 'more data' confound for adaptation benefit.
For each (user, train_cond, stream_cond, seed):
    - Per-session 2/3 train, 1/3 held-out split on sessions 1-3 of train_cond
      (reused from run_stage4b_exp1.per_session_split — strict per-session,
      no cross-session leakage).
    - Train a frozen VanillaRNN via BPTT (Stage 1 protocol) on the 2/3 portion.
    - Build TWO interleaved 1:1 evaluation streams of EQUAL length:
        no-drift stream  = held-out windows from sessions 1-3
        drift stream     = session 4 windows from stream_cond
      Truncate to min(len_nodrift, len_drift).
    - Run M1 (frozen) and M4 (BPTT fine-tune, lr=0.002) on each stream.
    - Compute drift magnitude (KS, classifier accuracy, MMD) between the
      raw sessions 1-3 and session 4 legit-user feature distributions.

Resume-safe: skip experiments whose key already exists in the summary CSV.
Chunkable: --chunk K --num-chunks N partitions users for parallel runs.
"""

import os
import csv
import time
import argparse
import numpy as np
from dotenv import load_dotenv

from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist

from data_loader import get_sessions_by_condition
from preprocessor import fit_normalizer, normalize, create_dataset
from run_experiment import load_user_windows, compute_eer
from run_streaming import (
    get_impostors, get_pairs, classify_pair, clone_rnn,
    SENSORS, MAX_NORM,
)
from run_stage1 import AdamState, WINDOW_SIZE, STEP_SIZE
from run_stage4b import train_numpy_rnn, load_streaming_data
from run_stage4b_exp1 import (
    per_session_split, balance_one_to_one, interleave_for_eval,
)

load_dotenv()

LR_ONLINE = 0.002
SEEDS = [42, 123, 456]
EXCLUDE_USERS = {
    '207969',  # directory-naming mismatch
    '733162',  # 1-row sensor CSVs in WRITING and MAP conditions
}

OUTPUT_DIR = 'results_confound'
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'results_confound_summary.csv')
FAILED_CSV  = os.path.join(OUTPUT_DIR, 'results_confound_failed.csv')

ROLLING_WINDOW = 50 

SUMMARY_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'pair_category', 'seed',
    'holdout_mode',
    'n_windows_used',
    # No-drift stream
    'M1_nodrift_eer',  'M1_nodrift_initial_eer',  'M1_nodrift_end_eer',
    'M4_nodrift_eer',  'M4_nodrift_initial_eer',  'M4_nodrift_end_eer',
    'adaptation_benefit_nodrift',
    'adaptation_benefit_nodrift_ci_lo', 'adaptation_benefit_nodrift_ci_hi',
    # Drift stream
    'M1_drift_eer',    'M1_drift_initial_eer',    'M1_drift_end_eer',
    'M4_drift_eer',    'M4_drift_initial_eer',    'M4_drift_end_eer',
    'adaptation_benefit_drift',
    'adaptation_benefit_drift_ci_lo',   'adaptation_benefit_drift_ci_hi',
    # Drift magnitude
    'ks_mean', 'ks_max', 'n_significant_ks', 'n_significant_ks_bonf',
    'classifier_accuracy', 'mmd_statistic',
]

FAILED_FIELDS = [
    'target_user', 'train_condition', 'stream_condition', 'seed', 'reason',
]

def windows_to_features(windows):
    """(N, T, F) -> (N, F) per-window feature means. Same convention as
    run_drift_detection.py."""
    return np.mean(windows, axis=1)


def mmd_rbf(X, Y, seed=0):
    """Squared MMD with RBF kernel, median-heuristic bandwidth (Gretton 2012).

    Returns sqrt(max(MMD^2, 0)) so the result is on the same scale as a distance.
    Subsamples to <=500 per side for tractability."""
    rng = np.random.RandomState(seed)
    if len(X) > 500:
        X = X[rng.choice(len(X), 500, replace=False)]
    if len(Y) > 500:
        Y = Y[rng.choice(len(Y), 500, replace=False)]

    pooled = np.vstack([X, Y])
    if len(pooled) > 1000:
        pooled = pooled[rng.choice(len(pooled), 1000, replace=False)]
    median_dist = float(np.median(pdist(pooled)))
    if median_dist <= 0:
        return 0.0
    gamma = 1.0 / (2.0 * median_dist ** 2)

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    mmd2 = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2.0 * K_XY.sum() / (m * n)
    return float(np.sqrt(max(mmd2, 0.0)))


def compute_drift_metrics(s13_windows, s4_windows, seed):
    """KS / classifier-CV / MMD between the no-drift and drift legit pools.

    Inputs are the post-balance, post-normalize legit windows that feed the
    no-drift and drift evaluation streams (`nodrift_legit` and `drift_legit`
    in run_one). Both arrays are length n_per_class with shape (N, T, F);
    they are normalized with the SAME training-set mean/std so the metric
    captures the gap that the model actually observes. Target-user only —
    impostor distributions are not measured."""
    s13_feat = windows_to_features(s13_windows)
    s4_feat  = windows_to_features(s4_windows)

    # KS per feature
    stats, pvals = [], []
    for j in range(s13_feat.shape[1]):
        s, p = ks_2samp(s13_feat[:, j], s4_feat[:, j])
        stats.append(s); pvals.append(p)
    ks_mean = float(np.mean(stats))
    ks_max  = float(np.max(stats))
    n_sig   = int(sum(1 for p in pvals if p < 0.05))
    n_sig_bonf = int(sum(1 for p in pvals if p < 0.05 / 12))

    # Classifier-two-sample test
    X = np.vstack([s13_feat, s4_feat])
    y = np.array([0] * len(s13_feat) + [1] * len(s4_feat))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf_acc = float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring='accuracy')))

    mmd = mmd_rbf(s13_feat, s4_feat, seed=seed)

    return {
        'ks_mean': ks_mean, 'ks_max': ks_max,
        'n_significant_ks': n_sig,
        'n_significant_ks_bonf': n_sig_bonf,
        'classifier_accuracy': clf_acc, 'mmd_statistic': mmd,
    }
def _stream_summary(labels, scores):
    """Return (overall, initial, end) EER triple. Values are None if a slice
    is single-class or shorter than ROLLING_WINDOW."""
    n = len(labels)
    if len(np.unique(labels)) < 2:
        return None, None, None
    overall = compute_eer(labels, scores)
    init_n = min(ROLLING_WINDOW, n)
    initial = (compute_eer(labels[:init_n], scores[:init_n])
               if len(np.unique(labels[:init_n])) >= 2 else None)
    end = (compute_eer(labels[-ROLLING_WINDOW:], scores[-ROLLING_WINDOW:])
           if n >= ROLLING_WINDOW
              and len(np.unique(labels[-ROLLING_WINDOW:])) >= 2
           else None)
    return overall, initial, end

def bootstrap_benefit_ci(labels, scores_m1, scores_m4, n_boot=2000, seed=42):

    """Paired bootstrap 95% CI for EER(M1) - EER(M4) on the same stream.
    Resamples (label, m1_score, m4_score) tuples with replacement so the
    per-resample EER difference accounts for label/score correlation."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    diffs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        sub_l = labels[idx]
        if len(np.unique(sub_l)) < 2:
            continue
        e1 = compute_eer(sub_l, scores_m1[idx])
        e4 = compute_eer(sub_l, scores_m4[idx])
        diffs[i] = e1 - e4
    valid = diffs[~np.isnan(diffs)]
    if len(valid) < 2:
        return None, None
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def run_frozen_stream(rnn, X_stream, y_stream):
    """M1: forward only, no updates. Returns (overall, initial, end, scores)."""
    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        _, h_cache = rnn.forward(X_stream[w])
        h_final = h_cache[len(X_stream[w]) - 1]
        scores[w] = rnn.output(h_final)[0]
    overall, initial, end = _stream_summary(y_stream, scores)
    return overall, initial, end, scores


def run_m4_stream(rnn_init, X_stream, y_stream, lr=LR_ONLINE, max_norm=MAX_NORM):
    """M4: per-window BPTT fine-tune with Adam. Mutates a CLONE; original untouched.
    Hidden state is reset per window (forward() re-initializes h to zero).
    Returns overall EER or None."""
    rnn = clone_rnn(rnn_init)
    param_shapes = [rnn.W_xh.shape, rnn.W_hh.shape, rnn.W_hy.shape,
                    rnn.b_h.shape, rnn.b_y.shape]
    opt = AdamState(param_shapes, lr=lr)

    scores = np.empty(len(X_stream))
    for w in range(len(X_stream)):
        a_cache, h_cache = rnn.forward(X_stream[w])
        h_final = h_cache[len(X_stream[w]) - 1]
        pred = rnn.output(h_final)
        scores[w] = pred[0]
        grads = rnn.backward(y_stream[w], pred, h_cache, a_cache, X_stream[w])
        grads = rnn.clip_gradients(grads, max_norm)
        opt.step(rnn, grads)

    overall, initial, end = _stream_summary(y_stream, scores)
    return overall, initial, end, scores


def per_session_split_contiguous(target_user, impostor_users, train_sessions,
                                  train_activity, train_posture, seed):
    """Contiguous-block per-session split: first 2/3 train, last 1/3 holdout
    of each session, with NO within-session shuffle.

    Mirrors per_session_split's return shape. The seed argument is accepted
    for interface compatibility but not consumed (the split is deterministic).
    Eliminates the temporal autocorrelation between train and holdout windows
    that the random within-session shuffle introduces."""
    legit_train_pieces = []
    legit_holdout_pieces = []
    impostor_train_pieces = []
    impostor_holdout_pieces = []
    impostor_pairs_attempted = 0
    impostor_pairs_used = 0

    for session in train_sessions:
        legit_s = load_user_windows(target_user, [session], SENSORS,
                                     WINDOW_SIZE, STEP_SIZE)
        if len(legit_s) < 2:
            return None
        sp = max(1, int(len(legit_s) * 2 / 3))
        legit_train_pieces.append(legit_s[:sp])
        legit_holdout_pieces.append(legit_s[sp:])

        session_index = train_sessions.index(session)
        for impostor_user in impostor_users:
            impostor_pairs_attempted += 1
            imp_sessions = get_sessions_by_condition(
                impostor_user, train_activity, train_posture)
            if len(imp_sessions) <= session_index:
                continue
            imp_session = imp_sessions[session_index]
            imp_s = load_user_windows(impostor_user, [imp_session], SENSORS,
                                       WINDOW_SIZE, STEP_SIZE)
            if len(imp_s) < 2:
                continue
            isp = max(1, int(len(imp_s) * 2 / 3))
            impostor_train_pieces.append(imp_s[:isp])
            impostor_holdout_pieces.append(imp_s[isp:])
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


def _list_all_users():
    """Same scan as run_stage4.py / run_stage4b_exp1._list_all_users."""
    root = os.getenv('HMOG_ROOT')
    if not root:
        raise RuntimeError('HMOG_ROOT environment variable is not set')
    users = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if any(s.startswith(d + '_session_') for s in os.listdir(p)):
            users.append(d)
    return users


def run_one(target_user, train_cond, stream_cond, seed, all_users,
            holdout_mode='contiguous'):
    """One (user, pair, seed) experiment. Returns summary dict, or
    ('fail', reason) on data error, or None on benign skip.

    holdout_mode:
        'contiguous' — first 2/3 train, last 1/3 holdout per session (default;
                       eliminates train↔holdout temporal autocorrelation that
                       would bias the no-drift baseline downward via M1
                       saturation on near-IID windows).
        'shuffle'    — within-session random 2/3 train, 1/3 holdout (matches
                       per_session_split convention from exp1; sensitivity
                       check only)."""
    train_act, train_pos = train_cond.split('+')

    train_session_numbers = get_sessions_by_condition(target_user, train_act, train_pos)
    if len(train_session_numbers) < 3:
        return ('fail', 'too_few_train_sessions')
    

    train_sessions = train_session_numbers[:3]

    # Impostors — same selection as Stage 4 (user-derived seed)
    impostors = get_impostors(target_user, all_users, n=3,
                              seed=sum(ord(c) for c in target_user))

    # ── Per-session 2/3 / 1/3 split ─────────────────────────────────────
    split_fn = (per_session_split_contiguous
                if holdout_mode == 'contiguous'
                else per_session_split)
    split = split_fn(target_user, impostors, train_sessions,
                     train_act, train_pos, seed)
    if split is None:
        return ('fail', f'{split_fn.__name__} returned None (insufficient data)')

    legit_train     = split['legit_train']
    impostor_train  = split['impostor_train']
    legit_holdout   = split['legit_holdout']
    impostor_holdout = split['impostor_holdout']

    # Balance both train and holdout 1:1 (same convention as run_stage4b_exp1)
    legit_train, impostor_train = balance_one_to_one(
        legit_train, impostor_train, np.random.RandomState(seed + 7777))
    legit_holdout, impostor_holdout = balance_one_to_one(
        legit_holdout, impostor_holdout, np.random.RandomState(seed + 8888))
    
    if len(legit_train) == 0 or len(legit_holdout) == 0:
        return ('fail', 'empty_train_or_holdout_after_balance')

    # Normalize (fit on training portion only)
    mean, std = fit_normalizer(np.concatenate([legit_train, impostor_train]))
    legit_train_n     = normalize(legit_train, mean, std)
    impostor_train_n  = normalize(impostor_train, mean, std)
    legit_holdout_n   = normalize(legit_holdout, mean, std)
    impostor_holdout_n = normalize(impostor_holdout, mean, std)

    # 80/20 train/val for early stopping (same as Stage 4 / exp1)
    X_all, y_all = create_dataset(legit_train_n, impostor_train_n,
                                  rng=np.random.RandomState(seed))
    tv_rng = np.random.RandomState(seed + 1000)
    perm = tv_rng.permutation(len(X_all))
    cut = int(0.8 * len(X_all))
    X_tr, y_tr = X_all[perm[:cut]], y_all[perm[:cut]]
    X_va, y_va = X_all[perm[cut:]], y_all[perm[cut:]]
    input_dim = X_tr.shape[2]

    # ── Train BPTT VanillaRNN (Stage 1 protocol — same call as exp1) ────
    rnn_bptt, _, _ = train_numpy_rnn(X_tr, y_tr, X_va, y_va, input_dim, seed)

    # ── Build raw 1:1 pools for both streams BEFORE interleaving ────────
    # No-drift pool: held-out windows from sessions 1-3 (already balanced
    # 1:1 and normalized above).
    nodrift_legit = legit_holdout_n
    nodrift_imp   = impostor_holdout_n

    # Drift pool: load_streaming_data balances + interleaves internally;
    # split it back out by class so we can re-balance to the common count.
    X_drift_full, y_drift_full = load_streaming_data(
        target_user, impostors, stream_cond,
        train_sessions, train_cond, mean, std, seed)
    if X_drift_full is None:
        return ('fail', 'load_streaming_data_returned_none')
    
    drift_legit = X_drift_full[y_drift_full == 1.0]
    drift_imp   = X_drift_full[y_drift_full == 0.0]

    # Common per-class count: floor of min legit and min impostor across streams
    n_per_class = int(min(len(nodrift_legit), len(drift_legit),
                          len(nodrift_imp),   len(drift_imp)))
    N_MIN_PER_CLASS = 10   # 20 windows total per stream — minimum for stable EER
    if n_per_class < N_MIN_PER_CLASS:
        return ('fail', f'n_per_class={n_per_class} < {N_MIN_PER_CLASS}')

    def _take(arr, n, rng):
        if len(arr) == n:
            return arr
        idx = rng.choice(len(arr), size=n, replace=False)
        return arr[idx]

    nodrift_legit = _take(nodrift_legit, n_per_class, np.random.RandomState(seed + 11))
    nodrift_imp   = _take(nodrift_imp,   n_per_class, np.random.RandomState(seed + 12))
    drift_legit   = _take(drift_legit,   n_per_class, np.random.RandomState(seed + 13))
    drift_imp     = _take(drift_imp,     n_per_class, np.random.RandomState(seed + 14))

    X_nodrift, y_nodrift = interleave_for_eval(
        nodrift_legit, nodrift_imp, np.random.RandomState(seed + 9999))
    X_drift, y_drift = interleave_for_eval(
        drift_legit, drift_imp, np.random.RandomState(seed + 10000))

    n_use = len(X_nodrift)   # == len(X_drift) == 2 * n_per_class by construction
    if len(np.unique(y_nodrift)) < 2 or len(np.unique(y_drift)) < 2:
        return ('fail', 'single-class slice after balancing')

    # ── Drift metrics on the SAME stream-level samples M1/M4 evaluate on ─
    # nodrift_legit and drift_legit are the post-balance, post-normalize
    # legit windows that feed the no-drift and drift streams respectively.
    # Computing drift on these makes the §2 correlations in analyze_confound.py
    # comparable: drift magnitude and adaptation benefit reference the same data.
    drift = compute_drift_metrics(nodrift_legit, drift_legit, seed=42)

    # ── Run M1 frozen + M4 fine-tune on each stream (fresh clone every call) ─
    m1_nd_o, m1_nd_i, m1_nd_e, m1_nd_s = run_frozen_stream(clone_rnn(rnn_bptt), X_nodrift, y_nodrift)
    m4_nd_o, m4_nd_i, m4_nd_e, m4_nd_s = run_m4_stream    (rnn_bptt, X_nodrift, y_nodrift)
    m1_dr_o, m1_dr_i, m1_dr_e, m1_dr_s = run_frozen_stream(clone_rnn(rnn_bptt), X_drift,   y_drift)
    m4_dr_o, m4_dr_i, m4_dr_e, m4_dr_s = run_m4_stream    (rnn_bptt, X_drift,   y_drift)

    if None in (m1_nd_o, m4_nd_o, m1_dr_o, m4_dr_o):
        return ('fail', 'overall EER undefined on at least one stream')

    # Per-row paired bootstrap CIs on the adaptation benefit
    nd_lo, nd_hi = bootstrap_benefit_ci(y_nodrift, m1_nd_s, m4_nd_s, seed=seed)
    dr_lo, dr_hi = bootstrap_benefit_ci(y_drift,   m1_dr_s, m4_dr_s, seed=seed)

    return {
        'target_user': target_user,
        'train_condition': train_cond,
        'stream_condition': stream_cond,
        'pair_category': classify_pair(train_cond, stream_cond),
        'seed': seed,
        'holdout_mode': holdout_mode,
        'n_windows_used': n_use,
        'M1_nodrift_eer': m1_nd_o, 'M1_nodrift_initial_eer': m1_nd_i, 'M1_nodrift_end_eer': m1_nd_e,
        'M4_nodrift_eer': m4_nd_o, 'M4_nodrift_initial_eer': m4_nd_i, 'M4_nodrift_end_eer': m4_nd_e,
        'adaptation_benefit_nodrift':       m1_nd_o - m4_nd_o,
        'adaptation_benefit_nodrift_ci_lo': nd_lo,
        'adaptation_benefit_nodrift_ci_hi': nd_hi,
        'M1_drift_eer':   m1_dr_o, 'M1_drift_initial_eer':   m1_dr_i, 'M1_drift_end_eer':   m1_dr_e,
        'M4_drift_eer':   m4_dr_o, 'M4_drift_initial_eer':   m4_dr_i, 'M4_drift_end_eer':   m4_dr_e,
        'adaptation_benefit_drift':         m1_dr_o - m4_dr_o,
        'adaptation_benefit_drift_ci_lo':   dr_lo,
        'adaptation_benefit_drift_ci_hi':   dr_hi,
        **drift,
    }


def _load_completed(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            try:
                done.add((row['target_user'], row['train_condition'],
                          row['stream_condition'], int(row['seed'])))
            except (ValueError, KeyError, TypeError):
                continue   # skip malformed/partial rows from a killed worker
    return done

def _ensure_csv(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append(path, fields, row):
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writerow(
            {k: row.get(k, '') for k in fields})


def _chunk_users(users, chunk, num_chunks):
    """Same chunk math as run_stage4.py:661-668."""
    if chunk is None:
        return users, ''
    idx = chunk - 1
    base = len(users) // num_chunks
    rem  = len(users) %  num_chunks
    start = idx * base + min(idx, rem)
    end   = start + base + (1 if idx < rem else 0)
    return users[start:end], f'_chunk{chunk}'


def main():
    ap = argparse.ArgumentParser(description='Confound experiment')
    ap.add_argument('--users-file', default='test_users.txt')
    ap.add_argument('--seeds', default=','.join(str(s) for s in SEEDS))
    ap.add_argument('--pairs', default='all',
                    choices=['all', 'temporal', 'cross-posture',
                             'cross-activity', 'cross-both'])
    ap.add_argument('--chunk', type=int, default=None)
    ap.add_argument('--num-chunks', type=int, default=12)
    ap.add_argument('--holdout-mode', choices=['shuffle', 'contiguous'],
                    default='contiguous',
                    help='Within-session split mode for the no-drift baseline. '
                         'Default is contiguous (last 1/3 in time), which removes '
                         'the train↔holdout autocorrelation that biases the no-drift '
                         'baseline downward. Use --holdout-mode shuffle for the '
                         'within-session-shuffle sensitivity check.')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    pairs = get_pairs(args.pairs)

    with open(args.users_file) as f:
        users = sorted(u.strip() for u in f if u.strip())
    users = [u for u in users if u not in EXCLUDE_USERS]
    users, tag = _chunk_users(users, args.chunk, args.num_chunks)

    mode_tag = '' if args.holdout_mode == 'contiguous' else f'_{args.holdout_mode}'

    summary_path = SUMMARY_CSV.replace('.csv', f'{mode_tag}{tag}.csv')
    failed_path  = FAILED_CSV.replace('.csv',  f'{mode_tag}{tag}.csv')


    _ensure_csv(summary_path, SUMMARY_FIELDS)
    _ensure_csv(failed_path,  FAILED_FIELDS)
    completed = _load_completed(summary_path)

    all_users = _list_all_users()
    total = len(users) * len(pairs) * len(seeds)
    n_done = n_skip = n_fail = 0

    print(f'[confound] users={len(users)} pairs={len(pairs)} seeds={seeds} '
          f'chunk={args.chunk}/{args.num_chunks} -> {summary_path}')

    for user in users:
        for seed in seeds:
            for tc, sc in pairs:
                key = (user, tc, sc, seed)
                if key in completed:
                    continue
                t0 = time.time()
                try:
                    out = run_one(user, tc, sc, seed, all_users,
                                  holdout_mode=args.holdout_mode)
                except Exception as e:
                    n_fail += 1
                    _append(failed_path, FAILED_FIELDS, {
                        'target_user': user, 'train_condition': tc,
                        'stream_condition': sc, 'seed': seed,
                        'reason': f'{type(e).__name__}: {e}',
                    })
                    print(f'  FAIL {user} {tc}->{sc} s={seed} '
                          f'in {time.time()-t0:.1f}s: {e}')
                    continue

                if out is None:
                    n_skip += 1
                    continue
                if isinstance(out, tuple) and out[0] == 'fail':
                    n_fail += 1
                    _append(failed_path, FAILED_FIELDS, {
                        'target_user': user, 'train_condition': tc,
                        'stream_condition': sc, 'seed': seed, 'reason': out[1],
                    })
                    continue

                _append(summary_path, SUMMARY_FIELDS, out)
                n_done += 1
                print(f'  [{n_done}] {user} {tc}->{sc} s={seed} '
                      f'N={out["n_windows_used"]} '
                      f'benefit_nd={out["adaptation_benefit_nodrift"]:+.4f} '
                      f'benefit_dr={out["adaptation_benefit_drift"]:+.4f} '
                      f'mmd={out["mmd_statistic"]:.3f} '
                      f'({time.time()-t0:.1f}s)')

    print(f'\n[done] total={total} done={n_done} skipped={n_skip} failed={n_fail}')
    print(f'  summary: {summary_path}')
    print(f'  failed:  {failed_path}')


if __name__ == '__main__':
    main()