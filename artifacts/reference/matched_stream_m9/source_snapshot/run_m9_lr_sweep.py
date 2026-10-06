"""
run_m9_lr_sweep.py
M9 learning-rate sweep on the 10 validation users.

Configurations:
    M2 @ lr=0.002 (control - original tuned baseline)
    M9 @ lr=0.0001
    M9 @ lr=0.0005
    M9 @ lr=0.001
    M9 @ lr=0.002

Per config: 10 validation users x 12 cross-both pairs x 3 seeds = 360 experiments.
Total: 5 configs x 360 = 1,800 experiments.

Implementation:
    Imports run_stage4_experiment unchanged. Before running each config,
    monkey-patches run_stage4.LR_ONLINE so the AdamState used for the M2 / M9
    online updates is constructed with the swept learning rate. All other
    determinism (impostor selection, normalization, train/val split, stream
    interleaving, BPTT-trained weights) is driven by the same per-user / per-seed
    RNG seeds inside run_stage4_experiment, so configs differ ONLY in lr_online.

Resume-safe: each config writes its own summary CSV. Re-running this script
skips any (target_user, train_condition, stream_condition, seed) keys already
present in the summary CSV for that config.

Output: results_m9_lr_sweep/
    M2_lr_0.002_summary.csv      (+ _detail.csv, _training_curves.csv)
    M9_lr_0.0001_summary.csv     (+ _detail.csv, _training_curves.csv)
    M9_lr_0.0005_summary.csv     (...)
    M9_lr_0.001_summary.csv      (...)
    M9_lr_0.002_summary.csv      (...)
    comparison.csv               (mean overall EER per config)
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from dotenv import load_dotenv

import run_stage4
from run_stage4 import (
    run_stage4_experiment,
    SUMMARY_FIELDS, DETAIL_FIELDS, TRAINING_CURVES_FIELDS,
)
from run_streaming import get_impostors, classify_pair, get_pairs


load_dotenv()
DATASET_PATH = os.getenv('HMOG_ROOT')

VALIDATION_USERS_FILE = 'validation_users.txt'
OUTPUT_DIR = 'results_m9_lr_sweep'
SEEDS = [42, 123, 456]
PAIR_FILTER = 'cross-both'

# (model_id, lr_online) - order matters: M2 control first, then M9 sweep
CONFIGS = [
    ('M2', 0.002),
    ('M9', 0.0001),
    ('M9', 0.0005),
    ('M9', 0.001),
    ('M9', 0.002),
]


def output_paths(model_id, lr):
    tag = f'{model_id}_lr_{lr:g}'
    base = os.path.join(OUTPUT_DIR, tag)
    return {
        'summary': f'{base}_summary.csv',
        'detail': f'{base}_detail.csv',
        'curves': f'{base}_training_curves.csv',
    }


def load_completed(summary_csv):
    completed = set()
    if not os.path.exists(summary_csv):
        return completed
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get('target_user'), row.get('train_condition'),
                   row.get('stream_condition'), row.get('seed'))
            completed.add(key)
    return completed


def init_csv(path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def run_config(model_id, lr, users, all_users, pairs):
    paths = output_paths(model_id, lr)
    init_csv(paths['summary'], SUMMARY_FIELDS)
    init_csv(paths['detail'], DETAIL_FIELDS)
    init_csv(paths['curves'], TRAINING_CURVES_FIELDS)

    completed = load_completed(paths['summary'])

    # Override the online learning rate used inside run_stage4_experiment.
    # AdamState is constructed inside the function, so this takes effect on
    # every subsequent call.
    run_stage4.LR_ONLINE = lr

    print(f'\n=== {model_id} @ lr_online={lr} ===')
    print(f'  Summary: {paths["summary"]}')
    print(f'  Resuming: {len(completed)} experiments already complete')

    total = len(users) * len(pairs) * len(SEEDS)
    count = 0
    skipped = 0
    failed = 0

    for user in users:
        impostors = get_impostors(user, all_users, n=3, seed=sum(ord(c) for c in user))
        for seed_val in SEEDS:
            for train_cond, stream_cond in pairs:
                count += 1
                key = (user, train_cond, stream_cond, str(seed_val))
                if key in completed:
                    continue

                pair_cat = classify_pair(train_cond, stream_cond)
                print(f'\n[{model_id} lr={lr}] [{count}/{total}] user={user} seed={seed_val} | '
                      f'{train_cond} -> {stream_cond} ({pair_cat})')

                start_time = time.time()
                try:
                    summary, detail_log, training_curves = run_stage4_experiment(
                        target_user=user,
                        impostor_users=impostors,
                        train_condition=train_cond,
                        stream_condition=stream_cond,
                        seed=seed_val,
                        models_to_run=[model_id],
                    )
                    elapsed = time.time() - start_time

                    if summary is None:
                        skipped += 1
                        print(f'    SKIPPED in {elapsed:.1f}s')
                        continue

                    eer = summary.get(f'{model_id}_overall_eer')
                    eer_s = f'{eer:.4f}' if eer is not None else 'N/A'
                    print(f'    Done in {elapsed:.1f}s | {model_id}_overall_eer={eer_s}')

                    # Write detail FIRST, then curves, then summary (resume marker last).
                    with open(paths['detail'], 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
                        for row in detail_log:
                            row['target_user'] = user
                            row['train_condition'] = train_cond
                            row['stream_condition'] = stream_cond
                            row['pair_category'] = pair_cat
                            row['seed'] = seed_val
                            writer.writerow({k: row.get(k, '') for k in DETAIL_FIELDS})

                    if training_curves:
                        with open(paths['curves'], 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=TRAINING_CURVES_FIELDS)
                            for row in training_curves:
                                writer.writerow({k: row.get(k, '') for k in TRAINING_CURVES_FIELDS})

                    with open(paths['summary'], 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                        writer.writerow({k: summary.get(k, '') for k in SUMMARY_FIELDS})

                except Exception as e:
                    elapsed = time.time() - start_time
                    failed += 1
                    print(f'    FAILED in {elapsed:.1f}s: {e}')
                    with open(paths['summary'], 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writerow({
                            'target_user': user,
                            'train_condition': train_cond,
                            'stream_condition': stream_cond,
                            'pair_category': pair_cat,
                            'seed': seed_val,
                        })

    print(f'\n  {model_id} @ lr={lr} done | total={count} skipped={skipped} failed={failed}')


def write_comparison(configs, valid_users):
    """Build comparison.csv: mean / std of <model>_overall_eer per config."""
    rows = []
    for model_id, lr in configs:
        path = output_paths(model_id, lr)['summary']
        if not os.path.exists(path):
            print(f'  WARNING: {path} not found, skipping in comparison')
            continue
        eers = []
        n_total_rows = 0
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get('target_user') not in valid_users:
                    continue
                n_total_rows += 1
                v = r.get(f'{model_id}_overall_eer', '')
                if v in (None, ''):
                    continue
                try:
                    eers.append(float(v))
                except ValueError:
                    pass
        rows.append({
            'model': model_id,
            'lr_online': lr,
            'n_rows': n_total_rows,
            'n_with_eer': len(eers),
            'mean_overall_eer': float(np.mean(eers)) if eers else None,
            'std_overall_eer': float(np.std(eers)) if eers else None,
            'min_overall_eer': float(np.min(eers)) if eers else None,
            'max_overall_eer': float(np.max(eers)) if eers else None,
        })

    out_path = os.path.join(OUTPUT_DIR, 'comparison.csv')
    fields = ['model', 'lr_online', 'n_rows', 'n_with_eer',
              'mean_overall_eer', 'std_overall_eer',
              'min_overall_eer', 'max_overall_eer']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'\nComparison CSV: {out_path}')
    print(f'  {"model":<6} {"lr":>8} {"n":>5} {"mean_eer":>10} {"std":>8}')
    for r in rows:
        mean_s = f'{r["mean_overall_eer"]:.4f}' if r['mean_overall_eer'] is not None else 'N/A'
        std_s = f'{r["std_overall_eer"]:.4f}' if r['std_overall_eer'] is not None else 'N/A'
        print(f'  {r["model"]:<6} {r["lr_online"]:>8g} {r["n_with_eer"]:>5} {mean_s:>10} {std_s:>8}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M9 learning-rate sweep on validation users')
    parser.add_argument('--summary-only', action='store_true',
                        help='Skip running; only build comparison.csv from existing per-config summaries')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated subset to run (e.g. "M9:0.001,M9:0.002"). Default: all 5')
    args = parser.parse_args()

    if not DATASET_PATH:
        print('ERROR: HMOG_ROOT environment variable not set.')
        sys.exit(1)
    if not os.path.isdir(DATASET_PATH):
        print(f'ERROR: HMOG_ROOT={DATASET_PATH} is not a directory.')
        sys.exit(1)

    if not os.path.exists(VALIDATION_USERS_FILE):
        print(f'ERROR: {VALIDATION_USERS_FILE} not found.')
        sys.exit(1)
    with open(VALIDATION_USERS_FILE) as f:
        users = sorted([line.strip() for line in f if line.strip()])
    print(f'Validation users ({len(users)}): {users}')

    # Build all_users from HMOG_ROOT (same logic as run_stage4.py)
    all_users = []
    for d in sorted(os.listdir(DATASET_PATH)):
        user_path = os.path.join(DATASET_PATH, d)
        if not os.path.isdir(user_path):
            continue
        session_dirs = [s for s in os.listdir(user_path) if s.startswith(d + '_session_')]
        if len(session_dirs) > 0:
            all_users.append(d)
    print(f'Total users in dataset: {len(all_users)}')

    pairs = get_pairs(PAIR_FILTER)
    print(f'Cross-both pairs: {len(pairs)}')
    print(f'Seeds: {SEEDS}')
    print(f'Per-config experiments: {len(users) * len(pairs) * len(SEEDS)}')

    configs_to_run = CONFIGS
    if args.configs:
        configs_to_run = []
        for c in args.configs.split(','):
            m, lr_str = c.split(':')
            configs_to_run.append((m.strip(), float(lr_str)))
        print(f'Running subset: {configs_to_run}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.summary_only:
        for model_id, lr in configs_to_run:
            run_config(model_id, lr, users, all_users, pairs)

    # Always rebuild comparison from whatever summaries exist
    write_comparison(CONFIGS, set(users))

    print('\n=== Sweep script finished ===')