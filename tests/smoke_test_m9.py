"""
smoke_test_m9.py

End-to-end smoke test for M9 (VanillaRNN-RTRL-Persistent).

Runs ONE Stage 4 experiment with {M1, M2, M9} on a single user, single
condition, single seed. Verifies:
  1. No crash — the full pipeline executes.
  2. No NaN/Inf/out-of-range in M9's overall EER — stability at stream length.
  3. M9 differs from M2 by more than float noise — persistence is active.

This is a sanity check before launching a 25-user comparison run.

Run:
    python tests/smoke_test_m9.py
"""

import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv('HMOG_ROOT')
if not DATASET_PATH or not os.path.isdir(DATASET_PATH):
    print(f"ERROR: HMOG_ROOT not set or not a directory: {DATASET_PATH}")
    sys.exit(1)

from run_stage4 import run_stage4_experiment
from run_streaming import get_impostors


# ── Config ──────────────────────────────────────────────────────────────────
TARGET_USER      = '151985'           # first user in test_users.txt
TRAIN_CONDITION  = 'READING+SITTING'
STREAM_CONDITION = 'WRITING+WALKING'  # cross-both drift — forces adaptation
SEED             = 42

MODELS           = ['M1', 'M2', 'M9']
EER_DIFF_MIN     = 1e-4               # M9 must differ from M2 by at least this


def get_all_users(dataset_path):
    """Enumerate users that have sessions in dataset_path."""
    users = []
    for d in sorted(os.listdir(dataset_path)):
        user_path = os.path.join(dataset_path, d)
        if not os.path.isdir(user_path):
            continue
        if any(s.startswith(d + '_session_') for s in os.listdir(user_path)):
            users.append(d)
    return users


def main():
    print("Smoke test: M9 (VanillaRNN-RTRL-Persistent)")
    print(f"  Target user:      {TARGET_USER}")
    print(f"  Train condition:  {TRAIN_CONDITION}")
    print(f"  Stream condition: {STREAM_CONDITION}")
    print(f"  Seed:             {SEED}")
    print(f"  Models:           {MODELS}")
    print("-" * 72)

    all_users = get_all_users(DATASET_PATH)
    if TARGET_USER not in all_users:
        print(f"ERROR: target user {TARGET_USER} not found in dataset at {DATASET_PATH}")
        sys.exit(1)

    impostors = get_impostors(
        TARGET_USER, all_users, n=3,
        seed=sum(ord(c) for c in TARGET_USER),
    )
    print(f"  Impostors:        {impostors}")
    print("-" * 72)
    print("Running experiment (expect ~30-120 seconds on laptop)...")

    t0 = time.time()
    summary, detail_log, _ = run_stage4_experiment(
        target_user=TARGET_USER,
        impostor_users=impostors,
        train_condition=TRAIN_CONDITION,
        stream_condition=STREAM_CONDITION,
        seed=SEED,
        models_to_run=MODELS,
    )
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    print("-" * 72)

    if summary is None:
        print("ERROR: experiment returned None (skipped due to insufficient data, or failed)")
        sys.exit(1)

    # Extract overall EERs
    m1 = summary.get('M1_overall_eer')
    m2 = summary.get('M2_overall_eer')
    m9 = summary.get('M9_overall_eer')

    def fmt(x):
        return f"{x:.6f}" if isinstance(x, (int, float)) else str(x)

    n_windows = summary.get('total_windows')
    print(f"  Total windows:            {n_windows}")
    print(f"  M1 (frozen)               overall_eer = {fmt(m1)}")
    print(f"  M2 (rtrl, buggy reset)    overall_eer = {fmt(m2)}")
    print(f"  M9 (rtrl, persistent)     overall_eer = {fmt(m9)}")
    print("-" * 72)

    failures = []

    # Check 1: M9 produced a result
    if m9 is None:
        failures.append("M9_overall_eer is None (no valid windows?)")

    # Check 2: M9 is a finite number in [0, 1]
    if isinstance(m9, (int, float)):
        if math.isnan(m9) or math.isinf(m9):
            failures.append(f"M9_overall_eer is non-finite: {m9}")
        elif m9 < 0.0 or m9 > 1.0:
            failures.append(f"M9_overall_eer out of [0,1]: {m9}")

    # Check 3: per-window score divergence between M2 and M9 (bit-level diagnostic).
    # This is far more sensitive than aggregate EER, which can be identical
    # between different models when predictions happen to cross the same decision
    # threshold the same number of times.
    m2_scores = [row.get('M2_score') for row in detail_log]
    m9_scores = [row.get('M9_score') for row in detail_log]
    paired = [(a, b) for a, b in zip(m2_scores, m9_scores)
              if a is not None and b is not None]

    n_paired = len(paired)
    n_identical = sum(1 for a, b in paired if abs(a - b) < 1e-12)
    n_different = n_paired - n_identical
    max_score_diff = max((abs(a - b) for a, b in paired), default=0.0)

    first_diverge = None
    for i, (a, b) in enumerate(paired):
        if abs(a - b) >= 1e-12:
            first_diverge = i
            break

    print(f"  Per-window score comparison (M2 vs M9):")
    print(f"    windows compared          = {n_paired}")
    print(f"    bit-identical windows     = {n_identical}")
    print(f"    differing windows         = {n_different}")
    print(f"    max |M2_score - M9_score| = {max_score_diff:.3e}")
    if first_diverge is not None:
        print(f"    first divergence at window = {first_diverge}")
    else:
        print(f"    first divergence at window = NEVER (scores identical everywhere)")

    if n_different == 0:
        failures.append(
            "M2 and M9 produce BIT-IDENTICAL scores on every window — "
            "persistence is NOT propagating; this is an integration bug"
        )

    # Aggregate EER diff (informational, not a pass/fail criterion)
    if isinstance(m9, (int, float)) and isinstance(m2, (int, float)):
        diff = abs(m9 - m2)
        print(f"  |M9_EER - M2_EER|          = {diff:.6e}  (informational)")

    print("-" * 72)


    print("-" * 72)
    if failures:
        print("SMOKE TEST: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("SMOKE TEST: PASS")


if __name__ == '__main__':
    main()
