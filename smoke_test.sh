#!/usr/bin/env bash
# Smoke test for run_confound.py before the multi-day run.
# Exits non-zero on any failure.
set -uo pipefail
cd "$(dirname "$0")"

OUT=results_confound/results_confound_summary.csv
FAIL=results_confound/results_confound_failed.csv
[ -f "$OUT" ] && rm -f "$OUT"
[ -f "$FAIL" ] && rm -f "$FAIL"

echo "[smoke] running confound on smoke_users.txt..."
python run_confound.py --users-file smoke_users.txt \
    --seeds 42 --pairs all --num-chunks 1 || exit 10

echo "[smoke] verifying CSV..."
python - <<'PY'
import sys, pandas as pd

df = pd.read_csv('results_confound/results_confound_summary.csv')
errors = []

# Assertion 1: rows present
if len(df) == 0:
    errors.append('no rows produced')

# Assertion 2: every row has at least 20 windows
bad_n = (df.n_windows_used < 20).sum()
if bad_n > 0:
    errors.append(f'{bad_n} rows with n_windows_used < 20')

# Assertion 3: no-drift benefit should be small in magnitude
m_benefit_nd = df.adaptation_benefit_nodrift.mean()
if abs(m_benefit_nd) >= 0.15:
    errors.append(f'mean adaptation_benefit_nodrift={m_benefit_nd:+.4f} '
                  f'is not small (|x| >= 0.15) — sanity check failed')

# Assertion 4: M1_drift_eer in [0.0, 0.6]
m1d = df.M1_drift_eer.dropna()
out_of_range = ((m1d < 0.0) | (m1d > 0.6)).sum()
if out_of_range > 0:
    errors.append(f'{out_of_range} rows with M1_drift_eer outside [0.0, 0.6]')

# Assertion 5: mean MMD on non-temporal pairs should be > 0
non_temp = df[df.pair_category != 'temporal']
if len(non_temp) == 0:
    errors.append('no non-temporal rows to check MMD on')
else:
    m_mmd = non_temp.mmd_statistic.mean()
    if not (m_mmd > 0):
        errors.append(f'mean mmd_statistic on non-temporal pairs={m_mmd:.4f} '
                      f'is not > 0 — drift should be detectable')

print(f'  rows produced       : {len(df)}')
print(f'  unique users        : {df.target_user.nunique()}')
print(f'  rows by category    : {df.pair_category.value_counts().to_dict()}')
print(f'  mean benefit_nd     : {m_benefit_nd:+.4f}')
print(f'  mean benefit_dr     : {df.adaptation_benefit_drift.mean():+.4f}')
print(f'  M1_drift_eer range  : [{m1d.min():.4f}, {m1d.max():.4f}]')
print(f'  mean mmd (overall)  : {df.mmd_statistic.mean():.4f}')
if len(non_temp) > 0:
    print(f'  mean mmd (non-temp) : {non_temp.mmd_statistic.mean():.4f}')

if errors:
    print('\n[smoke] FAILED:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
print('\n[smoke] all assertions passed')
PY

status=$?
echo "[smoke] exit status $status"
exit $status