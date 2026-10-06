#!/usr/bin/env bash
# Launch 12 chunked confound experiments in parallel.
# After all chunks finish, merge per-chunk CSVs into the canonical paths
# read by analyze_confound.py.

set -u
cd "$(dirname "$0")"
mkdir -p results_confound/logs

NUM=12
for k in $(seq 1 "$NUM"); do
    log="results_confound/logs/chunk${k}.log"
    nohup python -u run_confound.py \
        --users-file test_users.txt \
        --chunk "$k" --num-chunks "$NUM" \
        > "$log" 2>&1 &
    echo "started chunk $k -> $log (pid $!)"
done
wait
echo "all chunks finished"

echo "merging per-chunk CSVs..."
ts=$(date +%Y%m%d_%H%M%S)
for f in results_confound/results_confound_summary.csv \
         results_confound/results_confound_failed.csv; do
    if [ -f "$f" ]; then
        backup="${f%.csv}_backup_${ts}.csv"
        echo "  backing up existing $f -> $backup"
        mv "$f" "$backup"
    fi
done

# Fail loudly if any chunk's summary is missing
for k in $(seq 1 "$NUM"); do
    if [ ! -f "results_confound/results_confound_summary_chunk${k}.csv" ]; then
        echo "  WARNING: chunk ${k} summary missing — chunk likely crashed"
    fi
done

awk 'FNR==1 && NR!=1 {next} {print}' \
    results_confound/results_confound_summary_chunk*.csv \
    > results_confound/results_confound_summary.csv
awk 'FNR==1 && NR!=1 {next} {print}' \
    results_confound/results_confound_failed_chunk*.csv \
    > results_confound/results_confound_failed.csv

# Row-count validation: expected = n_users × 36 pairs × 3 seeds, minus
# rows that landed in failed.csv. Warn if completed rows fall >10% short.
n_users=$(grep -cve '^$' test_users.txt)
# Subtract anyone in EXCLUDE_USERS = {'207969', '733162'} that's in test_users.txt
for u in 207969 733162; do
    if grep -qx "$u" test_users.txt; then
        n_users=$((n_users - 1))
    fi
done
expected=$((n_users * 36 * 3))
n_summary=$(($(wc -l < results_confound/results_confound_summary.csv) - 1))   # minus header
n_failed=$(($(wc -l < results_confound/results_confound_failed.csv) - 1))
n_total=$((n_summary + n_failed))
threshold=$((expected * 90 / 100))
echo "row-count check: expected=$expected  summary=$n_summary  failed=$n_failed  total=$n_total"
if [ "$n_total" -lt "$threshold" ]; then
    echo "  WARNING: completed rows ($n_total) more than 10% below expected ($expected)."
    echo "  At least one chunk likely lost work. Inspect logs/chunk*.log before analysis."
fi