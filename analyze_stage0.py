"""
analyze_stage0.py
Analyze Stage 0 results and pick the best (window_size, step_size).
Run anytime — works on partial results too.
"""

import pandas as pd

OUTPUT_FILE = 'results_stage0_preprocessing.csv'

df = pd.read_csv(OUTPUT_FILE)
print(f'Loaded {len(df)} rows from {OUTPUT_FILE}\n')

# --- Progress check ---
expected_combos = 3 * 2  # 3 window sizes x 2 step sizes
seeds = df['seed'].nunique()
users = df['target_user'].nunique()
conditions = df['condition'].nunique()
print(f'Seeds: {seeds}, Users: {users}, Conditions: {conditions}')
print(f'Unique (ws, ss) combos: {df.groupby(["window_size", "step_size"]).ngroups} / {expected_combos}\n')

# --- Mean EER per (window_size, step_size), averaged across seeds/users/conditions ---
summary = (
    df.groupby(['window_size', 'step_size'])['eer']
    .agg(['mean', 'std', 'count'])
    .sort_values('mean')
)
summary.columns = ['mean_eer', 'std_eer', 'n_experiments']
print('=== EER by (window_size, step_size) ===')
print(summary.to_string())
print()

# --- Best combo ---
best = summary.index[0]
best_eer = summary.iloc[0]['mean_eer']
best_std = summary.iloc[0]['std_eer']
print(f'>>> Best: window_size={best[0]}, step_size={best[1]}  '
      f'(mean EER = {best_eer:.4f} +/- {best_std:.4f})')
print()

# --- Breakdown by condition ---
print('=== EER by (window_size, step_size, condition) ===')
by_cond = (
    df.groupby(['window_size', 'step_size', 'condition'])['eer']
    .agg(['mean', 'std', 'count'])
    .sort_values('mean')
)
by_cond.columns = ['mean_eer', 'std_eer', 'n']
print(by_cond.to_string())
print()

# --- Other metrics for the best combo ---
best_rows = df[(df['window_size'] == best[0]) & (df['step_size'] == best[1])]
print(f'=== All metrics for best combo (ws={best[0]}, ss={best[1]}) ===')
print(best_rows[['eer', 'accuracy', 'precision', 'recall', 'f1']].describe().round(4))
