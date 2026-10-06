"""
analyze_stage1.py
Analyze Stage 1 results and pick the best (optimizer, lr_train).
Decision: Lowest mean EER. Convergence_epoch as tiebreaker (fewer = more stable).
Run anytime — works on partial results too.
"""

import pandas as pd

OUTPUT_FILE = 'results_stage1_training_hparams.csv'

df = pd.read_csv(OUTPUT_FILE)
print(f'Loaded {len(df)} rows from {OUTPUT_FILE}\n')

# --- Progress check ---
expected_combos = 2 * 3  # 2 optimizers x 3 learning rates
seeds = df['seed'].nunique()
users = df['target_user'].nunique()
conditions = df['condition'].nunique()
print(f'Seeds: {seeds}, Users: {users}, Conditions: {conditions}')
print(f'Unique (optimizer, lr) combos: {df.groupby(["optimizer", "lr_train"]).ngroups} / {expected_combos}\n')

# --- Mean EER per (optimizer, lr_train), averaged across seeds/users/conditions ---
summary = (
    df.groupby(['optimizer', 'lr_train'])
    .agg(
        mean_eer=('eer', 'mean'),
        std_eer=('eer', 'std'),
        mean_convergence=('convergence_epoch', 'mean'),
        std_convergence=('convergence_epoch', 'std'),
        mean_final_loss=('final_loss', 'mean'),
        n_experiments=('eer', 'count'),
    )
    .sort_values(['mean_eer', 'mean_convergence'])
)
print('=== EER by (optimizer, lr_train) ===')
print(summary.to_string())
print()

# --- Best combo ---
best = summary.index[0]
best_row = summary.iloc[0]
print(f'>>> Best: optimizer={best[0]}, lr_train={best[1]}  '
      f'(mean EER = {best_row["mean_eer"]:.4f} +/- {best_row["std_eer"]:.4f}, '
      f'converges @ epoch {best_row["mean_convergence"]:.1f})')
print()

# --- Breakdown by condition ---
print('=== EER by (optimizer, lr_train, condition) ===')
by_cond = (
    df.groupby(['optimizer', 'lr_train', 'condition'])
    .agg(
        mean_eer=('eer', 'mean'),
        std_eer=('eer', 'std'),
        mean_convergence=('convergence_epoch', 'mean'),
        n=('eer', 'count'),
    )
    .sort_values('mean_eer')
)
print(by_cond.to_string())
print()

# --- Convergence analysis: did any combo hit max epochs frequently? ---
print('=== Convergence check (did training finish or early-stop?) ===')
conv = (
    df.groupby(['optimizer', 'lr_train'])['convergence_epoch']
    .agg(['mean', 'min', 'max'])
)
conv.columns = ['mean_epoch', 'min_epoch', 'max_epoch']
print(conv.to_string())
print()
max_epoch_used = df['convergence_epoch'].max()
n_maxed = (df['convergence_epoch'] == max_epoch_used).sum()
print(f'{n_maxed}/{len(df)} experiments hit max epochs ({max_epoch_used})')
