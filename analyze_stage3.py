"""
analyze_stage3.py
Analyze Stage 3 results — architecture and hidden size comparison.
Produces the capacity argument table for the paper.
"""

import pandas as pd

OUTPUT_FILE = 'results_stage3_architecture_search.csv'

df = pd.read_csv(OUTPUT_FILE)
print(f'Loaded {len(df)} rows from {OUTPUT_FILE}\n')

# --- Progress check ---
print(f'Architectures: {sorted(df["architecture"].unique())}')
print(f'Hidden dims: {sorted(df["hidden_dim"].unique())}')
print(f'Seeds: {df["seed"].nunique()}, Users: {df["target_user"].nunique()}, Conditions: {df["condition"].nunique()}')
print()

# --- Main table: Mean EER by (architecture, hidden_dim) ---
print('=== Mean EER by (architecture, hidden_dim) ===')
pivot = (
    df.groupby(['architecture', 'hidden_dim'])
    .agg(
        mean_eer=('eer', 'mean'),
        std_eer=('eer', 'std'),
        mean_convergence=('convergence_epoch', 'mean'),
        n_experiments=('eer', 'count'),
    )
    .sort_values(['architecture', 'hidden_dim'])
)
print(pivot.to_string())
print()

# --- Pivot table for paper (architecture rows, hidden_dim columns) ---
print('=== Paper Table: Mean EER ± Std ===')
paper = df.groupby(['architecture', 'hidden_dim'])['eer'].agg(['mean', 'std']).unstack('hidden_dim')
paper.columns = [f'H={hd} (mean)' if stat == 'mean' else f'H={hd} (std)' for stat, hd in paper.columns]
print(paper.to_string())
print()

# --- Best frozen model ---
best = pivot.sort_values('mean_eer').iloc[0]
best_idx = pivot.sort_values('mean_eer').index[0]
print(f'>>> Best frozen model: {best_idx[0]} H={best_idx[1]} '
      f'(mean EER = {best["mean_eer"]:.4f} ± {best["std_eer"]:.4f})')
print()

# --- Breakdown by condition ---
print('=== EER by (architecture, hidden_dim, condition) ===')
by_cond = (
    df.groupby(['architecture', 'hidden_dim', 'condition'])
    .agg(mean_eer=('eer', 'mean'), std_eer=('eer', 'std'), n=('eer', 'count'))
    .sort_values('mean_eer')
)
print(by_cond.to_string())
print()

# --- Convergence check ---
print('=== Convergence by (architecture, hidden_dim) ===')
conv = (
    df.groupby(['architecture', 'hidden_dim'])['convergence_epoch']
    .agg(['mean', 'min', 'max'])
)
conv.columns = ['mean_epoch', 'min_epoch', 'max_epoch']
print(conv.to_string())
