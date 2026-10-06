"""
analyze_stage4b.py — Stage 4 Drift Visualization (Experiment 3)
Generates publication-quality figures for IEEE TIFS paper.

EER values are displayed as percentages (0–100) throughout.
Non-EER metrics (precision, recall, recovery, forgetting rate, raw anomaly scores)
are unchanged.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_CSV = os.path.join(SCRIPT_DIR, 'results_stage4_merged_summary_with_m7.csv')
DETAIL_CSV = os.path.join(SCRIPT_DIR, 'results_stage4_merged_detail_with_m7.csv')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Publication style
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

MODEL_NAMES = {
    'M1': 'M1 RNN-Frozen',
    'M2': 'M2 RNN-RTRL',
    'M3': 'M3 RNN-RTRL-Legit',
    'M4': 'M4 RNN-BPTT-FT',
    'M5': 'M5 LSTM-Frozen',
    'M6': 'M6 GRU-Frozen',
    'M7': 'M7 GRU-RTRL',
    'M8': 'M8 GRU-BPTT-FT',
}

MODELS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']
DRIFT_CATS = ['temporal', 'cross-posture', 'cross-activity', 'cross-both']
PALETTE = sns.color_palette('colorblind', n_colors=8)

def save_fig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: figures/{name}.png and .pdf')


# ── Load summary data ────────────────────────────────────────────────────────
print('Loading summary CSV...')
df = pd.read_csv(SUMMARY_CSV)
print(f'  {len(df)} rows, {df["target_user"].nunique()} users')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: Box plots of EER by model
# ══════════════════════════════════════════════════════════════════════════════
print('\nFig 3: EER box plots by model...')

eer_data = []
for m in MODELS:
    col = f'{m}_overall_eer'
    if col in df.columns:
        vals = df[col].dropna()
        for v in vals:
            eer_data.append({'Model': MODEL_NAMES[m], 'EER': v * 100})

eer_df = pd.DataFrame(eer_data)

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=eer_df, x='Model', y='EER', palette=PALETTE, ax=ax,
            fliersize=2, linewidth=0.8)
ax.set_ylabel('Equal Error Rate (%)')
ax.set_xlabel('')
ax.set_title('Overall EER Distribution by Model')
ax.tick_params(axis='x', rotation=30)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Random (50%)')
ax.set_ylim(-2, 55)
fig.tight_layout()
save_fig(fig, 'fig3_eer_by_model')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: EER by drift category (frozen vs adapting)
# ══════════════════════════════════════════════════════════════════════════════
print('\nFig 4: EER by drift category...')

plot_models = ['M1', 'M2', 'M4', 'M6', 'M7', 'M8']
plot_names = {m: MODEL_NAMES[m] for m in plot_models}

cat_data = []
for _, row in df.iterrows():
    cat = row['pair_category']
    if cat not in DRIFT_CATS:
        continue
    for m in plot_models:
        col = f'{m}_overall_eer'
        if col in df.columns and pd.notna(row[col]):
            cat_data.append({
                'Drift Category': cat,
                'Model': plot_names[m],
                'EER': row[col] * 100,
            })

cat_df = pd.DataFrame(cat_data)

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=cat_df, x='Drift Category', y='EER', hue='Model',
            palette=PALETTE[:len(plot_models)], ax=ax,
            fliersize=1.5, linewidth=0.7,
            order=DRIFT_CATS)
ax.set_ylabel('Equal Error Rate (%)')
ax.set_xlabel('')
ax.set_title('EER by Drift Category and Model')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
ax.set_ylim(-2, 55)
fig.tight_layout()
save_fig(fig, 'fig4_eer_by_drift_category')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: EER heatmap by condition pair
# ══════════════════════════════════════════════════════════════════════════════
print('\nFig 2: EER heatmap by condition pair...')

CONDITIONS = [
    'READING+SITTING', 'READING+WALKING',
    'WRITING+SITTING', 'WRITING+WALKING',
    'MAP+SITTING',     'MAP+WALKING',
]
SHORT_CONDS = ['Read+Sit', 'Read+Walk', 'Write+Sit', 'Write+Walk', 'Map+Sit', 'Map+Walk']

def build_heatmap(df, eer_col):
    """Build a 6x6 heatmap matrix of mean EER values, scaled to percentage (0-100)."""
    matrix = np.full((6, 6), np.nan)
    for i, tc in enumerate(CONDITIONS):
        for j, sc in enumerate(CONDITIONS):
            subset = df[(df['train_condition'] == tc) & (df['stream_condition'] == sc)]
            if len(subset) > 0 and eer_col in subset.columns:
                vals = subset[eer_col].dropna()
                if len(vals) > 0:
                    matrix[i, j] = vals.mean() * 100
    return matrix

m1_heat = build_heatmap(df, 'M1_overall_eer')
m2_heat = build_heatmap(df, 'M2_overall_eer')
improvement = m1_heat - m2_heat  # already in percentage points

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, data, title, cmap, vmin, vmax in [
    (axes[0], m1_heat, 'M1 Frozen RNN (EER %)', 'YlOrRd', 0, 50),
    (axes[1], m2_heat, 'M2 RTRL-RNN (EER %)', 'YlOrRd', 0, 50),
    (axes[2], improvement, 'Improvement M1 − M2 (pp)', 'RdYlGn', -10, 30),
]:
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(SHORT_CONDS, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(SHORT_CONDS, fontsize=8)
    ax.set_xlabel('Stream Condition')
    ax.set_ylabel('Train Condition')
    ax.set_title(title)
    # Annotate cells
    for i in range(6):
        for j in range(6):
            if not np.isnan(data[i, j]):
                color = 'white' if data[i, j] > (vmax - vmin) * 0.6 + vmin else 'black'
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle('EER by Train/Stream Condition Pair (%)', fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, 'fig2_eer_heatmap')

# ── Fig 5: EER by Model ──────────────────────────────────────────────────────
print('\n=== Fig 5: EER by Model ===')

model_labels = [MODEL_NAMES[m] for m in MODELS]
eer_means = []
eer_sems = []
for m in MODELS:
    col = f'{m}_overall_eer'
    vals = pd.to_numeric(df[col], errors='coerce').dropna() * 100
    eer_means.append(vals.mean())
    eer_sems.append(vals.sem())

fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(model_labels))
bars = ax.bar(x, eer_means, yerr=eer_sems, capsize=3, width=0.6,
              color=[PALETTE[i] for i in range(len(MODELS))],
              edgecolor='black', linewidth=0.5)

ax.set_ylabel('Equal Error Rate (%)', fontsize=12)
ax.set_title('EER by Model (lower is better)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=20, ha='right', fontsize=9)

ax.set_ylim(0, max(eer_means) * 1.25)

for bar, val in zip(bars, eer_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

fig.tight_layout()
save_fig(fig, 'fig5_eer_by_model')

# # ── Fig 5: Multi-Metric Bar Chart ─────────────────────────────────────────────
# print('\n=== Fig 5: Multi-metric comparison across models ===')

# METRICS = {
#     'overall_eer':  ('EER ↓',    True),   # (display name, lower_is_better)
#     'accuracy':     ('Accuracy',  False),
#     'precision':    ('Precision', False),
#     'recall':       ('Recall',    False),
#     'f1':           ('F1',        False),
#     'mcc':          ('MCC',       False),
#     'auc_roc':      ('AUC-ROC',  False),
# }

# # Compute mean across all users/conditions/seeds for each model × metric
# metric_data = []
# for m in MODELS:
#     for metric_key, (display_name, _) in METRICS.items():
#         col = f'{m}_{metric_key}'
#         if col in df.columns:
#             vals = pd.to_numeric(df[col], errors='coerce').dropna()
#             metric_data.append({
#                 'Model': MODEL_NAMES[m],
#                 'Metric': display_name,
#                 'Mean': vals.mean(),
#                 'SEM': vals.sem(),
#             })

# mdf = pd.DataFrame(metric_data)

# # Split into two panels: EER (lower is better) and all others (higher is better)
# eer_df = mdf[mdf['Metric'] == 'EER ↓']
# other_df = mdf[mdf['Metric'] != 'EER ↓']

# fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2.2]})

# # --- Panel (a): EER (lower is better) ---
# ax = axes[0]
# model_labels = [MODEL_NAMES[m] for m in MODELS]
# eer_means = []
# eer_sems = []
# for mlabel in model_labels:
#     row = eer_df[eer_df['Model'] == mlabel]
#     eer_means.append(row['Mean'].values[0] if len(row) else 0)
#     eer_sems.append(row['SEM'].values[0] if len(row) else 0)

# x = np.arange(len(model_labels))
# bars = ax.bar(x, eer_means, yerr=eer_sems, capsize=3, width=0.6,
#               color=[PALETTE[i] for i in range(8)], edgecolor='black', linewidth=0.5)
# ax.set_ylabel('Equal Error Rate')
# ax.set_title('EER by Model (lower is better)')
# ax.set_xticks(x)
# ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
# ax.set_ylim(0, max(eer_means) * 1.25)
# # Add value labels
# for bar, val in zip(bars, eer_means):
#     ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
#             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# # --- Panel (b): Higher-is-better metrics ---
# ax = axes[1]
# higher_metrics = [name for name, (_, lib) in METRICS.items() if not lib]
# display_names = [METRICS[m][0] for m in higher_metrics]
# n_metrics = len(display_names)
# n_models = len(MODELS)
# bar_width = 0.8 / n_metrics
# x = np.arange(n_models)

# for j, metric_name in enumerate(display_names):
#     subset = other_df[other_df['Metric'] == metric_name]
#     means = []
#     sems = []
#     for mlabel in model_labels:
#         row = subset[subset['Model'] == mlabel]
#         means.append(row['Mean'].values[0] if len(row) else 0)
#         sems.append(row['SEM'].values[0] if len(row) else 0)
#     offset = (j - n_metrics/2 + 0.5) * bar_width
#     ax.bar(x + offset, means, bar_width, yerr=sems, capsize=2,
#            label=metric_name, edgecolor='black', linewidth=0.3)

# ax.set_ylabel('Metric Value')
# ax.set_title('(b) Classification Metrics by Model (higher is better)')
# ax.set_xticks(x)
# ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=9)
# ax.legend(loc='lower right', ncol=3, framealpha=0.9)
# ax.set_ylim(0, 1.05)

# fig.tight_layout()
# save_fig(fig, 'fig5_multi_metric_comparison')

# ── Fig 6: Score Distributions ────────────────────────────────────────────────
# NOTE: These are raw anomaly scores, NOT EER — left unchanged.
print('\n=== Fig 6: Score distributions (legit vs impostor) ===')

# Read detail CSV in chunks — only need score columns + true_label
SHOW_MODELS = ['M1', 'M2', 'M7', 'M8']  # frozen vs adaptive comparisons
score_cols = [f'{m}_score' for m in SHOW_MODELS] + ['true_label']

print('  Reading detail CSV in chunks (this may take a minute)...')
legit_scores = {m: [] for m in SHOW_MODELS}
impostor_scores = {m: [] for m in SHOW_MODELS}

for chunk in pd.read_csv(DETAIL_CSV, chunksize=500_000, usecols=score_cols):
    chunk['true_label'] = pd.to_numeric(chunk['true_label'], errors='coerce')
    legit_mask = chunk['true_label'] == 1
    impostor_mask = chunk['true_label'] == 0
    for m in SHOW_MODELS:
        col = f'{m}_score'
        vals = pd.to_numeric(chunk[col], errors='coerce')
        legit_scores[m].append(vals[legit_mask].dropna().values)
        impostor_scores[m].append(vals[impostor_mask].dropna().values)

# Concatenate chunks
for m in SHOW_MODELS:
    legit_scores[m] = np.concatenate(legit_scores[m])
    impostor_scores[m] = np.concatenate(impostor_scores[m])

print(f'  Loaded: {len(legit_scores["M1"]):,} legit, {len(impostor_scores["M1"]):,} impostor windows')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, m in enumerate(SHOW_MODELS):
    ax = axes[idx // 2][idx % 2]
    ax.hist(legit_scores[m], bins=80, alpha=0.6, density=True,
            label='Legitimate', color='#2196F3', edgecolor='none')
    ax.hist(impostor_scores[m], bins=80, alpha=0.6, density=True,
            label='Impostor', color='#F44336', edgecolor='none')
    ax.set_title(MODEL_NAMES[m], fontsize=12, fontweight='bold')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

fig.suptitle('Score Distributions: Legitimate vs Impostor', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'fig6_score_distributions')

# ── Fig 7: Forgetting Analysis ────────────────────────────────────────────────
# NOTE: Forgetting rate is already a percentage, and the comparison end_eer > initial_eer
#       is unit-invariant — left unchanged.
print('\n=== Fig 7: Forgetting analysis (end EER > initial EER) ===')

ADAPTIVE_MODELS = ['M2', 'M3', 'M4', 'M7', 'M8']

forgetting_data = []
for m in ADAPTIVE_MODELS:
    init_col = f'{m}_initial_eer'
    end_col = f'{m}_end_eer'
    if init_col not in df.columns or end_col not in df.columns:
        continue
    sub = df[[init_col, end_col, 'pair_category']].copy()
    sub[init_col] = pd.to_numeric(sub[init_col], errors='coerce')
    sub[end_col] = pd.to_numeric(sub[end_col], errors='coerce')
    sub = sub.dropna(subset=[init_col, end_col])

    for cat in DRIFT_CATS:
        cat_sub = sub[sub['pair_category'] == cat] if cat != 'all' else sub
        n_total = len(cat_sub)
        if n_total == 0:
            continue
        n_forgot = (cat_sub[end_col] > cat_sub[init_col]).sum()
        forgetting_data.append({
            'Model': MODEL_NAMES[m],
            'Category': cat.replace('-', '-\n'),
            'Forgetting %': 100 * n_forgot / n_total,
            'n_forgot': n_forgot,
            'n_total': n_total,
        })

    # Also add 'all' category
    n_total = len(sub)
    n_forgot = (sub[end_col] > sub[init_col]).sum()
    forgetting_data.append({
        'Model': MODEL_NAMES[m],
        'Category': 'all',
        'Forgetting %': 100 * n_forgot / n_total,
        'n_forgot': n_forgot,
        'n_total': n_total,
    })

fdf = pd.DataFrame(forgetting_data)
cats_order = [c.replace('-', '-\n') for c in DRIFT_CATS] + ['all']

fig, ax = plt.subplots(figsize=(12, 5))
n_cats = len(cats_order)
n_models = len(ADAPTIVE_MODELS)
bar_width = 0.8 / n_models
x = np.arange(n_cats)

for j, m in enumerate(ADAPTIVE_MODELS):
    subset = fdf[fdf['Model'] == MODEL_NAMES[m]]
    vals = []
    for cat in cats_order:
        row = subset[subset['Category'] == cat]
        vals.append(row['Forgetting %'].values[0] if len(row) else 0)
    offset = (j - n_models/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, vals, bar_width, label=MODEL_NAMES[m],
                  edgecolor='black', linewidth=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Drift Category')
ax.set_ylabel('Cases with Forgetting (%)')
ax.set_title('Forgetting Rate: % of Condition Pairs Where End EER > Initial EER')
ax.set_xticks(x)
ax.set_xticklabels(cats_order, fontsize=10)
ax.legend(loc='upper left', framealpha=0.9)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4, label='_nolegend_')
ax.set_ylim(0, max(fdf['Forgetting %']) * 1.2)

fig.tight_layout()
save_fig(fig, 'fig7_forgetting_analysis')

# ── Fig 8: Recovery by Model and Drift Category ──────────────────────────────
# NOTE: Recovery is a unitless ratio (1.0 = full recovery) — left unchanged.
print('\n=== Fig 8: Recovery metric by model and drift category ===')

recovery_data = []
for m in ADAPTIVE_MODELS:
    rec_col = f'{m}_recovery'
    if rec_col not in df.columns:
        continue
    for cat in DRIFT_CATS + ['all']:
        sub = df if cat == 'all' else df[df['pair_category'] == cat]
        vals = pd.to_numeric(sub[rec_col], errors='coerce').dropna()
        # Clip recovery to [0, 2] for display (some outliers can be very large)
        vals = vals.clip(0, 2)
        recovery_data.append({
            'Model': MODEL_NAMES[m],
            'Category': cat,
            'Mean': vals.mean(),
            'Median': vals.median(),
            'SEM': vals.sem(),
        })

rdf = pd.DataFrame(recovery_data)
cats_order = DRIFT_CATS + ['all']

fig, ax = plt.subplots(figsize=(12, 5))
n_cats = len(cats_order)
n_models = len(ADAPTIVE_MODELS)
bar_width = 0.8 / n_models
x = np.arange(n_cats)

for j, m in enumerate(ADAPTIVE_MODELS):
    subset = rdf[rdf['Model'] == MODEL_NAMES[m]]
    means = []
    sems = []
    for cat in cats_order:
        row = subset[subset['Category'] == cat]
        means.append(row['Mean'].values[0] if len(row) else 0)
        sems.append(row['SEM'].values[0] if len(row) else 0)
    offset = (j - n_models/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, means, bar_width, yerr=sems, capsize=2,
                  label=MODEL_NAMES[m], edgecolor='black', linewidth=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Drift Category')
ax.set_ylabel('Recovery (1.0 = full recovery)')
ax.set_title('Drift Recovery by Model and Category')
ax.set_xticks(x)
ax.set_xticklabels(cats_order, fontsize=10)
ax.legend(loc='upper right', framealpha=0.9)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(0, min(max(rdf['Mean']) * 1.3, 2.0))

fig.tight_layout()
save_fig(fig, 'fig8_recovery_by_drift')

# ── Fig 9: Precision vs Recall Scatter ────────────────────────────────────────
# NOTE: Precision and recall are kept as fractions (0–1) — left unchanged.
print('\n=== Fig 9: Precision vs Recall scatter ===')

fig, ax = plt.subplots(figsize=(8, 7))

for i, m in enumerate(MODELS):
    prec_col = f'{m}_precision'
    rec_col = f'{m}_recall'
    if prec_col not in df.columns or rec_col not in df.columns:
        continue
    prec = pd.to_numeric(df[prec_col], errors='coerce')
    rec = pd.to_numeric(df[rec_col], errors='coerce')
    valid = prec.notna() & rec.notna()
    prec, rec = prec[valid], rec[valid]

    # Plot mean ± std as ellipse-like point with error bars
    mean_p, mean_r = prec.mean(), rec.mean()
    std_p, std_r = prec.std(), rec.std()

    ax.errorbar(mean_r, mean_p, xerr=std_r, yerr=std_p,
                fmt='o', markersize=10, color=PALETTE[i],
                capsize=4, capthick=1.5, linewidth=1.5,
                label=MODEL_NAMES[m], zorder=5)

# Ideal point
ax.plot(1, 1, marker='*', markersize=18, color='gold', markeredgecolor='black',
        markeredgewidth=0.8, zorder=10, label='Ideal (1,1)')

# F1 iso-curves
for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
    r_range = np.linspace(0.01, 1, 200)
    p_curve = (f1_val * r_range) / (2 * r_range - f1_val)
    mask = (p_curve >= 0) & (p_curve <= 1)
    ax.plot(r_range[mask], p_curve[mask], '--', color='gray', alpha=0.3, linewidth=0.8)
    # Label the curve
    label_idx = np.argmin(np.abs(r_range[mask] - 0.95))
    if label_idx < len(r_range[mask]):
        ax.text(r_range[mask][label_idx], p_curve[mask][label_idx] + 0.01,
                f'F1={f1_val}', fontsize=7, color='gray', alpha=0.6)

ax.set_xlabel('Recall (1 − FRR)', fontsize=12)
ax.set_ylabel('Precision (1 − FAR impact)', fontsize=12)
ax.set_title('Precision–Recall Tradeoff by Model')
ax.set_xlim(0.3, 1.02)
ax.set_ylim(0.3, 1.02)
ax.legend(loc='lower left', framealpha=0.9)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

fig.tight_layout()
save_fig(fig, 'fig9_precision_recall_scatter')

# ── Fig 1: Rolling EER Over Time ──────────────────────────────────────────────
print('\n=== Fig 1: Rolling EER over time (streaming adaptation curves) ===')

# We'll average rolling EER across all users/conditions/seeds at each window_index
# Read detail CSV in chunks, accumulating sum and count per (model, window_index)
PLOT_MODELS = ['M1', 'M2', 'M4', 'M6', 'M7', 'M8']
eer_cols = [f'{m}_rolling_eer' for m in PLOT_MODELS]
use_cols = ['window_index'] + eer_cols

print('  Reading detail CSV in chunks...')
eer_sum = {m: {} for m in PLOT_MODELS}
eer_count = {m: {} for m in PLOT_MODELS}

for chunk in pd.read_csv(DETAIL_CSV, chunksize=500_000, usecols=use_cols):
    chunk['window_index'] = pd.to_numeric(chunk['window_index'], errors='coerce')
    for m in PLOT_MODELS:
        col = f'{m}_rolling_eer'
        valid = chunk[['window_index', col]].dropna()
        valid[col] = pd.to_numeric(valid[col], errors='coerce')
        valid = valid.dropna()
        for wi, eer_val in zip(valid['window_index'].astype(int), valid[col]):
            eer_sum[m][wi] = eer_sum[m].get(wi, 0.0) + eer_val
            eer_count[m][wi] = eer_count[m].get(wi, 0) + 1

# Compute means (in decimal), then scale to percentage at plot time
eer_mean = {}
for m in PLOT_MODELS:
    windows = sorted(eer_sum[m].keys())
    eer_mean[m] = {w: eer_sum[m][w] / eer_count[m][w] for w in windows}

print(f'  Max window index: {max(max(eer_mean[m].keys()) for m in PLOT_MODELS)}')

# Plot
fig, ax = plt.subplots(figsize=(12, 5))

# Style: frozen models dashed, adaptive models solid
line_styles = {
    'M1': ('--', PALETTE[0], 1.5),
    'M6': ('--', PALETTE[5], 1.5),
    'M2': ('-',  PALETTE[1], 2.0),
    'M4': ('-',  PALETTE[3], 1.5),
    'M7': ('-',  PALETTE[4], 2.0),
    'M8': ('-',  PALETTE[6], 1.5),
}

for m in PLOT_MODELS:
    windows = sorted(eer_mean[m].keys())
    vals = [eer_mean[m][w] * 100 for w in windows]  # scale to percentage
    ls, color, lw = line_styles[m]
    # Smooth with rolling average for readability
    if len(vals) > 20:
        smooth_vals = pd.Series(vals).rolling(window=10, min_periods=1).mean().values
    else:
        smooth_vals = vals
    ax.plot(windows, smooth_vals, ls, color=color, linewidth=lw,
            label=MODEL_NAMES[m], alpha=0.9)

ax.set_xlabel('Window Index (streaming order)')
ax.set_ylabel('Rolling EER (%)')
ax.set_title('Rolling EER During Streaming Evaluation')
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 50)

fig.tight_layout()
save_fig(fig, 'fig1_rolling_eer_over_time')