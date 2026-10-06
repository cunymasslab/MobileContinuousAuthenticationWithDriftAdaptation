"""
analyze_confound.py — Analysis for the confounding-variable test.

Outputs:
  results_confound/analysis_confound.txt
  results_confound/figures/{benefit_diff_vs_*.png, paired_bar.png}

Statistical conventions:
  * Paired tests aggregate to one row per (user × pair_category) so pairs
    are independent. Headline n is users × categories.
  * Cluster bootstrap 95% CI on the mean difference, resampling at the
    user level (10000 resamples).
  * Holm-Bonferroni correction across the 6 correlation tests
    (3 drift metrics × {Pearson, Spearman}).
  * Bonferroni-corrected KS feature count (n_significant_ks_bonf) is
    reported alongside the raw count.
  * Missingness sensitivity: paired test re-run on users with complete
    coverage across all 4 categories.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon, t as student_t
import statsmodels.formula.api as smf
import warnings

OUT_DIR = 'results_confound'
SUMMARY = os.path.join(OUT_DIR, 'results_confound_summary.csv')
FAILED  = os.path.join(OUT_DIR, 'results_confound_failed.csv')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TXT     = os.path.join(OUT_DIR, 'analysis_confound.txt')

CATEGORIES = ['temporal', 'cross-posture', 'cross-activity', 'cross-both']
N_BOOT = 10000
RNG = np.random.default_rng(42)


def section(title):
    return f'\n{"=" * 72}\n{title}\n{"=" * 72}\n'


def cohens_dz(diffs):
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    return diffs.mean() / sd if sd > 0 else float('nan')


def parametric_ci(diffs, alpha=0.05):
    n = len(diffs)
    if n < 2:
        return float('nan'), float('nan')
    sd = diffs.std(ddof=1)
    h  = student_t.ppf(1 - alpha/2, df=n - 1) * sd / np.sqrt(n)
    return diffs.mean() - h, diffs.mean() + h


def cluster_bootstrap_ci(diffs_by_cluster, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI of the mean, resampling clusters with replacement.
    User-weighted: each cluster contributes one number per resample (the mean
    of its values), so users with more observations don't dominate the estimate.
    diffs_by_cluster: dict cluster_id -> 1D array of values."""
    clusters = list(diffs_by_cluster.keys())
    if len(clusters) == 0:
        return float('nan'), float('nan')
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = RNG.choice(clusters, size=len(clusters), replace=True)
        per_cluster = [diffs_by_cluster[c].mean() for c in sample]
        means[i] = float(np.mean(per_cluster))
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1 - alpha/2))

def holm_bonferroni(pvals):
    """Holm step-down, returns adjusted p-values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        cand = pvals[idx] * (n - rank)
        running = max(running, cand)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    df['benefit_diff'] = df['adaptation_benefit_drift'] - df['adaptation_benefit_nodrift']
    out = []

    # ── Coverage ───────────────────────────────────────────────────────
    out.append(section('Coverage'))
    out.append(f'rows: {len(df)}   unique users: {df.target_user.nunique()}')
    out.append('rows by category:')
    out.append(df.pair_category.value_counts().to_string())

    # ── E1) Missingness audit ──────────────────────────────────────────
    out.append(section('E1) Missingness audit'))
    if os.path.exists(FAILED) and os.path.getsize(FAILED) > 0:
        fdf = pd.read_csv(FAILED)
        out.append(f'failed rows: {len(fdf)}')
        if len(fdf) > 0:
            out.append('top 10 users by failure count:')
            out.append(fdf['target_user'].value_counts().head(10).to_string())
            out.append('\nfailures by reason:')
            out.append(fdf['reason'].value_counts().to_string())
    else:
        out.append('no failures recorded')

    cov = df.groupby('target_user')['pair_category'].nunique()
    full_users = set(cov[cov == len(CATEGORIES)].index)
    out.append(f'\nusers with all {len(CATEGORIES)} categories present: '
               f'{len(full_users)}/{df.target_user.nunique()}')

    # ── 1) Drift magnitude by category ─────────────────────────────────
    out.append(section('1) Drift magnitude by category (mean ± SD, n)'))
    drift_cols = ['ks_mean', 'classifier_accuracy', 'mmd_statistic']
    g = df.groupby('pair_category')[drift_cols]
    drift_tbl = pd.concat({'mean': g.mean(), 'std': g.std(), 'n': g.count()},
                          axis=1).reindex(CATEGORIES)
    out.append(drift_tbl.round(4).to_string())

    # ── 2) Correlation: benefit_diff vs each drift metric ──────────────
    # Drift is deterministic per (user, condition); collapse y over seeds so
    # n is the count of distinct (user, condition) tuples, not 3× that.
    out.append(section('2) Correlation: (benefit_drift - benefit_nodrift) vs drift metric'))
    out.append('Drift metrics are deterministic per (user, train, stream) (computed with')
    out.append('seed=42). y (benefit_diff) is averaged over the 3 experiment seeds before')
    out.append('correlating, so n equals the count of distinct (user, train, stream) tuples.')
    out.append('Holm-Bonferroni adjusts across 6 tests (3 metrics × {Pearson, Spearman}).\n')
    df_corr = (df.groupby(['target_user', 'train_condition', 'stream_condition'])
                 .agg(benefit_diff=('benefit_diff', 'mean'),
                      ks_mean=('ks_mean', 'first'),
                      classifier_accuracy=('classifier_accuracy', 'first'),
                      mmd_statistic=('mmd_statistic', 'first'))
                 .reset_index())
    rows = []
    p_p, p_s = [], []
    for col in drift_cols:
        x = df_corr[col].to_numpy(); y = df_corr['benefit_diff'].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        r, rp = pearsonr(x[m], y[m]);  p_p.append(rp)
        s, sp = spearmanr(x[m], y[m]); p_s.append(sp)
        rows.append({'metric': col, 'n': int(m.sum()),
                     'pearson_r': r, 'pearson_p_raw': rp,
                     'spearman_rho': s, 'spearman_p_raw': sp})
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x[m], y[m], s=8, alpha=0.4)
        ax.set_xlabel(col); ax.set_ylabel('benefit_drift − benefit_nodrift')
        ax.set_title(f'{col}: r={r:+.2f}, ρ={s:+.2f}')
        ax.axhline(0, color='k', lw=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f'benefit_diff_vs_{col}.png'), dpi=150)
        plt.close(fig)
    cor_tbl = pd.DataFrame(rows)
    cor_tbl['pearson_p_holm']  = holm_bonferroni(p_p)
    cor_tbl['spearman_p_holm'] = holm_bonferroni(p_s)
    out.append(cor_tbl.round(4).to_string(index=False))

    # ── 3) Paired by category, aggregated to one row per user×category ─
    out.append(section('3) Paired comparison by category (one row per user × category)'))
    agg = (df.groupby(['target_user', 'pair_category'])
             .agg(benefit_nodrift=('adaptation_benefit_nodrift', 'mean'),
                  benefit_drift  =('adaptation_benefit_drift',   'mean'),
                  mmd_mean       =('mmd_statistic', 'mean'))
             .reset_index())
    agg['difference'] = agg['benefit_drift'] - agg['benefit_nodrift']

    rows = []
    for cat in CATEGORIES:
        sub = agg[agg['pair_category'] == cat]
        if len(sub) == 0:
            continue
        diffs = sub['difference'].to_numpy()
        n = len(diffs)
        lo_p, hi_p = parametric_ci(diffs)
        diffs_by_user = {u: np.array([d]) for u, d in
                         zip(sub['target_user'], sub['difference'])}
        lo_b, hi_b = cluster_bootstrap_ci(diffs_by_user)
        rows.append({
            'category':         cat,
            'n_users':          n,
            'mmd_mean':         sub['mmd_mean'].mean(),
            'benefit_nodrift':  sub['benefit_nodrift'].mean(),
            'benefit_drift':    sub['benefit_drift'].mean(),
            'difference':       diffs.mean(),
            'ci95_param_lo':    lo_p,
            'ci95_param_hi':    hi_p,
            'ci95_boot_lo':     lo_b,
            'ci95_boot_hi':     hi_b,
            'cohens_dz':        cohens_dz(diffs),
        })

    paired_tbl = pd.DataFrame(rows)
    out.append(paired_tbl.round(4).to_string(index=False))

    # Per-category paired tests with Bonferroni × 4
    out.append('\nPer-category paired tests (n_users = rows per category)')
    out.append('Bonferroni correction across 4 categories applied to t and W p-values.')
    cat_results = []
    for cat in CATEGORIES:
        sub = agg[agg['pair_category'] == cat].dropna(
            subset=['benefit_nodrift', 'benefit_drift'])
        if len(sub) < 2:
            cat_results.append({'category': cat, 'n_users': len(sub),
                                't': np.nan, 't_p_raw': np.nan,
                                'W': np.nan, 'W_p_raw': np.nan})
            continue
        a = sub['benefit_drift'].to_numpy()
        b = sub['benefit_nodrift'].to_numpy()
        t_stat, t_p = ttest_rel(a, b)
        try:
            w_stat, w_p = wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        cat_results.append({'category': cat, 'n_users': len(sub),
                            't': t_stat, 't_p_raw': t_p,
                            'W': w_stat, 'W_p_raw': w_p})
    cat_tbl = pd.DataFrame(cat_results)
    n_tests = int(cat_tbl['t_p_raw'].notna().sum())
    if n_tests > 0:
        cat_tbl['t_p_bonf'] = (cat_tbl['t_p_raw'] * n_tests).clip(upper=1.0)
        cat_tbl['W_p_bonf'] = (cat_tbl['W_p_raw'] * n_tests).clip(upper=1.0)
    out.append(cat_tbl.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(paired_tbl))
    w = 0.35
    ax.bar(xpos - w/2, paired_tbl['benefit_nodrift'], w, label='no-drift')
    ax.bar(xpos + w/2, paired_tbl['benefit_drift'],   w, label='drift')
    ax.set_xticks(xpos); ax.set_xticklabels(paired_tbl['category'], rotation=15)
    ax.set_ylabel('mean adaptation benefit (M1 − M4 EER)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'paired_bar.png'), dpi=150)
    plt.close(fig)

    # ── 4) Headline test (two views) ────────────────────────────────────
    out.append(section('4) Headline paired test'))
    out.append('Two complementary views:')
    out.append('  (a) USER-WEIGHTED: one value per user (mean across categories).')
    out.append('      All tests below are valid because pairs are independent.')
    out.append('  (b) USER × CATEGORY: 4 rows per user. Cluster bootstrap (user-weighted)')
    out.append('      is the principled CI; t/Wilcoxon are reported but anti-conservative')
    out.append('      because (user, category) rows within a user are not independent.\n')

    # ── (a) one value per user ──────────────────────────────────────────
    user_diff = (agg.dropna(subset=['benefit_drift', 'benefit_nodrift'])
                    .groupby('target_user')
                    .apply(lambda g: g['benefit_drift'].mean() - g['benefit_nodrift'].mean()))
    user_diff = user_diff[np.isfinite(user_diff)]
    if len(user_diff) >= 2:
        ut, utp = ttest_rel(
            agg.dropna(subset=['benefit_drift', 'benefit_nodrift'])
               .groupby('target_user')['benefit_drift'].mean(),
            agg.dropna(subset=['benefit_drift', 'benefit_nodrift'])
               .groupby('target_user')['benefit_nodrift'].mean())
        try:
            uw, uwp = wilcoxon(
                agg.dropna(subset=['benefit_drift', 'benefit_nodrift'])
                   .groupby('target_user')['benefit_drift'].mean(),
                agg.dropna(subset=['benefit_drift', 'benefit_nodrift'])
                   .groupby('target_user')['benefit_nodrift'].mean())
        except ValueError:
            uw, uwp = float('nan'), float('nan')
        u_lo, u_hi = parametric_ci(user_diff.to_numpy())
        out.append(f'(a) USER-WEIGHTED — one row per user (n={len(user_diff)})')
        out.append(f'  mean difference    : {user_diff.mean():+.4f}')
        out.append(f'  parametric 95% CI  : [{u_lo:+.4f}, {u_hi:+.4f}]')
        out.append(f"  Cohen's d_z        : {cohens_dz(user_diff.to_numpy()):+.3f}")
        out.append(f'  paired t           : t={ut:+.3f}  p={utp:.3e}')
        out.append(f'  Wilcoxon signed    : W={uw:.1f}  p={uwp:.3e}')
    else:
        out.append('(a) USER-WEIGHTED — insufficient users')

    # ── (b) one row per user × category ─────────────────────────────────
    a = agg['benefit_drift'].to_numpy()
    b = agg['benefit_nodrift'].to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    diffs = a[m] - b[m]
    n = len(diffs)
    t_stat, t_p = ttest_rel(a[m], b[m])
    try:
        w_stat, w_p = wilcoxon(a[m], b[m])
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')
    lo_p, hi_p = parametric_ci(diffs)
    diffs_by_user = {}
    for u in agg['target_user'].unique():
        d = agg.loc[agg['target_user'] == u, 'difference'].to_numpy()
        d = d[np.isfinite(d)]
        if len(d) > 0:
            diffs_by_user[u] = d
    lo_b, hi_b = cluster_bootstrap_ci(diffs_by_user)
    out.append(f'\n(b) USER × CATEGORY — n={n} pairs across {len(diffs_by_user)} users')
    out.append(f'  mean(benefit_drift)        : {a[m].mean():+.4f}')
    out.append(f'  mean(benefit_nodrift)      : {b[m].mean():+.4f}')
    out.append(f'  mean difference            : {diffs.mean():+.4f}')
    out.append(f'  cluster-bootstrap 95% CI   : [{lo_b:+.4f}, {hi_b:+.4f}]   (PRINCIPLED)')
    out.append(f'  parametric 95% CI          : [{lo_p:+.4f}, {hi_p:+.4f}]   (anti-conservative — pairs nested)')
    out.append(f"  Cohen's d_z                : {cohens_dz(diffs):+.3f}")
    out.append(f'  paired t                   : t={t_stat:+.3f}  p={t_p:.3e}   (anti-conservative)')
    out.append(f'  Wilcoxon signed-rank       : W={w_stat:.1f}  p={w_p:.3e}     (anti-conservative)')

    # ── 5) Sensitivity to missingness ──────────────────────────────────
    out.append(section('5) Sensitivity — restrict to users with all 4 categories'))
    sub = agg[agg['target_user'].isin(full_users)]
    if len(sub) > 0:
        a2 = sub['benefit_drift'].to_numpy()
        b2 = sub['benefit_nodrift'].to_numpy()
        d2 = a2 - b2
        t2, tp2 = ttest_rel(a2, b2)
        w2, wp2 = wilcoxon(a2, b2)
        out.append(f'  n_users complete : {len(full_users)}')
        out.append(f'  n rows           : {len(d2)}')
        out.append(f'  mean difference  : {d2.mean():+.4f}')
        out.append(f"  Cohen's d_z      : {cohens_dz(d2):+.3f}")
        out.append(f'  paired t         : t={t2:+.3f}  p={tp2:.3e}')
        out.append(f'  Wilcoxon         : W={w2:.1f}  p={wp2:.3e}')
    else:
        out.append('  no users with complete coverage')

    # ── 6) KS Bonferroni ──────────────────────────────────────────────
    out.append(section('6) KS multiple-comparison summary'))
    if 'n_significant_ks_bonf' in df.columns:
        out.append(f'  mean n_significant_ks      (raw α=0.05)        '
                   f': {df["n_significant_ks"].mean():.2f} / 12')
        out.append(f'  mean n_significant_ks_bonf (α=0.05/12≈0.0042)  '
                   f': {df["n_significant_ks_bonf"].mean():.2f} / 12')
    else:
        out.append('  Bonferroni column missing (run_confound.py predates fix 1g).')

    # ── 6.5) Linear mixed-effects model (parametric corroboration) ─────
    out.append(section('6.5) Linear mixed-effects model (parametric corroboration)'))
    out.append('Long-format: rows = (user, category, stream_type, benefit).')
    out.append('Model: benefit ~ stream_type + pair_category, random intercept by user.')
    out.append('Fixed effect on stream_type[T.drift] is the adaptation-benefit difference')
    out.append('when training distribution and test distribution differ, controlling for')
    out.append('between-user variance.\n')
    long = pd.melt(
        agg, id_vars=['target_user', 'pair_category'],
        value_vars=['benefit_nodrift', 'benefit_drift'],
        var_name='stream_type', value_name='benefit',
    )
    long['stream_type'] = long['stream_type'].map({
        'benefit_nodrift': 'nodrift', 'benefit_drift': 'drift',
    })
    long = long.dropna(subset=['benefit'])

    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            md  = smf.mixedlm('benefit ~ C(stream_type, Treatment("nodrift")) '
                              '+ C(pair_category, Treatment("temporal"))',
                              long, groups=long['target_user'],
                              re_formula='~C(stream_type, Treatment("nodrift"))')
            res = md.fit(reml=True, method='lbfgs')
        out.append(f'  converged          : {bool(res.converged)}')
        if wlist:
            for w in wlist:
                out.append(f'  WARNING ({w.category.__name__}): {w.message}')
        out.append(res.summary().as_text())


        params = res.params
        stream_key = [k for k in params.index if 'stream_type' in k and 'drift' in k]
        if stream_key:
            k = stream_key[0]
            coef = params[k]
            se   = res.bse[k]
            pval = res.pvalues[k]
            ci_lo, ci_hi = res.conf_int().loc[k]
            out.append(f'\nHeadline mixed-effects estimate (stream_type[T.drift]):')
            out.append(f'  coef                : {coef:+.4f}')
            out.append(f'  std err             : {se:.4f}')
            out.append(f'  95% CI              : [{ci_lo:+.4f}, {ci_hi:+.4f}]')
            out.append(f'  Wald p              : {pval:.3e}')
            out.append(f'  random user variance: {res.cov_re.iloc[0, 0]:.4f}')
    except Exception as e:
        out.append(f'mixed-effects model failed: {type(e).__name__}: {e}')

    # ── 7) Initial-vs-end EER (adaptation kinetics) ───────────────────


    if 'M4_drift_initial_eer' in df.columns:
        out.append(section('7) Adaptation kinetics on the drift stream (M4)'))
        ki = df.groupby('pair_category').agg(
            init_mean =('M4_drift_initial_eer', 'mean'),
            end_mean  =('M4_drift_end_eer',     'mean'),
        ).reindex(CATEGORIES)
        ki['recovery'] = (df.groupby('pair_category')
                            .apply(lambda g: ((g['M4_drift_initial_eer'] -
                                               g['M4_drift_end_eer']) /
                                              g['M4_drift_initial_eer'].replace(0, np.nan)).mean())
                            .reindex(CATEGORIES).values)
        # How many rows were dropped per category (initial_eer == 0 or NaN)?
        dropped = (df.groupby('pair_category')
                     .apply(lambda g: int(((g['M4_drift_initial_eer'] == 0) |
                                           g['M4_drift_initial_eer'].isna() |
                                           g['M4_drift_end_eer'].isna()).sum()))
                     .reindex(CATEGORIES))
        ki['n_dropped'] = dropped.values
        out.append(ki.round(4).to_string())

    # ── 8) Sanity check — M1 baseline vs Stage 4 ───────────────────────
    out.append(section('8) Sanity check — M1 baseline vs Stage 4'))
    STAGE4 = 'results_stage4_merged_summary.csv'
    if not os.path.exists(STAGE4):
        out.append(f'{STAGE4} not found — sanity check skipped.')
    else:
        s4 = pd.read_csv(STAGE4)
        required = ['target_user', 'train_condition', 'stream_condition', 'seed', 'M1_overall_eer']
        missing = [c for c in required if c not in s4.columns]
        if missing:
            out.append(f'Stage 4 CSV missing columns {missing}; sanity check skipped.')
        else:
            keys = ['target_user', 'train_condition', 'stream_condition', 'seed']
            df_keys = df[keys + ['M1_drift_eer', 'pair_category']].copy()
            df_keys['seed'] = df_keys['seed'].astype(int)
            s4['seed'] = s4['seed'].astype(int)
            merged = pd.merge(df_keys, s4[keys + ['M1_overall_eer']],
                              on=keys, how='inner')
            out.append(f'paired rows (confound × Stage 4): {len(merged)}')
            if len(merged) > 0:
                comp = merged.groupby('pair_category').agg(
                    confound_M1_mean=('M1_drift_eer',    'mean'),
                    stage4_M1_mean=  ('M1_overall_eer',  'mean'),
                    n=               ('target_user',     'count'),
                ).reindex(CATEGORIES)
                comp['mean_diff_(confound-stage4)'] = comp['confound_M1_mean'] - comp['stage4_M1_mean']
                out.append(comp.round(4).to_string())
                x = merged['M1_drift_eer'].to_numpy()
                y = merged['M1_overall_eer'].to_numpy()
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= 3:
                    r, p = pearsonr(x[m], y[m])
                    out.append(f'\nPearson r (paired M1 EERs): {r:+.3f}  p={p:.3e}  n={int(m.sum())}')
                    out.append('Interpretation: if r > 0.7 the 67%-trained baselines track the 100%-')
                    out.append('trained Stage 4 baselines and the confound deltas should generalize.')

    txt = '\n'.join(out) + '\n'
    with open(TXT, 'w') as f:
        f.write(txt)
    print(txt)
    print(f'\nWrote {TXT} and figures to {FIG_DIR}/')


if __name__ == '__main__':
    main()