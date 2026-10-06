"""
generate_recovery_curve.py — Headline recovery curve for the TIFS paper.

Plots mean rolling EER over normalized stream progress (0–100%) for four
models: M1 (VanillaRNN frozen), M2 (VanillaRNN adapting), M6 (GRU frozen),
M7 (GRU adapting), from the merged Stage 4 detail CSV.

Usage:
    python generate_recovery_curve.py results_stage4_merged_detail_with_m7.csv

CRITICAL: the input file is multi-GB. This script streams it in chunks via
pandas (chunksize=500_000) and never holds more than a chunk in memory.
Two passes:
    Pass 1: find max window_index per (user, train, stream, seed) → each
            experiment's length. Stored in a small dict.
    Pass 2: for each row, compute normalized progress = window_index/length,
            place into one of 100 percent-bins, accumulate per-model rolling
            EER sums and counts.

Outputs (figures/ in cwd, created if missing):
    figures/recovery_curve.png   (dpi=150)
    figures/recovery_curve.pdf
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CHUNK_SIZE = 500_000
N_BINS     = 100
FIG_DIR    = 'figures'

KEY_COLS = ['target_user', 'train_condition', 'stream_condition', 'seed']

# (model_id, legend_label, linestyle, color)
PLOT_MODELS = [
    ('M1', 'VanillaRNN frozen (M1)',   'dashed', 'tab:blue'),
    ('M2', 'VanillaRNN adapting (M2)', 'solid',  'tab:blue'),
    ('M6', 'GRU frozen (M6)',          'dashed', 'tab:orange'),
    ('M7', 'GRU adapting (M7)',        'solid',  'tab:orange'),
]
MODEL_IDS    = [m[0] for m in PLOT_MODELS]
ROLLING_COLS = [f'{m}_rolling_eer' for m in MODEL_IDS]

# Consistent dtypes so dict-tuple keys from pass 1 match merge keys in pass 2.
READ_DTYPES = {
    'target_user':      str,
    'train_condition':  str,
    'stream_condition': str,
    'seed':             'Int64',
    'window_index':     'Int64',
}


def pass1_max_window(csv_path):
    """Stream file, return dict {(user, train, stream, seed): max window_index}."""
    print('PASS 1: scanning for per-experiment max window_index...')
    max_window = {}
    n_rows = 0
    t0 = time.time()

    usecols = KEY_COLS + ['window_index']
    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE,
                         usecols=usecols, dtype=READ_DTYPES)

    for chunk_i, chunk in enumerate(reader, start=1):
        grp = chunk.groupby(KEY_COLS, sort=False)['window_index'].max()
        for key, wmax in grp.items():
            if pd.isna(wmax):
                continue
            wmax = int(wmax)
            cur = max_window.get(key)
            if cur is None or wmax > cur:
                max_window[key] = wmax
        n_rows += len(chunk)
        print(f'  pass1 chunk {chunk_i:>3d}  rows={n_rows:>12,}  '
              f'experiments_so_far={len(max_window):>6,}  '
              f'elapsed={time.time()-t0:6.1f}s')

    print(f'PASS 1 done. total_rows={n_rows:,}  '
          f'experiments_found={len(max_window):,}  '
          f'elapsed={time.time()-t0:.1f}s')
    print()
    return max_window


def pass2_accumulate(csv_path, max_window):
    """Stream file, accumulate per-bin per-model rolling EER sums + counts."""
    print('PASS 2: accumulating per-bin per-model means...')
    n_models = len(MODEL_IDS)
    sums   = np.zeros((n_models, N_BINS), dtype=np.float64)
    counts = np.zeros((n_models, N_BINS), dtype=np.int64)

    # Build a small lookup DataFrame for vectorized merge.
    if max_window:
        records = [(*k, v) for k, v in max_window.items()]
        lookup = pd.DataFrame(records,
                              columns=KEY_COLS + ['_exp_max_window'])
        # Match dtypes so the merge key alignment is exact.
        lookup['seed'] = lookup['seed'].astype('Int64')
    else:
        lookup = pd.DataFrame(columns=KEY_COLS + ['_exp_max_window'])

    t0 = time.time()
    n_rows = 0

    usecols = KEY_COLS + ['window_index'] + ROLLING_COLS
    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE,
                         usecols=usecols, dtype=READ_DTYPES)

    for chunk_i, chunk in enumerate(reader, start=1):
        chunk = chunk.merge(lookup, on=KEY_COLS, how='left')
        lengths = chunk['_exp_max_window'].to_numpy(dtype=np.float64)
        windex  = chunk['window_index'].to_numpy(dtype=np.float64)

        # progress = window_index / max_window_index, clamped into 100 bins.
        # length == 0 (single-window experiments) → assign to bin 0.
        safe_len = np.where(lengths > 0, lengths, 1.0)
        progress = np.where(lengths > 0, windex / safe_len, 0.0)
        bins = np.clip((progress * N_BINS).astype(np.int64), 0, N_BINS - 1)

        for m_idx, col in enumerate(ROLLING_COLS):
            if col not in chunk.columns:
                continue
            vals = chunk[col].to_numpy(dtype=np.float64)
            valid = np.isfinite(vals)
            if not valid.any():
                continue
            np.add.at(sums[m_idx],   bins[valid], vals[valid])
            np.add.at(counts[m_idx], bins[valid], 1)

        n_rows += len(chunk)
        print(f'  pass2 chunk {chunk_i:>3d}  rows={n_rows:>12,}  '
              f'elapsed={time.time()-t0:6.1f}s')

    print(f'PASS 2 done. total_rows={n_rows:,}  '
          f'elapsed={time.time()-t0:.1f}s')
    print()
    return sums, counts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('csv_path',
                    help='Path to results_stage4_merged_detail_with_m7.csv '
                         '(streamed; never loaded whole)')
    args = ap.parse_args()

    # ── File-open sanity check ─────────────────────────────────────────────
    try:
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(args.csv_path)
        with open(args.csv_path, 'r') as f:
            header = f.readline()
        cols = [c.strip() for c in header.strip().split(',')]
        print(f'opened: {args.csv_path}')
        print(f'header columns: {len(cols)}')
        missing_req = [c for c in KEY_COLS + ['window_index'] if c not in cols]
        if missing_req:
            print(f'ERROR: required columns missing from header: {missing_req}',
                  file=sys.stderr)
            sys.exit(1)
        missing_models = [c for c in ROLLING_COLS if c not in cols]
        if missing_models:
            print(f'WARNING: rolling-EER columns missing: {missing_models} '
                  f'(those models will be skipped)', file=sys.stderr)
    except FileNotFoundError:
        print(f'ERROR: CSV not found: {args.csv_path}', file=sys.stderr)
        print('Check the path; expected an existing file.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'ERROR opening {args.csv_path}: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(1)
    print()

    # ── Pass 1 ─────────────────────────────────────────────────────────────
    max_window = pass1_max_window(args.csv_path)
    n_exp = len(max_window)
    print(f'TOTAL EXPERIMENTS FOUND IN PASS 1: {n_exp}')
    print()

    # ── Pass 2 ─────────────────────────────────────────────────────────────
    sums, counts = pass2_accumulate(args.csv_path, max_window)

    # ── Per-bin means ──────────────────────────────────────────────────────
    means = np.full(sums.shape, np.nan)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]

    # ── Console report: first-bin vs last-bin per model ────────────────────
    print('Per-model rolling EER (mean) at bin 0 (start) vs bin 99 (end):')
    for m_idx, (mid, _, _, _) in enumerate(PLOT_MODELS):
        first = means[m_idx, 0]
        last  = means[m_idx, -1]
        n_first = int(counts[m_idx, 0])
        n_last  = int(counts[m_idx, -1])
        if np.isnan(first) or np.isnan(last):
            print(f'  {mid}  first={first}  last={last}  '
                  f'(n_first={n_first}, n_last={n_last}) — insufficient data')
            continue
        print(f'  {mid}  first={first:.4f}  last={last:.4f}  '
              f'change={last - first:+.4f}  '
              f'(n_first={n_first}, n_last={n_last})')
    print()
    print('(Draft expects M2 change near -0.208; this is the percent-bin')
    print(' analogue, directionally similar but not identical to the')
    print(' first-50-windows-vs-last-50-windows metric.)')
    print()

    # ── Plot ───────────────────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)
    x = np.arange(N_BINS)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m_idx, (mid, label, linestyle, color) in enumerate(PLOT_MODELS):
        y = means[m_idx]
        ax.plot(x, y, label=label, linestyle=linestyle,
                color=color, linewidth=1.8)
    ax.set_xlabel('Stream progress (percent)')
    ax.set_ylabel('Rolling EER')
    ax.set_title('Authentication Performance Over the Streaming Session')
    ax.set_xlim(0, N_BINS - 1)
    ax.legend(loc='best', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    png_path = os.path.join(FIG_DIR, 'recovery_curve.png')
    pdf_path = os.path.join(FIG_DIR, 'recovery_curve.pdf')
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    print('Wrote figures:')
    print(f'  {png_path}')
    print(f'  {pdf_path}')


if __name__ == '__main__':
    main()