import os, csv
import numpy as np

OUTPUT_DIR = "results_m9_lr_sweep"
CONFIGS = [('M2', 0.002), ('M9', 0.0001), ('M9', 0.0005), ('M9', 0.001), ('M9', 0.002)]

def output_paths(model_id, lr):
    tag = f'{model_id}_lr_{lr:g}'
    base = os.path.join(OUTPUT_DIR, tag)
    return {
        'summary': f'{base}_summary.csv',
        'detail': f'{base}_detail.csv',
        'curves': f'{base}_training_curves.csv',
    }

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

if __name__ == "__main__":
    with open("validation_users.txt") as f:
        users = {line.strip() for line in f if line.strip()}
    write_comparison(CONFIGS, users)
