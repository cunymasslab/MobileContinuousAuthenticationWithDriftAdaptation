#!/usr/bin/env python3
"""
Generate HMOG vs Touchalytics drift comparison table.

Reads Touchalytics drift results from touchalytics_drift_results.csv
and outputs a formatted comparison table. HMOG values are left blank
for manual entry.

Usage:
    python touchalytics_hmog_comparison.py /path/to/data/directory
"""

import argparse
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="HMOG vs Touchalytics comparison table")
    parser.add_argument('data_dir', help="Directory containing touchalytics_drift_results.csv")
    args = parser.parse_args()

    path = os.path.join(args.data_dir, 'touchalytics_drift_results.csv')
    if not os.path.exists(path):
        sys.exit(f"Error: {path} not found. Run touchalytics_drift_check.py first.")

    df = pd.read_csv(path)

    # Compute means per drift type
    touch = {}
    for dt in ['temporal', 'cross_activity', 'cross_both']:
        subset = df[df['drift_type'] == dt]
        if len(subset) > 0:
            touch[dt] = {
                'ks_mean': subset['ks_mean'].mean(),
                'classifier_accuracy': subset['classifier_accuracy'].mean(),
                'mmd_statistic': subset['mmd_statistic'].mean(),
            }
        else:
            touch[dt] = {'ks_mean': float('nan'), 'classifier_accuracy': float('nan'), 'mmd_statistic': float('nan')}

    metrics = ['ks_mean', 'classifier_accuracy', 'mmd_statistic']
    metric_labels = ['Mean KS statistic', 'Mean classifier accuracy', 'Mean MMD']
    drift_types = ['temporal', 'cross_activity', 'cross_both']
    drift_labels = ['Temporal', 'Cross-Activity', 'Cross-Both']

    # --- Markdown table ---
    print("## Drift Magnitude Comparison: HMOG vs Touchalytics\n")
    header = f"| {'Metric':<28s} | {'HMOG Temp':>10s} | {'HMOG XAct':>10s} | {'HMOG XBoth':>10s} | {'Touch Temp':>10s} | {'Touch XAct':>10s} | {'Touch XBoth':>10s} |"
    sep =    f"|{'-'*30}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|"
    print(header)
    print(sep)
    for m, ml in zip(metrics, metric_labels):
        vals = [f"{touch[dt][m]:.4f}" for dt in drift_types]
        print(f"| {ml:<28s} | {'':>10s} | {'':>10s} | {'':>10s} | {vals[0]:>10s} | {vals[1]:>10s} | {vals[2]:>10s} |")

    # --- LaTeX table ---
    print("\n\n% LaTeX table for paper")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Drift magnitude comparison across datasets and drift types.}")
    print(r"\label{tab:cross-dataset-drift}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{HMOG (IMU)}} & \multicolumn{3}{c}{\textbf{Touchalytics (Touch)}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    print(r"\textbf{Metric} & Temporal & Cross-Act. & Cross-Both & Temporal & Cross-Act. & Cross-Both \\")
    print(r"\midrule")
    for m, ml in zip(metrics, metric_labels):
        vals = [f"{touch[dt][m]:.3f}" for dt in drift_types]
        print(f"{ml} & --- & --- & --- & {vals[0]} & {vals[1]} & {vals[2]} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # --- CSV output ---
    csv_path = os.path.join(args.data_dir, 'hmog_touchalytics_comparison.csv')
    rows = []
    for m, ml in zip(metrics, metric_labels):
        row = {'metric': ml}
        for dt, dl in zip(drift_types, drift_labels):
            row[f'hmog_{dl.lower()}'] = ''
            row[f'touch_{dl.lower()}'] = f"{touch[dt][m]:.4f}"
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
