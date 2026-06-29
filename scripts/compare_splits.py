#!/usr/bin/env python3
"""Compute and summarize regression metrics (n, RMSE, MAE, R^2) for train/validation/test CSVs.

This script looks for CSVs matching *train*.csv, *validation*.csv, *test*.csv and computes metrics
using the same L2-norm scalar for multi-dim attributes (matching error_analysis). It writes
a summary CSV and a PNG bar chart.
"""
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics_from_csv(csv_path):
    with open(csv_path, 'r', newline='') as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        gt_cols = [c for c in header if c.startswith('gt_')]
        pred_cols = [c for c in header if c.startswith('pred_')]
        if not gt_cols or not pred_cols:
            return None
        gts = []
        preds = []
        for row in r:
            try:
                gt = np.array([float(row[c]) for c in gt_cols], dtype=np.float32)
                pred = np.array([float(row[c]) for c in pred_cols], dtype=np.float32)
            except Exception:
                continue
            gts.append(np.linalg.norm(gt))
            preds.append(np.linalg.norm(pred))
    if len(gts) == 0:
        return None
    gts = np.array(gts)
    preds = np.array(preds)
    residuals = preds - gts
    sse = float((residuals ** 2).sum())
    rmse = math.sqrt(sse / len(gts))
    mae = float(np.mean(np.abs(residuals)))
    sst = float(((gts - gts.mean()) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst != 0 else float('nan')
    return {'n': int(len(gts)), 'rmse': rmse, 'mae': mae, 'r2': r2}


def main():
    cwd = Path('.')
    rows = []
    for label in ['train', 'validation', 'test']:
        # prefer a *_full.csv if present, otherwise pick the first matching csv
        full_files = sorted(cwd.glob(f'*{label}*full*.csv'))
        if full_files:
            chosen = full_files[0]
        else:
            files = sorted(cwd.glob(f'*{label}*.csv'))
            if not files:
                continue
            chosen = files[0]
        metrics = compute_metrics_from_csv(chosen)
        if metrics:
            rows.append({'split': label, **metrics, 'csv': str(chosen)})

    if not rows:
        print('No matching CSVs found (train/validation/test)')
        return

    out_csv = 'split_metrics_summary.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'n', 'rmse', 'mae', 'r2', 'csv'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # bar chart for RMSE/MAE/R2
    labels = [r['split'] for r in rows]
    rmses = [r['rmse'] for r in rows]
    maes = [r['mae'] for r in rows]
    r2s = [r['r2'] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].bar(labels, rmses, color='C0')
    ax[0].set_title('RMSE')
    ax[1].bar(labels, maes, color='C1')
    ax[1].set_title('MAE')
    ax[2].bar(labels, r2s, color='C2')
    ax[2].set_title('R^2')
    for a in ax:
        a.set_xlabel('split')

    fig.tight_layout()
    out_png = 'split_metrics_summary.png'
    fig.savefig(out_png, dpi=200)
    print('Wrote', out_csv, out_png)


if __name__ == '__main__':
    main()
