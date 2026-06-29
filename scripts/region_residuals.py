#!/usr/bin/env python3
"""Compute per-region residuals and summary for evaluation CSVs.

Produces per-region metrics (n, rmse, mae, r2) for a chosen gt/pred dimension
and writes per-split CSVs under `evaluations/region_metrics_<split>_dim<k>.csv`.

Usage:
  python3 scripts/region_residuals.py evaluations/eval_size_gt_pred_test_full_with_midpoints.csv --dim 1
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

REGION_NAMES = ['upper-left','upper-middle','upper-right',
                'middle-left','middle','middle-right',
                'lower-left','lower-middle','lower-right']


def region_label_from_row(r, region_n=3):
    try:
        cx = float(r['midpoint_x']) / float(r['canvas_width'])
        cy = float(r['midpoint_y']) / float(r['canvas_height'])
    except Exception:
        return 'unknown'
    cx = min(max(cx, 0.0), 0.9999)
    cy = min(max(cy, 0.0), 0.9999)
    col = int(cx * region_n)
    rowi = int(cy * region_n)
    rid = rowi * region_n + col
    if region_n == 3:
        return REGION_NAMES[rid]
    return f'region_{rid}'


def compute_metrics_for_df(df, dim=1, region_n=3):
    gt_col = f'gt_{dim}'
    pred_col = f'pred_{dim}'
    if gt_col not in df.columns or pred_col not in df.columns:
        raise ValueError(f'Missing columns {gt_col}/{pred_col}')
    # ensure region column
    if 'region' not in df.columns:
        if all(c in df.columns for c in ('midpoint_x','midpoint_y','canvas_width','canvas_height')):
            df['region'] = df.apply(lambda r: region_label_from_row(r, region_n=region_n), axis=1)
        else:
            df['region'] = 'unknown'

    rows = []
    for region, g in df.groupby('region'):
        try:
            gt = g[gt_col].astype(float).to_numpy()
            pred = g[pred_col].astype(float).to_numpy()
        except Exception:
            continue
        n = len(gt)
        if n == 0:
            continue
        resid = pred - gt
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        rows.append(dict(region=region, n=n, rmse=rmse, mae=mae, r2=r2))

    dfm = pd.DataFrame(rows).sort_values('rmse')
    return dfm


def process_file(csv_path, dim=1, region_n=3, out_dir=Path('evaluations')):
    p = Path(csv_path)
    df = pd.read_csv(p)
    metrics = compute_metrics_for_df(df, dim=dim, region_n=region_n)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'region_metrics_{p.stem}_dim{dim}.csv'
    metrics.to_csv(out_csv, index=False)
    # pick best region: highest r2 (or lowest rmse)
    best_by_rmse = metrics.sort_values('rmse').iloc[0]
    best_by_r2 = metrics.sort_values('r2', ascending=False).iloc[0]
    summary = dict(file=p.name, best_rmse_region=best_by_rmse['region'], best_rmse=float(best_by_rmse['rmse']),
                   best_r2_region=best_by_r2['region'], best_r2=float(best_by_r2['r2']))
    return out_csv, metrics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvs', nargs='+')
    parser.add_argument('--dim', type=int, default=1, help='which gt/pred dim to analyze (default 1)')
    args = parser.parse_args()

    summaries = []
    for csv in args.csvs:
        out_csv, metrics, summary = process_file(csv, dim=args.dim)
        print('Wrote metrics to', out_csv)
        print(metrics)
        summaries.append(summary)

    print('\nSummary across files:')
    for s in summaries:
        print(s)


if __name__ == '__main__':
    main()
