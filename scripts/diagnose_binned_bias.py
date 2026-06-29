#!/usr/bin/env python3
"""Compute binned bias and per-region bias from an eval CSV.

Usage:
  python3 scripts/diagnose_binned_bias.py <csv> --attr size --out evaluations/binned_region_bias.csv

This script writes a CSV with columns: bin_left, bin_right, region, n, mean_gt, mean_pred, mean_residual, rmse
and prints a short summary. It also saves a scatter PNG of residual vs gt for GT<0.2.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_region(df):
    # If region column exists, use it. Otherwise compute 3x3 grid from midpoint / canvas.
    if 'region' in df.columns:
        return df['region'].astype(str)

    # Expect midpoint_x, midpoint_y, canvas_width, canvas_height columns
    if not {'midpoint_x', 'midpoint_y', 'canvas_width', 'canvas_height'}.issubset(df.columns):
        # fallback: single region
        return pd.Series(['R00'] * len(df), index=df.index)

    mx = df['midpoint_x'].astype(float)
    my = df['midpoint_y'].astype(float)
    cw = df['canvas_width'].astype(float)
    ch = df['canvas_height'].astype(float)

    # compute normalized coordinates robustly
    nx = mx / cw.replace(0, np.nan)
    ny = my / ch.replace(0, np.nan)
    # if any NaN or extreme values, also try dividing by 1000 (some exports scaled by 1000)
    nx = nx.fillna(mx / cw.max())
    ny = ny.fillna(my / ch.max())

    nx = nx.clip(0.0, 0.9999)
    ny = ny.clip(0.0, 0.9999)

    ix = (nx * 3).astype(int).clip(0, 2)
    iy = (ny * 3).astype(int).clip(0, 2)

    regions = [f"R{i}{j}" for i in range(3) for j in range(3)]
    return pd.Series([regions[y * 3 + x] for x, y in zip(ix, iy)], index=df.index)


def compute_bin_stats(df, gt_col, pred_col, bins, region_series):
    rows = []
    # Build global bins first
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        mask_bin = (df[gt_col] >= left) & (df[gt_col] < right)
        if mask_bin.sum() == 0:
            continue
        for region in sorted(region_series.unique()):
            mask = mask_bin & (region_series == region)
            n = int(mask.sum())
            if n == 0:
                continue
            gt = df.loc[mask, gt_col].to_numpy(dtype=float)
            pred = df.loc[mask, pred_col].to_numpy(dtype=float)
            res = pred - gt
            rmse = float(np.sqrt(np.mean(res ** 2)))
            rows.append({
                'bin_left': left,
                'bin_right': right,
                'region': region,
                'n': n,
                'mean_gt': float(gt.mean()),
                'mean_pred': float(pred.mean()),
                'mean_residual': float(res.mean()),
                'rmse': rmse,
            })

    return pd.DataFrame(rows)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Evaluation CSV path')
    p.add_argument('--attr', default='size', help='attribute name (default: size)')
    p.add_argument('--dim', type=int, default=None, help='numeric dimension index if CSV uses gt_0/pred_0 columns')
    p.add_argument('--out', default=None, help='Output CSV path')
    p.add_argument('--png', default=None, help='Output scatter PNG for GT<0.2')
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print('CSV not found:', csv_path)
        sys.exit(2)

    df = pd.read_csv(csv_path)

    # Determine gt/pred column names. Support either gt_<attr>/pred_<attr> or gt_<dim>/pred_<dim> styles.
    gt_col = f'gt_{args.attr}'
    pred_col = f'pred_{args.attr}'
    if gt_col not in df.columns or pred_col not in df.columns:
        if args.dim is not None:
            gt_col = f'gt_{args.dim}'
            pred_col = f'pred_{args.dim}'
        else:
            # try to auto-detect numeric gt_* column
            numeric_gt = [c for c in df.columns if c.startswith('gt_') and c[3:].isdigit()]
            if numeric_gt:
                gt_col = numeric_gt[0]
                pred_col = 'pred_' + gt_col.split('_', 1)[1]
                print(f"Auto-detected numeric dim columns: {gt_col}, {pred_col}")
            else:
                print('CSV does not contain expected columns:', f'gt_{args.attr}', f'pred_{args.attr}')
                print('Available columns:', list(df.columns))
                sys.exit(2)

    if gt_col not in df.columns or pred_col not in df.columns:
        print('Final check failed, missing columns:', gt_col, pred_col)
        print('Available columns:', list(df.columns))
        sys.exit(2)

    region_series = make_region(df)

    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    stats = compute_bin_stats(df, gt_col, pred_col, bins, region_series)

    # overall summary for GT < 0.1
    mask_small = df[gt_col] < 0.1
    n_small = int(mask_small.sum())
    if n_small > 0:
        res = (df.loc[mask_small, pred_col].to_numpy(dtype=float) - df.loc[mask_small, gt_col].to_numpy(dtype=float))
        print(f'GT < 0.1: n={n_small}, mean_residual={res.mean():.6f}, rmse={np.sqrt((res**2).mean()):.6f}')
    else:
        print('No examples with GT < 0.1')

    # per-region small-bias summary
    rows_region = []
    for region in sorted(region_series.unique()):
        mask = (region_series == region) & mask_small
        n = int(mask.sum())
        if n == 0:
            continue
        gt = df.loc[mask, gt_col].to_numpy(dtype=float)
        pred = df.loc[mask, pred_col].to_numpy(dtype=float)
        res = pred - gt
        rows_region.append({'region': region, 'n': n, 'mean_residual': float(res.mean()), 'rmse': float(np.sqrt((res**2).mean()))})

    reg_df = pd.DataFrame(rows_region).sort_values('region')
    if len(reg_df):
        print('\nPer-region summary for GT < 0.1:')
        print(reg_df.to_string(index=False))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(outp, index=False)
        print('\nWrote binned-region CSV to', outp)

    # quick scatter plot for GT<0.2
    if args.png:
        try:
            import matplotlib.pyplot as plt

            mask_focus = df[gt_col] < 0.2
            plt.figure(figsize=(6, 4))
            res = df.loc[mask_focus, pred_col] - df.loc[mask_focus, gt_col]
            plt.scatter(df.loc[mask_focus, gt_col], res, s=8, alpha=0.4)
            # binned means
            bins_plot = np.linspace(0.0, 0.2, 21)
            inds = np.digitize(df.loc[mask_focus, gt_col], bins_plot)
            xs = []
            ys = []
            for i in range(1, len(bins_plot)):
                m = inds == i
                if m.sum() == 0:
                    continue
                xs.append(df.loc[mask_focus, gt_col].values[m].mean())
                ys.append((df.loc[mask_focus, pred_col].values[m] - df.loc[mask_focus, gt_col].values[m]).mean())
            if xs:
                plt.plot(xs, ys, color='red', marker='o')
            plt.axhline(0, color='k', lw=0.8)
            plt.xlabel('GT')
            plt.ylabel('pred - gt')
            plt.title(f'Residual vs GT (<0.2) for {args.attr}')
            plt.tight_layout()
            p = Path(args.png)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=150)
            print('Wrote scatter PNG to', p)
        except Exception as e:
            print('Could not save PNG:', e)


if __name__ == '__main__':
    main(sys.argv[1:])
