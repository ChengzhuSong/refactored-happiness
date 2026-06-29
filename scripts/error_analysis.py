#!/usr/bin/env python3
"""Simple error analysis: top-k absolute-error outliers and residual-by-GT bins.

Usage (interactive):
  from scripts.error_analysis import analyze_csv
  analyze_csv('eval_size_gt_pred_train.csv', out_prefix='train_analysis', top_k=50)

Or run as script to process multiple CSVs in the cwd.
"""
import csv
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


def analyze_csv(csv_path, out_prefix=None, top_k=50, n_bins=20, gt_col='gt_size', pred_col='pred_size'):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)

    rows = []
    with open(p, 'r', newline='') as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        gt_cols = [c for c in header if c.startswith('gt_')]
        pred_cols = [c for c in header if c.startswith('pred_')]
        # fall back to single-named columns if present
        if not gt_cols and gt_col in header:
            gt_cols = [gt_col]
        if not pred_cols and pred_col in header:
            pred_cols = [pred_col]

        if len(gt_cols) == 0 or len(pred_cols) == 0:
            # no matching columns
            for row in r:
                pass
            print('No gt_/pred_ columns found in', csv_path)
            return None

        for row in r:
            try:
                gt_vec = np.array([float(row[c]) for c in gt_cols], dtype=np.float32)
                pred_vec = np.array([float(row[c]) for c in pred_cols], dtype=np.float32)
            except Exception:
                continue
            # L2 residual
            err = float(np.linalg.norm(pred_vec - gt_vec))
            row['_gt_vec'] = gt_vec
            row['_pred_vec'] = pred_vec
            row['_err'] = err
            # for convenience, store scalar gt_norm
            row['_gt'] = float(np.linalg.norm(gt_vec))
            row['_pred'] = float(np.linalg.norm(pred_vec))
            rows.append(row)

    if len(rows) == 0:
        print('No valid rows in', csv_path)
        return None

    # Top-k outliers by absolute error
    rows_sorted = sorted(rows, key=lambda r: -r['_err'])
    topk = rows_sorted[:top_k]

    # Residuals by GT bins
    gts = np.array([r['_gt'] for r in rows])
    preds = np.array([r['_pred'] for r in rows])
    errs = np.array([r['_err'] for r in rows])
    bins = np.linspace(gts.min(), gts.max(), n_bins + 1)
    bin_idx = np.digitize(gts, bins) - 1
    bin_stats = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() == 0:
            bin_stats.append({'bin': b, 'gt_lo': bins[b], 'gt_hi': bins[b+1], 'count': 0, 'rmse': None, 'mae': None})
            continue
        sel_errs = preds[sel] - gts[sel]
        # rmse in L2 space
        rmse = math.sqrt((sel_errs ** 2).mean())
        mae = float(np.mean(np.abs(sel_errs)))
        bin_stats.append({'bin': b, 'gt_lo': bins[b], 'gt_hi': bins[b+1], 'count': int(sel.sum()), 'rmse': rmse, 'mae': mae})

    out_pref = out_prefix if out_prefix else p.stem
    # save top-k CSV
    topk_csv = f'{out_pref}_topk.csv'
    # write a compact top-k CSV with selected fields
    with open(topk_csv, 'w', newline='') as f:
        # choose a few useful columns if present
        all_keys = list(topk[0].keys())
        # ensure _err/_gt/_pred present
        keys = [k for k in ['poster_idx', 'slot_idx', 'n_text_elements'] if k in all_keys]
        keys += [k for k in all_keys if k.startswith('slot_pos_')]
        keys += ['_gt', '_pred', '_err']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in topk:
            outrow = {k: r.get(k, '') for k in keys}
            w.writerow(outrow)

    # save bin stats CSV
    bins_csv = f'{out_pref}_bins.csv'
    with open(bins_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['bin', 'gt_lo', 'gt_hi', 'count', 'rmse', 'mae'])
        w.writeheader()
        for b in bin_stats:
            w.writerow(b)

    # quick plots: RMSE vs GT bin (bar) and residual scatter
    try:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        xs = [0.5 * (b['gt_lo'] + b['gt_hi']) for b in bin_stats]
        rmses = [b['rmse'] if b['rmse'] is not None else 0 for b in bin_stats]
        axs[0].bar(xs, rmses, width=(bins[1] - bins[0]) * 0.9)
        axs[0].set_title('RMSE by GT bin')
        axs[0].set_xlabel('GT size')
        axs[0].set_ylabel('RMSE')

        axs[1].scatter(gts, preds, s=8, alpha=0.6)
        axs[1].plot([gts.min(), gts.max()], [gts.min(), gts.max()], color='red', linewidth=1)
        axs[1].set_title('Residual scatter')
        axs[1].set_xlabel('GT')
        axs[1].set_ylabel('Pred')

        fig.tight_layout()
        fig_path = f'{out_pref}_analysis.png'
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print('Failed to create plots:', e)
        fig_path = None

    print('Analysis written:', topk_csv, bins_csv, fig_path)
    return {'topk_csv': topk_csv, 'bins_csv': bins_csv, 'fig': fig_path}


def main():
    # process train/validation/test CSVs if present
    cwd = Path('.')
    for name in ['train', 'validation', 'test']:
        for f in cwd.glob(f'*{name}*.csv'):
            try:
                print('Analyzing', f)
                analyze_csv(f, out_prefix=f.stem)
            except Exception as e:
                print('Failed to analyze', f, e)


if __name__ == '__main__':
    main()
