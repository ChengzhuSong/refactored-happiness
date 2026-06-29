#!/usr/bin/env python3
"""Plot train and validation scatter plots using plot_from_csv.plot_from_csv

Usage examples:
  python3 scripts/plot_split_scatter.py --train eval_size_gt_pred_train.csv --val eval_size_gt_pred_val.csv --out-dir plots/
  python3 scripts/plot_split_scatter.py --dir ./ --pattern size
"""
import argparse
from pathlib import Path
import sys

try:
    from scripts.plot_from_csv import plot_from_csv
except Exception:
    # allow running as a script from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_from_csv import plot_from_csv


def main():
    # Interactive convenience: auto-detect train and val CSVs in the current directory
    cwd = Path('.').resolve()
    train_candidates = sorted(cwd.glob('*train*.csv'))
    val_candidates = sorted(cwd.glob('*val*.csv'))

    if len(train_candidates) == 0 and len(val_candidates) == 0:
        print('No train/val CSVs found in current directory. Place files matching *train*.csv and/or *val*.csv')
        return

    def plot_one_interactive(csv_path, label):
        print('Plotting', csv_path, ' (interactive)')
        metrics = plot_from_csv(csv_path, out=None, show=True, size_scale=10.0,
                                pos_2d=False, pos_2d_dims=(0, 1), pos_2d_radius_scale=1.0)
        print(f'{label} metrics:', metrics)

    # prefer exact matches if present
    if train_candidates:
        # pick first
        plot_one_interactive(train_candidates[0], 'train')
    if val_candidates:
        plot_one_interactive(val_candidates[0], 'val')


if __name__ == '__main__':
    main()
