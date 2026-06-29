#!/usr/bin/env python3
"""Compute midpoint for masked elements in an eval CSV.

For each row this script computes midpoint = [left + 0.5*width, top + 0.5*height].
It looks for `slot_pos_0`, `slot_pos_1`, and `gt_0`,`gt_1` (the gt size dims) in the CSV
and writes a new CSV with two extra columns: `midpoint_x`, `midpoint_y`.

Usage:
  python3 scripts/compute_masked_midpoints.py input.csv [--out out.csv]
"""
import csv
import argparse
from pathlib import Path


def compute_midpoints(in_csv: Path, out_csv: Path):
    with in_csv.open('r', newline='') as inf:
        reader = csv.reader(inf)
        header = next(reader)
        idx = {n: i for i, n in enumerate(header)}

        # required columns
        for col in ('slot_pos_0', 'slot_pos_1', 'gt_0', 'gt_1'):
            if col not in idx:
                raise ValueError(f"Required column missing from CSV: {col}")

        rows_out = []
        for row in reader:
            try:
                left = float(row[idx['slot_pos_0']])
                top = float(row[idx['slot_pos_1']])
                width = float(row[idx['gt_0']])
                height = float(row[idx['gt_1']])
            except Exception:
                # preserve row but write empty midpoints
                rows_out.append(row + ['', ''])
                continue
            mx = left + 0.5 * width
            my = top + 0.5 * height
            rows_out.append(row + [f"{mx:.6f}", f"{my:.6f}"])

    with out_csv.open('w', newline='') as outf:
        writer = csv.writer(outf)
        new_header = header + ['midpoint_x', 'midpoint_y']
        writer.writerow(new_header)
        for r in rows_out:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_csv', help='eval CSV produced by plot_scatter_eval.py')
    p.add_argument('--out', help='output CSV path (default: input_midpoints.csv)')
    args = p.parse_args()

    inp = Path(args.input_csv)
    if args.out:
        outp = Path(args.out)
    else:
        outp = inp.with_name(inp.stem + '_midpoints.csv')

    compute_midpoints(inp, outp)
    print('Wrote midpoints to', outp)


if __name__ == '__main__':
    main()
