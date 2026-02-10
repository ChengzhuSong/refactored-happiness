#!/usr/bin/env python3
"""
Extract per-poster slot texts from crello_*_elements_per_image.parquet files and
write JSONL files where each line is a JSON array of length `--slots` (default 64)
containing strings or null for empty/missing slots.

Writes files to: data/crello/slot_texts_<split>.jsonl

Usage:
  python3 scripts/extract_slot_texts.py --slots 64

"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def extract_split(parquet_path: Path, out_path: Path, slots: int = 64):
    print(f"Reading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if 'poster_id' not in df.columns or 'element_index' not in df.columns or 'element_text' not in df.columns:
        raise ValueError('Parquet missing required columns: poster_id, element_index, element_text')

    # Group by poster_id and accumulate element_text by element_index
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poster_ids = df['poster_id'].unique()
    print(f"Found {len(poster_ids)} posters")

    # Build mapping poster_id -> dict(index->text)
    groups = df.groupby('poster_id')

    written = 0
    too_large_counts = 0
    with out_path.open('w', encoding='utf-8') as f:
        for pid, g in groups:
            slot_texts = [None] * slots
            max_idx = int(g['element_index'].max())
            if max_idx >= slots:
                # enlarge to hold all indices (warn)
                new_slots = max_idx + 1
                slot_texts = [None] * new_slots
                too_large_counts += 1

            for _, row in g.iterrows():
                idx = int(row['element_index'])
                txt = row['element_text']
                # treat empty strings / whitespace-only as missing
                if txt is None:
                    val = None
                else:
                    try:
                        txts = str(txt).strip()
                        val = txts if len(txts) > 0 else None
                    except Exception:
                        val = None
                if idx < 0:
                    continue
                if idx >= len(slot_texts):
                    # extend
                    extend_by = idx + 1 - len(slot_texts)
                    slot_texts.extend([None] * extend_by)
                slot_texts[idx] = val

            # ensure final length is at least slots
            if len(slot_texts) < slots:
                slot_texts.extend([None] * (slots - len(slot_texts)))

            # write JSON array
            f.write(json.dumps(slot_texts, ensure_ascii=False) + '\n')
            written += 1

    print(f"Wrote {written} lines to {out_path}; {too_large_counts} posters had indices >= {slots}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slots', type=int, default=64, help='number of slots (default 64)')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    parser.add_argument('--rawdir', type=str, default='rawdata', help='rawdata directory')
    parser.add_argument('--outdir', type=str, default='data/crello', help='output directory for JSONL')
    args = parser.parse_args()

    for split in args.splits:
        p = Path(args.rawdir) / f"crello_{split}_elements_per_image.parquet"
        if not p.exists():
            print(f"Skipping {split}: {p} not found")
            continue
        out = Path(args.outdir) / f"slot_texts_{split}.jsonl"
        extract_split(p, out, slots=args.slots)


if __name__ == '__main__':
    main()
