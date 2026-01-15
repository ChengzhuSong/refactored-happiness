#!/usr/bin/env python3
"""
Generate per-poster inputs using fused per-element embeddings (produced by
`scripts/train_element_attribute_encoder.py`).

Outputs (under data/crello):
 - poster_inputs_fused_X.npy  (float32: num_posters x max_elems x fused_dim+6)
 - poster_inputs_fused_mask.npy (uint8: num_posters x max_elems)
 - poster_inputs_fused_index.csv (poster_id,num_elements)

This script is a safe, idempotent helper that does not overwrite the existing
legacy `poster_inputs_X.npy` files. It reads `crello_element_fused_embeddings.npy`
and `crello_validation_elements_per_image.parquet` (or csv) and writes fused poster inputs.
"""
import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    base = Path('data') / 'crello'
    fused_p = base / 'crello_element_fused_embeddings.npy'
    assert fused_p.exists(), 'Fused embeddings not found; run scripts/train_element_attribute_encoder.py first'
    fused = np.load(fused_p)

    geom_cols = ['left', 'top', 'width', 'height', 'angle', 'opacity']

    # process each split's per-image table if it exists and has fused indices
    splits = ['train', 'validation', 'test']
    for split in splits:
        pname_parquet = base / f'crello_{split}_elements_per_image.parquet'
        pname_csv = base / f'crello_{split}_elements_per_image.csv'
        per_image = None
        if pname_parquet.exists():
            per_image = pname_parquet
        elif pname_csv.exists():
            per_image = pname_csv
        else:
            print(f'skip {split}: no per-image table found')
            continue

        print('Processing split', split, 'using', per_image)
        df = pd.read_parquet(per_image) if per_image.suffix == '.parquet' else pd.read_csv(per_image)

        if 'fused_embedding_idx' not in df.columns:
            print(f'skip {split}: per-image table lacks fused_embedding_idx')
            continue

        # geometry
        geom = np.zeros((len(df), len(geom_cols)), dtype=np.float32)
        for j, c in enumerate(geom_cols):
            if c in df.columns:
                col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
                geom[:, j] = (col.values / 1000.0)

        idxs = pd.to_numeric(df['fused_embedding_idx'], errors='coerce').fillna(-1).astype(int).values
        valid = (idxs >= 0) & (idxs < len(fused))
        fdim = fused.shape[1]
        fused_feats = np.zeros((len(df), fdim + geom.shape[1]), dtype=np.float32)
        fused_feats[valid, :fdim] = fused[idxs[valid]]
        fused_feats[:, fdim:] = geom

        # group by poster id
        poster_col = 'poster_id' if 'poster_id' in df.columns else ('poster_row_idx' if 'poster_row_idx' in df.columns else None)
        if poster_col is None:
            print(f'skip {split}: no poster id column')
            continue
        df['_orig_idx'] = np.arange(len(df))
        groups = df.groupby(poster_col)

        poster_ids = []
        seqs = []
        counts = []
        max_elems = 64
        for pid, g in groups:
            if 'element_index' in g.columns:
                g = g.sort_values('element_index')
            seq = fused_feats[g['_orig_idx'].values]
            if seq.shape[0] >= max_elems:
                seq = seq[:max_elems]
            else:
                pad = np.zeros((max_elems - seq.shape[0], seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])
            seqs.append(seq)
            counts.append(int(g.shape[0]))
            poster_ids.append(pid)

        Xf = np.stack(seqs, axis=0).astype(np.float32)
        maskf = (np.arange(max_elems)[None, :] < np.array(counts)[:, None]).astype(np.uint8)

        out_Xf = base / f'poster_inputs_{split}_fused_X.npy'
        out_maskf = base / f'poster_inputs_{split}_fused_mask.npy'
        out_idxf = base / f'poster_inputs_{split}_fused_index.csv'
        np.save(out_Xf, Xf)
        np.save(out_maskf, maskf)
        with open(out_idxf, 'w') as f:
            w = csv.writer(f)
            w.writerow(['poster_id', 'num_elements'])
            for pid, c in zip(poster_ids, counts):
                w.writerow([pid, c])
        print('Wrote fused poster inputs to', out_Xf, out_maskf, out_idxf)


if __name__ == '__main__':
    main()
