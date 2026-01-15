#!/usr/bin/env python3
"""
Prepare per-poster transformer inputs from the per-image elements table.

Saves:
 - data/crello/poster_inputs_X.npy  (float32: num_posters x max_elems x feature_dim)
 - data/crello/poster_inputs_mask.npy (uint8: num_posters x max_elems)
 - data/crello/poster_inputs_index.csv (poster_id,num_elements)

This script is intentionally lightweight: it uses only numpy/pandas and a deterministic
hash-based text projection as a fallback for text embeddings so it runs without HF heavy deps.
"""
import os
import sys
import json
import math
import numpy as np
import pandas as pd


def deterministic_text_embed(texts, dim=64, seed=42):
    """Create a deterministic lightweight text embedding for a list of texts.
    Each token is hashed and projected by a fixed random matrix (seeded) and summed.
    """
    rng = np.random.RandomState(seed)
    proj = rng.normal(scale=0.1, size=(100000, dim)).astype(np.float32)
    embeds = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        toks = str(t).split()
        acc = np.zeros((dim,), dtype=np.float32)
        for tok in toks:
            h = abs(hash(tok)) % 100000
            acc += proj[h]
        if len(toks) > 0:
            acc /= float(len(toks))
        # L2 normalize
        n = np.linalg.norm(acc)
        if n > 0:
            acc = acc / n
        embeds[i] = acc
    return embeds


def load_table(base_dir):
    p_parquet = os.path.join(base_dir, 'crello_validation_elements_per_image.parquet')
    p_csv = os.path.join(base_dir, 'crello_validation_elements_per_image.csv')
    if os.path.exists(p_parquet):
        print('Loading per-image table from parquet:', p_parquet)
        df = pd.read_parquet(p_parquet)
        return df
    if os.path.exists(p_csv):
        print('Loading per-image table from csv:', p_csv)
        df = pd.read_csv(p_csv)
        return df

    # fallback: attempt to rebuild per-image table from crello_validation_elements.parquet
    fallback = os.path.join(base_dir, 'crello_validation_elements.parquet')
    if os.path.exists(fallback):
        print('Per-image table not found; rebuilding from elements parquet:', fallback)
        elements = pd.read_parquet(fallback)
        # expect column 'image_embedding_idxs' which holds lists of indices
        if 'image_embedding_idxs' not in elements.columns:
            raise FileNotFoundError('Cannot rebuild per-image table: elements parquet missing image_embedding_idxs')
        rows = []
        for i, row in elements.iterrows():
            idxs = row['image_embedding_idxs']
            if not isinstance(idxs, (list, tuple)):
                # try to parse JSON-like string
                try:
                    idxs = json.loads(idxs)
                except Exception:
                    idxs = []
            for pos, emb_idx in enumerate(idxs):
                r = row.to_dict()
                r['image_embedding_idx'] = int(emb_idx)
                r['image_pos'] = int(pos)
                # remove the list column to avoid duplication
                rows.append(r)
        if not rows:
            raise FileNotFoundError('Rebuilt per-image table would be empty')
        df = pd.DataFrame(rows)
        return df

    raise FileNotFoundError('No per-image table found (parquet/csv) and cannot rebuild from elements parquet in ' + base_dir)


def main():
    # base_dir should be the project's data/crello directory relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))
    os.makedirs(base_dir, exist_ok=True)

    df = load_table(base_dir)

    # ensure poster id column
    poster_col = 'poster_id' if 'poster_id' in df.columns else ('poster_row_idx' if 'poster_row_idx' in df.columns else None)
    if poster_col is None:
        raise RuntimeError('No poster identifier column found in table')

    # image embedding: prefer explicit column 'image_embedding' (list) else load embeddings by index
    emb_arr = None
    if 'image_embedding' in df.columns:
        print('Using in-row image_embedding column')
        # convert list column to numpy array when needed later
        pass
    else:
        emb_path = os.path.join(base_dir, 'crello_element_image_embeddings.npy')
        if os.path.exists(emb_path):
            print('Loading embeddings array from', emb_path)
            emb_arr = np.load(emb_path)
            print('Embeddings shape:', emb_arr.shape)
        else:
            print('No external embeddings file found and no in-row embeddings; image features will be zeros')

    # text embeddings: prefer precomputed 'text_embedding' column, else fallback to deterministic or element_text
    if 'text_embedding' in df.columns:
        print("Using precomputed 'text_embedding' column")
        # convert list-like column into ndarray
        text_list = df['text_embedding'].tolist()
        try:
            text_embs = np.vstack([np.array(x, dtype=np.float32) if (x is not None and str(x) != 'nan') else np.zeros((64,), dtype=np.float32) for x in text_list])
        except Exception:
            # fallback: deterministic embedding
            print('Warning: failed to parse text_embedding; falling back to deterministic text embed')
            text_embs = deterministic_text_embed(df['element_text'].fillna('').tolist(), dim=64)
    elif 'element_text' in df.columns:
        print('Computing lightweight deterministic text embeddings (dim=64)')
        text_embs = deterministic_text_embed(df['element_text'].fillna('').tolist(), dim=64)
    else:
        text_embs = np.zeros((len(df), 64), dtype=np.float32)

    # geometry features: left, top, width, height, angle, opacity
    geom_cols = ['left', 'top', 'width', 'height', 'angle', 'opacity']
    geom = np.zeros((len(df), len(geom_cols)), dtype=np.float32)
    for j, c in enumerate(geom_cols):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
            # naive normalization: divide by 1000 to keep numbers in reasonable range
            geom[:, j] = (col.values / 1000.0)

    # image embeddings per row
    img_dim = 512
    img_feats = np.zeros((len(df), img_dim), dtype=np.float32)
    if 'image_embedding' in df.columns:
        for i, v in enumerate(df['image_embedding'].tolist()):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            try:
                arr = np.array(v, dtype=np.float32)
                if arr.ndim == 1 and arr.size == img_dim:
                    # ensure normalized
                    n = np.linalg.norm(arr)
                    if n > 0:
                        arr = arr / n
                    img_feats[i] = arr
            except Exception:
                continue
    elif 'image_embedding_idx' in df.columns or 'image_embedding_idx' in df.columns:
        if emb_arr is not None:
            idxs = pd.to_numeric(df['image_embedding_idx'], errors='coerce').fillna(-1).astype(int).values
            valid = (idxs >= 0) & (idxs < len(emb_arr))
            img_feats[valid] = emb_arr[idxs[valid]]
        else:
            pass
    elif 'image_embedding_idx' in df.columns:
        # redundant check; handled above
        pass

    # build per-element feature vector: [img(512) | text(64) | geom(6)]
    feat_dim = img_feats.shape[1] + text_embs.shape[1] + geom.shape[1]
    print('Per-element feature dim:', feat_dim)
    elem_feats = np.concatenate([img_feats, text_embs, geom], axis=1)

    # group by poster
    df['_orig_idx'] = np.arange(len(df))
    group_cols = [poster_col]
    groups = df.groupby(group_cols)
    poster_ids = []
    seqs = []
    counts = []
    max_elems = 64
    for pid, g in groups:
        # sort by element_index then image_pos if present
        sort_cols = []
        if 'element_index' in g.columns:
            sort_cols.append('element_index')
        if 'image_pos' in g.columns:
            sort_cols.append('image_pos')
        if 'text_embedding' in df.columns:
            print("Using precomputed 'text_embedding' column from per-image table")
            text_list = df['text_embedding'].tolist()
            try:
                text_embs = np.vstack([np.array(x, dtype=np.float32) if (x is not None and str(x) != 'nan') else np.zeros((64,), dtype=np.float32) for x in text_list])
            except Exception:
                print('Warning: failed to parse text_embedding in per-image table; will try to merge from elements parquet')
                df = df.drop(columns=['text_embedding'], errors='ignore')

        # If per-image table lacks text_embedding, try to merge from crello_validation_elements.parquet
        if 'text_embedding' not in df.columns:
            elems_path = os.path.join(base_dir, 'crello_validation_elements.parquet')
            if os.path.exists(elems_path):
                print('Merging text_embedding from', elems_path)
                elems = pd.read_parquet(elems_path)
                # Determine join keys: prefer poster_row_idx + element_index, else poster_id + element_index
                if 'poster_row_idx' in df.columns and 'element_index' in df.columns and 'poster_row_idx' in elems.columns:
                    join_keys = ['poster_row_idx', 'element_index']
                elif 'poster_id' in df.columns and 'element_index' in df.columns and 'poster_id' in elems.columns:
                    join_keys = ['poster_id', 'element_index']
                else:
                    join_keys = None

                if join_keys is not None and 'text_embedding' in elems.columns:
                    elems_small = elems[join_keys + ['text_embedding']].copy()
                    # ensure element_index type matches
                    if 'element_index' in join_keys:
                        elems_small['element_index'] = pd.to_numeric(elems_small['element_index'], errors='coerce')
                        df['element_index'] = pd.to_numeric(df['element_index'], errors='coerce')
                    df = df.merge(elems_small, on=join_keys, how='left', suffixes=('', '_from_elems'))
                    # if merge succeeded, use the merged column
                    if 'text_embedding' in df.columns and df['text_embedding'].isnull().all():
                        # if merged column is empty, try the fallback
                        df = df.drop(columns=['text_embedding'], errors='ignore')
                    else:
                        print('Merged text_embedding into per-image table')
                else:
                    print('Could not find suitable join keys or text_embedding column in elements parquet; will fallback to deterministic embedding')

        # After possible merge, if we now have text_embedding, convert it
        if 'text_embedding' in df.columns:
            print("Converting 'text_embedding' column to ndarray")
            text_list = df['text_embedding'].tolist()
            # infer dim from first non-null
            dim = None
            for x in text_list:
                if x is not None and str(x) != 'nan':
                    dim = len(x)
                    break
            if dim is None:
                print('No valid text embeddings found after merge; using deterministic fallback')
                text_embs = deterministic_text_embed(df.get('element_text', pd.Series([''] * len(df))).fillna('').tolist(), dim=64)
            else:
                text_embs = np.zeros((len(df), dim), dtype=np.float32)
                for i, x in enumerate(text_list):
                    try:
                        if x is None or str(x) == 'nan':
                            continue
                        arr = np.array(x, dtype=np.float32)
                        # L2-normalize if not already
                        n = np.linalg.norm(arr)
                        if n > 0:
                            arr = arr / n
                        text_embs[i, :arr.shape[0]] = arr
                    except Exception:
                        continue
        elif 'element_text' in df.columns:
            print('Computing lightweight deterministic text embeddings (dim=64)')
            text_embs = deterministic_text_embed(df['element_text'].fillna('').tolist(), dim=64)
        else:
            text_embs = np.zeros((len(df), 64), dtype=np.float32)

if __name__ == '__main__':
    main()
