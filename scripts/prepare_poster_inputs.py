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
import argparse
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
    parser = argparse.ArgumentParser(description='Prepare per-poster inputs from per-image elements table')
    parser.add_argument('--input', type=str, default=None, help='Explicit input per-image parquet/csv path (overrides default search)')
    parser.add_argument('--out-prefix', type=str, default=None, help='Output filename prefix within data/crello (e.g. poster_inputs_train)')
    args = parser.parse_args()

    # base_dir should be the project's data/crello directory relative to this script
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))
    os.makedirs(base_dir, exist_ok=True)

    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError('Provided input path not found: ' + args.input)
        print('Loading per-image table from explicit input:', args.input)
        # allow parquet or csv
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        out_prefix = args.out_prefix
    else:
        df = load_table(base_dir)
        out_prefix = args.out_prefix

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

    # geometry features split into position, size, angle, opacity
    # position: left, top  (normalized by 1000)
    # size: width, height  (normalized by 1000)
    # angle: normalized by 360
    # opacity: kept in [0,1]
    position_cols = ['left', 'top']
    size_cols = ['width', 'height']
    angle_col = ['angle']
    opacity_col = ['opacity']

    pos = np.zeros((len(df), len(position_cols)), dtype=np.float32)
    size = np.zeros((len(df), len(size_cols)), dtype=np.float32)
    angle = np.zeros((len(df), len(angle_col)), dtype=np.float32)
    opacity = np.zeros((len(df), len(opacity_col)), dtype=np.float32)

    for j, c in enumerate(position_cols):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
            pos[:, j] = (col.values / 1000.0)

    for j, c in enumerate(size_cols):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
            size[:, j] = (col.values / 1000.0)

    for j, c in enumerate(angle_col):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
            # normalize angle degrees to [0,1]
            angle[:, j] = (col.values / 360.0)

    for j, c in enumerate(opacity_col):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)
            # clamp opacity to [0,1]
            op = col.values
            op = np.clip(op, 0.0, 1.0)
            opacity[:, j] = op

    # font: prefer numeric font index column. Map original font ids to a dense 0..K-1
    # and save `font_idx` into the dataframe plus a `font_vocab.json` for reproducibility.
    if 'font' in df.columns:
        # attempt to interpret as integers (existing dataset stores font as index)
        raw = pd.to_numeric(df['font'], errors='coerce').fillna(0).astype(int).tolist()
        unique_vals = sorted(set(raw))
        # create dense mapping old_id -> dense_idx
        dense_map = {v: i for i, v in enumerate(unique_vals)}
        font_idx_arr = np.array([dense_map.get(v, 0) for v in raw], dtype=np.int32)
        df['font_idx'] = font_idx_arr
        # write vocab mapping (list of original ids in dense order)
        try:
            vocab_path = os.path.join(base_dir, 'font_vocab.json')
            with open(vocab_path, 'w', encoding='utf8') as f:
                json.dump(unique_vals, f, ensure_ascii=False, indent=2)
            print('Wrote font vocab to', vocab_path)
        except Exception as e:
            print('Warning: could not write font_vocab.json:', e)
    else:
        # no font column: create a font_idx column of zeros
        df['font_idx'] = np.zeros((len(df),), dtype=np.int32)
        unique_vals = [0]

    # element type handling: prefer an integer 'type' column if present
    # expected semantic codes (example): 0=vector shape, 1=text, 2=image, 3=pure color, 4=image_no_background
    if 'type' in df.columns:
        raw_t = pd.to_numeric(df['type'], errors='coerce').fillna(-1).astype(int).tolist()
        unique_type_vals = sorted(set(raw_t))
        # create dense mapping old_type -> dense_idx
        dense_type_map = {v: i for i, v in enumerate(unique_type_vals)}
        type_idx_arr = np.array([dense_type_map.get(v, 0) for v in raw_t], dtype=np.int32)
        df['type_idx'] = type_idx_arr
        # write type vocab mapping for reproducibility; include a small hint mapping when common codes present
        try:
            type_vocab_path = os.path.join(base_dir, 'type_vocab.json')
            hint_map = {}
            # add known hints if codes present
            known = {0: 'vector_shape', 1: 'text', 2: 'image', 3: 'pure_color', 4: 'image_no_background'}
            for code in unique_type_vals:
                if code in known:
                    hint_map[str(code)] = known[code]
            with open(type_vocab_path, 'w', encoding='utf8') as f:
                json.dump({'unique_vals': unique_type_vals, 'hints': hint_map}, f, ensure_ascii=False, indent=2)
            print('Wrote type vocab to', type_vocab_path)
        except Exception as e:
            print('Warning: could not write type_vocab.json:', e)
    else:
        df['type_idx'] = np.zeros((len(df),), dtype=np.int32)
        unique_type_vals = [0]

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

    # build per-element feature vector: [img(512) | text(text_dim) | pos(2) | size(2) | angle(1) | opacity(1)]
    feat_dim = img_feats.shape[1] + text_embs.shape[1] + pos.shape[1] + size.shape[1] + angle.shape[1] + opacity.shape[1]
    print('Per-element feature dim:', feat_dim)
    elem_feats = np.concatenate([img_feats, text_embs, pos, size, angle, opacity], axis=1)

    # group by poster and build fixed-length sequences
    df['_orig_idx'] = np.arange(len(df))
    group_cols = [poster_col]
    groups = df.groupby(group_cols)
    poster_ids = []
    counts = []
    max_elems = 64

    num_posters = len(groups)
    X = np.zeros((num_posters, max_elems, feat_dim), dtype=np.float32)
    MASK = np.zeros((num_posters, max_elems), dtype=np.uint8)
    FONT_IDX = np.zeros((num_posters, max_elems), dtype=np.int32)
    TYPE_IDX = np.zeros((num_posters, max_elems), dtype=np.int32)

    i = 0
    for pid, g in groups:
        # sort by element_index then image_pos if present
        sort_cols = []
        if 'element_index' in g.columns:
            sort_cols.append('element_index')
        if 'image_pos' in g.columns:
            sort_cols.append('image_pos')
        if sort_cols:
            g = g.sort_values(by=sort_cols)

        idxs = g['_orig_idx'].values.astype(int)
        seq = elem_feats[idxs]
        n = seq.shape[0]
        take = min(n, max_elems)
        if take > 0:
            X[i, :take] = seq[:take]
            MASK[i, :take] = 1
            # copy font indices for this poster's elements
            font_seq = df['font_idx'].values[idxs]
            FONT_IDX[i, :take] = font_seq[:take]
            # copy type indices for this poster's elements
            type_seq = df['type_idx'].values[idxs]
            TYPE_IDX[i, :take] = type_seq[:take]

        # store poster id and counts
        poster_ids.append(pid)
        counts.append(int(n))
        i += 1

    # trim arrays in case some groups were filtered (shouldn't happen) but keep safe
    if i != num_posters:
        X = X[:i]
        MASK = MASK[:i]
        poster_ids = poster_ids[:i]
        counts = counts[:i]

    # save outputs
    # choose output filenames; if out_prefix provided, use it to avoid overwrites for splits
    if out_prefix:
        out_X = os.path.join(base_dir, f'{out_prefix}_X.npy')
        out_mask = os.path.join(base_dir, f'{out_prefix}_mask.npy')
        out_font = os.path.join(base_dir, f'{out_prefix}_font_idx.npy')
        out_type = os.path.join(base_dir, f'{out_prefix}_type_idx.npy')
        out_index = os.path.join(base_dir, f'{out_prefix}_index.csv')
        schema_p = os.path.join(base_dir, f'{out_prefix}_schema.json')
    else:
        out_X = os.path.join(base_dir, 'poster_inputs_X.npy')
        out_mask = os.path.join(base_dir, 'poster_inputs_mask.npy')
        out_font = os.path.join(base_dir, 'poster_inputs_font_idx.npy')
        out_type = os.path.join(base_dir, 'poster_inputs_type_idx.npy')
        out_index = os.path.join(base_dir, 'poster_inputs_index.csv')
        schema_p = os.path.join(base_dir, 'poster_inputs_schema.json')
    np.save(out_X, X)
    np.save(out_mask, MASK)
    np.save(out_font, FONT_IDX)
    np.save(out_type, TYPE_IDX)
    df_index = pd.DataFrame({'poster_id': poster_ids, 'num_elements': counts})
    df_index.to_csv(out_index, index=False)
    print('Wrote poster inputs:', out_X, out_mask, out_font, out_index)

    # write schema to help downstream slicing
    schema = {}
    fields = [('image', img_dim), ('text', text_embs.shape[1]), ('pos', pos.shape[1]), ('size', size.shape[1]), ('angle', angle.shape[1]), ('opacity', opacity.shape[1])]
    offsets = {}
    cur = 0
    for name, d in fields:
        offsets[name] = [cur, cur + d]
        cur += d
    schema['fields'] = [{ 'name': n, 'dim': d, 'offset': offsets[n] } for n, d in fields]
    schema['feat_dim'] = feat_dim
    schema['max_elems'] = max_elems
    # font metadata: path and number of unique font ids
    schema['font'] = { 'path': os.path.basename(out_font), 'num_fonts': len(unique_vals) }
    # type metadata: path and number of unique type ids
    schema['type'] = { 'path': os.path.basename(out_type), 'num_types': len(unique_type_vals) }
    try:
        with open(schema_p, 'w', encoding='utf8') as f:
            json.dump(schema, f, indent=2)
        print('Wrote schema to', schema_p)
    except Exception as e:
        print('Warning: failed to write schema:', e)

if __name__ == '__main__':
    main()
