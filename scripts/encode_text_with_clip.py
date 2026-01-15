#!/usr/bin/env python3
"""
Encode element text using CLIP text encoder and attach embeddings to elements parquet.

Saves:
 - data/crello/crello_text_embeddings.npy
 - updates data/crello/crello_train_elements.parquet with a `text_embedding` column (list per row)
"""
import os
import sys
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd


def encode_texts_with_clip(texts: List[str], model, tokenizer, device: str, batch_size: int = 32, subset: Optional[int] = None) -> np.ndarray:
    """Encode a list of texts to CLIP text features (L2-normalized).

    Returns a numpy array of shape (len(texts), dim).
    """
    all_embs = []
    N = len(texts) if subset is None else min(len(texts), subset)
    for i in range(0, N, batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with __import__('torch').no_grad():
            feats = model.get_text_features(**enc)
        feats = feats.cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-12)
        all_embs.append(feats.astype(np.float32))
    embs = np.vstack(all_embs)
    return embs


def process_parquet(elems_path: str, tokenizer, model, device: str, batch_size: int, subset: Optional[int], out_npy_list: List[np.ndarray], start_idx: int, save_parquet: bool = True) -> int:
    """Load parquet, encode texts, append embeddings to out_npy_list, update parquet file with indices.

    Returns the next start_idx (for global indexing into combined bank).
    """
    if not os.path.exists(elems_path):
        print('Warning: parquet not found:', elems_path)
        return start_idx
    print('Loading elements from', elems_path)
    df = pd.read_parquet(elems_path)
    texts = df['element_text'].fillna('').astype(str).tolist() if 'element_text' in df.columns else [''] * len(df)
    embs = encode_texts_with_clip(texts, model, tokenizer, device, batch_size=batch_size, subset=subset)
    print('Encoded text embeddings shape for', os.path.basename(elems_path), embs.shape)

    # record indices into the combined bank
    n = embs.shape[0]
    indices = list(range(start_idx, start_idx + n))
    # If subset was used, only assign to the first n rows and warn
    if subset is not None and n < len(df):
        print(f'Note: subset used ({subset}); only updating first {n} rows of parquet {os.path.basename(elems_path)}')
        df.iloc[:n, df.columns.get_loc('text_embedding_idx') if 'text_embedding_idx' in df.columns else len(df.columns)] = np.nan
        df.iloc[:n, df.columns.get_loc('text_embedding_idx') if 'text_embedding_idx' in df.columns else len(df.columns)-1] = indices
        # assign embeddings via iloc to avoid index alignment issues
        df.iloc[:n, df.columns.get_loc('text_embedding') if 'text_embedding' in df.columns else len(df.columns)-1] = [row.tolist() for row in embs]
    else:
        df['text_embedding_idx'] = indices
        # store embeddings as lists for compatibility (optional)
        df['text_embedding'] = [row.tolist() for row in embs]

    if save_parquet:
        backup = os.path.join(os.path.dirname(elems_path), os.path.basename(elems_path) + '.bak')
        try:
            print('Backing up original parquet to', backup)
            df.to_parquet(backup, index=False)
        except Exception as e:
            print('Warning: failed to write backup parquet:', e)
        print('Saving updated elements parquet with text_embedding to', elems_path)
        df.to_parquet(elems_path, index=False)

    out_npy_list.append(embs)
    return start_idx + n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--elems-path', type=str, default=None, help='Path to a single elements parquet to encode')
    p.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test', 'all'], help='Which split to encode (default test)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--subset', type=int, default=None, help='Optional subset to encode for smoke tests')
    p.add_argument('--out-npy', type=str, default=None, help='Output .npy file for combined embeddings (default data/crello/crello_text_embeddings.npy)')
    p.add_argument('--no-save-parquet', action='store_true', help='Do not overwrite parquet files (just write embeddings .npy)')
    args = p.parse_args()

    try:
        import torch
        from transformers import CLIPTokenizer, CLIPModel
    except Exception as e:
        raise RuntimeError('transformers and torch are required: ' + str(e))

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))
    default_out = os.path.join(base_dir, 'crello_text_embeddings.npy')
    out_npy = args.out_npy or default_out

    model_name = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    splits = []
    if args.elems_path:
        splits = [args.elems_path]
    else:
        if args.split == 'all':
            splits = [
                os.path.join(base_dir, 'crello_train_elements.parquet'),
                os.path.join(base_dir, 'crello_validation_elements.parquet'),
                os.path.join(base_dir, 'crello_test_elements.parquet'),
            ]
        else:
            splits = [os.path.join(base_dir, f'crello_{args.split}_elements.parquet')]

    out_npy_list: List[np.ndarray] = []
    start_idx = 0
    for path in splits:
        start_idx = process_parquet(path, tokenizer=tokenizer, model=model, device=('cuda' if torch.cuda.is_available() else 'cpu'), batch_size=args.batch_size, subset=args.subset, out_npy_list=out_npy_list, start_idx=start_idx, save_parquet=not args.no_save_parquet)

    if out_npy_list:
        combined = np.vstack(out_npy_list)
        os.makedirs(os.path.dirname(out_npy), exist_ok=True)
        np.save(out_npy, combined)
        print('Saved combined text embeddings to', out_npy, 'shape', combined.shape)
    else:
        print('No embeddings were produced (no valid parquet files found)')


if __name__ == '__main__':
    main()
