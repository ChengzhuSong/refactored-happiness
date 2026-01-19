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
import numpy as np
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description='Encode element text with CLIP text encoder')
    parser.add_argument('--split', type=str, default='test',
                        help='Which split to process: train, validation (or val), or test')
    parser.add_argument('--input', type=str, default=None,
                        help='Explicit input parquet path (overrides --split)')
    parser.add_argument('--output', type=str, default=None,
                        help='Explicit output .npy path (overrides default naming)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for tokenization/encoding')
    args = parser.parse_args()

    # get torch and CLIP
    try:
        import torch
        from transformers import CLIPTokenizer, CLIPModel
    except Exception as e:
        raise RuntimeError('transformers and torch are required: ' + str(e))

    # base data directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))

    # decide input parquet
    if args.input:
        elems_path = args.input
        split_norm = os.path.splitext(os.path.basename(elems_path))[0]
    else:
        split = args.split.lower()
        if split in ('val', 'validation'):
            split_norm = 'validation'
        elif split == 'train':
            split_norm = 'train'
        else:
            split_norm = 'test'
        elems_path = os.path.join(base_dir, f'crello_{split_norm}_elements.parquet')

    if not os.path.exists(elems_path):
        raise FileNotFoundError('Expected elements parquet at: ' + elems_path)

    print('Loading elements from', elems_path)
    df = pd.read_parquet(elems_path)
    # texts contains a lot of empty strings if no text, maybe should filter those out?
    texts = df['element_text'].fillna('').astype(str).tolist() if 'element_text' in df.columns else [''] * len(df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device for CLIP text encoding:', device)

    model_name = 'openai/clip-vit-base-patch32'
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    batch_size = args.batch_size
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            feats = model.get_text_features(**enc)
        feats = feats.cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / (norms + 1e-12)
        all_embs.append(feats.astype(np.float32))

    embs = np.vstack(all_embs)
    print('Encoded text embeddings shape:', embs.shape)

    # attach to dataframe as lists
    df['text_embedding'] = [row.tolist() for row in embs]

    out_parquet = elems_path
    backup = os.path.join(os.path.dirname(elems_path), os.path.basename(elems_path) + '.bak')
    try:
        print('Backing up original parquet to', backup)
        df.to_parquet(backup, index=False)
    except Exception as e:
        print('Warning: failed to write backup parquet:', e)

    print('Saving updated elements parquet with text_embedding to', out_parquet)
    df.to_parquet(out_parquet, index=False)

    # decide output npy filename
    if args.output:
        out_npy = args.output
    else:
        out_npy = os.path.join(base_dir, f'crello_text_embeddings_{split_norm}.npy')
    np.save(out_npy, embs)
    print('Saved text embeddings to', out_npy)

if __name__ == '__main__':
    main()
