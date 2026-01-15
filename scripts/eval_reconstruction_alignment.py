#!/usr/bin/env python3
"""
Evaluate alignment between reconstructed image-slices from the masked-element model
and a CLIP image embedding bank.

Outputs a CSV with per-query cosine and rank, and prints aggregate P@1/P@5 and mean cosine.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pandas as pd


def l2_normalize(a, axis=1, eps=1e-12):
    norms = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norms + eps)


def load_model(ckpt_path, input_dim, max_len):
    class Model(nn.Module):
        def __init__(self, input_dim, model_dim=512, n_layers=4, n_heads=8, max_len=64, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, model_dim)
            self.mask_token = nn.Parameter(torch.randn(model_dim) * 0.02)
            self.pos_emb = nn.Parameter(torch.randn(max_len, model_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim*4, dropout=dropout, activation='gelu')
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = nn.Linear(model_dim, input_dim)

        def forward(self, x, valid_mask=None, mask_positions=None):
            b, l, d = x.shape
            h = self.input_proj(x)
            if mask_positions is not None:
                mask_tok = self.mask_token.unsqueeze(0).unsqueeze(0).expand(b, l, -1)
                h = torch.where(mask_positions.unsqueeze(-1), mask_tok, h)
            h = h + self.pos_emb.unsqueeze(0)
            h = h.transpose(0, 1)
            key_padding_mask = None
            if valid_mask is not None:
                key_padding_mask = ~ (valid_mask.bool())
            out = self.transformer(h, src_key_padding_mask=key_padding_mask)
            out = out.transpose(0, 1)
            return self.output_proj(out)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = Model(input_dim=input_dim, max_len=max_len)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-x', required=True)
    p.add_argument('--test-mask', required=True)
    p.add_argument('--ckpt', default='models/masked_element_model.best.pt')
    p.add_argument('--clip-bank', default='data/crello/crello_element_image_embeddings.npy')
    p.add_argument('--num-queries', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-dim', type=int, default=512, help='dimension of CLIP image vectors in the element vector prefix')
    p.add_argument('--out-dir', default='outputs/alignment_clip')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.test_x)
    M = np.load(args.test_mask)
    N, L, D = X.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, input_dim=D, max_len=L).to(device).eval()

    clip_bank = np.load(args.clip_bank)
    clip_bank_norm = l2_normalize(clip_bank, axis=1)

    rng = np.random.RandomState(args.seed)

    # build list of candidate masked positions across dataset
    mask_prob = 0.15
    mask_list = []  # tuples (poster_idx, pos)
    for i in range(N):
        valid = M[i]
        mask_rng = rng.random_sample((L,))
        mask_positions = (mask_rng < mask_prob) & (valid == 1)
        for p in np.where(mask_positions)[0]:
            mask_list.append((i, int(p)))

    if len(mask_list) == 0:
        print('No masked positions found; increase mask_prob or check mask file')
        return

    # sample queries
    rng = np.random.RandomState(args.seed)
    if len(mask_list) > args.num_queries:
        sel = rng.choice(len(mask_list), size=args.num_queries, replace=False)
        mask_list = [mask_list[i] for i in sel]

    rows = []

    for qi, (poster_idx, pos) in enumerate(mask_list):
        x_orig = X[poster_idx:poster_idx+1].astype('float32')
        valid_mask = M[poster_idx:poster_idx+1]
        mask_positions = np.zeros((1, L), dtype=bool)
        mask_positions[0, pos] = True

        xb = torch.from_numpy(x_orig).to(device)
        vm = torch.from_numpy(valid_mask).to(device)
        mp = torch.from_numpy(mask_positions).to(device)

        with torch.no_grad():
            recon_masked = model(xb, valid_mask=vm, mask_positions=mp).cpu().numpy()[0]

        recon_img = recon_masked[pos, :args.image_dim]
        true_img = x_orig[0, pos, :args.image_dim]

        # normalize
        rnorm = recon_img / (np.linalg.norm(recon_img) + 1e-12)
        tnorm = true_img / (np.linalg.norm(true_img) + 1e-12)
        cos = float(np.dot(rnorm, tnorm))

        sims = clip_bank_norm @ rnorm
        # rank of true image among bank: based on sim to true_img
        sim_true = float(np.dot(clip_bank_norm @ tnorm, np.ones(clip_bank_norm.shape[0]))[0]) if False else float(np.dot(rnorm, tnorm))
        # compute rank: number of bank entries with higher similarity than sim_true
        rank = int((sims > sim_true).sum()) + 1
        p1 = 1 if rank == 1 else 0
        p5 = 1 if rank <= 5 else 0

        rows.append({'poster_idx': int(poster_idx), 'pos': int(pos), 'cos_with_true': cos, 'rank_in_clip_bank': rank, 'p_at_1': p1, 'p_at_5': p5})

        if qi % 50 == 0:
            print(f'Processed {qi+1}/{len(mask_list)} queries')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, 'alignment_summary.csv')
    df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)

    print('Aggregate:')
    mean_cos = df['cos_with_true'].mean()
    p1 = df['p_at_1'].mean()
    p5 = df['p_at_5'].mean()
    print(f'  mean cosine: {mean_cos:.4f}')
    print(f'  P@1: {p1:.4f}, P@5: {p5:.4f}')


if __name__ == '__main__':
    main()
