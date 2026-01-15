#!/usr/bin/env python3
"""
Evaluate the trained masked-element model on a test poster_inputs set.

Usage:
  python3 scripts/eval_on_test.py --test-x path/to/poster_inputs_test_X.npy --test-mask path/to/poster_inputs_test_mask.npy

Outputs:
  - prints overall test MSE on masked positions
  - writes outputs/test_eval_per_poster.csv with per-poster masked MSE and masked count
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test-x', required=True)
    p.add_argument('--test-mask', required=True)
    p.add_argument('--ckpt', default='models/masked_element_model.best.pt')
    p.add_argument('--batch', type=int, default=64)
    return p.parse_args()


def load_model(ckpt_path, input_dim, max_len):
    # model definition must match train script
    import torch.nn as nn
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


def main():
    args = parse_args()
    assert os.path.exists(args.test_x) and os.path.exists(args.test_mask), 'test files missing'
    X = np.load(args.test_x, mmap_mode='r')
    M = np.load(args.test_mask, mmap_mode='r')
    N, L, D = X.shape
    print('Test set:', N, 'seq_len:', L, 'feat_dim:', D)

    model = load_model(args.ckpt, input_dim=D, max_len=L)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    rows = []
    total_sq = 0.0
    total_count = 0

    with torch.no_grad():
        for i in range(0, N, args.batch):
            xb = torch.from_numpy(X[i:i+args.batch].astype('float32')).to(device)
            maskb = torch.from_numpy(M[i:i+args.batch].astype('uint8')).to(device)
            # create random mask positions on valid positions
            rand = torch.rand_like(maskb.float())
            mask_positions = (rand < 0.15) & (maskb == 1)
            if not mask_positions.any():
                # record zeros
                for j in range(xb.shape[0]):
                    rows.append({'idx': i+j, 'masked_count': 0, 'mse': None})
                continue
            recon = model(xb, valid_mask=maskb, mask_positions=mask_positions)
            dif = ((recon - xb)**2)[mask_positions]
            # accumulate per-sample stats
            # mask_positions shape (B,L)
            for bi in range(xb.shape[0]):
                mp = mask_positions[bi]
                cnt = int(mp.sum().item())
                if cnt == 0:
                    rows.append({'idx': i+bi, 'masked_count': 0, 'mse': None})
                else:
                    sample_dif = ((recon - xb)**2)[bi][mp]
                    mse = float(sample_dif.mean().item())
                    rows.append({'idx': i+bi, 'masked_count': cnt, 'mse': mse})
            total_sq += float(dif.sum().item())
            total_count += int(dif.numel())

    overall = (total_sq / total_count) if total_count > 0 else float('nan')
    print('Overall test MSE (masked positions):', overall)
    df = pd.DataFrame(rows)
    os.makedirs('outputs', exist_ok=True)
    outcsv = os.path.join('outputs', 'test_eval_per_poster.csv')
    df.to_csv(outcsv, index=False)
    print('Wrote per-poster CSV:', outcsv)


if __name__ == '__main__':
    main()
