#!/usr/bin/env python3
"""
Experiment with hub-reduction strategies for reconstructed image vectors.
Runs the masked-element model to get reconstructed image-part vectors, then
computes nearest neighbors using:
 - raw cosine
 - CSLS
 - PCA-1 (remove top PC) + cosine
 - PCA-1 + CSLS

Prints top-20 most frequent top-1 emb_idx for each method.

Usage:
  python3 scripts/experiment_reduce_hubs.py --test-x data/crello/poster_inputs_test_X.npy --test-mask data/crello/poster_inputs_test_mask.npy

Requires torch installed (script will run the masked model).
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def l2(a, eps=1e-12):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)


def load_model_and_recon(test_x, test_mask):
    import torch
    import torch.nn as nn

    # replicate load_model() here to avoid package import issues
    def load_model_local(ckpt_path, input_dim, max_len):
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

    X = np.load(test_x)
    M = np.load(test_mask)
    N, L, D = X.shape
    ckpt = Path('models') / 'masked_element_model.best.pt'
    assert ckpt.exists(), 'model checkpoint missing'
    model = load_model_local(str(ckpt), input_dim=D, max_len=L)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    recon_list = []
    rng = np.random.RandomState(42)
    for idx in range(N):
        x = torch.from_numpy(X[idx:idx+1].astype('float32')).to(device)
        mask_valid = M[idx:idx+1]
        mask_rng = rng.random_sample((1, L))
        mask_positions = (mask_rng < 0.15) & (mask_valid == 1)
        mask_positions = torch.from_numpy(mask_positions).to(device)
        with torch.no_grad():
            recon = model(x, valid_mask=torch.from_numpy(mask_valid).to(device), mask_positions=mask_positions)
        recon_np = recon.cpu().numpy()[0]
        mask_np = mask_positions.cpu().numpy()[0].astype(bool)
        masked_idx = np.where(mask_np)[0].tolist()
        for pos in masked_idx:
            img_part = recon_np[pos][:512]
            recon_list.append((int(idx), int(pos), img_part.astype('float32')))

    Q = np.stack([r[2] for r in recon_list], axis=0)
    return recon_list, Q


def csls_scores(Qn, Bn, k=10):
    # Qn: (q, d), Bn: (b, d)
    S = Qn.dot(Bn.T)
    # r_q: mean top-k for each query
    k = min(k, Bn.shape[0])
    topk_q = np.partition(-S, k-1, axis=1)[:, :k]
    r_q = -topk_q.mean(axis=1)
    St = S.T
    topk_db = np.partition(-St, k-1, axis=1)[:, :k]
    r_db = -topk_db.mean(axis=1)
    CS = 2 * S - r_q[:, None] - r_db[None, :]
    return CS


def top1_freq_from_scores(score_mat):
    ids = np.argmax(score_mat, axis=1)
    uniques, counts = np.unique(ids, return_counts=True)
    order = np.argsort(-counts)
    return list(zip(uniques[order][:20], counts[order][:20]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-x', required=True)
    p.add_argument('--test-mask', required=True)
    p.add_argument('--csls-k', type=int, default=10)
    args = p.parse_args()

    per_image_p = Path('data') / 'crello' / 'crello_train_elements_per_image.parquet'
    emb_img_p = Path('data') / 'crello' / 'crello_element_image_embeddings.npy'
    assert per_image_p.exists() and emb_img_p.exists(), 'per-image or emb bank missing'

    per_image = pd.read_parquet(per_image_p)
    emb = np.load(emb_img_p)
    emb_n = l2(emb)

    print('Computing reconstructions...')
    recon_list, Q = load_model_and_recon(args.test_x, args.test_mask)
    Qn = l2(Q)

    print('Baseline cosine...')
    S = Qn.dot(emb_n.T)
    base_top = top1_freq_from_scores(S)
    print('Top-20 baseline top1 (emb_idx, count):')
    print(base_top)

    print('\nCSLS...')
    CS = csls_scores(Qn, emb_n, k=args.csls_k)
    csls_top = top1_freq_from_scores(CS)
    print('Top-20 CSLS top1:')
    print(csls_top)

    print('\nPCA-1 removal (compute top PC of emb bank)')
    # compute top PC of emb
    # center
    mean = emb.mean(axis=0, keepdims=True)
    emb_center = emb - mean
    # top PC via svd on centered
    u, s, vh = np.linalg.svd(emb_center, full_matrices=False)
    pc1 = vh[0:1]
    # subtract projection
    emb_pc = emb - (emb.dot(pc1.T) * pc1)
    emb_pc_n = l2(emb_pc)
    Q_pc = Q - (Q.dot(pc1.T) * pc1)
    Q_pc_n = l2(Q_pc)

    S_pc = Q_pc_n.dot(emb_pc_n.T)
    pc_top = top1_freq_from_scores(S_pc)
    print('Top-20 PCA-1 removed top1:')
    print(pc_top)

    print('\nPCA-1 + CSLS...')
    CS_pc = csls_scores(Q_pc_n, emb_pc_n, k=args.csls_k)
    cs_pc_top = top1_freq_from_scores(CS_pc)
    print('Top-20 PCA-1+CSLS top1:')
    print(cs_pc_top)


if __name__ == '__main__':
    main()
