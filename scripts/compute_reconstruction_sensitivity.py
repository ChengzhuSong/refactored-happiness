#!/usr/bin/env python3
"""
Compute reconstruction diagnostics:

- Masked vs Unmasked Gap: difference between MSE when the model reconstructs a masked
  position vs when the model is run without masking for the same positions.
- Random Context Baseline: MSE when the context (all other valid positions) is replaced
  by random element vectors sampled from the embedding bank.
- Element Ablation Sensitivity: for each masked query, ablate (zero) up to K context
  elements one-at-a-time and measure increase in MSE; report mean and top deltas.

Writes outputs to `outputs/sensitivity/` as CSVs and prints a short summary.

Designed to run on a sampled subset (default 500 queries) to keep runtime reasonable.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def l2_normalize(a, axis=1, eps=1e-12):
    norms = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norms + eps)


def compute_attention_average(model: nn.Module, x_np: np.ndarray, valid_mask: np.ndarray, device: torch.device) -> np.ndarray:
    """Return averaged attention matrix (L x L) summed across encoder layers for a single example.

    Uses model.transformer.layers and layer.self_attn to extract attention weights.
    """
    with torch.no_grad():
        h = model.input_proj(torch.from_numpy(x_np).to(device))
        pos_emb = model.pos_emb.unsqueeze(0).to(device)
        h = h + pos_emb
        h_t = h.transpose(0, 1)
        key_padding_mask = ~ (torch.from_numpy(valid_mask).to(device).bool())
        attn_acc = None
        for layer in model.transformer.layers:
            out, attn_w = layer.self_attn(h_t, h_t, h_t, key_padding_mask=key_padding_mask)
            aw = attn_w.cpu().numpy()[0]
            if attn_acc is None:
                attn_acc = aw
            else:
                attn_acc += aw
        attn_avg = attn_acc / len(model.transformer.layers)
    return attn_avg


def compute_attention_rollout(model: nn.Module, x_np: np.ndarray, valid_mask: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute attention-rollout scores (Abnar & Zuidema) for one example.

    Returns a vector of length L with contributions from each input token to each output token.
    """
    with torch.no_grad():
        h = model.input_proj(torch.from_numpy(x_np).to(device))
        pos_emb = model.pos_emb.unsqueeze(0).to(device)
        h = h + pos_emb
        h_t = h.transpose(0, 1)
        key_padding_mask = ~ (torch.from_numpy(valid_mask).to(device).bool())
        attn_mats: List[np.ndarray] = []
        for layer in model.transformer.layers:
            out, attn_w = layer.self_attn(h_t, h_t, h_t, key_padding_mask=key_padding_mask)
            aw = attn_w.cpu().numpy()[0]
            attn_mats.append(aw)

        # Compose rollout: A_hat = A + I, row-normalize, then multiply across layers
        Augs = []
        for A in attn_mats:
            I = np.eye(A.shape[0], dtype=A.dtype)
            A_hat = A + I
            row_sums = A_hat.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A_hat = A_hat / row_sums
            Augs.append(A_hat)

        rollout = Augs[0]
        for r in Augs[1:]:
            rollout = r @ rollout

    return rollout


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
    p.add_argument('--use-fused', action='store_true', help='Use fused element embeddings as retrieval bank (and fused indices)')
    p.add_argument('--emb-bank', default='data/crello/crello_element_image_embeddings.npy', help='Embedding bank for random sampling')
    p.add_argument('--num-queries', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-ablate', type=int, default=8, help='Max context elements to ablate per query')
    p.add_argument('--out-dir', default='outputs/sensitivity')
    p.add_argument('--ablation-selection', type=str, default='random', choices=['random','attention','rollout'], help='How to pick context elements to ablate')
    return p.parse_args()


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.test_x)
    M = np.load(args.test_mask)
    N, L, D = X.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, input_dim=D, max_len=L).to(device).eval()

    # load embedding bank for random sampling
    emb_bank = np.load(args.emb_bank)
    emb_bank_n = l2_normalize(emb_bank, axis=1)

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

    batch_size = 16
    # Precompute normalized emb bank for random sampling
    emb_choices = emb_bank

    for qi, (poster_idx, pos) in enumerate(mask_list):
        # build input batch of size 1
        x_orig = X[poster_idx:poster_idx+1].astype('float32')
        valid_mask = M[poster_idx:poster_idx+1]
        # create mask_positions for this single pos
        mask_positions = np.zeros((1, L), dtype=bool)
        mask_positions[0, pos] = True

        xb = torch.from_numpy(x_orig).to(device)
        vm = torch.from_numpy(valid_mask).to(device)
        mp = torch.from_numpy(mask_positions).to(device)

        # recon when masked
        with torch.no_grad():
            recon_masked = model(xb, valid_mask=vm, mask_positions=mp).cpu().numpy()[0]
        mse_masked = mse(recon_masked[pos], x_orig[0, pos])

        # recon when unmasked (no mask_positions)
        with torch.no_grad():
            recon_unmasked = model(xb, valid_mask=vm, mask_positions=None).cpu().numpy()[0]
        mse_unmasked = mse(recon_unmasked[pos], x_orig[0, pos])

        # Random context baseline: replace all valid context positions (except pos) with random embeddings
        x_rand = x_orig.copy()
        # sample random vectors for each context position from emb_bank (use normalized bank if dims match)
        for j in range(L):
            if j == pos: continue
            if valid_mask[0, j] == 0: continue
            ri = rng.randint(0, emb_choices.shape[0])
            # if embedding bank dim matches element image slice, place accordingly; else try using full vector
            bvec = emb_choices[ri]
            # place bvec into prefix of the element vector (assume image-first layout)
            x_rand[0, j, :bvec.shape[0]] = bvec
        xb_rand = torch.from_numpy(x_rand).to(device)
        with torch.no_grad():
            recon_rand = model(xb_rand, valid_mask=vm, mask_positions=mp).cpu().numpy()[0]
        mse_rand = mse(recon_rand[pos], x_orig[0, pos])

        # Element ablation sensitivity: sample up to K context indices and zero them one-by-one
        context_idx = [j for j in range(L) if j != pos and valid_mask[0, j] == 1]
        if len(context_idx) == 0:
            mean_ablate_delta = 0.0
            top_ablate_delta = 0.0
        else:
            # choose context elements based on selection strategy
            if args.ablation_selection == 'attention':
                # compute average self-attention from encoder layers for this input
                attn_avg = compute_attention_average(model, x_orig, valid_mask, device)
                scores = attn_avg[pos]  # attention from pos -> all keys
                # mask out invalid positions and the position itself
                valid_bool = (valid_mask[0] == 1)
                scores = np.array(scores)
                scores[~valid_bool] = -np.inf
                scores[pos] = -np.inf
                k = min(args.max_ablate, int(valid_bool.sum()) - 1)
                if k <= 0:
                    context_sample = []
                else:
                    # pick top-k by attention score
                    context_sample = [int(i) for i in np.argsort(-scores)[:k]]
            elif args.ablation_selection == 'rollout':
                rollout = compute_attention_rollout(model, x_orig, valid_mask, device)
                scores = rollout[pos]
                valid_bool = (valid_mask[0] == 1)
                scores = np.array(scores)
                scores[~valid_bool] = -np.inf
                scores[pos] = -np.inf
                k = min(args.max_ablate, int(valid_bool.sum()) - 1)
                if k <= 0:
                    context_sample = []
                else:
                    context_sample = [int(i) for i in np.argsort(-scores)[:k]]
            else:
                # random selection (previous behavior)
                if len(context_idx) > args.max_ablate:
                    context_sample = list(rng.choice(context_idx, size=args.max_ablate, replace=False))
                else:
                    context_sample = context_idx

            deltas = []
            for cj in context_sample:
                x_ab = x_orig.copy()
                x_ab[0, cj, :] = 0.0
                xb_ab = torch.from_numpy(x_ab).to(device)
                with torch.no_grad():
                    recon_ab = model(xb_ab, valid_mask=vm, mask_positions=mp).cpu().numpy()[0]
                mse_ab = mse(recon_ab[pos], x_orig[0, pos])
                deltas.append(mse_ab - mse_masked)
            mean_ablate_delta = float(np.mean(deltas)) if deltas else 0.0
            top_ablate_delta = float(np.max(deltas)) if deltas else 0.0

        rows.append({'poster_idx': int(poster_idx), 'pos': int(pos), 'mse_masked': mse_masked, 'mse_unmasked': mse_unmasked, 'gap_masked_unmasked': mse_masked - mse_unmasked, 'mse_random_context': mse_rand, 'rand_gap': mse_rand - mse_masked, 'mean_ablate_delta': mean_ablate_delta, 'top_ablate_delta': top_ablate_delta})

        if qi % 50 == 0:
            print(f'Processed {qi+1}/{len(mask_list)} queries')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, 'sensitivity_summary.csv')
    df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)

    # print summary
    print('Summary (mean over queries):')
    print(df[['mse_masked','mse_unmasked','gap_masked_unmasked','mse_random_context','rand_gap','mean_ablate_delta','top_ablate_delta']].mean())


if __name__ == '__main__':
    main()
