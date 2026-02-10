#!/usr/bin/env python3
"""
Compute MSE between predicted text embeddings and ground-truth text embeddings on a split.
By default this uses exact-K masking (mask-count) and the same sampling RNG as
`nn_decode_texts.py` so you can compute the MSE for the same masked positions.

Example:
  PYTHONPATH=. python3 scripts/compute_mse_test.py --model-ckpt checkpoints/scratch/best_epoch.pth \
      --split test --base-dir data/crello --mask-count 1 --seed 123 --batch-size 64 --device cpu

Output: prints MSE (mean squared error per embedding dimension) and number of entries used.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.two_stage_transformer import AttributeStage, ElementStage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--base-dir', default='data/crello')
    p.add_argument('--mask-count', type=int, default=1, help='exact K masked slots per poster')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    tokenizer_order = ckpt.get('tokenizer_order', ['image','text','pos','size','angle','opacity','font'])

    # data paths
    prefix = f"poster_input_{args.split}"
    base = args.base_dir
    Xp = os.path.join(base, f"{prefix}_X.npy")
    Mp = os.path.join(base, f"{prefix}_mask.npy")
    Fp = os.path.join(base, f"{prefix}_font_idx.npy")
    Tp = os.path.join(base, f"{prefix}_type_idx.npy")
    Sp = os.path.join(base, f"{prefix}_schema.json")

    X_all = np.load(Xp, mmap_mode='r')
    M_all = np.load(Mp, mmap_mode='r')
    FONT_all = np.load(Fp, mmap_mode='r')
    TYPE_all = np.load(Tp, mmap_mode='r')
    with open(Sp, 'r') as f:
        schema = json.load(f)
    fields = {f['name']: f for f in schema['fields']}
    S = X_all.shape[1]

    # robust num_fonts like earlier
    num_fonts = schema.get('font', {}).get('num_fonts', int(FONT_all.max() + 1))
    if 'attr_stage' in ckpt and 'tokenizer.font_emb.weight' in ckpt['attr_stage']:
        num_fonts = int(ckpt['attr_stage']['tokenizer.font_emb.weight'].shape[0])

    # model components
    d_attr = 128
    D_elem = 256
    num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))
    attr_stage = AttributeStage(img_dim=schema['fields'][0]['dim'], txt_dim=fields['text']['dim'], d_attr=d_attr, D_elem=D_elem, num_fonts=num_fonts)
    elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(schema['fields']) + 1)
    decoders = {name: nn.Linear(D_elem, fields[name]['dim']) for name in [f['name'] for f in schema['fields']]}

    # load weights
    attr_stage.load_state_dict(ckpt['attr_stage'])
    elem_stage.load_state_dict(ckpt['elem_stage'])
    if 'decoders' in ckpt:
        for n in decoders:
            if n in ckpt['decoders']:
                decoders[n].load_state_dict(ckpt['decoders'][n])

    attr_stage.to(device).eval()
    elem_stage.to(device).eval()
    for d in decoders.values():
        d.to(device).eval()

    tstart, tend = fields['text']['offset']

    N = X_all.shape[0]
    bs = args.batch_size

    total_sse = 0.0
    n_dims = 0

    for i in range(0, N, bs):
        batch_idx = list(range(i, min(i+bs, N)))
        Xb = torch.from_numpy(np.array(X_all[batch_idx])).float().to(device)
        MASKb = torch.from_numpy(np.array(M_all[batch_idx])).to(device)
        FONTb = torch.from_numpy(np.array(FONT_all[batch_idx])).long().to(device)
        TYPEb = torch.from_numpy(np.array(TYPE_all[batch_idx])).long().to(device)

        img = Xb[:, :, fields['image']['offset'][0] : fields['image']['offset'][1]]
        text = Xb[:, :, fields['text']['offset'][0] : fields['text']['offset'][1]]
        pos = Xb[:, :, fields['pos']['offset'][0] : fields['pos']['offset'][1]]
        size = Xb[:, :, fields['size']['offset'][0] : fields['size']['offset'][1]]
        angle = Xb[:, :, fields['angle']['offset'][0] : fields['angle']['offset'][1]]
        opacity = Xb[:, :, fields['opacity']['offset'][0] : fields['opacity']['offset'][1]]

        valid_mask = (MASKb == 1)
        text_present = (FONTb != 0)

        # sampling: exact-K per poster from valid & text_present
        k = int(args.mask_count)
        sampled = torch.zeros((len(batch_idx), S), dtype=torch.bool, device=device)
        for mbi in range(len(batch_idx)):
            valid_idxs = torch.nonzero(valid_mask[mbi] & text_present[mbi], as_tuple=False).view(-1)
            n_valid = valid_idxs.numel()
            if n_valid == 0:
                continue
            choose_k = min(k, int(n_valid))
            g = torch.Generator(device=device)
            g.manual_seed(int(args.seed) + i + mbi)
            perm = torch.randperm(n_valid, generator=g, device=device)
            sel = valid_idxs[perm[:choose_k]]
            sampled[mbi, sel] = True

        slot_attr_mask = torch.zeros((len(batch_idx), S, len(tokenizer_order)), dtype=torch.bool, device=device)
        masked_attr_id = torch.zeros((len(batch_idx), S), dtype=torch.long, device=device)
        if 'text' in tokenizer_order:
            tok_idx = tokenizer_order.index('text')
            slot_attr_mask[:, :, tok_idx] = sampled
            masked_attr_id[sampled] = tok_idx + 1

        with torch.no_grad():
            elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=slot_attr_mask)
            ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=valid_mask, masked_attr_id=masked_attr_id)
            pred_text_emb = decoders['text'](ctx)  # (B, S, D)

        B = len(batch_idx)
        D = pred_text_emb.shape[-1]

        # accumulate SSE
        for mbi, bi in enumerate(batch_idx):
            for s in range(S):
                if not sampled[mbi, s]:
                    continue
                gt_vec = X_all[bi, s, tstart:tend].astype(np.float32)
                if np.all(gt_vec == 0):
                    continue
                pred_vec = pred_text_emb[mbi, s].cpu().numpy().astype(np.float32)
                diff = pred_vec - gt_vec
                total_sse += (diff * diff).sum()
                n_dims += D

    if n_dims > 0:
        mse = total_sse / n_dims
        print(f'MSE (per-dimension mean squared error): {mse}')
        print(f'Number of vectors compared: {n_dims // D}, embedding dim: {D}')
    else:
        print('No entries compared (no masked positions matched)')


if __name__ == '__main__':
    main()
