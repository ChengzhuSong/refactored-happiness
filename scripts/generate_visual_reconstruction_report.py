#!/usr/bin/env python3
"""
Run masked reconstruction on a dataset and produce a visual HTML report showing
poster previews, masked positions, original element thumbnails (if present), and
top-k nearest image candidates for each reconstructed element.

Usage:
  python3 scripts/generate_visual_reconstruction_report.py --test-x data/crello/poster_inputs_test_X.npy --test-mask data/crello/poster_inputs_test_mask.npy --topk 5

Outputs:
  - outputs/reconstruction_visual_report.html
  - outputs/reconstruction_visual_report.csv
"""
import os
import argparse
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from pathlib import Path


def l2_normalize(a, axis=1, eps=1e-12):
    norms = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norms + eps)


def load_model(ckpt_path: str, input_dim: int, max_len: int):
    """Load the masked-element model from checkpoint (same architecture used in training)."""
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


def build_report(test_x: str, test_mask: str, topk: int = 5, out_dir: str = 'outputs', seed: int = 42, use_csls: bool = False, csls_k: int = 10, use_fused: bool = False) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # load inputs
    X = np.load(test_x)
    M = np.load(test_mask)
    N, L, D = X.shape

    # load assets
    base = Path('data/crello')
    per_image_p = base / 'crello_train_elements_per_image.parquet'
    assert per_image_p.exists(), 'per-image table missing'
    per_image = pd.read_parquet(per_image_p)

    emb_img_p = base / ('crello_element_fused_embeddings.npy' if use_fused else 'crello_element_image_embeddings.npy')
    assert emb_img_p.exists(), 'embedding bank missing'
    emb_img = np.load(emb_img_p)
    emb_img_n = l2_normalize(emb_img, axis=1)

    # build lookup: embedding idx -> per_image rows (may be multiple)
    per_image = per_image.reset_index(drop=True)
    emb_idx_col = 'fused_embedding_idx' if use_fused else 'image_embedding_idx'
    idx_to_rows: Dict[int, List[Dict[str, Any]]] = {}
    for i, row in per_image.iterrows():
        key = int(row.get(emb_idx_col, -1))
        idx_to_rows.setdefault(key, []).append(row)

    # load model
    ckpt = Path('models') / 'masked_element_model.best.pt'
    assert ckpt.exists(), 'best checkpoint not found'
    model = load_model(str(ckpt), input_dim=D, max_len=L)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    rng = np.random.RandomState(seed)

    # First pass: run model on all posters and collect reconstructed vectors for masked positions
    recon_list = []  # will store (poster_idx, pos, recon_vec)
    rows = []
    html_parts = ['<html><head><meta charset="utf-8"><title>Reconstruction Visual Report</title></head><body>']
    html_parts.append('<h1>Reconstruction Visual Report</h1>')

    for idx in range(N):
        x = torch.from_numpy(X[idx:idx+1].astype('float32')).to(device)
        mask_valid = M[idx:idx+1]
        # deterministic random mask per poster
        mask_rng = rng.random_sample((1, L))
        mask_positions = (mask_rng < 0.15) & (mask_valid == 1)
        mask_positions = torch.from_numpy(mask_positions).to(device)
        with torch.no_grad():
            recon = model(x, valid_mask=torch.from_numpy(mask_valid).to(device), mask_positions=mask_positions)
        recon_np = recon.cpu().numpy()[0]
        mask_np = mask_positions.cpu().numpy()[0].astype(bool)
        masked_idx = np.where(mask_np)[0].tolist()
        if not masked_idx:
            continue

        # poster preview path: find first per_image row with poster_row_idx == idx
        preview_row = per_image[per_image['poster_row_idx'] == int(idx)]
        preview_path = None
        if not preview_row.empty:
            preview_path = preview_row.iloc[0].get('preview', None)
        preview_html = f'<img src="{os.path.join("..", "data", "crello", preview_path)}" style="width:320px">' if preview_path else ''
        html_parts.append(f'<h2>Poster {idx}</h2>')
        if preview_path:
            html_parts.append(preview_html)

        for pos in masked_idx:
            recon_vec = recon_np[pos]
            # determine image (retrieval) dim: for fused retrieval this is emb_img dim, otherwise 512
            image_dim = emb_img.shape[1]
            geom_dim = D - image_dim
            img_part = recon_vec[:image_dim]
            recon_list.append((int(idx), int(pos), img_part.astype('float32')))

    if not recon_list:
        # nothing to report
        out_html = os.path.join(out_dir, 'reconstruction_visual_report.html')
        with open(out_html, 'w', encoding='utf8') as f:
            f.write('\n'.join(html_parts) + '\n</body></html>')
        return out_html, os.path.join(out_dir, 'reconstruction_visual_report.csv')

    # Build matrix of reconstructions and normalize
    Q = np.stack([r[2] for r in recon_list], axis=0)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

    # emb_img_n already normalized
    # compute similarity matrix S = Qn @ emb_img_n.T
    S = Qn.dot(emb_img_n.T)
    if use_csls:
        k = int(csls_k)
        # r_q: mean of top-k similarities for each query
        topk_q = np.partition(-S, k-1, axis=1)[:, :k]
        r_q = -topk_q.mean(axis=1)
        # r_db: mean of top-k similarities for each db vector across queries
        # compute column-wise top-k
        St = S.T
        topk_db = np.partition(-St, k-1, axis=1)[:, :k]
        r_db = -topk_db.mean(axis=1)
        # CSLS score
        CS = 2 * S - r_q[:, None] - r_db[None, :]
        score_mat = CS
    else:
        score_mat = S

    # For each reconstruction (row in score_mat), pick topk indices
    topk_ids = np.argsort(-score_mat, axis=1)[:, :topk]

    # mutual-NN filtering: if enabled, require that candidate's top-m queries include this query
    # we'll expose mutual options via CLI flags in main()
    # Note: we will compute reverse top-m lists on the fly when requested (cost OK for current sizes)
    mutual_enabled = getattr(build_report, '_mutual_enabled', False)
    mutual_k = getattr(build_report, '_mutual_k', 1)
    if mutual_enabled:
        q_count, db_count = score_mat.shape
        # build reverse top-k sets for each db vector
        rev_sets = [set(np.argsort(-score_mat[:, j])[:mutual_k]) for j in range(db_count)]
        # filter each query's candidate list
        filtered_topk = []
        for qi in range(topk_ids.shape[0]):
            kept = [iid for iid in topk_ids[qi] if qi in rev_sets[int(iid)]]
            if not kept:
                # fallback to original topk for this query
                kept = list(topk_ids[qi])
            # pad/truncate to topk
            kept = kept[:topk]
            # if fewer than topk, append next best non-duplicate ids
            if len(kept) < topk:
                extras = list(np.argsort(-score_mat[qi]))
                for e in extras:
                    if e not in kept:
                        kept.append(int(e))
                    if len(kept) >= topk:
                        break
            filtered_topk.append(kept[:topk])
        topk_ids = np.array(filtered_topk, dtype=int)

    # Now assemble HTML and CSV rows using recon_list order
    for qi, (poster_idx, pos, img_part) in enumerate(recon_list):
        html_parts.append(f'<h3>Position {pos} (masked)</h3>')
        # original element image if available
        orig_row = per_image[(per_image['poster_row_idx'] == int(poster_idx)) & (per_image['element_index'] == pos)]
        if not orig_row.empty:
            orig_img = orig_row.iloc[0].get('image')
            if orig_img:
                src = os.path.join('..', 'data', 'crello', str(orig_img))
                html_parts.append(f'<div>Original element: <img src="{src}" style="width:160px"></div>')

        html_parts.append('<div style="display:flex;gap:8px;flex-wrap:wrap;">')
        ids = topk_ids[qi]
        sims = score_mat[qi, ids]
        for k_i, iid in enumerate(ids):
            sim = float(sims[k_i])
            candidate_rows = idx_to_rows.get(int(iid), [])
            img_src = None
            meta = ''
            if candidate_rows:
                r = candidate_rows[0]
                img_path = r.get('image')
                if img_path:
                    img_src = os.path.join('..', 'data', 'crello', str(img_path))
                meta = f"poster_id={r.get('poster_id')} elem_idx={r.get('element_index')}"

            decoded_src = None
            decoded_fname = f'emb_{int(iid):06d}.png'
            decoded_path = os.path.join('outputs', 'decoded_bank', decoded_fname)
            if os.path.exists(decoded_path):
                rel = os.path.relpath(decoded_path, start=out_dir)
                decoded_src = rel.replace('\\', '/')

            caption = f'#{int(iid)} sim={sim:.4f} {meta}'
            block_parts = []
            if decoded_src:
                block_parts.append(f'<div style="text-align:center"><div>Decoded</div><img src="{decoded_src}" style="width:160px"></div>')
            if img_src and os.path.exists(os.path.join('data','crello', candidate_rows[0].get('image'))):
                block_parts.append(f'<div style="text-align:center"><div>Nearest</div><img src="{img_src}" style="width:160px"></div>')
            if not block_parts:
                block_parts.append(f'<div style="width:160px;height:90px;border:1px solid #ccc;display:flex;align-items:center;justify-content:center">{caption}</div>')
            html_parts.append('<div style="margin:4px;">' + '\n'.join(block_parts) + f'<div style="font-size:12px">{caption}</div></div>')
            rows.append({'poster_idx': poster_idx, 'pos': int(pos), 'rank': k_i+1, 'emb_idx': int(iid), 'sim': sim, 'meta': meta})
        html_parts.append('</div>')

    html_parts.append('</body></html>')
    html = '\n'.join(html_parts)
    out_html = os.path.join(out_dir, 'reconstruction_visual_report.html')
    with open(out_html, 'w', encoding='utf8') as f:
        f.write(html)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, 'reconstruction_visual_report.csv')
    df.to_csv(out_csv, index=False)
    return out_html, out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test-x', required=True)
    p.add_argument('--test-mask', required=True)
    p.add_argument('--topk', type=int, default=3)
    p.add_argument('--use-csls', action='store_true', help='Use CSLS re-ranking for nearest neighbors')
    p.add_argument('--csls-k', type=int, default=10, help='k parameter for CSLS (top-k average)')
    p.add_argument('--mutual', action='store_true', help='Enforce mutual nearest neighbor filtering')
    p.add_argument('--mutual-k', type=int, default=1, help='k parameter for mutual-NN (db->queries)')
    p.add_argument('--use-fused', action='store_true', help='Use fused element embeddings (crello_element_fused_embeddings.npy) as retrieval DB')
    args = p.parse_args()
    # pass mutual options via attributes on build_report to avoid changing signature everywhere
    setattr(build_report, '_mutual_enabled', bool(args.mutual))
    setattr(build_report, '_mutual_k', int(args.mutual_k))
    html, csvp = build_report(args.test_x, args.test_mask, topk=args.topk, use_csls=args.use_csls, csls_k=args.csls_k, use_fused=args.use_fused)
    print('Wrote', html, csvp)


if __name__ == '__main__':
    main()
