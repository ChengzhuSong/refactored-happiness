#!/usr/bin/env python3
"""
Map reconstructed element vectors back to nearest images/text in the dataset.

This script:
 - Loads a checkpoint and poster inputs (or uses the demo sample),
 - Reconstructs masked element vectors for a small set of posters,
 - Splits each reconstructed vector into image/text/geom parts (infers dims: image=512, geom=6),
 - Finds nearest image embeddings in `data/crello/crello_element_image_embeddings.npy` (cosine similarity),
 - Looks up metadata (poster_id, element_text, image path) from `crello_train_elements_per_image.parquet` and
   prints the top-k nearest candidates for each reconstructed element.

Run: python3 scripts/map_recon_to_asset.py
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path


def l2_normalize(a, axis=1, eps=1e-12):
    norms = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norms + eps)


def load_data():
    base = Path(__file__).resolve().parents[1] / 'data' / 'crello'
    Xp = base / 'poster_inputs_X.npy'
    Maskp = base / 'poster_inputs_mask.npy'
    per_image_p = base / 'crello_train_elements_per_image.parquet'
    emb_img_p = base / 'crello_element_image_embeddings.npy'

    assert Xp.exists() and Maskp.exists(), 'poster inputs missing'
    X = np.load(Xp)
    MASK = np.load(Maskp)

    per_image = None
    if per_image_p.exists():
        per_image = pd.read_parquet(per_image_p)

    emb_img = None
    if emb_img_p.exists():
        emb_img = np.load(emb_img_p)

    return X, MASK, per_image, emb_img


def reconstruct_sample(checkpoint_path=None, sample_indices=None, mask_prob=0.15):
    # load trained model checkpoint and run reconstruction on selected posters
    base = Path(__file__).resolve().parents[0]
    X, MASK, per_image, emb_img = load_data()
    N, L, D = X.shape

    # build model identically to training
    class Model(torch.nn.Module):
        def __init__(self, input_dim, model_dim=512, n_layers=4, n_heads=8, max_len=64):
            super().__init__()
            self.input_proj = torch.nn.Linear(input_dim, model_dim)
            self.mask_token = torch.nn.Parameter(torch.randn(model_dim) * 0.02)
            self.pos_emb = torch.nn.Parameter(torch.randn(max_len, model_dim) * 0.02)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim*4, dropout=0.1, activation='gelu')
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = torch.nn.Linear(model_dim, input_dim)

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
            recon = self.output_proj(out)
            return recon

    # locate checkpoint
    if checkpoint_path is None:
        ck1 = Path('models') / 'masked_element_model.best.pt'
        ck2 = Path('models') / 'masked_element_model.pt'
        checkpoint_path = ck1 if ck1.exists() else (ck2 if ck2.exists() else None)
    assert checkpoint_path is not None and Path(checkpoint_path).exists(), 'checkpoint not found'
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    model = Model(input_dim=D, model_dim=512, n_layers=4, n_heads=8, max_len=L)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if sample_indices is None:
        sample_indices = [0, min(10, N-1), N-1]

    recon_results = []
    for idx in sample_indices:
        x = torch.from_numpy(X[idx:idx+1]).to(device)
        valid = torch.from_numpy(MASK[idx:idx+1]).to(device)
        rng = torch.rand(valid.shape, device=device)
        mask_pos = (rng < mask_prob) & (valid == 1)
        with torch.no_grad():
            recon = model(x, valid_mask=valid, mask_positions=mask_pos)
        recon_np = recon.cpu().numpy()[0]
        mask_np = mask_pos.cpu().numpy()[0].astype(bool)
        orig_np = X[idx]
        recon_results.append({'poster_idx': idx, 'orig': orig_np, 'recon': recon_np, 'mask': mask_np})

    return recon_results, per_image, emb_img


def find_nearest_images(recon_img_vec, emb_img, topk=5):
    # emb_img: (M, 512), recon_img_vec: (512,)
    if emb_img is None:
        return []
    # ensure normalized
    emb_img_n = l2_normalize(emb_img, axis=1)
    q = recon_img_vec / (np.linalg.norm(recon_img_vec) + 1e-12)
    sims = emb_img_n.dot(q)
    ids = np.argsort(-sims)[:topk]
    return list(zip(ids, sims[ids]))


def main():
    recon_results, per_image, emb_img = reconstruct_sample()

    # infer split dims
    example = recon_results[0]['orig']
    D = example.shape[1]
    image_dim = 512
    geom_dim = 6
    text_dim = D - image_dim - geom_dim
    print('Inferred dims -> image:', image_dim, 'text:', text_dim, 'geom:', geom_dim)

    for res in recon_results:
        idx = res['poster_idx']
        mask_idx = np.where(res['mask'])[0]
        print('\nPoster', idx, 'masked positions:', mask_idx.tolist())
        for pos in mask_idx:
            recon_vec = res['recon'][pos]
            img_part = recon_vec[:image_dim]
            text_part = recon_vec[image_dim:image_dim+text_dim]
            geom_part = recon_vec[-geom_dim:]

            print(f' Position {pos}: geom={geom_part.tolist()[:6]}')
            # nearest image embeddings
            nn = find_nearest_images(img_part, emb_img, topk=5)
            if len(nn) == 0:
                print('  No image embeddings available to match.')
                continue
            print('  Top image matches (emb_idx, sim):')
            for emb_idx, sim in nn:
                info = None
                if per_image is not None:
                    # try to find rows that reference this embedding index
                    if 'image_embedding_idx' in per_image.columns:
                        rows = per_image[per_image['image_embedding_idx'] == int(emb_idx)]
                    else:
                        rows = per_image.iloc[[int(emb_idx)]] if emb_idx < len(per_image) else pd.DataFrame()
                    if len(rows) > 0:
                        # show first matching row's useful columns if present
                        r = rows.iloc[0]
                        pid = r.get('poster_id', r.get('poster_row_idx', 'N/A'))
                        et = r.get('element_text', None)
                        imgpath = r.get('image', r.get('image_path', None))
                        info = {'poster': pid, 'text': et, 'image': imgpath}
                print('   ', emb_idx, float(sim), info)


if __name__ == '__main__':
    main()
