#!/usr/bin/env python3
"""
Generate a human-friendly HTML + CSV report for masked-element reconstruction.

This script:
 - Loads the trained masked-element model checkpoint,
 - Reconstructs masked elements for a handful of posters,
 - Finds nearest image embeddings and nearest text embeddings (CLIP) for each reconstructed element,
 - Writes `outputs/reconstruction_report.html` and `outputs/reconstruction_report.csv` with the results.

Run: python3 scripts/generate_reconstruction_report.py
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import html


def l2_normalize(a, axis=1, eps=1e-12):
    norms = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / (norms + eps)


def load_assets():
    base = Path(__file__).resolve().parents[1] / 'data' / 'crello'
    Xp = base / 'poster_inputs_X.npy'
    Maskp = base / 'poster_inputs_mask.npy'
    per_image_p = base / 'crello_train_elements_per_image.parquet'
    elems_p = base / 'crello_train_elements.parquet'
    emb_img_p = base / 'crello_element_image_embeddings.npy'
    emb_text_p = base / 'crello_text_embeddings.npy'

    assert Xp.exists() and Maskp.exists(), 'poster inputs missing'
    X = np.load(Xp)
    MASK = np.load(Maskp)

    per_image = pd.read_parquet(per_image_p) if per_image_p.exists() else None
    elems = pd.read_parquet(elems_p) if elems_p.exists() else None
    emb_img = np.load(emb_img_p) if emb_img_p.exists() else None
    emb_text = np.load(emb_text_p) if emb_text_p.exists() else None

    return X, MASK, per_image, elems, emb_img, emb_text


def load_model_checkpoint(ckpt_path=None, input_dim=None, max_len=64):
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

    if ckpt_path is None:
        c1 = Path('models') / 'masked_element_model.best.pt'
        c2 = Path('models') / 'masked_element_model.pt'
        ckpt_path = c1 if c1.exists() else (c2 if c2.exists() else None)
    assert ckpt_path is not None and Path(ckpt_path).exists(), 'checkpoint not found'
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    if input_dim is None:
        raise ValueError('input_dim required')
    model = Model(input_dim=input_dim, model_dim=512, n_layers=4, n_heads=8, max_len=max_len)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def reconstruct_and_map(sample_idxs=None, topk=5, mask_prob=0.15):
    X, MASK, per_image, elems, emb_img, emb_text = load_assets()
    N, L, D = X.shape

    if sample_idxs is None:
        sample_idxs = [0, min(10, N-1), N-1]
    sample_idxs = sorted(set(sample_idxs))

    model = load_model_checkpoint(input_dim=D, max_len=L)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    results = []

    # prepare text bank if available (elements parquet with text_embedding)
    text_bank = None
    text_texts = None
    if emb_text is not None and elems is not None:
        text_texts = elems['element_text'].fillna('').astype(str).tolist()
        # normalize bank
        text_bank = l2_normalize(emb_text, axis=1)

    # normalize image bank
    emb_img_n = l2_normalize(emb_img, axis=1) if emb_img is not None else None

    for idx in sample_idxs:
        x = torch.from_numpy(X[idx:idx+1]).to(device)
        valid = torch.from_numpy(MASK[idx:idx+1]).to(device)
        rng = torch.rand(valid.shape, device=device)
        mask_pos = (rng < mask_prob) & (valid == 1)
        with torch.no_grad():
            recon = model(x, valid_mask=valid, mask_positions=mask_pos)
        recon_np = recon.cpu().numpy()[0]
        mask_np = mask_pos.cpu().numpy()[0].astype(bool)
        orig_np = X[idx]

        masked_idx = np.where(mask_np)[0].tolist()
        for pos in masked_idx:
            recon_vec = recon_np[pos]
            # split parts
            image_dim = 512
            geom_dim = 6
            text_dim = D - image_dim - geom_dim
            img_part = recon_vec[:image_dim]
            text_part = recon_vec[image_dim:image_dim+text_dim]
            geom_part = recon_vec[-geom_dim:]

            # nearest images
            img_matches = []
            if emb_img_n is not None:
                q = img_part / (np.linalg.norm(img_part) + 1e-12)
                sims = emb_img_n.dot(q)
                ids = np.argsort(-sims)[:topk]
                img_matches = [(int(i), float(sims[i])) for i in ids]

            # nearest texts (only if dims align)
            text_matches = []
            if text_bank is not None:
                # bank dim vs query dim
                bank_dim = text_bank.shape[1]
                qdim = text_part.shape[0]
                if bank_dim == qdim:
                    q = text_part / (np.linalg.norm(text_part) + 1e-12)
                    sims = text_bank.dot(q)
                    ids = np.argsort(-sims)[:topk]
                    text_matches = [(int(i), float(sims[i]), text_texts[int(i)]) for i in ids]
                else:
                    # dimension mismatch; skip text matching
                    text_matches = []

            results.append({
                'poster_idx': int(idx),
                'pos': int(pos),
                'geom': geom_part.tolist(),
                'img_matches': img_matches,
                'text_matches': text_matches,
            })

    return results


def write_report(results, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    csv_rows = []
    html_rows = []
    for r in results:
        pid = r['poster_idx']
        pos = r['pos']
        geom = r['geom']
        # text matches
        t_html = '<ol>' + ''.join([f"<li>{html.escape(str(tm[2]))} (sim={tm[1]:.4f})</li>" for tm in r['text_matches']]) + '</ol>' if r['text_matches'] else 'N/A'
        # image matches
        img_html = '<ol>' + ''.join([f"<li>emb#{im[0]} (sim={im[1]:.4f})</li>" for im in r['img_matches']]) + '</ol>' if r['img_matches'] else 'N/A'
        html_rows.append(f"<h3>Poster {pid} - position {pos}</h3>Geom: {geom}<br/>Nearest texts:{t_html}<br/>Nearest images:{img_html}")

        # flatten CSV rows: include top-3
        for i, (im_idx, sim) in enumerate(r['img_matches'][:3]):
            tmatch = r['text_matches'][i] if i < len(r['text_matches']) else (None, None, None)
            csv_rows.append({'poster_idx': pid, 'pos': pos, 'rank': i+1, 'img_emb_idx': im_idx, 'img_sim': sim, 'text_sim': tmatch[1], 'text': tmatch[2]})

    html_doc = '<html><body><h1>Reconstruction report</h1>' + '\n<hr/>\n'.join(html_rows) + '</body></html>'
    with open(os.path.join(out_dir, 'reconstruction_report.html'), 'w', encoding='utf8') as f:
        f.write(html_doc)
    df = pd.DataFrame(csv_rows)
    df.to_csv(os.path.join(out_dir, 'reconstruction_report.csv'), index=False)
    return os.path.join(out_dir, 'reconstruction_report.html'), os.path.join(out_dir, 'reconstruction_report.csv')


def main():
    print('Reconstructing and mapping...')
    results = reconstruct_and_map()
    print('Found', len(results), 'reconstructed masked elements')
    html_path, csv_path = write_report(results)
    print('Wrote report:', html_path, csv_path)


if __name__ == '__main__':
    main()
