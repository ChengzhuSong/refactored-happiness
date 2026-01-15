#!/usr/bin/env python3
"""
Train a Transformer-based element attribute encoder that fuses per-element
attributes (image embedding, text embedding, geometry) into a single vector.

The model is trained as an autoencoder: project each modality to tokens,
run a Transformer encoder, pool to a single vector, and reconstruct the
original concatenated element vector with an MSE loss.

Output:
 - data/crello/crello_element_fused_embeddings.npy (num_elements x model_dim)
 - data/crello/crello_train_elements_per_image.parquet updated with fused index column

Usage (smoke):
  python3 scripts/train_element_attribute_encoder.py --epochs 1 --subset 2048 --batch 256
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def deterministic_text_embed(texts, dim=64, seed=42):
    rng = np.random.RandomState(seed)
    proj = rng.normal(scale=0.1, size=(100000, dim)).astype(np.float32)
    embeds = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        toks = str(t).split()
        acc = np.zeros((dim,), dtype=np.float32)
        for tok in toks:
            h = abs(hash(tok)) % 100000
            acc += proj[h]
        if len(toks) > 0:
            acc /= float(len(toks))
        n = np.linalg.norm(acc)
        if n > 0:
            acc = acc / n
        embeds[i] = acc
    return embeds


class ElementDataset(Dataset):
    def __init__(self, per_image_parquet, emb_npy, subset=None):
        self.df = pd.read_parquet(per_image_parquet).reset_index(drop=True)
        self.emb = np.load(emb_npy) if Path(emb_npy).exists() else None
        if subset is not None:
            self.df = self.df.iloc[:subset].reset_index(drop=True)
        # prepare image features
        img_dim = self.emb.shape[1] if self.emb is not None else 512
        self.img_feats = np.zeros((len(self.df), img_dim), dtype=np.float32)
        if 'image_embedding' in self.df.columns:
            for i, v in enumerate(self.df['image_embedding'].tolist()):
                try:
                    arr = np.array(v, dtype=np.float32)
                    if arr.size == img_dim:
                        n = np.linalg.norm(arr)
                        if n > 0:
                            arr = arr / n
                        self.img_feats[i] = arr
                except Exception:
                    pass
        elif 'image_embedding_idx' in self.df.columns and self.emb is not None:
            idxs = pd.to_numeric(self.df['image_embedding_idx'], errors='coerce').fillna(-1).astype(int).values
            valid = (idxs >= 0) & (idxs < len(self.emb))
            self.img_feats[valid] = self.emb[idxs[valid]]

        # text features
        if 'text_embedding' in self.df.columns:
            text_list = self.df['text_embedding'].tolist()
            first = next((x for x in text_list if x is not None and str(x) != 'nan'), None)
            if first is not None:
                td = len(first)
                self.text_feats = np.zeros((len(self.df), td), dtype=np.float32)
                for i, x in enumerate(text_list):
                    try:
                        if x is None or str(x) == 'nan':
                            continue
                        arr = np.array(x, dtype=np.float32)
                        n = np.linalg.norm(arr)
                        if n > 0:
                            arr = arr / n
                        self.text_feats[i, :arr.shape[0]] = arr
                    except Exception:
                        pass
            else:
                self.text_feats = deterministic_text_embed(self.df.get('element_text', [''] * len(self.df)), dim=64)
        elif 'element_text' in self.df.columns:
            self.text_feats = deterministic_text_embed(self.df['element_text'].fillna('').tolist(), dim=64)
        else:
            self.text_feats = np.zeros((len(self.df), 64), dtype=np.float32)

        # geometry
        geom_cols = ['left', 'top', 'width', 'height', 'angle', 'opacity']
        geom = np.zeros((len(self.df), len(geom_cols)), dtype=np.float32)
        for j, c in enumerate(geom_cols):
            if c in self.df.columns:
                col = pd.to_numeric(self.df[c], errors='coerce').fillna(0).astype(np.float32)
                geom[:, j] = (col.values / 1000.0)
        self.geom = geom

        # final concatenated target (image|text|geom)
        self.targets = np.concatenate([self.img_feats, self.text_feats, self.geom], axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'img': self.img_feats[idx],
            'text': self.text_feats[idx],
            'geom': self.geom[idx],
            'target': self.targets[idx],
            'row_idx': int(self.df.iloc[idx].name)
        }


class AttributeFusionModel(nn.Module):
    def __init__(self, img_dim=512, text_dim=64, geom_dim=6, model_dim=512, n_layers=4, n_heads=8):
        super().__init__()
        self.model_dim = model_dim
        # modality projections to token space
        self.img_proj = nn.Linear(img_dim, model_dim)
        self.text_proj = nn.Linear(text_dim, model_dim)
        self.geom_proj = nn.Linear(geom_dim, model_dim)
        # optional learned cls token
        self.cls = nn.Parameter(torch.randn(model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # pooling -> fused embedding
        self.pool_proj = nn.Linear(model_dim, model_dim)
        # decoder: fused -> reconstruct concatenated target dim
        self.decoder = nn.Linear(model_dim, img_dim + text_dim + geom_dim)

    def forward(self, img, text, geom):
        # inputs: (B, img_dim), (B, text_dim), (B, geom_dim)
        b = img.shape[0]
        t_img = self.img_proj(img).unsqueeze(1)  # (B,1,M)
        t_text = self.text_proj(text).unsqueeze(1)
        t_geom = self.geom_proj(geom).unsqueeze(1)
        # tokens: [cls, img, text, geom]
        cls_tok = self.cls.unsqueeze(0).unsqueeze(1).expand(b, 1, self.model_dim)
        tokens = torch.cat([cls_tok, t_img, t_text, t_geom], dim=1)  # (B,4,M)
        # transformer expects (S,B,E) if using default; we'll use (S,B,E)
        h = tokens.transpose(0, 1)
        h = self.transformer(h)
        h = h.transpose(0, 1)
        pooled = h[:, 0, :]  # CLS pooling
        fused = self.pool_proj(pooled)
        recon = self.decoder(fused)
        return fused, recon


def train(args):
    base = Path('data') / 'crello'
    # default per-image path for training (train split)
    per_image_p = base / 'crello_train_elements_per_image.parquet'
    emb_img_p = base / 'crello_element_image_embeddings.npy'
    assert per_image_p.exists(), 'per-image parquet missing'
    if not emb_img_p.exists():
        print('Warning: image embedding bank missing, image tokens will be zeros')

    ds = ElementDataset(str(per_image_p), str(emb_img_p), subset=args.subset)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dim = ds.img_feats.shape[1]
    text_dim = ds.text_feats.shape[1]
    geom_dim = ds.geom.shape[1]
    model = AttributeFusionModel(img_dim=img_dim, text_dim=text_dim, geom_dim=geom_dim, model_dim=args.model_dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    # If encode-only, skip training and load checkpoint
    if args.encode_only:
        ckpt_p = Path('models') / 'element_attribute_encoder.best.pt'
        if not ckpt_p.exists():
            raise RuntimeError('encode-only requested but checkpoint not found at ' + str(ckpt_p))
        ck = torch.load(str(ckpt_p), map_location='cpu')
        model.load_state_dict(ck['model_state'])
        print('Loaded checkpoint for encode-only from', ckpt_p)
    else:
        for epoch in range(args.epochs):
            model.train()
            running = []
            for b in dl:
                # DataLoader default collate may already convert numpy arrays to tensors.
                img = b['img']
                if not torch.is_tensor(img):
                    img = torch.from_numpy(img)
                img = img.to(device)
                text = b['text']
                if not torch.is_tensor(text):
                    text = torch.from_numpy(text)
                text = text.to(device)
                geom = b['geom']
                if not torch.is_tensor(geom):
                    geom = torch.from_numpy(geom)
                geom = geom.to(device)
                target = b['target']
                if not torch.is_tensor(target):
                    target = torch.from_numpy(target)
                target = target.to(device)
                fused, recon = model(img, text, geom)
                loss = criterion(recon, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running.append(float(loss.item()))
        epoch_loss = float(np.mean(running)) if running else 0.0
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.6f}')
        # save best
        ck = {'model_state': model.state_dict(), 'epoch': epoch+1}
        os.makedirs('models', exist_ok=True)
        torch.save(ck, 'models/element_attribute_encoder.pt')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ck, 'models/element_attribute_encoder.best.pt')

    # after training or loading checkpoint, encode specified splits and update per-image parquets
    model.eval()
    out_p = base / 'crello_element_fused_embeddings.npy'
    existing_fused = None
    if out_p.exists():
        existing_fused = np.load(out_p)
        next_idx = existing_fused.shape[0]
        fused_list = [existing_fused]
        print('Loaded existing fused embeddings, count=', next_idx)
    else:
        fused_list = []
        next_idx = 0

    splits = [s.strip() for s in args.splits.split(',')] if args.splits else ['train']
    with torch.no_grad():
        for split in splits:
            per_image_path = base / f'crello_{split}_elements_per_image.parquet'
            per_image_csv = base / f'crello_{split}_elements_per_image.csv'
            if per_image_path.exists():
                per_image_file = per_image_path
            elif per_image_csv.exists():
                per_image_file = per_image_csv
            else:
                print('Skipping split', split, ': per-image table not found')
                continue
            print('Encoding split', split, 'from', per_image_file)
            per_df = pd.read_parquet(per_image_file) if str(per_image_file).endswith('.parquet') else pd.read_csv(per_image_file)
            if 'fused_embedding_idx' in per_df.columns and not args.force:
                print('Split', split, 'already has fused_embedding_idx; skipping (use --force to overwrite)')
                continue

            ds_full = ElementDataset(str(per_image_file), str(emb_img_p), subset=None)
            dl_full = DataLoader(ds_full, batch_size=args.batch, shuffle=False, num_workers=4)
            split_fused = []
            row_order = []
            for b in dl_full:
                img = b['img']
                if not torch.is_tensor(img):
                    img = torch.from_numpy(img)
                img = img.to(device)
                text = b['text']
                if not torch.is_tensor(text):
                    text = torch.from_numpy(text)
                text = text.to(device)
                geom = b['geom']
                if not torch.is_tensor(geom):
                    geom = torch.from_numpy(geom)
                geom = geom.to(device)
                fused_vec, _ = model(img, text, geom)
                split_fused.append(fused_vec.cpu().numpy())
                row_order.extend(b['row_idx'])
            if split_fused:
                split_fused = np.concatenate(split_fused, axis=0)
            else:
                split_fused = np.zeros((len(per_df), args.model_dim), dtype=np.float32)

            # append to fused_list and update per_df with indices
            fused_list.append(split_fused)
            start = next_idx
            indices = np.arange(start, start + len(per_df)).astype(int)
            per_df = per_df.reset_index(drop=True)
            per_df['fused_embedding_idx'] = indices
            # write back
            if str(per_image_file).endswith('.parquet'):
                per_df.to_parquet(per_image_file)
            else:
                per_df.to_csv(per_image_file, index=False)
            print('Updated', per_image_file, 'with fused_embedding_idx for', len(per_df), 'rows')
            next_idx += len(per_df)

    if fused_list:
        fused_all = np.concatenate(fused_list, axis=0)
        np.save(out_p, fused_all)
        print('Wrote fused embeddings to', out_p, 'total count=', fused_all.shape[0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--subset', type=int, default=None)
    p.add_argument('--model-dim', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--encode-only', action='store_true', help='Load saved checkpoint and only run encoding for specified splits')
    p.add_argument('--splits', type=str, default='train', help='Comma-separated splits to encode: train,validation,test')
    p.add_argument('--force', action='store_true', help='Overwrite existing fused_embedding_idx in per-image tables')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
