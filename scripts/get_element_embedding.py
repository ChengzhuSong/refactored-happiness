#!/usr/bin/env python3
"""Load prepared poster inputs and extract an element embedding via a small transformer.

Usage examples:
  python3 scripts/get_element_embedding.py --X data/crello/poster_inputs_X.npy \
      --font data/crello/poster_inputs_font_idx.npy --mask data/crello/poster_inputs_mask.npy \
      --schema data/crello/poster_inputs_schema.json --poster 0 --slot 0

The script also accepts an --out argument to save the embedding as a .npy file.
"""
import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print('This script requires PyTorch. Install with pip install torch')
    raise


class ElementEncoder(nn.Module):
    """Project per-attribute slices of a concatenated element vector into model_dim.

    Combines image/text/pos/size/angle/opacity projections plus a font embedding.
    Sums projected attributes, adds slot positional embeddings and optional mask token.
    """
    def __init__(self, schema: Dict, model_dim: int = 256, font_emb_dim: int = 32, use_type_embeddings: bool = True, dropout: float = 0.0):
        super().__init__()
        # parse schema
        # schema['fields'] is list of {name, dim, offset: [start, end]}
        self.model_dim = model_dim
        self.field_info = {}
        for f in schema['fields']:
            name = f['name']
            off = f['offset']
            self.field_info[name] = (off[0], off[1])

        # per-field projections
        self.projs = nn.ModuleDict()
        for name, (s, e) in self.field_info.items():
            dim = e - s
            self.projs[name] = nn.Linear(dim, model_dim)

        # font embedding + projection
        num_fonts = int(schema.get('font', {}).get('num_fonts', 1))
        self.font_emb = nn.Embedding(max(1, num_fonts), font_emb_dim)
        self.font_proj = nn.Linear(font_emb_dim, model_dim)

        # optional small learned type embeddings
        self.use_type_embeddings = use_type_embeddings
        if use_type_embeddings:
            self.type_embs = nn.ParameterDict()
            for name in list(self.field_info.keys()) + ['font']:
                self.type_embs[name] = nn.Parameter(torch.zeros(model_dim), requires_grad=True)

        # mask token for absent slots
        self.mask_token = nn.Parameter(torch.zeros(model_dim), requires_grad=True)

        # slot positional embeddings
        self.max_elems = int(schema.get('max_elems', 64))
        self.slot_pos_emb = nn.Embedding(self.max_elems, model_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, font_idx: torch.LongTensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # X: (B, L, feat_dim)
        B, L, _ = X.shape
        device = X.device
        out = torch.zeros((B, L, self.model_dim), device=device, dtype=X.dtype)

        for name, (s, e) in self.field_info.items():
            slice_ = X[:, :, s:e]
            proj = self.projs[name](slice_)
            out = out + proj
            if self.use_type_embeddings:
                out = out + self.type_embs[name].view(1, 1, -1)

        # font
        if font_idx is None:
            font_idx = torch.zeros((B, L), dtype=torch.long, device=device)
        font_vec = self.font_emb(font_idx)
        out = out + self.font_proj(font_vec)
        if self.use_type_embeddings:
            out = out + self.type_embs['font'].view(1, 1, -1)

        # slot pos
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        out = out + self.slot_pos_emb(pos_ids)

        # mask token for absent slots
        if mask is not None:
            absent = (mask == 0)
            if absent.any():
                out[absent] = self.mask_token.view(1, 1, -1).expand_as(out[absent])

        out = self.dropout(out)
        return out


def build_transformer(model_dim: int = 256, nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 1024, dropout: float = 0.1) -> nn.Module:
    layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def load_schema(path: str) -> Dict:
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--X', type=str, required=False, help='Path to poster_inputs_X.npy')
    p.add_argument('--font', type=str, required=False, help='Path to poster_inputs_font_idx.npy')
    p.add_argument('--mask', type=str, required=False, help='Path to poster_inputs_mask.npy')
    p.add_argument('--schema', type=str, required=False, help='Path to poster_inputs_schema.json')
    p.add_argument('--poster', type=int, default=0, help='Poster index to inspect')
    p.add_argument('--slot', type=int, default=0, help='Slot index within poster')
    p.add_argument('--model-dim', type=int, default=256)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--out', type=str, default=None, help='Optional path to save embedding (.npy)')
    p.add_argument('--use-random', action='store_true', help='If set, run with random inputs instead of loading files (useful for smoke tests)')
    args = p.parse_args()

    if args.use_random:
        # small random smoke test
        B, L = 2, 8
        feat_dim = 512 + 64 + 2 + 2 + 1 + 1
        X = np.random.randn(B, L, feat_dim).astype(np.float32)
        FONT = np.zeros((B, L), dtype=np.int32)
        MASK = np.zeros((B, L), dtype=np.uint8)
        MASK[:, :4] = 1
        schema = {
            'fields': [
                {'name': 'image', 'dim': 512, 'offset': [0, 512]},
                {'name': 'text', 'dim': 64, 'offset': [512, 576]},
                {'name': 'pos', 'dim': 2, 'offset': [576, 578]},
                {'name': 'size', 'dim': 2, 'offset': [578, 580]},
                {'name': 'angle', 'dim': 1, 'offset': [580, 581]},
                {'name': 'opacity', 'dim': 1, 'offset': [581, 582]},
            ],
            'feat_dim': feat_dim,
            'max_elems': L,
            'font': {'path': '', 'num_fonts': 1}
        }
    else:
        # require files
        if not args.X or not args.mask or not args.schema:
            print('When not using --use-random you must provide --X, --mask and --schema (font optional).')
            sys.exit(2)

        if not os.path.exists(args.X):
            print('X file not found:', args.X)
            sys.exit(2)
        X = np.load(args.X)
        MASK = np.load(args.mask)
        schema = load_schema(args.schema)

        if args.font and os.path.exists(args.font):
            FONT = np.load(args.font)
        else:
            # if schema contains font.path in same folder, try that
            font_path = schema.get('font', {}).get('path')
            if font_path:
                maybe = os.path.join(os.path.dirname(args.schema), font_path)
                if os.path.exists(maybe):
                    FONT = np.load(maybe)
                else:
                    FONT = np.zeros((X.shape[0], X.shape[1]), dtype=np.int32)
            else:
                FONT = np.zeros((X.shape[0], X.shape[1]), dtype=np.int32)

    # convert to torch
    device = torch.device('cpu')
    X_t = torch.from_numpy(X).to(device)
    FONT_t = torch.from_numpy(FONT).long().to(device)
    MASK_t = torch.from_numpy(MASK).to(device)

    model_dim = args.model_dim
    enc = ElementEncoder(schema, model_dim=model_dim)
    transformer = build_transformer(model_dim=model_dim, num_layers=args.num_layers)

    enc.to(device)
    transformer.to(device)

    # encode
    seq = enc(X_t, font_idx=FONT_t, mask=MASK_t)
    # transformer expects (S, B, E)
    src_key_padding_mask = (MASK_t == 0)  # True for padding
    seq_t = seq.permute(1, 0, 2)
    out_t = transformer(seq_t, src_key_padding_mask=src_key_padding_mask)
    out = out_t.permute(1, 0, 2)

    pidx = int(args.poster)
    sidx = int(args.slot)
    if pidx < 0 or pidx >= out.shape[0]:
        print('Poster index out of range:', pidx)
        sys.exit(2)
    if sidx < 0 or sidx >= out.shape[1]:
        print('Slot index out of range:', sidx)
        sys.exit(2)

    emb = out[pidx, sidx].cpu().numpy()
    print('Extracted embedding shape:', emb.shape)
    if args.out:
        np.save(args.out, emb)
        print('Saved embedding to', args.out)


if __name__ == '__main__':
    main()
