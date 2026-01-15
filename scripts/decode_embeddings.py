#!/usr/bin/env python3
"""
Decode a bank of CLIP image embeddings into pixel thumbnails using a trained decoder.

Usage:
  python3 scripts/decode_embeddings.py --ckpt models/image_decoder.best.pt --outdir outputs/decoded_bank --limit 512
"""
import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from scripts.train_image_decoder import Decoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default='models/image_decoder.best.pt')
    p.add_argument('--emb', type=str, default='data/crello/crello_element_image_embeddings.npy')
    p.add_argument('--outdir', type=str, default='outputs/decoded_bank')
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    emb = np.load(args.emb)
    if args.limit:
        emb = emb[:args.limit]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Decoder(emb_dim=emb.shape[1]).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    os.makedirs(args.outdir, exist_ok=True)
    with torch.no_grad():
        batch = 64
        for i in range(0, emb.shape[0], batch):
            z = torch.tensor(emb[i:i+batch], dtype=torch.float32, device=device)
            out = model(z).cpu()
            for j in range(out.shape[0]):
                im = (out[j].permute(1,2,0).numpy() * 255).astype('uint8')
                Image.fromarray(im).save(os.path.join(args.outdir, f'emb_{i+j:06d}.png'))


if __name__ == '__main__':
    main()
