#!/usr/bin/env python3
"""
Train a decoder that maps CLIP image embeddings -> RGB thumbnails (128x128).
Uses L1 pixel loss + perceptual loss (VGG features).

Usage (smoke test):
  python3 scripts/train_image_decoder.py --epochs 1 --subset 512 --batch 64

Outputs:
  - models/image_decoder.pt (latest)
  - models/image_decoder.best.pt (best val)
  - outputs/decoded_samples/ (example reconstructions)
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


class ImageEmbeddingDataset(Dataset):
    def __init__(self, per_image_parquet, emb_npy, subset=None, transform=None):
        df = pd.read_parquet(per_image_parquet)
        # Only keep rows with an image path and valid embedding idx
        df = df[df['image'].notnull()]
        self.df = df.reset_index(drop=True)
        self.emb = np.load(emb_npy)
        # l2-normalize embeddings to stabilize decoder training
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = (self.emb / norms).astype('float32')
        if subset is not None:
            self.df = self.df.iloc[:subset].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image']
        # embedding idx stored in image_embedding_idx or image_embedding_idxs
        if 'image_embedding_idx' in row and not pd.isna(row['image_embedding_idx']):
            emb_idx = int(row['image_embedding_idx'])
        else:
            emb_idx = int(row['image_embedding_idxs'][0])
        emb = self.emb[emb_idx].astype('float32')
        img_full = Image.open(os.path.join('data','crello', img_path)).convert('RGB')
        if self.transform:
            img = self.transform(img_full)
        else:
            img = T.ToTensor()(img_full)
        return emb, img


class Decoder(nn.Module):
    def __init__(self, emb_dim=512, ngf=256, out_size=128):
        super().__init__()
        # project to 8x8 feature map
        self.fc = nn.Linear(emb_dim, ngf * 8 * 8)
        self.net = nn.Sequential(
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf//2, 4, 2, 1), # 8->16
            nn.BatchNorm2d(ngf//2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf//2, ngf//4, 4, 2, 1), # 16->32
            nn.BatchNorm2d(ngf//4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf//4, ngf//8, 4, 2, 1), # 32->64
            nn.BatchNorm2d(ngf//8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf//8, ngf//16, 4, 2, 1), # 64->128
            nn.BatchNorm2d(ngf//16), nn.ReLU(True),
            nn.Conv2d(ngf//16, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        b = z.shape[0]
        h = self.fc(z).view(b, -1, 8, 8)
        return self.net(h)


def perceptual_loss(vgg, x, y):
    # x,y in [0,1], compute features
    # vgg expects normalized inputs
    return nn.functional.mse_loss(vgg(x), vgg(y))


def get_vgg_feature_extractor(device):
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    # wrap to accept [B,3,H,W] and return features
    return vgg


def train(args):
    base = Path('data') / 'crello'
    per_image = base / 'crello_train_elements_per_image.parquet'
    emb_npy = base / 'crello_element_image_embeddings.npy'
    assert per_image.exists() and emb_npy.exists(), 'per-image parquet or embeddings missing'

    transform = T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
    ])
    ds = ImageEmbeddingDataset(str(per_image), str(emb_npy), subset=args.subset, transform=transform)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Decoder(emb_dim=args.emb_dim).to(device)
    vgg = get_vgg_feature_extractor(device) if args.perceptual_weight > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/decoded_samples', exist_ok=True)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running = []
        running_l1 = []
        running_perc = []
        for emb, img in dl:
            emb = emb.to(device)
            img = img.to(device)
            recon = model(emb)
            l1 = nn.functional.l1_loss(recon, img)
            perc = torch.tensor(0.0, device=device)
            if vgg is not None:
                # normalize to VGG's expected input (ImageNet)
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
                re_n = (recon - mean) / std
                im_n = (img - mean) / std
                perc = nn.functional.mse_loss(vgg(re_n), vgg(im_n))
            loss = l1 + args.perceptual_weight * perc
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(float(loss.item()))
            running_l1.append(float(l1.item()))
            running_perc.append(float(perc.item()))

        epoch_loss = float(np.mean(running)) if running else 0.0
        epoch_l1 = float(np.mean(running_l1)) if running_l1 else 0.0
        epoch_perc = float(np.mean(running_perc)) if running_perc else 0.0
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.6f} (l1={epoch_l1:.6f} perc={epoch_perc:.6f})')
        # save latest
        ckpt = {'model_state': model.state_dict(), 'epoch': epoch+1}
        torch.save(ckpt, 'models/image_decoder.pt')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, 'models/image_decoder.best.pt')
            # write sample outputs
            model.eval()
            with torch.no_grad():
                sample_emb, sample_img = next(iter(dl))
                sample_emb = sample_emb.to(device)
                recon = model(sample_emb).cpu()
                sample_img = sample_img.cpu()
                for i in range(min(8, recon.shape[0])):
                    out = (recon[i].permute(1,2,0).numpy() * 255).astype('uint8')
                    Image.fromarray(out).save(f'outputs/decoded_samples/epoch{epoch+1}_sample{i}.png')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--subset', type=int, default=None)
    p.add_argument('--emb-dim', type=int, default=512)
    p.add_argument('--perceptual-weight', type=float, default=1.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
