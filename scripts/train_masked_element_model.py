#!/usr/bin/env python3
"""
Train a transformer to reconstruct masked element vectors (masked element modeling).

Usage (quick smoke test):
  python3 scripts/train_masked_element_model.py --epochs 1 --subset 128 --batch 8

Outputs:
 - models/masked_element_model.pt  (latest checkpoint)

This script is lightweight and suitable for experimentation. It expects
`data/crello/poster_inputs_X.npy` and `data/crello/poster_inputs_mask.npy` created earlier.
"""
import os
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
except Exception:
    wandb = None


class PosterDataset(Dataset):
    """Dataset that reads precomputed poster inputs from .npy files.

    Each item is a tuple (X, mask) where X is float32 array of shape (L, D)
    and mask is an integer mask array of shape (L,) with 1 for valid tokens.
    """

    def __init__(self, x_path: str, mask_path: str, subset: Optional[int] = None):
        self.X = np.load(x_path, mmap_mode='r')
        self.mask = np.load(mask_path, mmap_mode='r')
        if subset is not None:
            self.X = self.X[:subset]
            self.mask = self.mask[:subset]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].astype(np.float32)
        m = self.mask[idx].astype(np.uint8)
        return x, m


class MaskedElementTransformer(nn.Module):
    """Transformer that reconstructs masked element vectors.

    Inputs:
        x: Tensor(B, L, D)
        valid_mask: Tensor(B, L) with 1 for valid tokens
        mask_positions: optional Tensor(B, L) boolean indicating positions to replace with mask token
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.mask_token = nn.Parameter(torch.randn(model_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(max_len, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 4, dropout=dropout, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, mask_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, l, d = x.shape
        h = self.input_proj(x)  # (B,L,M)
        if mask_positions is not None:
            # mask_positions: bool tensor (B,L) where True means we should replace with mask token
            mask_tok = self.mask_token.unsqueeze(0).unsqueeze(0).expand(b, l, -1)
            h = torch.where(mask_positions.unsqueeze(-1), mask_tok, h)
        # add positional embeddings
        h = h + self.pos_emb.unsqueeze(0)
        # transformer expects (L,B,M)
        h = h.transpose(0, 1)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~(valid_mask.bool())
        out = self.transformer(h, src_key_padding_mask=key_padding_mask)
        out = out.transpose(0, 1)
        recon = self.output_proj(out)
        return recon


def collate_fn(batch):
    """Collate that returns torch tensors with correct dtypes.

    Converts numpy arrays to torch tensors and ensures float32 for inputs and uint8 for masks.
    """
    xs = np.stack([b[0] for b in batch], axis=0)
    ms = np.stack([b[1] for b in batch], axis=0)
    return torch.from_numpy(xs.astype(np.float32)), torch.from_numpy(ms.astype(np.uint8))


def build_model(input_dim: int, args, max_len: Optional[int] = None) -> MaskedElementTransformer:
    if max_len is None:
        # fall back to an explicit arg if provided, else default to 64
        max_len = getattr(args, 'max_len', 64)
    return MaskedElementTransformer(
        input_dim=input_dim, model_dim=args.model_dim, n_layers=args.layers, n_heads=args.heads, max_len=max_len, dropout=args.dropout
    )


def train(args):
    if wandb is not None:
        wandb.init(project="masked_elements_model_training", config=vars(args))
    else:
        print('WandB not available — continuing without remote logging')

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))
    # allow overriding the training files via CLI (useful after splitting)
    if args.train_x and args.train_mask:
        x_path = args.train_x
        mask_path = args.train_mask
    else:
        x_path = os.path.join(base_dir, 'poster_inputs_X.npy')
        mask_path = os.path.join(base_dir, 'poster_inputs_mask.npy')
    assert os.path.exists(x_path) and os.path.exists(mask_path), 'poster_inputs files missing'

    dataset = PosterDataset(x_path, mask_path, subset=args.subset)
    dl = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # optional validation dataset
    val_loader = None
    if args.val_x and args.val_mask:
        assert os.path.exists(args.val_x) and os.path.exists(args.val_mask), 'validation poster_inputs files missing'
        val_ds = PosterDataset(args.val_x, args.val_mask, subset=None)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # infer dims from a sample
    sample_x, sample_mask = dataset[0]
    L, D = sample_x.shape
    print(f'Dataset size: {len(dataset)}, seq_len: {L}, feat_dim: {D}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(input_dim=D, args=args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs('models', exist_ok=True)
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_ctr = 0

    for epoch in range(args.epochs):
        model.train()
        running = []
        for xb, maskb in dl:
            xb = xb.to(device)
            maskb = maskb.to(device)
            # create masking positions using mask_prob only on valid positions
            rand = torch.rand_like(maskb.float())
            mask_positions = (rand < args.mask_prob) & (maskb == 1)
            # forward: replace masked positions with mask token inside model
            recon = model(xb, valid_mask=maskb, mask_positions=mask_positions)
            # compute MSE only on masked positions
            if mask_positions.any():
                loss = ((recon - xb)**2)[mask_positions].mean()
            else:
                # no masked positions in batch (rare), skip
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())

        epoch_loss = float(np.mean(running)) if running else 0.0
        print(f'Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.6f}')

        wandb.log({'epoch': epoch+1, 'loss': epoch_loss})
        # save checkpoint
        ckpt = {'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'epoch': epoch+1}
        torch.save(ckpt, os.path.join('models', 'masked_element_model.pt'))
        # evaluate on validation set if provided (every eval_every epochs)
        val_loss = None
        if val_loader is not None and ((epoch + 1) % args.eval_every == 0):
            model.eval()
            total_sq = 0.0
            total_count = 0
            with torch.no_grad():
                for xb, maskb in val_loader:
                    xb = xb.to(device)
                    maskb = maskb.to(device)
                    rand = torch.rand_like(maskb.float())
                    mask_positions = (rand < args.mask_prob) & (maskb == 1)
                    if not mask_positions.any():
                        continue
                    recon = model(xb, valid_mask=maskb, mask_positions=mask_positions)
                    # sum squared error and count
                    dif = ((recon - xb)**2)[mask_positions]
                    total_sq += float(dif.sum().item())
                    total_count += int(dif.numel())
            if total_count > 0:
                val_loss = total_sq / total_count
            else:
                val_loss = float('inf')
            print(f'  Validation loss: {val_loss:.6f}')
            wandb.log({'epoch': epoch+1, 'val_loss': val_loss})

            # early stopping logic
            if val_loss + args.min_delta < best_val_loss:
                best_val_loss = val_loss
                patience_ctr = 0
                torch.save(ckpt, os.path.join('models', 'masked_element_model.best.pt'))
                print('  New best validation loss, saving checkpoint')
            else:
                patience_ctr += 1
                print(f'  No improvement (patience {patience_ctr}/{args.patience})')
                if patience_ctr >= args.patience:
                    print('Early stopping triggered')
                    break
        else:
            # no validation: fall back to training-loss-based checkpoint
            if val_loader is None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(ckpt, os.path.join('models', 'masked_element_model.best.pt'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--mask_prob', type=float, default=0.15)
    p.add_argument('--subset', type=int, default=None, help='optional subset of posters for quick tests')
    p.add_argument('--model_dim', type=int, default=512)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    # validation / early stopping
    p.add_argument('--val-x', type=str, default=None, help='path to validation poster_inputs_X.npy')
    p.add_argument('--val-mask', type=str, default=None, help='path to validation poster_inputs_mask.npy')
    p.add_argument('--patience', type=int, default=50, help='early stopping patience (eval rounds)')
    p.add_argument('--min-delta', type=float, default=1e-4, help='minimum improvement to reset patience')
    p.add_argument('--eval-every', type=int, default=1, help='evaluate validation every N epochs')
    p.add_argument('--train-x', type=str, default=None, help='optional path to training poster_inputs_X.npy')
    p.add_argument('--train-mask', type=str, default=None, help='optional path to training poster_inputs_mask.npy')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
