#!/usr/bin/env python3
"""
Inference demo for masked-element reconstruction.

Loads the best checkpoint from training, masks some element positions in a few posters,
reconstructs them with the trained model, and prints MSE statistics and sample vectors.

Usage: python3 scripts/inference_demo_masked_recon.py
"""
import os
import numpy as np
import torch


class MaskedElementTransformer(torch.nn.Module):
    def __init__(self, input_dim, model_dim=512, n_layers=4, n_heads=8, max_len=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.input_proj = torch.nn.Linear(input_dim, model_dim)
        self.mask_token = torch.nn.Parameter(torch.randn(model_dim) * 0.02)
        self.pos_emb = torch.nn.Parameter(torch.randn(max_len, model_dim) * 0.02)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim*4, dropout=dropout, activation='gelu')
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


def load_best_checkpoint(path):
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    raise FileNotFoundError(path)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'crello'))
    x_path = os.path.join(base_dir, 'poster_inputs_X.npy')
    mask_path = os.path.join(base_dir, 'poster_inputs_mask.npy')
    assert os.path.exists(x_path) and os.path.exists(mask_path), 'Need poster_inputs files'

    X = np.load(x_path)
    MASK = np.load(mask_path)
    print('Loaded X', X.shape, 'MASK', MASK.shape)

    # infer dims
    N, L, D = X.shape

    # load model
    ckpt_path = os.path.join('models', 'masked_element_model.best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join('models', 'masked_element_model.pt')
    ckpt = load_best_checkpoint(ckpt_path)
    # instantiate model with matching dims (use defaults consistent with training)
    model = MaskedElementTransformer(input_dim=D, model_dim=512, n_layers=4, n_heads=8, max_len=L)
    # load state dict safely
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # choose a few posters to demo
    indices = [0, min(10, N-1), max(0, N-1)]
    indices = sorted(set(indices))

    for idx in indices:
        x = torch.from_numpy(X[idx:idx+1]).to(device)
        valid = torch.from_numpy(MASK[idx:idx+1]).to(device)
        # create mask positions: randomly mask 15% of valid positions
        prob = 0.15
        rng = torch.rand(valid.shape, device=device)
        mask_pos = (rng < prob) & (valid == 1)
        with torch.no_grad():
            recon = model(x, valid_mask=valid, mask_positions=mask_pos)
        recon_np = recon.cpu().numpy()
        x_np = x.cpu().numpy()
        mask_np = mask_pos.cpu().numpy().astype(bool)

        masked_idx = np.where(mask_np[0])[0]
        print('\nPoster idx', idx, 'masked positions count', len(masked_idx))
        if len(masked_idx) == 0:
            print('No masked positions for this poster (rare)')
            continue
        # MSE per masked position
        mses = np.mean((recon_np[0, masked_idx] - x_np[0, masked_idx])**2, axis=1)
        print('Masked position MSEs (first 10):', mses[:10])
        print('Mean MSE:', float(mses.mean()))
        # show first masked position original vs recon (first 12 values)
        i0 = masked_idx[0]
        print('Example original (first 12 vals):', x_np[0, i0, :12].tolist())
        print('Example recon     (first 12 vals):', recon_np[0, i0, :12].tolist())

    # save a small output file
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'inference_demo_sample.npz')
    np.savez_compressed(out_path, X_sample=X[indices], MASK_sample=MASK[indices])
    print('\nSaved sample to', out_path)


if __name__ == '__main__':
    main()
