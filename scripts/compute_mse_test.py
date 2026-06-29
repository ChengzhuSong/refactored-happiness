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
try:
    import torch
    import torch.nn as nn
    import subprocess
except ModuleNotFoundError:
    print('PyTorch is not installed in this Python environment.')
    print('Please run this script in the same environment you used for training (where torch is available).')
    print('Example conda install (CPU-only): conda install -c pytorch pytorch cpuonly -y')
    print('Or see https://pytorch.org for installation instructions matching your CUDA/drivers.')
    import sys
    sys.exit(1)

# Ensure repo root is on sys.path so `from models...` works when running
# this file as `python scripts/compute_mse_test.py` (sys.path[0] would be scripts/)
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.two_stage_transformer import AttributeStage, ElementStage
from models.heads import make_decoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--base-dir', default='data/crello')
    p.add_argument('--mask-count', type=int, default=1, help='exact K masked slots per poster')
    p.add_argument('--mask-attr', type=str, default='text', help='Which attribute to mask/evaluate (default: text). Use attribute name from schema, e.g. size')
    p.add_argument('--mask-gate-font', action='store_true', help='Gate masking for font-linked attributes (e.g. size) to slots where FONT != 0')
    p.add_argument('--only-mask-attr', action='store_true', help='Only compute MSE for the masked attribute (do not accumulate other attribute MSEs)')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--make-scatter', action='store_true', help='Run plotting script after evaluation to save GT vs Pred scatter PNG')
    p.add_argument('--scatter-out-file', default='eval_scatter.png', help='Output PNG path when --make-scatter is used')
    p.add_argument('--scatter-save-csv', default=None, help='Optional CSV path to save GT/Pred when making scatter')
    p.add_argument('--scatter-plain-out-file', default=None, help='Optional plain scatter PNG path (no metrics box)')
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
    # separate continuous-attribute decoders (exclude 'font') and a font classifier
    attr_names = [f['name'] for f in schema['fields']]
    cont_attrs = [n for n in attr_names if n != 'font']
    decoders = {name: make_decoder(D_elem, fields[name]['dim']) for name in cont_attrs}
    font_classifier = nn.Linear(D_elem, num_fonts)

    # load weights
    attr_stage.load_state_dict(ckpt['attr_stage'])
    elem_stage.load_state_dict(ckpt['elem_stage'])
    if 'decoders' in ckpt:
        for n in decoders:
            if n in ckpt['decoders']:
                decoders[n].load_state_dict(ckpt['decoders'][n])
        # try to load a font classifier if it was saved under decoders or separately
        if 'font' in ckpt['decoders']:
            try:
                font_classifier.load_state_dict(ckpt['decoders']['font'])
            except Exception:
                pass
    # also accept a top-level font_classifier key in the checkpoint
    if 'font_classifier' in ckpt:
        try:
            font_classifier.load_state_dict(ckpt['font_classifier'])
        except Exception:
            pass

    attr_stage.to(device).eval()
    elem_stage.to(device).eval()
    for d in decoders.values():
        d.to(device).eval()
    font_classifier.to(device).eval()

    # default slice for text (used earlier for legacy reasons); we'll compute per-attribute offsets below
    tstart, tend = fields['text']['offset']
    mask_attr = args.mask_attr
    if mask_attr not in tokenizer_order:
        raise ValueError(f"--mask-attr '{mask_attr}' not in tokenizer_order {tokenizer_order}")

    N = X_all.shape[0]
    bs = args.batch_size

    # per-attribute accumulators
    total_sse = {name: 0.0 for name in cont_attrs}
    n_dims = {name: 0 for name in cont_attrs}
    # font accuracy counters
    font_correct = 0
    font_total = 0

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
        # compute presence mask depending on requested mask attribute
        # By default, only 'text' and 'font' are gated by FONT!=0. Use --mask-gate-font
        # to gate other font-linked attributes (e.g. 'size') to slots where FONT != 0.
        if mask_attr in ('text', 'font'):
            present_mask = (FONTb != 0)
        elif args.mask_gate_font and mask_attr == 'size':
            present_mask = (FONTb != 0)
        else:
            # assume other attributes are present wherever the slot is valid
            present_mask = torch.ones_like(valid_mask)

        # sampling: exact-K per poster from valid & present_mask
        k = int(args.mask_count)
        sampled = torch.zeros((len(batch_idx), S), dtype=torch.bool, device=device)
        for mbi in range(len(batch_idx)):
            valid_idxs = torch.nonzero(valid_mask[mbi] & present_mask[mbi], as_tuple=False).view(-1)
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
        # mark the requested mask_attr in slot_attr_mask / masked_attr_id
        tok_idx = tokenizer_order.index(mask_attr)
        slot_attr_mask[:, :, tok_idx] = sampled
        masked_attr_id[sampled] = tok_idx + 1

        with torch.no_grad():
            elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=slot_attr_mask)
            ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=valid_mask, masked_attr_id=masked_attr_id)
            # predictions for continuous attributes
            preds = {name: decoders[name](ctx) for name in cont_attrs}
            # font logits
            font_logits = font_classifier(ctx)

        B = len(batch_idx)

        # accumulate SSE per attribute and font accuracy
        for mbi, bi in enumerate(batch_idx):
            for s in range(S):
                if not sampled[mbi, s]:
                    continue
                # skip if ground-truth vector is all zeros (as before)
                for name in cont_attrs:
                    # if only_mask_attr is set, skip other attributes
                    if args.only_mask_attr and name != mask_attr:
                        continue
                    astart, aend = fields[name]['offset']
                    gt_vec = X_all[bi, s, astart:aend].astype(np.float32)
                    if np.all(gt_vec == 0):
                        continue
                    pred_vec = preds[name][mbi, s].cpu().numpy().astype(np.float32)
                    diff = pred_vec - gt_vec
                    total_sse[name] += (diff * diff).sum()
                    n_dims[name] += pred_vec.size
                # font accuracy
                if 'font' in fields and (not args.only_mask_attr or mask_attr == 'font'):
                    # only consider slots with a real font (non-zero) for evaluation
                    gt_font = int(FONT_all[bi, s])
                    if gt_font == 0:
                        continue
                    logits = font_logits[mbi, s].cpu().numpy()
                    pred_font = int(logits.argmax())
                    font_total += 1
                    if pred_font == gt_font:
                        font_correct += 1

    any_compared = any(n_dims[name] > 0 for name in cont_attrs)
    if any_compared:
        print('Per-attribute MSE (per-dimension):')
        for name in cont_attrs:
            if n_dims[name] > 0:
                mse = total_sse[name] / n_dims[name]
                print(f'  {name}: {mse:.6e} (n_vectors: {n_dims[name] // (fields[name]["offset"][1]-fields[name]["offset"][0])})')
            else:
                print(f'  {name}: (no comparisons)')
    else:
        print('No continuous attribute comparisons (no masked positions matched)')

    if font_total > 0:
        acc = font_correct / font_total
        print(f'Font accuracy on masked positions: {acc:.6f} ({font_correct}/{font_total})')
    else:
        print('No font positions compared (no masked font positions with gt font != 0)')

    # optionally call plotting script to save scatter PNG (uses same masking sampling)
    if args.make_scatter:
        cmd = [
            'python3', 'scripts/plot_scatter_eval.py',
            '--model-ckpt', args.model_ckpt,
            '--split', args.split,
            '--mask-attr', args.mask_attr,
            '--mask-count', str(args.mask_count),
            '--seed', str(args.seed),
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--out-file', args.scatter_out_file,
            '--max-points', str(args.batch_size * 100)  # heuristic cap
        ]
        if args.scatter_save_csv:
            cmd += ['--save-csv', args.scatter_save_csv]
        if args.scatter_plain_out_file:
            cmd += ['--plain-out-file', args.scatter_plain_out_file]
        print('Running scatter plot:', ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print('Failed to run scatter plotting script:', e)


if __name__ == '__main__':
    main()
