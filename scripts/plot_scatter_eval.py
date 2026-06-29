#!/usr/bin/env python3
"""Plot GT vs predicted scatter for a masked attribute across a split.

This script samples masked slots (exact-K per poster) the same way as
`compute_mse_test.py`, collects ground-truth and predicted vectors for the
requested attribute, and saves a PNG scatter where x=gt and y=pred.

Example:
  conda activate test
  python3 scripts/plot_scatter_eval.py --model-ckpt checkpoints/best_epoch.pth \
      --split test --mask-attr size --mask-count 1 --seed 123 --batch-size 64 \
      --device cpu --out-file eval_size_scatter.png
"""
import argparse
import json
import csv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import torch
    import torch.nn as nn
except Exception as e:
    print('Missing dependency for plotting:', e)
    print('Run this in the environment used for training (conda activate test)')
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except Exception:
    print('matplotlib is required for plotting. Install it in your env: pip install matplotlib')
    sys.exit(1)

from models.two_stage_transformer import AttributeStage, ElementStage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--base-dir', default='data/crello')
    p.add_argument('--mask-count', type=int, default=1)
    p.add_argument('--mask-attr', type=str, default='size')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out-file', default='eval_scatter.png')
    p.add_argument('--mask-gate-font', action='store_true', help='Gate masking for font-linked attributes (e.g. size) to slots where FONT != 0')
    p.add_argument('--plain-out-file', default=None, help='Optional path to save a plain scatter (no metrics)')
    p.add_argument('--save-csv', default=None, help='Path to save GT/Pred CSV (optional)')
    p.add_argument('--max-points', type=int, default=5000, help='Maximum number of points to plot')
    p.add_argument('--no-plot', action='store_true', help='If set, do not generate PNG plots; only write CSV')
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

    # model components (dims match training defaults)
    d_attr = 128
    D_elem = 256
    num_fonts = schema.get('font', {}).get('num_fonts', int(FONT_all.max() + 1))
    # if checkpoint contains a tokenizer.font_emb.weight, prefer that size
    if 'attr_stage' in ckpt and 'tokenizer.font_emb.weight' in ckpt['attr_stage']:
        try:
            num_fonts = int(ckpt['attr_stage']['tokenizer.font_emb.weight'].shape[0])
        except Exception:
            pass
    num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))
    attr_stage = AttributeStage(img_dim=schema['fields'][0]['dim'], txt_dim=fields['text']['dim'], d_attr=d_attr, D_elem=D_elem, num_fonts=num_fonts)
    elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(schema['fields']) + 1)

    attr_stage.load_state_dict(ckpt['attr_stage'])
    elem_stage.load_state_dict(ckpt['elem_stage'])
    attr_stage.to(device).eval()
    elem_stage.to(device).eval()

    # create decoder for the one attribute we care about
    name = args.mask_attr
    if name not in fields:
        raise ValueError(f"Unknown attribute '{name}' in schema")
    cont_dim = fields[name]['dim']
    decoder = nn.Linear(D_elem, cont_dim)
    if 'decoders' in ckpt and name in ckpt['decoders']:
        try:
            decoder.load_state_dict(ckpt['decoders'][name])
        except Exception:
            pass
    decoder.to(device).eval()

    N = X_all.shape[0]
    bs = args.batch_size

    gt_list = []
    pred_list = []
    meta_list = []

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
        if name in ('text', 'font'):
            present_mask = (FONTb != 0)
        elif args.mask_gate_font and name == 'size':
            present_mask = (FONTb != 0)
        else:
            present_mask = torch.ones_like(valid_mask)

        # exact-K sampling
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

        tok_idx = tokenizer_order.index(name)
        slot_attr_mask = torch.zeros((len(batch_idx), S, len(tokenizer_order)), dtype=torch.bool, device=device)
        masked_attr_id = torch.zeros((len(batch_idx), S), dtype=torch.long, device=device)
        slot_attr_mask[:, :, tok_idx] = sampled
        masked_attr_id[sampled] = tok_idx + 1

        with torch.no_grad():
            elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=slot_attr_mask)
            ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=valid_mask, masked_attr_id=masked_attr_id)
            preds = decoder(ctx)

        for mbi, bi in enumerate(batch_idx):
            for s in range(S):
                if not sampled[mbi, s]:
                    continue
                astart, aend = fields[name]['offset']
                gt_vec = X_all[bi, s, astart:aend].astype(np.float32)
                if np.all(gt_vec == 0):
                    continue
                pred_vec = preds[mbi, s].cpu().numpy().astype(np.float32)
                gt_list.append(gt_vec)
                pred_list.append(pred_vec)
                # store poster index (bi) and slot index (s) to trace back
                meta_list.append((bi, s))

        # cap points to max_points for plotting speed
        if len(gt_list) >= args.max_points:
            break

    if len(gt_list) == 0:
        print('No points collected for plotting (no masked positions matched).')
        return

    gt_arr = np.vstack(gt_list)
    pred_arr = np.vstack(pred_list)
    meta_arr = np.array(meta_list, dtype=int)

    # optionally save CSV with poster_idx, slot_idx, gt_*, pred_*, abs_err_*
    if args.save_csv:
        csv_path = args.save_csv
        # load poster index -> poster_id mapping (poster_input_<split>_index.csv)
        idx_csv = Path(base) / f'poster_input_{args.split}_index.csv'
        def _norm_pid(s):
            if s is None:
                return ''
            s = str(s)
            # remove common wrapper characters anywhere in the string
            for ch in ('"', "'", '(', ')', ','):
                s = s.replace(ch, '')
            return s.strip()

        poster_idx_to_id = {}
        if idx_csv.exists():
            with idx_csv.open() as f:
                r = csv.reader(f)
                try:
                    hdr = next(r)
                except StopIteration:
                    hdr = []
                for i, row in enumerate(r):
                    if len(row) >= 1:
                        pid = _norm_pid(row[0])
                        poster_idx_to_id[i] = pid
        # load crello elements CSV to map poster_id -> canvas sizes (first occurrence)
        crello_elements = Path(base) / f'crello_{args.split}_elements.csv'
        poster_canvas = {}
        if crello_elements.exists():
            with crello_elements.open() as f:
                r = csv.reader(f)
                hdr = next(r)
                # find columns
                col_idx = {c: j for j,c in enumerate(hdr)}
                w_col = None
                h_col = None
                pid_col = None
                for colname in col_idx:
                    if colname.strip() == 'canvas_width':
                        w_col = col_idx[colname]
                    if colname.strip() == 'canvas_height':
                        h_col = col_idx[colname]
                    if colname.strip() == 'poster_id':
                        pid_col = col_idx[colname]
                if pid_col is not None and w_col is not None and h_col is not None:
                    for row in r:
                        try:
                            pid = _norm_pid(row[pid_col])
                            if pid and pid not in poster_canvas:
                                poster_canvas[pid] = (float(row[w_col]), float(row[h_col]))
                        except Exception:
                            continue
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # header
            dim = gt_arr.shape[1]
            # include slot positional attribute dims (pos_*) and number of text elements on the poster
            pos_start, pos_end = fields['pos']['offset']
            pos_dim = pos_end - pos_start
            header = ['poster_idx', 'poster_id', 'slot_idx']
            for p in range(pos_dim):
                header.append(f'slot_pos_{p}')
            # midpoint columns and canvas dims
            header += ['midpoint_x', 'midpoint_y', 'canvas_width', 'canvas_height']
            header.append('n_text_elements')
            for d in range(dim):
                header.append(f'gt_{d}')
            for d in range(dim):
                header.append(f'pred_{d}')
            for d in range(dim):
                header.append(f'abs_err_{d}')
            writer.writerow(header)
            for i in range(len(gt_arr)):
                bi = int(meta_arr[i,0])
                s = int(meta_arr[i,1])
                pid = poster_idx_to_id.get(bi, '')
                row = [bi, pid, s]
                # slot position vector from the full X_all array for this slot
                pos_vec = X_all[bi, s, pos_start:pos_end].astype(np.float32)
                for v in pos_vec.tolist():
                    row.append(float(v))
                # midpoint using GT width/height (scale by 1000 as requested)
                astart, aend = fields[name]['offset']
                gt_vec = X_all[bi, s, astart:aend].astype(np.float32)
                # guard for 2D gt (width,height)
                if gt_vec.shape[0] >= 2:
                    left = float(pos_vec[0])
                    top = float(pos_vec[1])
                    width = float(gt_vec[0])
                    height = float(gt_vec[1])
                    mx = (left + 0.5 * width) * 1000.0
                    my = (top + 0.5 * height) * 1000.0
                else:
                    mx = ''
                    my = ''
                row += [mx, my]
                # canvas sizes
                cw, ch = ('', '')
                if pid and pid in poster_canvas:
                    try:
                        cw, ch = poster_canvas[pid]
                    except Exception:
                        cw, ch = ('', '')
                row += [cw, ch]
                # number of text elements in this poster (FONT != 0)
                n_text = int((FONT_all[bi] != 0).sum())
                row.append(n_text)
                row += [float(x) for x in gt_arr[i].tolist()]
                row += [float(x) for x in pred_arr[i].tolist()]
                abs_err = np.abs(pred_arr[i] - gt_arr[i])
                row += [float(x) for x in abs_err.tolist()]
                writer.writerow(row)
        print('Saved CSV to', csv_path)

    # if multidimensional, plot first dim and warn
    if gt_arr.shape[1] > 1:
        print('Warning: attribute has multiple dimensions; plotting first dimension only.')

    x = gt_arr[:, 0]
    y = pred_arr[:, 0]

    # prepare simple stats used by plotting
    n = len(x)

    # optionally save a plain scatter (no metrics)
    if args.plain_out_file:
        plt.figure(figsize=(6, 6))
        if n > 2000:
            hb = plt.hexbin(x, y, gridsize=100, cmap='Blues', mincnt=1)
            plt.colorbar(hb, label='counts')
        else:
            plt.scatter(x, y, s=6, alpha=0.7)
        plt.plot([min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02], [min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02], color='red', linewidth=1)
        plt.xlabel('ground-truth')
        plt.ylabel('predicted')
        plt.title(f"Scatter GT vs Pred — {args.mask_attr} (n={n})")
        plt.tight_layout()
        plt.savefig(args.plain_out_file, dpi=200)
        print('Saved plain scatter to', args.plain_out_file)

    # compute simple metrics
    n = len(x)
    diff = y - x
    mse = float((diff ** 2).mean())
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    plt.figure(figsize=(6, 6))
    # prefer hexbin when there are many overlapping points
    if n > 2000:
        hb = plt.hexbin(x, y, gridsize=100, cmap='Blues', mincnt=1)
        plt.colorbar(hb, label='counts')
    else:
        plt.scatter(x, y, s=6, alpha=0.7)

    # identity line
    minv = min(x.min(), y.min())
    maxv = max(x.max(), y.max())
    rng = maxv - minv if maxv > minv else maxv if maxv != 0 else 1.0
    plt.plot([minv - 0.02 * rng, maxv + 0.02 * rng], [minv - 0.02 * rng, maxv + 0.02 * rng], color='red', linewidth=1)

    plt.xlabel('ground-truth')
    plt.ylabel('predicted')
    plt.title(f"Scatter GT vs Pred — {args.mask_attr} (n={n})")

    # annotation box with metrics
    textstr = f"n={n}\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nR^2={r2:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(args.out_file, dpi=200)
    print('Saved scatter plot to', args.out_file)


if __name__ == '__main__':
    main()
