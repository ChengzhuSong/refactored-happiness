#!/usr/bin/env python3
"""Fit independent linear readouts from element-context (ctx) to continuous attributes.

Usage:
  python3 scripts/fit_independent_readouts.py --model-ckpt checkpoints/best_epoch.pth --train-split train --test-split test

This script:
 - loads attr_stage and elem_stage from checkpoint
 - computes ctx for all text slots (FONT != 0 and valid mask) on train and test
 - fits independent OLS linear maps from ctx -> gt_0 and ctx -> gt_1 on train
 - applies to test and compares to model decoder predictions
 - writes CSV: evaluations/independent_readouts_<test_split>.csv
"""
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_schema(base, split):
    prefix = f'poster_input_{split}'
    with open(os.path.join(base, f'{prefix}_schema.json'), 'r') as f:
        return json.load(f)


def make_model_from_ckpt(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    # build attr_stage and elem_stage as in train.py
    from models.two_stage_transformer import AttributeStage, ElementStage

    schema = ckpt.get('schema', None)
    # fallback: we need schema fields; try to load data/crello schema via train code path
    # but compute offsets later from disk when we know split.
    # create placeholder sizes; we'll re-create modules with correct dims after reading schema in main
    return ckpt


def compute_ctx_for_split(ckpt, ckpt_path, base, split, device='cpu'):
    # load arrays and schema
    prefix = f'poster_input_{split}'
    X_path = os.path.join(base, f'{prefix}_X.npy')
    mask_path = os.path.join(base, f'{prefix}_mask.npy')
    font_path = os.path.join(base, f'{prefix}_font_idx.npy')
    type_path = os.path.join(base, f'{prefix}_type_idx.npy')
    schema_path = os.path.join(base, f'{prefix}_schema.json')

    X_all = np.load(X_path, mmap_mode='r')
    MASK_all = np.load(mask_path, mmap_mode='r')
    FONT_all = np.load(font_path, mmap_mode='r')
    TYPE_all = np.load(type_path, mmap_mode='r')
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    N = X_all.shape[0]
    S = X_all.shape[1]

    # fields mapping
    fields = {f['name']: f for f in schema['fields']}

    # build models
    d_attr = 128
    D_elem = 256
    # try to infer num_fonts from checkpoint if present to avoid shape mismatch
    num_fonts = None
    if isinstance(ckpt, dict) and 'attr_stage' in ckpt and hasattr(ckpt['attr_stage'], 'keys'):
        # state_dict style: search for a key that includes 'font_emb' or 'font' emb weights
        for k in ckpt['attr_stage'].keys():
            if 'font_emb' in k and k.endswith('weight'):
                num_fonts = int(ckpt['attr_stage'][k].shape[0])
                break
    if num_fonts is None:
        num_fonts = schema.get('font', {}).get('num_fonts', int(FONT_all.max() + 1))
    num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))

    from models.two_stage_transformer import AttributeStage, ElementStage
    attr_stage = AttributeStage(
        img_dim=schema['fields'][0]['dim'],
        txt_dim=[f for f in schema['fields'] if f['name'] == 'text'][0]['dim'],
        d_attr=d_attr,
        D_elem=D_elem,
        num_fonts=num_fonts,
    ).to(device)
    elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(schema['fields']) + 1).to(device)

    # load ckpt parts
    if 'attr_stage' in ckpt:
        attr_stage.load_state_dict(ckpt['attr_stage'])
    if 'elem_stage' in ckpt:
        elem_stage.load_state_dict(ckpt['elem_stage'])

    attr_stage.eval(); elem_stage.eval()

    # prepare to compute ctx vectors
    Bs = 64
    ctx_list = []
    meta_rows = []

    for i in range(0, N, Bs):
        batch_idx = np.arange(i, min(i + Bs, N))
        Xb = torch.from_numpy(np.array(X_all[batch_idx])).float().to(device)
        MASKb = torch.from_numpy(np.array(MASK_all[batch_idx])).to(device)
        FONTb = torch.from_numpy(np.array(FONT_all[batch_idx])).long().to(device)
        TYPEb = torch.from_numpy(np.array(TYPE_all[batch_idx])).long().to(device)

        img = Xb[:, :, fields['image']['offset'][0] : fields['image']['offset'][1]]
        text = Xb[:, :, fields['text']['offset'][0] : fields['text']['offset'][1]]
        pos = Xb[:, :, fields['pos']['offset'][0] : fields['pos']['offset'][1]]
        size = Xb[:, :, fields['size']['offset'][0] : fields['size']['offset'][1]]
        angle = Xb[:, :, fields['angle']['offset'][0] : fields['angle']['offset'][1]]
        opacity = Xb[:, :, fields['opacity']['offset'][0] : fields['opacity']['offset'][1]]

        valid_mask = (MASKb == 1)
        has_font = (FONTb != 0)

        # compute elem_emb and ctx
        with torch.no_grad():
            elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=None)
            ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=valid_mask, masked_attr_id=None)

        # ctx shape (B, S, D_elem)
        ctx_np = ctx.cpu().numpy()
        # gather only text slots (has_font True and valid_mask True)
        mask_slots = (valid_mask.cpu().numpy() == 1) & (has_font.cpu().numpy() != 0)
        for bi, global_idx in enumerate(batch_idx):
            for s in range(S):
                if not mask_slots[bi, s]:
                    continue
                row = {
                    'poster_idx': int(global_idx),
                    'slot_idx': int(s),
                    'canvas_width': float(fields.get('canvas_width', {}).get('max', 1.0)),
                }
                meta_rows.append(row)
                ctx_list.append(ctx_np[bi, s].astype(np.float32))

    ctx_arr = np.vstack(ctx_list) if len(ctx_list) else np.zeros((0, D_elem), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return ctx_arr, meta_df, fields, schema


def fit_and_eval(args):
    import torch
    device = torch.device('cpu')
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    base = 'data/crello'

    # compute train ctx and collect gt targets
    print('Computing train ctx...')
    ctx_train, meta_train, fields, schema = compute_ctx_for_split(ckpt, args.model_ckpt, base, args.train_split, device='cpu')

    # load train arrays to get gt targets
    prefix = f'poster_input_{args.train_split}'
    X_train = np.load(os.path.join(base, f'{prefix}_X.npy'))
    FONT_train = np.load(os.path.join(base, f'{prefix}_font_idx.npy'))
    MASK_train = np.load(os.path.join(base, f'{prefix}_mask.npy'))

    # Build list of gt targets aligned with ctx_train by iterating same order (simple approach)
    gt_list = []
    N = X_train.shape[0]
    S = X_train.shape[1]
    # fields offsets for size
    a,b = fields['size']['offset']
    # iterate same as compute function
    Bs = 64
    for i in range(0, N, Bs):
        batch_idx = np.arange(i, min(i + Bs, N))
        Xb = np.array(X_train[batch_idx])
        FONTb = np.array(FONT_train[batch_idx])
        MASKb = np.array(MASK_train[batch_idx])
        mask_slots = (MASKb == 1) & (FONTb != 0)
        for bi, gidx in enumerate(batch_idx):
            for s in range(S):
                if not mask_slots[bi, s]:
                    continue
                targ = Xb[bi, s, a:b]
                # targ may be multi-dim; ensure 1D
                gt_list.append(np.array(targ, dtype=np.float32).reshape(-1))

    gt_arr = np.vstack(gt_list) if len(gt_list) else np.zeros((0, b-a), dtype=np.float32)
    print('Train ctx shape', ctx_train.shape, 'train gt shape', gt_arr.shape)

    # Fit independent OLS for each dimension (baseline) unless we'll fit heads
    W = []
    intercepts = []
    for dim in range(gt_arr.shape[1]):
        y = gt_arr[:, dim]
        A = np.hstack([ctx_train, np.ones((ctx_train.shape[0], 1), dtype=np.float32)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        w = coef[:-1]
        b0 = coef[-1]
        W.append(w)
        intercepts.append(b0)
        print(f'Fitted linear readout dim{dim}: weights shape {w.shape} intercept {b0:.6f}')

    W = np.vstack(W)  # shape (dim, D_elem)
    intercepts = np.array(intercepts)

    # Optionally train small independent linear heads (optimizing MSE) on top of frozen ctx
    if getattr(args, 'fit_heads', False):
        print('\nTraining independent linear heads (torch) on top of frozen ctx...')
        import torch
        D = ctx_train.shape[1]
        device = torch.device('cpu')
        # prepare tensors
        X = torch.from_numpy(ctx_train).to(device)
        Y = torch.from_numpy(gt_arr).to(device)

        # two independent heads
        head0 = nn.Linear(D, 1).to(device)
        head1 = nn.Linear(D, 1).to(device)
        params = list(head0.parameters()) + list(head1.parameters())
        opt = torch.optim.Adam(params, lr=getattr(args, 'lr', 1e-3), weight_decay=getattr(args, 'weight_decay', 0.0))
        loss_fn = nn.MSELoss()
        batch = int(getattr(args, 'batch', 1024))
        epochs = int(getattr(args, 'epochs', 20))
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            for i in range(0, n, batch):
                ids = perm[i:i+batch]
                xb = X[ids]
                yb = Y[ids]
                opt.zero_grad()
                p0 = head0(xb).squeeze(-1)
                p1 = head1(xb).squeeze(-1)
                loss = loss_fn(p0, yb[:, 0]) + loss_fn(p1, yb[:, 1])
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            print(f' epoch {ep+1}/{epochs} avg_loss={epoch_loss/n:.6f}')

        # replace W/intercepts with trained heads
        W = np.vstack([head0.weight.detach().cpu().numpy(), head1.weight.detach().cpu().numpy()])
        intercepts = np.array([float(head0.bias.detach().cpu().numpy()), float(head1.bias.detach().cpu().numpy())])
        print('Trained heads intercepts:', intercepts)

    # compute test ctx
    print('Computing test ctx...')
    ctx_test, meta_test, _, _ = compute_ctx_for_split(ckpt, args.model_ckpt, base, args.test_split, device='cpu')

    # load test arrays and model preds from CSV for comparison
    test_csv = f'evaluations/eval_size_gt_pred_{args.test_split}_text_only_with_midpoints.csv'
    df_test = pd.read_csv(test_csv)
    # detect columns
    gt_cols = [c for c in df_test.columns if c.startswith('gt_')]
    gt0_col = 'gt_0' if 'gt_0' in df_test.columns else gt_cols[0]
    gt1_col = 'gt_1' if 'gt_1' in df_test.columns else gt_cols[min(1, len(gt_cols)-1)]
    pred0_col = 'pred_0' if 'pred_0' in df_test.columns else 'pred_'+gt0_col.split('_',1)[1]
    pred1_col = 'pred_1' if 'pred_1' in df_test.columns else 'pred_'+gt1_col.split('_',1)[1]

    # apply linear maps to ctx_test
    lin_preds = (W @ ctx_test.T).T + intercepts[None, :]
    # build comparison dataframe: align ordering should match the order we collected ctx_test (it does)
    out_rows = []
    for i in range(lin_preds.shape[0]):
        out_rows.append({
            'idx': i,
            'lin_pred_0': float(lin_preds[i, 0]),
            'lin_pred_1': float(lin_preds[i, 1]),
        })
    df_lin = pd.DataFrame(out_rows)

    # compare with model preds from CSV by taking their pred_0/pred_1 in the same order (CSV was produced in the same sampling scheme)
    # We'll truncate to min length
    n_compare = min(len(df_lin), len(df_test))
    cmp_df = pd.DataFrame({
        'gt_0': df_test[gt0_col].astype(float).values[:n_compare],
        'gt_1': df_test[gt1_col].astype(float).values[:n_compare],
        'model_pred_0': df_test[pred0_col].astype(float).values[:n_compare],
        'model_pred_1': df_test[pred1_col].astype(float).values[:n_compare],
        'lin_pred_0': df_lin['lin_pred_0'].values[:n_compare],
        'lin_pred_1': df_lin['lin_pred_1'].values[:n_compare],
    })

    # metrics
    def metrics(y_true, y_pred):
        res = y_pred - y_true
        return {'n': len(res), 'mean_residual': float(res.mean()), 'rmse': float(np.sqrt((res**2).mean()))}

    for dim in [0,1]:
        print('\nDimension', dim)
        m_model = metrics(cmp_df[f'gt_{dim}'].values, cmp_df[f'model_pred_{dim}'].values)
        m_lin = metrics(cmp_df[f'gt_{dim}'].values, cmp_df[f'lin_pred_{dim}'].values)
        print('  model: ', m_model)
        print('  lin  : ', m_lin)

    outp = Path(f'evaluations/independent_readouts_{args.test_split}.csv')
    cmp_df.to_csv(outp, index=False)
    print('Wrote comparison CSV to', outp)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--train-split', default='train')
    p.add_argument('--test-split', default='test')
    p.add_argument('--fit-heads', action='store_true', help='Train independent linear heads on top of frozen ctx')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    args = p.parse_args()
    fit_and_eval(args)
