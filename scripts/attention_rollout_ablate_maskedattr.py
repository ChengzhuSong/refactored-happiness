#!/usr/bin/env python3
"""
Compute attention rollout for ElementStage, with option to zero the
masked-attribute embedding before running. Writes per-poster JSONL where each
line is a list of dicts like:
{"slot": masked_slot_idx, "mask_attr": "size", "top": [[idx, score], ...]}

By default this masks the size attribute only on text slots (FONT != 0) and
hides the masked slot from the element-stage key/value mask, matching train.py.
Pass `--allow-non-text-size` to sample size from any valid element. Pass
`--keep-masked-slot-visible` to reproduce the old rollout script behavior.
"""
import argparse
import csv
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

# When this script is run as `python scripts/...` sys.path[0] is the scripts/ dir,
# which means sibling packages (like `models/`) are not importable. Prepend the
# repository root to sys.path so `from models...` works regardless of invocation.
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.two_stage_transformer import AttributeStage, ElementStage
from models.heads import make_decoder, load_decoder_state


def normalize_poster_id(value):
    if value is None:
        return ""
    value = str(value)
    for ch in ('"', "'", "(", ")", ","):
        value = value.replace(ch, "")
    return value.strip()


def load_poster_ids(index_path: str):
    poster_ids = []
    if not os.path.exists(index_path):
        return poster_ids
    with open(index_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return poster_ids
        for row in reader:
            poster_ids.append(normalize_poster_id(row[0]) if row else "")
    return poster_ids


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--base-dir', default='data/crello')
    p.add_argument('--out', required=True)
    p.add_argument('--max-posters', type=int, default=200)
    p.add_argument('--mask-count', type=int, default=1)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--device', default='cpu')
    p.add_argument('--mask-attr', default='size', help='Attribute to mask for rollout, e.g. size or text')
    p.add_argument('--mask-gate-font', action='store_true', help='Deprecated compatibility flag; size masking is text-only by default')
    p.add_argument('--allow-non-text-size', action='store_true', help='Allow size masking on non-text elements too')
    p.add_argument('--keep-masked-slot-visible', action='store_true', help='Keep the masked slot as an attention key/value, matching the old script instead of train.py')
    p.add_argument('--include-self', action='store_true', help='Allow the masked slot itself to appear in top-k rollout sources')
    p.add_argument('--zero-masked-attr', action='store_true', help='zero masked_attr_emb before running')
    return p.parse_args()


def register_attention_hooks(elem_stage: ElementStage, store: List):
    """Wrap each TransformerEncoderLayer's self_attn.forward so we capture
    attention weights (attn_output_weights) per forward call.
    Appends a list of attn matrices per layer to `store` on each forward.
    """
    # elem_stage.transformer is nn.TransformerEncoder -> has .layers list
    for layer in elem_stage.transformer.layers:
        mha = layer.self_attn
        orig_forward = mha.forward

        def make_forward(orig):
            def wrapped_forward(query, key, value, *args, **kwargs):
                # force need_weights=True to get attn weights
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                out = orig(query, key, value, *args, **kwargs)
                # out is (attn_output, attn_output_weights)
                if isinstance(out, tuple) and len(out) >= 2:
                    attn_w = out[1]
                    # detach and store (we'll process later)
                    store.append(attn_w.detach().cpu())
                return out

            return wrapped_forward

        mha.forward = make_forward(orig_forward)


def compute_rollout(attns: List[torch.Tensor], n_head: int, S: int) -> np.ndarray:
    """Compute attention rollout from a list of attention weight tensors.
    attns: list of tensors captured in order of forward calls. Each attn
    may have shape (B*num_heads, S, S) or (B, S, S) depending on PyTorch.

    We'll first reshape/average heads to (B, S, S) per layer then for each
    sample multiply (A + I) cumulatively to get final attribution matrix.
    Returns a (B, S, S) numpy array of rollout matrices.
    """
    # Determine number of layers: we'll assume attns are grouped per layer
    # and were appended sequentially for each forward. For TransformerEncoder
    # each layer's self_attn called once per forward, so attns list length == L
    L = len(attns)
    # convert each to (B, S, S)
    mats = []
    for a in attns:
        # a may be shape (B, H, S, S), (B, S, S), or (B*num_heads, S, S)
        if a.dim() == 4:
            mats.append(a.mean(dim=1).numpy())
        elif a.dim() == 3:
            Bx, M, N = a.shape
            if n_head > 1 and (Bx % n_head == 0):
                B = Bx // n_head
                a = a.view(B, n_head, M, N)
                mats.append(a.mean(dim=1).numpy())
            else:
                mats.append(a.numpy())
        else:
            mats.append(a.numpy())

    # mats: list of (B, S, S)
    mats = np.stack(mats, axis=0)  # (L, B, S, S)
    L, B, S, _ = mats.shape
    # add identity and normalize rows
    for l in range(L):
        mats[l] = mats[l] + np.eye(S)[None, :, :]
        # row-normalize
        row_sums = mats[l].sum(axis=-1, keepdims=True) + 1e-12
        mats[l] = mats[l] / row_sums

    # cumulative multiplication: R = mats[0] @ mats[1] @ ... @ mats[L-1]
    # We'll compute per sample
    R = np.eye(S)[None, :, :].repeat(B, axis=0)
    for l in range(L):
        R = np.matmul(R, mats[l])

    return R  # (B, S, S)


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    tokenizer_order = ckpt.get('tokenizer_order', ['image','text','pos','size','angle','opacity','font'])
    if args.mask_attr not in tokenizer_order:
        raise ValueError(f"--mask-attr '{args.mask_attr}' not in tokenizer_order {tokenizer_order}")

    # data paths
    prefix = f"poster_input_{args.split}"
    base = args.base_dir
    Xp = os.path.join(base, f"{prefix}_X.npy")
    Mp = os.path.join(base, f"{prefix}_mask.npy")
    Fp = os.path.join(base, f"{prefix}_font_idx.npy")
    Tp = os.path.join(base, f"{prefix}_type_idx.npy")
    Sp = os.path.join(base, f"{prefix}_schema.json")
    Ip = os.path.join(base, f"{prefix}_index.csv")

    X_all = np.load(Xp, mmap_mode='r')
    M_all = np.load(Mp, mmap_mode='r')
    FONT_all = np.load(Fp, mmap_mode='r')
    TYPE_all = np.load(Tp, mmap_mode='r')
    with open(Sp, 'r') as f:
        schema = json.load(f)
    fields = {f['name']: f for f in schema['fields']}
    S = X_all.shape[1]
    poster_ids = load_poster_ids(Ip)

    num_fonts = schema.get('font', {}).get('num_fonts', int(FONT_all.max() + 1))
    if 'attr_stage' in ckpt and 'tokenizer.font_emb.weight' in ckpt['attr_stage']:
        num_fonts = int(ckpt['attr_stage']['tokenizer.font_emb.weight'].shape[0])

    # build model
    d_attr = 128
    D_elem = 256
    num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))
    attr_stage = AttributeStage(img_dim=schema['fields'][0]['dim'], txt_dim=fields['text']['dim'], d_attr=d_attr, D_elem=D_elem, num_fonts=num_fonts)
    elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(schema['fields']) + 1)
    decoder = None
    if args.mask_attr in fields and 'decoders' in ckpt and args.mask_attr in ckpt['decoders']:
        decoder = make_decoder(D_elem, fields[args.mask_attr]['dim'])
        load_decoder_state(decoder, ckpt['decoders'][args.mask_attr])

    # load state
    elem_stage.load_state_dict(ckpt['elem_stage'])
    attr_stage.load_state_dict(ckpt['attr_stage'])

    # optionally zero masked_attr_emb
    if args.zero_masked_attr:
        if hasattr(elem_stage, 'masked_attr_emb'):
            with torch.no_grad():
                elem_stage.masked_attr_emb.weight.data.zero_()
        else:
            print('ElementStage has no masked_attr_emb to zero')

    attr_stage.to(device).eval()
    elem_stage.to(device).eval()
    if decoder is not None:
        decoder.to(device).eval()

    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # process posters sequentially and capture rollout per-poster
    N = X_all.shape[0]
    max_posters = min(args.max_posters, N)

    # helper to capture attn weights per forward call
    attn_store = []
    register_attention_hooks(elem_stage, attn_store)

    with open(out_path, 'w', encoding='utf-8') as fout:
        for i in range(max_posters):
            # load single poster
            X = torch.from_numpy(np.array(X_all[i : i + 1])).float().to(device)  # (1, S, F)
            M = torch.from_numpy(np.array(M_all[i : i + 1])).to(device)
            FONT = torch.from_numpy(np.array(FONT_all[i : i + 1])).long().to(device)
            TYPE = torch.from_numpy(np.array(TYPE_all[i : i + 1])).long().to(device)

            img = X[:, :, fields['image']['offset'][0] : fields['image']['offset'][1]]
            text = X[:, :, fields['text']['offset'][0] : fields['text']['offset'][1]]
            pos = X[:, :, fields['pos']['offset'][0] : fields['pos']['offset'][1]]
            size = X[:, :, fields['size']['offset'][0] : fields['size']['offset'][1]]
            angle = X[:, :, fields['angle']['offset'][0] : fields['angle']['offset'][1]]
            opacity = X[:, :, fields['opacity']['offset'][0] : fields['opacity']['offset'][1]]

            valid_mask = (M == 1)
            if args.mask_attr in ('text', 'font'):
                present_mask = (FONT != 0)
            elif args.mask_attr == 'size' and not args.allow_non_text_size:
                present_mask = (FONT != 0)
            else:
                present_mask = torch.ones_like(valid_mask)

            # sample exactly k masked slots from valid slots where the attribute is present
            k = int(args.mask_count)
            sampled = torch.zeros((1, S), dtype=torch.bool, device=device)
            valid_idxs = torch.nonzero(valid_mask[0] & present_mask[0], as_tuple=False).view(-1)
            if valid_idxs.numel() == 0:
                fout.write(json.dumps([]) + "\n")
                continue
            g = torch.Generator(device=device)
            g.manual_seed(int(args.seed) + i)
            perm = torch.randperm(valid_idxs.numel(), generator=g, device=device)
            sel = valid_idxs[perm[:min(k, valid_idxs.numel())]]
            sampled[0, sel] = True

            # build masked_attr_id
            slot_attr_mask = torch.zeros((1, S, len(tokenizer_order)), dtype=torch.bool, device=device)
            masked_attr_id = torch.zeros((1, S), dtype=torch.long, device=device)
            tok_idx = tokenizer_order.index(args.mask_attr)
            slot_attr_mask[:, :, tok_idx] = sampled
            masked_attr_id[sampled] = tok_idx + 1

            input_mask = valid_mask.clone()
            if not args.keep_masked_slot_visible:
                input_mask[sampled] = 0

            # clear attn_store
            attn_store.clear()

            with torch.no_grad():
                elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONT, slot_attr_mask=slot_attr_mask)
                ctx = elem_stage(elem_emb, role_idx=TYPE, mask=input_mask, masked_attr_id=masked_attr_id)
                pred = decoder(ctx) if decoder is not None else None

            # attn_store contains a list of tensors captured during forward (one per layer)
            # Each entry may be shape (B, H, S, S), (num_heads*B, S, S), or (B, S, S)
            if len(attn_store) == 0:
                # nothing captured
                fout.write(json.dumps([]) + "\n")
                continue

            # compute rollout
            first = attn_store[0]
            S_local = first.shape[-1]
            n_head = elem_stage.transformer.layers[0].self_attn.num_heads
            R = compute_rollout(attn_store, n_head=n_head, S=S_local)
            records = []
            for masked_pos in sampled.nonzero(as_tuple=False):
                masked_idx = int(masked_pos[1].item())
                row = R[0, masked_idx].copy()
                if not args.include_self:
                    row[masked_idx] = -np.inf
                valid_sources = valid_mask[0].detach().cpu().numpy().astype(bool)
                row[~valid_sources] = -np.inf
                topk = int(args.top_k)
                idxs = np.argsort(-row)[:topk]
                top = [
                    [int(ii), float(row[ii])]
                    for ii in idxs
                    if np.isfinite(row[ii])
                ]
                rec = {
                    "poster_idx": i,
                    "poster_id": poster_ids[i] if i < len(poster_ids) else "",
                    "slot": masked_idx,
                    "font_idx": int(FONT[0, masked_idx].item()),
                    "is_text": bool(FONT[0, masked_idx].item() != 0),
                    "mask_attr": args.mask_attr,
                    "top": top,
                }
                if pred is not None and fields[args.mask_attr]['dim'] <= 8:
                    astart, aend = fields[args.mask_attr]['offset']
                    gt_vec = X[0, masked_idx, astart:aend].detach().cpu().numpy().astype(float)
                    pred_vec = pred[0, masked_idx].detach().cpu().numpy().astype(float)
                    rec["gt"] = gt_vec.tolist()
                    rec["pred"] = pred_vec.tolist()
                    rec["abs_err"] = np.abs(pred_vec - gt_vec).tolist()
                records.append(rec)
            fout.write(json.dumps(records) + "\n")

    print('Wrote attention rollout to', out_path)


if __name__ == '__main__':
    main()
