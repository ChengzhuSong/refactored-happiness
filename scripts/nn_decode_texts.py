#!/usr/bin/env python3
"""
Nearest-neighbor decoding for predicted text embeddings.

Builds a candidate pool from training slot texts and their precomputed text
embeddings (from poster_input_train_X.npy). For each predicted text embedding
from a two-stage model checkpoint, finds the nearest candidate by cosine
similarity and writes per-poster JSONL of decoded strings.

Usage (smoke run):
  PYTHONPATH=. python3 scripts/nn_decode_texts.py --model-ckpt checkpoints/best_epoch.pth --split test --out data/crello/decoded_texts_test_nn_small.jsonl --max-posters 100 --device cuda

"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.two_stage_transformer import AttributeStage, ElementStage
from models.heads import make_decoder


def build_candidate_pool(base_dir='data/crello', slots=64, max_candidates=None):
    """Return (texts_list, embeddings_np) gathered from train split.
    embeddings_np shape: (C, D)
    """
    slot_texts_path = Path(base_dir) / 'slot_texts_train.jsonl'
    Xp = Path(base_dir) / 'poster_input_train_X.npy'
    Mp = Path(base_dir) / 'poster_input_train_mask.npy'
    Sp = Path(base_dir) / 'poster_input_train_schema.json'
    if not slot_texts_path.exists() or not Xp.exists():
        raise FileNotFoundError('train slot_texts or X.npy missing; run extract_slot_texts.py first')

    # load schema to get text offsets
    with open(Sp, 'r') as f:
        schema = json.load(f)
    fields = {f['name']: f for f in schema['fields']}
    tstart, tend = fields['text']['offset']

    # load arrays (mmap)
    X_all = np.load(str(Xp), mmap_mode='r')
    M_all = np.load(str(Mp), mmap_mode='r')

    texts_to_emb = OrderedDict()
    # iterate posters; for each non-null slot text add mapping text->embedding (first seen)
    with open(slot_texts_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            arr = json.loads(line)
            arr_X = X_all[i]
            mask = M_all[i]
            for s, txt in enumerate(arr[:slots]):
                if txt is None:
                    continue
                if not mask[s]:
                    continue
                if txt in texts_to_emb:
                    continue
                emb = arr_X[s, tstart:tend].astype(np.float32)
                texts_to_emb[txt] = emb
            if max_candidates is not None and len(texts_to_emb) >= max_candidates:
                break

    texts = list(texts_to_emb.keys())
    embs = np.stack([texts_to_emb[t] for t in texts], axis=0)
    return texts, embs


def nn_decode(args):
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # load two-stage ckpt
    ckpt = torch.load(args.model_ckpt, map_location='cpu')
    tokenizer_order = ckpt.get('tokenizer_order', ['image','text','pos','size','angle','opacity','font'])

    # prepare data paths
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

    # build model components
    d_attr = 128
    D_elem = 256
    num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))
    attr_stage = AttributeStage(img_dim=schema['fields'][0]['dim'], txt_dim=fields['text']['dim'], d_attr=d_attr, D_elem=D_elem, num_fonts=num_fonts)
    elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(schema['fields']) + 1)
    decoders = {name: make_decoder(D_elem, fields[name]['dim']) for name in [f['name'] for f in schema['fields']]}

    # load state
    attr_stage.load_state_dict(ckpt['attr_stage'])
    elem_stage.load_state_dict(ckpt['elem_stage'])
    if 'decoders' in ckpt:
        for n in decoders:
            if n in ckpt['decoders']:
                decoders[n].load_state_dict(ckpt['decoders'][n])

    # Try to move models and decoders to the requested device. If CUDA
    # initialization fails (e.g. driver/hardware incompatibility), catch
    # the error and fall back to CPU so the script can still run.
    try:
        attr_stage.to(device).eval()
        elem_stage.to(device).eval()
        for k in decoders:
            decoders[k].to(device).eval()
    except Exception as e:
        # Common CUDA initialization errors raise during .to('cuda').
        # Fall back to CPU and continue.
        print(f"Warning: failed to move models to device={device}! Falling back to CPU. Error: {e}")
        device = torch.device('cpu')
        attr_stage.to(device).eval()
        elem_stage.to(device).eval()
        for k in decoders:
            decoders[k].to(device).eval()

    # build candidate pool (texts + embeddings)
    print('Building candidate pool...')
    texts, emb_np = build_candidate_pool(base_dir=args.base_dir, slots=S, max_candidates=args.max_candidates)
    print('Candidate count:', len(texts))
    # normalize embeddings
    emb_t = torch.from_numpy(emb_np).float().to(device)
    emb_t = emb_t / (emb_t.norm(dim=1, keepdim=True) + 1e-8)

    out_path = args.out or os.path.join(args.base_dir, f"decoded_texts_{args.split}_nn.jsonl")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # load slot_texts for GT (prefer split-specific, fall back to train)
    slot_texts_path = Path(args.base_dir) / f"slot_texts_{args.split}.jsonl"
    if not slot_texts_path.exists():
        slot_texts_path = Path(args.base_dir) / 'slot_texts_train.jsonl'
    if not slot_texts_path.exists():
        raise FileNotFoundError('slot_texts file not found for split or train; run extract_slot_texts.py')
    # read into list of per-poster lists (may be large; we read once)
    slot_texts = []
    with open(slot_texts_path, 'r', encoding='utf-8') as f:
        for line in f:
            slot_texts.append(json.loads(line))

    N = X_all.shape[0]
    max_posters = args.max_posters or N
    bs = args.batch_size

    with open(out_path, 'w', encoding='utf-8') as fout:
        for i in range(0, min(N, max_posters), bs):
            batch_idx = list(range(i, min(i + bs, min(N, max_posters))))
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
            # determine which slots actually contain text. Historically we
            # treat font==0 as 'no text' (presence gating) so only mask
            # those slots that have a real text/font index != 0.
            text_present = (FONTb != 0)
            # choose masked positions. If mask_count is set, select exactly
            # that many valid slots per poster (without replacement). Else
            # fall back to mask_prob sampling.
            if args.mask_count is not None:
                k = int(args.mask_count)
                sampled = torch.zeros((len(batch_idx), S), dtype=torch.bool, device=device)
                for mbi in range(len(batch_idx)):
                    # only consider slots that are valid and actually contain text
                    valid_idxs = torch.nonzero(valid_mask[mbi] & text_present[mbi], as_tuple=False).view(-1)
                    n_valid = valid_idxs.numel()
                    if n_valid == 0:
                        continue
                    choose_k = min(k, int(n_valid))
                    if args.seed is not None:
                        g = torch.Generator(device=device)
                        g.manual_seed(int(args.seed) + i + mbi)
                        perm = torch.randperm(n_valid, generator=g, device=device)
                    else:
                        perm = torch.randperm(n_valid, device=device)
                    sel = valid_idxs[perm[:choose_k]]
                    sampled[mbi, sel] = True
            else:
                # choose a random subset of valid slots to mask (so we only
                # predict some slots instead of decoding every slot)
                mask_prob = float(args.mask_prob)
                if args.seed is not None:
                    # make selection deterministic per-run; only mask slots that
                    # are valid and have real text.
                    g = torch.Generator(device=device)
                    g.manual_seed(int(args.seed) + i)
                    sampled = (torch.rand((len(batch_idx), S), device=device, generator=g) < mask_prob) & valid_mask & text_present
                else:
                    sampled = (torch.rand((len(batch_idx), S), device=device) < mask_prob) & valid_mask & text_present

            slot_attr_mask = torch.zeros((len(batch_idx), S, len(tokenizer_order)), dtype=torch.bool, device=device)
            masked_attr_id = torch.zeros((len(batch_idx), S), dtype=torch.long, device=device)
            if 'text' in tokenizer_order:
                tok_idx = tokenizer_order.index('text')
                slot_attr_mask[:, :, tok_idx] = sampled
                masked_attr_id[sampled] = tok_idx + 1

            # input_mask indicates which slots are present/valid for attention
            input_mask = valid_mask.clone()

            with torch.no_grad():
                elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=slot_attr_mask)
                ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=input_mask, masked_attr_id=masked_attr_id)
                pred_text_emb = decoders['text'](ctx)  # (B, S, D)

            B = len(batch_idx)
            # flatten queries and filter only the masked positions we want
            pred_flat = pred_text_emb.view(B * S, -1)
            mask_flat = sampled.view(-1)
            queries = pred_flat[mask_flat]
            if queries.numel() == 0:
                # no valid slots (or no sampled masked slots) -> write empty list per poster
                for _ in batch_idx:
                    fout.write(json.dumps([], ensure_ascii=False) + "\n")
                continue

            # normalize queries
            qn = queries / (queries.norm(dim=1, keepdim=True) + 1e-8)
            # compute cosine sim: (Q, D) @ (D, C) -> (Q, C)
            sims = torch.matmul(qn, emb_t.t())
            top_idx = sims.argmax(dim=1).cpu().numpy()

            # map top_idx back into per-poster lists containing only predicted masked entries
            ptr = 0
            for mbi, bi in enumerate(batch_idx):
                out_list = []
                gt_list = slot_texts[bi]
                for s in range(S):
                    if not valid_mask[mbi, s]:
                        continue
                    if not sampled[mbi, s]:
                        # not masked -> we won't predict; skip
                        continue
                    # this position was masked and predicted
                    sel = top_idx[ptr]
                    pred_str = texts[int(sel)]
                    gt_str = None
                    try:
                        gt_str = gt_list[s]
                    except Exception:
                        gt_str = None
                    out_list.append({"gt": gt_str, "pred": pred_str})
                    ptr += 1
                fout.write(json.dumps(out_list, ensure_ascii=False) + "\n")

    print('Wrote NN decoded texts to', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-ckpt', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--base-dir', default='data/crello')
    p.add_argument('--out', default=None)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--max-posters', type=int, default=None)
    p.add_argument('--max-candidates', type=int, default=None, help='limit candidate pool size (train texts)')
    p.add_argument('--mask-prob', type=float, default=0.25, help='probability of masking each valid slot for prediction')
    p.add_argument('--seed', type=int, default=None, help='random seed for masked-slot selection')
    p.add_argument('--mask-count', type=int, default=None, help='exact number of slots to mask per poster (overrides --mask-prob when set)')
    args = p.parse_args()
    nn_decode(args)


if __name__ == '__main__':
    main()
