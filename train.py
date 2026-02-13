"""Training script for the two-stage transformer.

This script supports running over a named split (e.g. --split test) which expects
files like data/crello/poster_input_test_X.npy, etc. If --split is omitted it will
fall back to legacy poster_inputs_*.npy names.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from models.two_stage_transformer import AttributeStage, ElementStage

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default=None, help='Which poster_input split to use, e.g. "test" or "train". If omitted uses legacy poster_inputs_ files')
parser.add_argument('--val-split', type=str, default='validation', help='Which split to use for validation (default: validation).')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--mask-prob', type=float, default=0.25)
parser.add_argument('--mask-attrs', type=str, default='text', help='Comma-separated attribute names to mask, or "random" to pick random attr per masked slot, or "all" to mask all token types')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--use-wandb', action='store_true', help='Log metrics to Weights & Biases if installed')
parser.add_argument('--patience', type=int, default=50, help='Early stopping patience on validation loss (epochs)')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save best model checkpoints')
args = parser.parse_args()

device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

# choose files based on split
base = "data/crello"
if args.split:
    prefix = f"poster_input_{args.split}"
    X_path = os.path.join(base, f"{prefix}_X.npy")
    mask_path = os.path.join(base, f"{prefix}_mask.npy")
    font_path = os.path.join(base, f"{prefix}_font_idx.npy")
    type_path = os.path.join(base, f"{prefix}_type_idx.npy")
    schema_path = os.path.join(base, f"{prefix}_schema.json")
else:
    X_path = os.path.join(base, 'poster_inputs_X.npy')
    mask_path = os.path.join(base, 'poster_inputs_mask.npy')
    font_path = os.path.join(base, 'poster_inputs_font_idx.npy')
    type_path = os.path.join(base, 'poster_inputs_type_idx.npy')
    schema_path = os.path.join(base, 'poster_inputs_schema.json')

print('Using:', X_path)

# mmap large arrays
X_all = np.load(X_path, mmap_mode='r')
MASK_all = np.load(mask_path, mmap_mode='r')
FONT_all = np.load(font_path, mmap_mode='r')
TYPE_all = np.load(type_path, mmap_mode='r')
with open(schema_path, 'r') as f:
    schema = json.load(f)

N = X_all.shape[0]
S = X_all.shape[1]
F = X_all.shape[2]
print(f'dataset shape: N={N}, S={S}, F={F}')

# model hyperparams (smaller dims for quick runs; tune as needed)
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
d_attr = 128
D_elem = 256
attr_names = [f['name'] for f in schema['fields']]

num_fonts = schema.get('font', {}).get('num_fonts', int(FONT_all.max() + 1))
num_roles = schema.get('type', {}).get('num_types', int(TYPE_all.max() + 1))

print('num_fonts', num_fonts, 'num_roles', num_roles)

# build models
attr_stage = AttributeStage(
    img_dim=schema['fields'][0]['dim'],
    txt_dim=[f for f in schema['fields'] if f['name'] == 'text'][0]['dim'],
    d_attr=d_attr,
    D_elem=D_elem,
    num_fonts=num_fonts,
).to(device)

elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=S, num_attributes=len(attr_names) + 1).to(device)

# per-attribute decoders (regressors) for continuous attributes
fields = {f['name']: f for f in schema['fields']}
decoders = {name: nn.Linear(D_elem, fields[name]['dim']).to(device) for name in attr_names}

# font classification head (replace regression with classification)
font_classifier = nn.Linear(D_elem, num_fonts).to(device)

params = list(attr_stage.parameters()) + list(elem_stage.parameters()) + [p for d in decoders.values() for p in d.parameters()] + list(font_classifier.parameters())
opt = torch.optim.Adam(params, lr=lr)

# tokenizer order (explicit) includes font at end
tokenizer_order = ['image', 'text', 'pos', 'size', 'angle', 'opacity', 'font']

# parse mask attrs argument
if args.mask_attrs == 'random':
    mask_attrs_mode = 'random'
elif args.mask_attrs == 'all':
    mask_attrs_mode = 'all'
else:
    mask_attrs_mode = 'list'
    mask_attr_list = [a.strip() for a in args.mask_attrs.split(',') if a.strip()]
    # validate
    for a in mask_attr_list:
        if a not in tokenizer_order:
            raise ValueError(f"Unknown attribute '{a}' in --mask-attrs. Valid: {tokenizer_order}")

# wandb init (optional)
use_wandb = args.use_wandb
wandb = None
if use_wandb:
    try:
        import wandb
        wandb.init(project='two-stage-poster', config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'mask_prob': args.mask_prob,
            'mask_attrs': args.mask_attrs,
        })
    except Exception as e:
        print('wandb requested but not available or failed to init:', e)
        use_wandb = False

step = 0
best_val = float('inf')
best_epoch = -1
no_improve = 0
os.makedirs(args.save_dir, exist_ok=True)
for epoch in range(epochs):
    indices = np.arange(N)
    np.random.shuffle(indices)
    epoch_loss = 0.0
    attr_losses_acc = {name: 0.0 for name in attr_names}
    attr_counts = {name: 0 for name in attr_names}
    for i in range(0, N, batch_size):
        batch_idx = indices[i : i + batch_size]
        Xb = torch.from_numpy(np.array(X_all[batch_idx])).float().to(device)
        MASKb = torch.from_numpy(np.array(MASK_all[batch_idx])).to(device)
        FONTb = torch.from_numpy(np.array(FONT_all[batch_idx])).long().to(device)
        TYPEb = torch.from_numpy(np.array(TYPE_all[batch_idx])).long().to(device)

        B = Xb.shape[0]
        # slice attributes using schema offsets
        img = Xb[:, :, fields['image']['offset'][0] : fields['image']['offset'][1]]
        text = Xb[:, :, fields['text']['offset'][0] : fields['text']['offset'][1]]
        pos = Xb[:, :, fields['pos']['offset'][0] : fields['pos']['offset'][1]]
        size = Xb[:, :, fields['size']['offset'][0] : fields['size']['offset'][1]]
        angle = Xb[:, :, fields['angle']['offset'][0] : fields['angle']['offset'][1]]
        opacity = Xb[:, :, fields['opacity']['offset'][0] : fields['opacity']['offset'][1]]

        # determine which slots actually contain text by inspecting the font index.
        # In this dataset font==0 denotes a non-text element; font>0 indicates a text font.
        # Use FONTb to decide presence instead of checking text embedding norm (which may
        # be non-zero for non-text elements).
        has_text = (FONTb != 0)
        # also compute font presence explicitly so we can gate font masking early
        has_font = (FONTb != 0)

        # build masks
        valid_mask = (MASKb == 1)
        rand = torch.rand(valid_mask.shape, device=device)
        target_mask = (rand < args.mask_prob) & valid_mask
        # per-attribute mask construction
        slot_attr_mask = torch.zeros((B, S, len(tokenizer_order)), dtype=torch.bool, device=device)
        masked_attr_id = torch.zeros((B, S), dtype=torch.long, device=device)
        if mask_attrs_mode == 'random':
            # pick random attribute indices per masked slot
            rand_idx = torch.randint(0, len(tokenizer_order), (B, S), device=device)
            for tt in range(len(tokenizer_order)):
                m = (rand_idx == tt) & target_mask
                # gate masking for attributes that may be absent: text and font
                if tokenizer_order[tt] == 'text':
                    m = m & has_text
                elif tokenizer_order[tt] == 'font':
                    m = m & has_font
                slot_attr_mask[:, :, tt] = m
                masked_attr_id[m] = tt + 1
        elif mask_attrs_mode == 'all':
            # mask all attributes for masked slots, but only mask text where text exists
            for tt in range(len(tokenizer_order)):
                if tokenizer_order[tt] == 'text':
                    slot_attr_mask[:, :, tt] = target_mask & has_text
                elif tokenizer_order[tt] == 'font':
                    slot_attr_mask[:, :, tt] = target_mask & has_font
                else:
                    slot_attr_mask[:, :, tt] = target_mask
            # mark masked_attr_id for text (and font will only be set where has_font True)
            masked_attr_id[target_mask & has_text] = 1  # non-zero indicates masked (text only when has_text)
        else:
            # list mode: cycle through provided attrs for masked slots (or apply all of them)
            for a in mask_attr_list:
                tt = tokenizer_order.index(a)
                if tokenizer_order[tt] == 'text':
                    m = target_mask & has_text
                elif tokenizer_order[tt] == 'font':
                    m = target_mask & has_font
                else:
                    m = target_mask
                slot_attr_mask[:, :, tt] = m
                masked_attr_id[m] = tt + 1

        input_mask = valid_mask.clone()
        input_mask[target_mask] = 0

        # --- presence gating for text and font ---
        # has_text is determined from FONTb above. For font presence (used when gating
        # font masking), treat font==0 as absent and font!=0 as present.
        has_font = (FONTb != 0)

        # masked_attr_id uses tt+1 encoding where tt is tokenizer_order index
        # build mask_block for slots that were selected to mask text/font but actually lack the attribute
        device = masked_attr_id.device
        text_mask_id = tokenizer_order.index('text') + 1 if 'text' in tokenizer_order else None
        font_mask_id = tokenizer_order.index('font') + 1 if 'font' in tokenizer_order else None

        mask_block = torch.zeros_like(masked_attr_id, dtype=torch.bool, device=device)
        if text_mask_id is not None:
            mask_block |= (masked_attr_id == int(text_mask_id)) & (~has_text.to(device))
        if font_mask_id is not None:
            mask_block |= (masked_attr_id == int(font_mask_id)) & (~has_font.to(device))

        # Clear masked_attr_id for blocked slots (0 == no-mask)
        if mask_block.any():
            masked_attr_id = masked_attr_id.masked_fill(mask_block, 0)
            # also clear the slot_attr_mask channels corresponding to text/font so the attr_stage won't see them
            if text_mask_id is not None:
                tt = text_mask_id - 1
                slot_attr_mask[:, :, tt] = slot_attr_mask[:, :, tt] & (~mask_block)
            if font_mask_id is not None:
                tt = font_mask_id - 1
                slot_attr_mask[:, :, tt] = slot_attr_mask[:, :, tt] & (~mask_block)

        # forward
        elem_emb = attr_stage(img, text, pos, size, angle, opacity, FONTb, slot_attr_mask=slot_attr_mask)
        ctx = elem_stage(elem_emb, role_idx=TYPEb, mask=input_mask, masked_attr_id=masked_attr_id)

        # decode per-attribute losses
        total_loss = 0.0
        any_masked = False
        # compute losses for continuous attributes + font classification
        for idx, name in enumerate(attr_names + ['font']):
            if name == 'font':
                # classification target comes from FONTb (integers)
                targ_font = FONTb  # shape (B, S)
                logits = font_classifier(ctx)  # (B, S, num_fonts)
                C = logits.shape[-1]
                # flatten and compute per-position CE without reduction
                ce = nn.CrossEntropyLoss(reduction='none')
                logits_flat = logits.view(-1, C)
                targ_flat = targ_font.view(-1)
                loss_flat = ce(logits_flat, targ_flat)  # (B*S,)
                loss_pos = loss_flat.view(B, -1)  # (B, S)
                # masked slots for font are where masked_attr_id == (tok_idx+1) with tok_idx for 'font'
                if 'font' in tokenizer_order:
                    tok_idx = tokenizer_order.index('font')
                    mask_slots = (masked_attr_id == (tok_idx + 1))
                else:
                    mask_slots = torch.zeros_like(masked_attr_id, dtype=torch.bool)
                if mask_slots.any():
                    any_masked = True
                    loss_attr = loss_pos[mask_slots].mean()
                    total_loss = total_loss + loss_attr
                    attr_losses_acc['font'] = attr_losses_acc.get('font', 0.0) + float(loss_attr.item())
                    attr_counts['font'] = attr_counts.get('font', 0) + 1
            else:
                a, b = fields[name]['offset']
                targ = Xb[..., a:b]
                pred = decoders[name](ctx)
                mse = (pred - targ).pow(2).mean(dim=-1)
                # check masked slots for this attribute token index
                if name in tokenizer_order:
                    tok_idx = tokenizer_order.index(name)
                    mask_slots = (masked_attr_id == (tok_idx + 1))
                else:
                    mask_slots = torch.zeros_like(masked_attr_id, dtype=torch.bool)
                if mask_slots.any():
                    any_masked = True
                    loss_attr = mse[mask_slots].mean()
                    total_loss = total_loss + loss_attr
                    attr_losses_acc[name] += float(loss_attr.item())
                    attr_counts[name] += 1
        if not any_masked:
            continue
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        step += 1
        epoch_loss += float(total_loss.item())
        if step % 10 == 0:
            #print(f'epoch {epoch} step {step} batch_loss {total_loss.item():.6f}')
            if use_wandb:
                wandb.log({'train/batch_loss': float(total_loss.item()), 'step': step})
                wandb.log({f'train/epoch_loss': epoch_loss / step, 'step': step})


    if step > 0:
        print('epoch', epoch, 'avg batch loss', epoch_loss / max(1, step))
        for name in attr_names:
            if attr_counts[name] > 0:
                print(f'  {name} avg loss: {attr_losses_acc[name] / attr_counts[name]:.6f} (n={attr_counts[name]})')
    else:
        print('No masked slots encountered in epoch')

    # run validation pass at epoch end
    # load validation split
    val_loss = None
    val_steps = 0
    val_epoch_loss = 0.0
    # include 'font' in validation loss tracking so we can accumulate font CE loss
    val_attr_losses = {name: 0.0 for name in (attr_names + ['font'])}
    val_attr_counts = {name: 0 for name in (attr_names + ['font'])}
    # try to locate val files
    val_prefix = f"poster_input_{args.val_split}"
    val_X_path = os.path.join(base, f"{val_prefix}_X.npy")
    if not os.path.exists(val_X_path):
        # try 'val'
        val_prefix = f"poster_input_val"
        val_X_path = os.path.join(base, f"{val_prefix}_X.npy")
    if os.path.exists(val_X_path):
        print('Running validation on', val_prefix)
        val_X_all = np.load(val_X_path, mmap_mode='r')
        val_MASK_all = np.load(os.path.join(base, f"{val_prefix}_mask.npy"), mmap_mode='r')
        val_FONT_all = np.load(os.path.join(base, f"{val_prefix}_font_idx.npy"), mmap_mode='r')
        val_TYPE_all = np.load(os.path.join(base, f"{val_prefix}_type_idx.npy"), mmap_mode='r')
        Nval = val_X_all.shape[0]
        for j in range(0, Nval, batch_size):
            bj = slice(j, min(j + batch_size, Nval))
            Xv = torch.from_numpy(np.array(val_X_all[bj])).float().to(device)
            MASKv = torch.from_numpy(np.array(val_MASK_all[bj])).to(device)
            FONTv = torch.from_numpy(np.array(val_FONT_all[bj])).long().to(device)
            TYPEv = torch.from_numpy(np.array(val_TYPE_all[bj])).long().to(device)
            Bv = Xv.shape[0]
            img = Xv[:, :, fields['image']['offset'][0] : fields['image']['offset'][1]]
            text = Xv[:, :, fields['text']['offset'][0] : fields['text']['offset'][1]]
            pos = Xv[:, :, fields['pos']['offset'][0] : fields['pos']['offset'][1]]
            size = Xv[:, :, fields['size']['offset'][0] : fields['size']['offset'][1]]
            angle = Xv[:, :, fields['angle']['offset'][0] : fields['angle']['offset'][1]]
            opacity = Xv[:, :, fields['opacity']['offset'][0] : fields['opacity']['offset'][1]]

            valid_mask_v = (MASKv == 1)
            rand_v = torch.rand(valid_mask_v.shape, device=device)
            target_mask_v = (rand_v < args.mask_prob) & valid_mask_v

            # determine which slots actually contain text in validation by checking font index
            # font==0 denotes non-text element
            has_text_v = (FONTv != 0)
            # also compute font presence for validation so we can gate font masking early
            has_font_v = (FONTv != 0)

            # construct slot_attr_mask for val using mask_attrs_mode
            slot_attr_mask_v = torch.zeros((Bv, S, len(tokenizer_order)), dtype=torch.bool, device=device)
            masked_attr_id_v = torch.zeros((Bv, S), dtype=torch.long, device=device)
            if mask_attrs_mode == 'random':
                rand_idx = torch.randint(0, len(tokenizer_order), (Bv, S), device=device)
                for tt in range(len(tokenizer_order)):
                    m = (rand_idx == tt) & target_mask_v
                    # gate text/font presence in validation masking as well
                    if tokenizer_order[tt] == 'text':
                        m = m & has_text_v
                    elif tokenizer_order[tt] == 'font':
                        m = m & has_font_v
                    slot_attr_mask_v[:, :, tt] = m
                    masked_attr_id_v[m] = tt + 1
            elif mask_attrs_mode == 'all':
                for tt in range(len(tokenizer_order)):
                    if tokenizer_order[tt] == 'text':
                        slot_attr_mask_v[:, :, tt] = target_mask_v & has_text_v
                    elif tokenizer_order[tt] == 'font':
                        slot_attr_mask_v[:, :, tt] = target_mask_v & has_font_v
                    else:
                        slot_attr_mask_v[:, :, tt] = target_mask_v
                masked_attr_id_v[target_mask_v & has_text_v] = 1
            else:
                for a in mask_attr_list:
                    tt = tokenizer_order.index(a)
                    if tokenizer_order[tt] == 'text':
                        m = target_mask_v & has_text_v
                    elif tokenizer_order[tt] == 'font':
                        m = target_mask_v & has_font_v
                    else:
                        m = target_mask_v
                    slot_attr_mask_v[:, :, tt] = m
                    masked_attr_id_v[m] = tt + 1

            input_mask_v = valid_mask_v.clone()
            input_mask_v[target_mask_v] = 0

            # --- presence gating for text and font in validation ---
            # has_text_v computed above (Bv, S). For font presence, treat font==0 as absent.
            has_font_v = (FONTv != 0)

            device_v = masked_attr_id_v.device
            text_mask_id_v = tokenizer_order.index('text') + 1 if 'text' in tokenizer_order else None
            font_mask_id_v = tokenizer_order.index('font') + 1 if 'font' in tokenizer_order else None

            mask_block_v = torch.zeros_like(masked_attr_id_v, dtype=torch.bool, device=device_v)
            if text_mask_id_v is not None:
                mask_block_v |= (masked_attr_id_v == int(text_mask_id_v)) & (~has_text_v.to(device_v))
            if font_mask_id_v is not None:
                mask_block_v |= (masked_attr_id_v == int(font_mask_id_v)) & (~has_font_v.to(device_v))

            if mask_block_v.any():
                masked_attr_id_v = masked_attr_id_v.masked_fill(mask_block_v, 0)
                if text_mask_id_v is not None:
                    tt = text_mask_id_v - 1
                    slot_attr_mask_v[:, :, tt] = slot_attr_mask_v[:, :, tt] & (~mask_block_v)
                if font_mask_id_v is not None:
                    tt = font_mask_id_v - 1
                    slot_attr_mask_v[:, :, tt] = slot_attr_mask_v[:, :, tt] & (~mask_block_v)

            with torch.no_grad():
                elem_emb_v = attr_stage(img, text, pos, size, angle, opacity, FONTv, slot_attr_mask=slot_attr_mask_v)
                ctx_v = elem_stage(elem_emb_v, role_idx=TYPEv, mask=input_mask_v, masked_attr_id=masked_attr_id_v)

                total_val_batch = 0.0
                any_masked_v = False
                for idx, name in enumerate(attr_names + ['font']):
                    if name == 'font':
                        # font classification
                        targ_font_v = FONTv
                        logits_v = font_classifier(ctx_v)  # (Bv, S, num_fonts)
                        C = logits_v.shape[-1]
                        ce = nn.CrossEntropyLoss(reduction='none')
                        logits_flat_v = logits_v.view(-1, C)
                        targ_flat_v = targ_font_v.view(-1)
                        loss_flat_v = ce(logits_flat_v, targ_flat_v)
                        loss_pos_v = loss_flat_v.view(Bv, -1)
                        if 'font' in tokenizer_order:
                            tok_idx = tokenizer_order.index('font')
                            mask_slots_v = (masked_attr_id_v == (tok_idx + 1))
                        else:
                            mask_slots_v = torch.zeros_like(masked_attr_id_v, dtype=torch.bool)
                        if mask_slots_v.any():
                            any_masked_v = True
                            loss_attr_v = loss_pos_v[mask_slots_v].mean()
                            total_val_batch = total_val_batch + loss_attr_v
                            val_attr_losses['font'] += float(loss_attr_v.item())
                            val_attr_counts['font'] += 1
                    else:
                        a, b = fields[name]['offset']
                        targ = Xv[..., a:b]
                        pred = decoders[name](ctx_v)
                        mse = (pred - targ).pow(2).mean(dim=-1)
                        if name in tokenizer_order:
                            tok_idx = tokenizer_order.index(name)
                            mask_slots_v = (masked_attr_id_v == (tok_idx + 1))
                        else:
                            mask_slots_v = torch.zeros_like(masked_attr_id_v, dtype=torch.bool)
                        if mask_slots_v.any():
                            any_masked_v = True
                            loss_attr_v = mse[mask_slots_v].mean()
                            total_val_batch = total_val_batch + loss_attr_v
                            val_attr_losses[name] += float(loss_attr_v.item())
                            val_attr_counts[name] += 1
                if any_masked_v:
                    val_steps += 1
                    val_epoch_loss += float(total_val_batch.item())

        if val_steps > 0:
            val_loss = val_epoch_loss / val_steps
            print(f'val loss: {val_loss:.6e}')
            if use_wandb:
                wandb.log({'val/loss': val_loss, 'epoch': epoch})
            # early stopping logic
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                # save checkpoint
                ckpt = {
                    'epoch': epoch,
                    'best_val': best_val,
                    'attr_stage': attr_stage.state_dict(),
                    'elem_stage': elem_stage.state_dict(),
                    'decoders': {n: d.state_dict() for n, d in decoders.items()},
                    'optimizer': opt.state_dict(),
                    'tokenizer_order': tokenizer_order,
                    'attr_names': attr_names,
                }
                ckpt_path = os.path.join(args.save_dir, f'best_epoch.pth')
                torch.save(ckpt, ckpt_path)
                print('Saved new best checkpoint to', ckpt_path)
            else:
                no_improve += 1
                print(f'No improvement for {no_improve} epochs (patience {args.patience})')
            if no_improve >= args.patience:
                print(f'Early stopping: no improvement for {no_improve} epochs. Best val {best_val} at epoch {best_epoch}')
                if use_wandb:
                    wandb.finish()
                exit(0)
        else:
            print('Validation had no masked slots; skipping early-stop update')

print('Done. Steps:', step)