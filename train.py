"""Training script for the two-stage transformer.

This script supports running over a named split (e.g. --split test) which expects
files like data/crello/poster_input_test_X.npy, etc. If --split is omitted it will
fall back to legacy poster_inputs_*.npy names.
"""

import argparse
import atexit
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from models.two_stage_transformer import AttributeStage, ElementStage
from models.heads import make_decoder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_preprocessing_metadata(base_dir, split):
    metadata = {'variant': split or 'legacy'}
    if not split or '_' not in split:
        return metadata
    source_split, suffix = split.split('_', 1)
    path = os.path.join(base_dir, f'crello_{source_split}_image_embeddings_{suffix}.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            metadata.update(json.load(f))
        metadata['metadata_path'] = os.path.abspath(path)
    return metadata


def make_regression_stats(attr_names, fields):
    stats = {}
    for name in attr_names:
        dim = fields[name]['dim']
        stats[name] = {
            'sse': np.zeros(dim, dtype=np.float64),
            'sae': np.zeros(dim, dtype=np.float64),
            'target_sum': np.zeros(dim, dtype=np.float64),
            'target_sq_sum': np.zeros(dim, dtype=np.float64),
            'negative': np.zeros(dim, dtype=np.int64),
            'count': 0,
        }
    return stats


def update_regression_stats(stats, name, prediction, target, mask_slots):
    if not mask_slots.any():
        return
    selected_prediction = prediction.detach()[mask_slots]
    selected_target = target.detach()[mask_slots]
    difference = selected_prediction - selected_target
    entry = stats[name]
    entry['sse'] += difference.square().sum(dim=0).cpu().double().numpy()
    entry['sae'] += difference.abs().sum(dim=0).cpu().double().numpy()
    entry['target_sum'] += selected_target.sum(dim=0).cpu().double().numpy()
    entry['target_sq_sum'] += selected_target.square().sum(dim=0).cpu().double().numpy()
    entry['negative'] += (selected_prediction < 0).sum(dim=0).cpu().numpy()
    entry['count'] += int(selected_prediction.shape[0])


def dimension_labels(name, dimension):
    known = {
        'pos': ['left', 'top'],
        'size': ['width', 'height'],
        'angle': ['angle'],
        'opacity': ['opacity'],
    }
    labels = known.get(name, [])
    return labels if len(labels) == dimension else []


def regression_metrics(stats, prefix):
    metrics = {}
    for name, entry in stats.items():
        count = entry['count']
        if count == 0:
            continue
        dimension = len(entry['sse'])
        metrics[f'{prefix}/{name}/mse'] = float(entry['sse'].sum() / (count * dimension))
        metrics[f'{prefix}/{name}/mae'] = float(entry['sae'].sum() / (count * dimension))
        for index, label in enumerate(dimension_labels(name, dimension)):
            mse = entry['sse'][index] / count
            mae = entry['sae'][index] / count
            target_ss = entry['target_sq_sum'][index] - entry['target_sum'][index] ** 2 / count
            r2 = 1.0 - entry['sse'][index] / target_ss if target_ss > 0 else float('nan')
            metrics[f'{prefix}/{name}/{label}_mse'] = float(mse)
            metrics[f'{prefix}/{name}/{label}_mae'] = float(mae)
            metrics[f'{prefix}/{name}/{label}_r2'] = float(r2)
            metrics[f'{prefix}/{name}/{label}_negative_rate'] = float(entry['negative'][index] / count)
    return metrics

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir', default='data/crello', help='Directory containing poster_input files')
parser.add_argument('--split', type=str, default=None, help='Which poster_input split to use, e.g. "test" or "train". If omitted uses legacy poster_inputs_ files')
parser.add_argument('--val-split', type=str, default='validation', help='Which split to use for validation (default: validation).')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--mask-prob', type=float, default=0.25)
parser.add_argument('--mask-attrs', type=str, default='text', help='Comma-separated attribute names to mask, or "random" to pick random attr per masked slot, or "all" to mask all token types')
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--use-wandb', action='store_true', help='Log metrics to Weights & Biases if installed')
parser.add_argument('--wandb-project', default='poster-element-relations')
parser.add_argument('--wandb-entity', default=None)
parser.add_argument('--wandb-run-name', default=None)
parser.add_argument('--wandb-group', default=None)
parser.add_argument('--wandb-tags', default='', help='Comma-separated W&B tags')
parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
parser.add_argument('--wandb-dir', default='wandb', help='Local directory for W&B run files')
parser.add_argument('--wandb-log-every', type=int, default=10, help='Log batch metrics every N optimizer steps')
parser.add_argument('--wandb-log-model', action=argparse.BooleanOptionalAction, default=True, help='Upload the final best checkpoint as one W&B artifact')
parser.add_argument('--debug-masks', action='store_true', help='Print per-batch mask/debug counts')
parser.add_argument('--record-mse', action='store_true', help='Accumulate per-attribute SSE/counts during train/val and save MSEs to checkpoint')
parser.add_argument('--mask-gate-font', action='store_true', help='Gate masking of size (and other font-linked attrs) to slots where FONT != 0')
parser.add_argument('--patience', type=int, default=50, help='Early stopping patience on validation loss (epochs)')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save best model checkpoints')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

if args.wandb_log_every <= 0:
    parser.error('--wandb-log-every must be positive')

seed_everything(args.seed)

device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

# choose files based on split
base = args.base_dir
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
preprocessing_metadata = {
    'train': load_preprocessing_metadata(base, args.split),
    'validation': load_preprocessing_metadata(base, args.val_split),
}
task_metadata = {
    'target_image_visible': True,
    'target_image_jointly_masked': False,
    'size_representation': 'width_height_divided_by_1000',
    'validation_masks_fixed': True,
    'validation_mask_seed': args.seed + 2,
}
data_paths = {
    'train_X': os.path.abspath(X_path),
    'train_mask': os.path.abspath(mask_path),
    'train_font': os.path.abspath(font_path),
    'train_type': os.path.abspath(type_path),
    'train_schema': os.path.abspath(schema_path),
}

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
decoders = {name: make_decoder(D_elem, fields[name]['dim']).to(device) for name in attr_names}

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
wandb_run = None


def finish_wandb():
    global wandb_run
    if wandb_run is not None:
        wandb_run.finish()
        wandb_run = None


atexit.register(finish_wandb)
if use_wandb:
    try:
        wandb_dir = os.path.abspath(args.wandb_dir)
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ.setdefault('WANDB_CACHE_DIR', os.path.join(wandb_dir, 'cache'))
        os.environ.setdefault('WANDB_CONFIG_DIR', os.path.join(wandb_dir, 'config'))
        os.environ.setdefault('WANDB_DATA_DIR', os.path.join(wandb_dir, 'data'))
        import wandb
        tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
        run_config = vars(args).copy()
        run_config.update({
            'resolved_device': str(device),
            'dataset_shape': [N, S, F],
            'data_paths': data_paths,
            'preprocessing': preprocessing_metadata,
            'task': task_metadata,
            'schema': schema,
            'd_attr': d_attr,
            'D_elem': D_elem,
            'num_fonts': num_fonts,
            'num_roles': num_roles,
        })
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=tags,
            job_type='train',
            mode=args.wandb_mode,
            dir=wandb_dir,
            config=run_config,
        )
        wandb.define_metric('global_step')
        wandb.define_metric('train/batch_*', step_metric='global_step')
        wandb.define_metric('epoch')
        wandb.define_metric('train/epoch_*', step_metric='epoch')
        wandb.define_metric('train/masked_slots', step_metric='epoch')
        for name in attr_names:
            wandb.define_metric(f'train/{name}/*', step_metric='epoch')
        wandb.define_metric('val/*', step_metric='epoch')
        wandb.define_metric('optimizer/*', step_metric='epoch')
    except Exception as e:
        print('wandb requested but not available or failed to init:', e)
        use_wandb = False
        wandb_run = None

collect_metrics = use_wandb or args.record_mse
step = 0
best_val = float('inf')
best_epoch = -1
no_improve = 0
best_ckpt_path = None
stopped_early = False
shuffle_rng = np.random.default_rng(args.seed)
train_mask_generator = torch.Generator(device=device)
train_mask_generator.manual_seed(args.seed + 1)
os.makedirs(args.save_dir, exist_ok=True)
for epoch in range(epochs):
    indices = shuffle_rng.permutation(N)
    epoch_loss = 0.0
    epoch_steps = 0
    epoch_masked_slots = 0
    attr_losses_acc = {name: 0.0 for name in attr_names}
    attr_counts = {name: 0 for name in attr_names}
    train_stats = make_regression_stats(attr_names, fields)
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
        rand = torch.rand(valid_mask.shape, device=device, generator=train_mask_generator)
        target_mask = (rand < args.mask_prob) & valid_mask
        # per-attribute mask construction
        slot_attr_mask = torch.zeros((B, S, len(tokenizer_order)), dtype=torch.bool, device=device)
        masked_attr_id = torch.zeros((B, S), dtype=torch.long, device=device)
        if mask_attrs_mode == 'random':
            # pick random attribute indices per masked slot
            rand_idx = torch.randint(
                0, len(tokenizer_order), (B, S), device=device, generator=train_mask_generator
            )
            for tt in range(len(tokenizer_order)):
                m = (rand_idx == tt) & target_mask
                # gate masking for attributes that may be absent: text and font
                if tokenizer_order[tt] == 'text':
                    m = m & has_text
                elif tokenizer_order[tt] == 'font':
                    m = m & has_font
                elif args.mask_gate_font and tokenizer_order[tt] == 'size':
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
                    elif args.mask_gate_font and tokenizer_order[tt] == 'size':
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
                elif args.mask_gate_font and tokenizer_order[tt] == 'size':
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

        # debug mask counts
        if args.debug_masks:
            valid_slots = valid_mask.sum().item()
            target_slots = target_mask.sum().item()
            masked_ids = (masked_attr_id != 0).sum().item()
            # per-attribute masked counts
            per_attr_counts = {tokenizer_order[t]: int(slot_attr_mask[:, :, t].sum().item()) for t in range(len(tokenizer_order))}
            print(f"[train debug] valid_slots={valid_slots} target_slots={target_slots} masked_ids={masked_ids} per_attr={per_attr_counts}")

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
                    if collect_metrics:
                        update_regression_stats(train_stats, name, pred, targ, mask_slots)
        if not any_masked:
            continue
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        step += 1
        epoch_steps += 1
        epoch_masked_slots += int((masked_attr_id != 0).sum().item())
        epoch_loss += float(total_loss.item())
        if use_wandb and step % args.wandb_log_every == 0:
            wandb_run.log({
                'train/batch_loss': float(total_loss.item()),
                'train/batch_running_loss': epoch_loss / epoch_steps,
                'global_step': step,
                'epoch': epoch,
            })

    train_epoch_loss = epoch_loss / epoch_steps if epoch_steps else None
    if epoch_steps > 0:
        print('epoch', epoch, 'avg batch loss', train_epoch_loss)
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
    val_masked_slots = 0
    val_stats = make_regression_stats(attr_names, fields)
    val_mask_generator = torch.Generator(device=device)
    val_mask_generator.manual_seed(args.seed + 2)
    epoch_log = {
        'epoch': epoch,
        'global_step': step,
        'train/epoch_loss': train_epoch_loss,
        'train/masked_slots': epoch_masked_slots,
        'optimizer/learning_rate': float(opt.param_groups[0]['lr']),
    }
    epoch_log.update(regression_metrics(train_stats, 'train'))
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
        data_paths.update({
            'validation_X': os.path.abspath(val_X_path),
            'validation_mask': os.path.abspath(os.path.join(base, f"{val_prefix}_mask.npy")),
            'validation_font': os.path.abspath(os.path.join(base, f"{val_prefix}_font_idx.npy")),
            'validation_type': os.path.abspath(os.path.join(base, f"{val_prefix}_type_idx.npy")),
        })
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
            rand_v = torch.rand(
                valid_mask_v.shape, device=device, generator=val_mask_generator
            )
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
                rand_idx = torch.randint(
                    0,
                    len(tokenizer_order),
                    (Bv, S),
                    device=device,
                    generator=val_mask_generator,
                )
                for tt in range(len(tokenizer_order)):
                    m = (rand_idx == tt) & target_mask_v
                    # gate text/font presence in validation masking as well
                    if tokenizer_order[tt] == 'text':
                        m = m & has_text_v
                    elif tokenizer_order[tt] == 'font':
                        m = m & has_font_v
                    elif args.mask_gate_font and tokenizer_order[tt] == 'size':
                        m = m & has_font_v
                    slot_attr_mask_v[:, :, tt] = m
                    masked_attr_id_v[m] = tt + 1
            elif mask_attrs_mode == 'all':
                for tt in range(len(tokenizer_order)):
                    if tokenizer_order[tt] == 'text':
                        slot_attr_mask_v[:, :, tt] = target_mask_v & has_text_v
                    elif tokenizer_order[tt] == 'font':
                        slot_attr_mask_v[:, :, tt] = target_mask_v & has_font_v
                    elif args.mask_gate_font and tokenizer_order[tt] == 'size':
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
                    elif args.mask_gate_font and tokenizer_order[tt] == 'size':
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
                            if collect_metrics:
                                update_regression_stats(
                                    val_stats, name, pred, targ, mask_slots_v
                                )
                if args.debug_masks:
                    valid_slots_v = valid_mask_v.sum().item()
                    target_slots_v = target_mask_v.sum().item()
                    masked_ids_v = (masked_attr_id_v != 0).sum().item()
                    per_attr_counts_v = {tokenizer_order[t]: int(slot_attr_mask_v[:, :, t].sum().item()) for t in range(len(tokenizer_order))}
                    print(f"[val debug] valid_slots={valid_slots_v} target_slots={target_slots_v} masked_ids={masked_ids_v} per_attr={per_attr_counts_v}")

                if any_masked_v:
                    val_steps += 1
                    val_masked_slots += int((masked_attr_id_v != 0).sum().item())
                    val_epoch_loss += float(total_val_batch.item())

        if val_steps > 0:
            val_loss = val_epoch_loss / val_steps
            val_metrics = regression_metrics(val_stats, 'val')
            val_mse = {
                name: float(entry['sse'].sum() / (entry['count'] * len(entry['sse'])))
                for name, entry in val_stats.items()
                if entry['count'] > 0
            }
            train_mse = {
                name: float(entry['sse'].sum() / (entry['count'] * len(entry['sse'])))
                for name, entry in train_stats.items()
                if entry['count'] > 0
            }
            print(f'val loss: {val_loss:.6e}')
            if args.record_mse:
                print('Validation per-attribute MSEs:')
                for name in attr_names:
                    if name in val_mse:
                        print(f'  {name}: {val_mse[name]:.6e} (n={val_stats[name]["count"]})')
                    else:
                        print(f'  {name}: (no comparisons)')
                print('Train per-attribute MSEs (epoch):')
                for name in attr_names:
                    if name in train_mse:
                        print(f'  {name}: {train_mse[name]:.6e} (n={train_stats[name]["count"]})')
                    else:
                        print(f'  {name}: (no comparisons)')

            epoch_log.update({
                'val/loss': val_loss,
                'val/masked_slots': val_masked_slots,
            })
            epoch_log.update(val_metrics)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                no_improve = 0
                ckpt = {
                    'epoch': epoch,
                    'global_step': step,
                    'best_val': best_val,
                    'attr_stage': attr_stage.state_dict(),
                    'elem_stage': elem_stage.state_dict(),
                    'decoders': {n: d.state_dict() for n, d in decoders.items()},
                    'font_classifier': font_classifier.state_dict(),
                    'optimizer': opt.state_dict(),
                    'tokenizer_order': tokenizer_order,
                    'attr_names': attr_names,
                    'val_mse': val_mse if args.record_mse else None,
                    'val_metrics': val_metrics,
                    'train_metrics': regression_metrics(train_stats, 'train'),
                    'args': vars(args).copy(),
                    'seed': args.seed,
                    'schema': schema,
                    'data_paths': data_paths.copy(),
                    'preprocessing': preprocessing_metadata,
                    'task': task_metadata,
                    'model_config': {
                        'd_attr': d_attr,
                        'D_elem': D_elem,
                        'num_fonts': num_fonts,
                        'num_roles': num_roles,
                        'max_slots': S,
                    },
                    'wandb': {
                        'run_id': wandb_run.id if wandb_run is not None else None,
                        'project': args.wandb_project if use_wandb else None,
                        'run_name': wandb_run.name if wandb_run is not None else None,
                    },
                }
                best_ckpt_path = os.path.join(args.save_dir, 'best_epoch.pth')
                torch.save(ckpt, best_ckpt_path)
                print('Saved new best checkpoint to', best_ckpt_path)
            else:
                no_improve += 1
                print(f'No improvement for {no_improve} epochs (patience {args.patience})')

            epoch_log.update({
                'val/best_loss': float(best_val),
                'val/best_epoch': best_epoch,
                'val/epochs_without_improvement': no_improve,
            })
            if no_improve >= args.patience:
                print(f'Early stopping: no improvement for {no_improve} epochs. Best val {best_val} at epoch {best_epoch}')
                stopped_early = True
        else:
            print('Validation had no masked slots; skipping early-stop update')

    if use_wandb:
        wandb_run.config.update({'data_paths': data_paths}, allow_val_change=True)
        wandb_run.log(epoch_log)
        wandb_run.summary['best_val'] = float(best_val)
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.summary['global_step'] = step
    if stopped_early:
        break

print('Done. Steps:', step)

if use_wandb and args.wandb_log_model and best_ckpt_path and os.path.exists(best_ckpt_path):
    try:
        artifact = wandb.Artifact(
            name=f'{wandb_run.id}-best-model',
            type='model',
            metadata={
                'best_val': float(best_val),
                'best_epoch': best_epoch,
                'split': args.split,
                'val_split': args.val_split,
                'seed': args.seed,
                'preprocessing': preprocessing_metadata,
                'task': task_metadata,
            },
        )
        artifact.add_file(best_ckpt_path, name='best_epoch.pth')
        wandb_run.log_artifact(artifact, aliases=['best'])
    except Exception as error:
        print('Warning: failed to log best checkpoint as a W&B artifact:', error)
        wandb_run.summary['artifact_error'] = str(error)

finish_wandb()
