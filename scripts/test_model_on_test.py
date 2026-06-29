"""Evaluate a saved checkpoint on a poster inputs split (test by default).

Usage example:
  python3 scripts/test_model_on_test.py --ckpt checkpoints/best_epoch.pth --split test --batch-size 32 --device cuda

This script mirrors the training masking and decoding logic and reports per-attribute
losses and font accuracy on masked positions.

It expects the repository's preprocessing files under data/crello with names like
`poster_input_<split>_X.npy`, `poster_input_<split>_mask.npy`, `poster_input_<split>_font_idx.npy`,
`poster_input_<split>_type_idx.npy`, and `poster_input_<split>_schema.json`.

Note: model size hyperparameters (d_attr, D_elem) should match those used when saving
the checkpoint (defaults here match train.py: d_attr=128, D_elem=256).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from models.two_stage_transformer import AttributeStage, ElementStage, slice_X_by_schema


def make_paths(base, split):
	prefix = f"poster_input_{split}"
	return (
		os.path.join(base, f"{prefix}_X.npy"),
		os.path.join(base, f"{prefix}_mask.npy"),
		os.path.join(base, f"{prefix}_font_idx.npy"),
		os.path.join(base, f"{prefix}_type_idx.npy"),
		os.path.join(base, f"{prefix}_schema.json"),
	)


def load_schema(schema_path):
	with open(schema_path, 'r') as f:
		return json.load(f)


def build_models(schema, device, d_attr=128, D_elem=256, num_fonts_override: int = None):
	# infer dims from schema
	fields = {f['name']: f for f in schema['fields']}
	img_dim = fields['image']['dim']
	txt_dim = fields['text']['dim']
	num_fonts = schema.get('font', {}).get('num_fonts', 1)
	if num_fonts_override is not None:
		num_fonts = int(num_fonts_override)
	num_roles = schema.get('type', {}).get('num_types', 1)

	attr_stage = AttributeStage(
		img_dim=img_dim,
		txt_dim=txt_dim,
		num_fonts=num_fonts,
		d_attr=d_attr,
		D_elem=D_elem,
	).to(device)

	elem_stage = ElementStage(D_elem=D_elem, num_roles=num_roles, max_slots=schema.get('max_elems', 64), num_attributes=len(schema['fields']) + 1).to(device)

	# decoders for continuous attributes
	attr_names = [f['name'] for f in schema['fields']]
	from models.heads import make_decoder
	decoders = {name: make_decoder(D_elem, fields[name]['dim']).to(device) for name in attr_names}
	# font classifier
	font_classifier = nn.Linear(D_elem, num_fonts).to(device)

	return attr_stage, elem_stage, decoders, font_classifier


def evaluate_checkpoint(
	ckpt_path,
	base_dir,
	split,
	device,
	batch_size=32,
	mask_prob=0.25,
	mask_attrs='text',
	max_samples=None,
	deterministic=False,
):
	Xp, Mp, Fp, Tp, Sp = make_paths(base_dir, split)
	for p in (Xp, Mp, Fp, Tp, Sp):
		if not os.path.exists(p):
			raise FileNotFoundError(f"Required file not found: {p}")

	schema = load_schema(Sp)

	# load checkpoint early so we can match tokenizer/font sizes
	ckpt = torch.load(ckpt_path, map_location=device)
	# try to infer num_fonts from checkpoint attr_stage tokenizer weights
	ckpt_num_fonts = None
	if 'attr_stage' in ckpt and isinstance(ckpt['attr_stage'], dict):
		for k, v in ckpt['attr_stage'].items():
			# look for the font embedding weight key (endswith 'font_emb.weight')
			if k.endswith('font_emb.weight'):
				ckpt_num_fonts = v.shape[0]
				break
	# fallback: if top-level 'tokenizer.font_emb.weight' present
	if ckpt_num_fonts is None:
		for k, v in ckpt.items():
			if k.endswith('tokenizer.font_emb.weight') or k.endswith('font_emb.weight'):
				try:
					ckpt_num_fonts = v.shape[0]
					break
				except Exception:
					pass

	if ckpt_num_fonts is not None:
		print(f'Checkpoint font embedding has {ckpt_num_fonts} entries; building model to match')

	attr_stage, elem_stage, decoders, font_classifier = build_models(schema, device, d_attr=128, D_elem=256, num_fonts_override=ckpt_num_fonts)
	if 'attr_stage' in ckpt:
		try:
			attr_stage.load_state_dict(ckpt['attr_stage'])
		except RuntimeError as e:
			# fallback: try non-strict loading
			print('Strict load failed for attr_stage, trying non-strict load:', e)
			attr_stage.load_state_dict(ckpt['attr_stage'], strict=False)
	if 'elem_stage' in ckpt:
		elem_stage.load_state_dict(ckpt['elem_stage'])
	if 'decoders' in ckpt:
		for n, d in decoders.items():
			if n in ckpt['decoders']:
				d.load_state_dict(ckpt['decoders'][n])
	if 'font_classifier' in ckpt:
		try:
			font_classifier.load_state_dict(ckpt['font_classifier'])
		except Exception:
			# older checkpoints may have font in 'decoders' or not saved; ignore
			pass
	else:
		# older train script saved font in decoders? handled above
		pass

	# load test arrays (mmap)
	X_all = np.load(Xp, mmap_mode='r')
	M_all = np.load(Mp, mmap_mode='r')
	F_all = np.load(Fp, mmap_mode='r')
	T_all = np.load(Tp, mmap_mode='r')

	N = X_all.shape[0]
	if max_samples is not None:
		N = min(N, max_samples)

	attr_names = [f['name'] for f in schema['fields']]
	tokenizer_order = ['image', 'text', 'pos', 'size', 'angle', 'opacity', 'font']

	# mask attrs parsing
	if mask_attrs == 'random':
		mask_mode = 'random'
		mask_list = None
	elif mask_attrs == 'all':
		mask_mode = 'all'
		mask_list = None
	else:
		mask_mode = 'list'
		mask_list = [a.strip() for a in mask_attrs.split(',') if a.strip()]

	# accumulators
	total_steps = 0
	total_loss = 0.0
	attr_losses = {name: 0.0 for name in attr_names}
	attr_counts = {name: 0 for name in attr_names}
	font_correct = 0
	font_total = 0

	ce = nn.CrossEntropyLoss(reduction='none')

	# optional deterministic RNG
	rng = np.random.RandomState(0) if deterministic else None

	for i in range(0, N, batch_size):
		idx = np.arange(i, min(i + batch_size, N))
		Xb = torch.from_numpy(np.array(X_all[idx])).float().to(device)
		Mb = torch.from_numpy(np.array(M_all[idx])).to(device)
		Fb = torch.from_numpy(np.array(F_all[idx])).long().to(device)
		Tb = torch.from_numpy(np.array(T_all[idx])).long().to(device)

		B = Xb.shape[0]
		# slice attrs
		img, text, pos, size, angle, opacity = slice_X_by_schema(Xb, schema, device=device)

		valid_mask = (Mb == 1)
		if rng is not None:
			rand = torch.from_numpy(rng.rand(*valid_mask.shape)).to(device)
		else:
			rand = torch.rand(valid_mask.shape, device=device)
		target_mask = (rand < mask_prob) & valid_mask

		# build slot_attr_mask and masked_attr_id
		slot_attr_mask = torch.zeros((B, Xb.shape[1], len(tokenizer_order)), dtype=torch.bool, device=device)
		masked_attr_id = torch.zeros((B, Xb.shape[1]), dtype=torch.long, device=device)
		if mask_mode == 'random':
			rand_idx = torch.randint(0, len(tokenizer_order), (B, Xb.shape[1]), device=device)
			for tt in range(len(tokenizer_order)):
				m = (rand_idx == tt) & target_mask
				slot_attr_mask[:, :, tt] = m
				masked_attr_id[m] = tt + 1
		elif mask_mode == 'all':
			for tt in range(len(tokenizer_order)):
				slot_attr_mask[:, :, tt] = target_mask
			masked_attr_id[target_mask] = 1
		else:
			for a in mask_list:
				tt = tokenizer_order.index(a)
				slot_attr_mask[:, :, tt] = target_mask
				masked_attr_id[target_mask] = tt + 1

		input_mask = valid_mask.clone()
		input_mask[target_mask] = 0

		with torch.no_grad():
			elem_emb = attr_stage(img, text, pos, size, angle, opacity, Fb, slot_attr_mask=slot_attr_mask)
			ctx = elem_stage(elem_emb, role_idx=Tb, mask=input_mask, masked_attr_id=masked_attr_id)

			# per-attribute evaluation
			for name in attr_names:
				a, b = next((f['offset'][0], f['offset'][1]) for f in schema['fields'] if f['name'] == name)
				targ = Xb[..., a:b]
				pred = decoders[name](ctx)
				mse = (pred - targ).pow(2).mean(dim=-1)
				if name in tokenizer_order:
					tok_idx = tokenizer_order.index(name)
					mask_slots = (masked_attr_id == (tok_idx + 1))
				else:
					mask_slots = torch.zeros_like(masked_attr_id, dtype=torch.bool)
				if mask_slots.any():
					loss_attr = mse[mask_slots].mean().item()
					attr_losses[name] += loss_attr
					attr_counts[name] += 1
					total_loss += loss_attr
			# font
			logits = font_classifier(ctx)  # (B, S, num_fonts)
			C = logits.shape[-1]
			logits_flat = logits.view(-1, C)
			targ_flat = Fb.view(-1)
			loss_flat = ce(logits_flat, targ_flat).view(B, -1)
			if 'font' in tokenizer_order:
				tok_idx = tokenizer_order.index('font')
				mask_slots = (masked_attr_id == (tok_idx + 1))
			else:
				mask_slots = torch.zeros_like(masked_attr_id, dtype=torch.bool)
			if mask_slots.any():
				loss_font = loss_flat[mask_slots].mean().item()
				total_loss += loss_font
				# accuracy
				preds = logits.argmax(dim=-1)
				correct = (preds[mask_slots] == Fb[mask_slots]).sum().item()
				font_correct += correct
				font_total += mask_slots.sum().item()

		total_steps += 1

	# report
	print('Evaluation results on split:', split)
	if total_steps == 0:
		print('No evaluation steps (empty split?)')
		return
	print('Total batches:', total_steps)
	print('Per-attribute losses (averaged over batches with masked positions):')
	for name in attr_names:
		if attr_counts[name] > 0:
			print(f'  {name}: {attr_losses[name] / attr_counts[name]:.6e} (n_batches={attr_counts[name]})')
		else:
			print(f'  {name}: n/a (no masked positions)')
	if font_total > 0:
		acc = font_correct / font_total
		print(f'Font accuracy on masked slots: {acc:.4f} ({font_correct}/{font_total})')
	else:
		print('Font accuracy: n/a (no masked font slots)')
	print('Aggregate masked loss (sum of attribute losses across masked attrs):', total_loss)


def main():
	p = argparse.ArgumentParser()
	p.add_argument('--ckpt', type=str, required=True)
	p.add_argument('--split', type=str, default='test')
	p.add_argument('--base-dir', type=str, default='data/crello')
	p.add_argument('--batch-size', type=int, default=32)
	p.add_argument('--device', type=str, default=None)
	p.add_argument('--mask-prob', type=float, default=0.25)
	p.add_argument('--mask-attrs', type=str, default='text')
	p.add_argument('--max-samples', type=int, default=None)
	p.add_argument('--deterministic', action='store_true')
	args = p.parse_args()

	device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

	evaluate_checkpoint(
		args.ckpt,
		args.base_dir,
		args.split,
		device,
		batch_size=args.batch_size,
		mask_prob=args.mask_prob,
		mask_attrs=args.mask_attrs,
		max_samples=args.max_samples,
		deterministic=args.deterministic,
	)


if __name__ == '__main__':
	main()

