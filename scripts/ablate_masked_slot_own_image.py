#!/usr/bin/env python3
"""Ablate the masked slot's own image feature during masked-size inference.

This measures a causal-ish dependency that attention rollout cannot answer:
for the same masked text-size reconstruction query, how much does the
prediction change if only the target slot's own image feature vector is set to
zero before AttributeStage?
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.heads import load_decoder_state, make_decoder
from models.two_stage_transformer import AttributeStage, ElementStage


def normalize_poster_id(value):
    if value is None:
        return ""
    value = str(value)
    for ch in ('"', "'", "(", ")", ","):
        value = value.replace(ch, "")
    return value.strip()


def load_poster_ids(index_path):
    poster_ids = []
    if not os.path.exists(index_path):
        return poster_ids
    with open(index_path, newline="", encoding="utf-8") as f:
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
    p.add_argument("--model-ckpt", default="checkpoints/best_epoch.pth")
    p.add_argument("--split", default="test")
    p.add_argument("--base-dir", default="data/crello")
    p.add_argument("--out-dir", default="evaluations/attributes")
    p.add_argument("--mask-attr", default="size")
    p.add_argument("--mask-count", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-non-text-size", action="store_true")
    p.add_argument("--keep-masked-slot-visible", action="store_true")
    return p.parse_args()


def make_models(ckpt, schema, fields, num_fonts, num_roles, slots):
    d_attr = 128
    d_elem = 256
    attr_stage = AttributeStage(
        img_dim=schema["fields"][0]["dim"],
        txt_dim=fields["text"]["dim"],
        d_attr=d_attr,
        D_elem=d_elem,
        num_fonts=num_fonts,
    )
    elem_stage = ElementStage(
        D_elem=d_elem,
        num_roles=num_roles,
        max_slots=slots,
        num_attributes=len(schema["fields"]) + 1,
    )
    return attr_stage, elem_stage


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    x = x[ok]
    y = y[ok]
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows):
    if not rows:
        return []
    arr = {
        k: np.array([float(r[k]) for r in rows], dtype=np.float64)
        for k in rows[0]
        if k.startswith(("gt_", "base_", "ablated_", "delta_"))
    }
    n = len(rows)
    out = []
    for dim, label in [(0, "width"), (1, "height")]:
        gt = arr[f"gt_{dim}"]
        base = arr[f"base_pred_{dim}"]
        abl = arr[f"ablated_pred_{dim}"]
        delta = arr[f"delta_pred_{dim}"]
        abs_delta = np.abs(delta)
        base_err = arr[f"base_abs_err_{dim}"]
        abl_err = arr[f"ablated_abs_err_{dim}"]
        err_delta = arr[f"delta_abs_err_{dim}"]
        out.append(
            {
                "dimension": label,
                "n": n,
                "base_mae": float(base_err.mean()),
                "ablated_mae": float(abl_err.mean()),
                "delta_mae": float(err_delta.mean()),
                "mean_delta_pred": float(delta.mean()),
                "mean_abs_delta_pred": float(abs_delta.mean()),
                "median_abs_delta_pred": float(np.median(abs_delta)),
                "p90_abs_delta_pred": float(np.quantile(abs_delta, 0.90)),
                "p95_abs_delta_pred": float(np.quantile(abs_delta, 0.95)),
                "max_abs_delta_pred": float(abs_delta.max()),
                "fraction_abs_delta_gt_0_01": float((abs_delta > 0.01).mean()),
                "fraction_abs_delta_gt_0_05": float((abs_delta > 0.05).mean()),
                "base_pred_gt_pearson": pearson(base, gt),
                "ablated_pred_gt_pearson": pearson(abl, gt),
            }
        )
    return out


def write_csv(path, rows, fieldnames=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path, rows):
    import matplotlib.pyplot as plt

    if not rows:
        return
    base_w = np.array([float(r["base_pred_0"]) for r in rows])
    base_h = np.array([float(r["base_pred_1"]) for r in rows])
    abl_w = np.array([float(r["ablated_pred_0"]) for r in rows])
    abl_h = np.array([float(r["ablated_pred_1"]) for r in rows])
    delta_w = abl_w - base_w
    delta_h = abl_h - base_h
    err_delta_w = np.array([float(r["delta_abs_err_0"]) for r in rows])
    err_delta_h = np.array([float(r["delta_abs_err_1"]) for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    for ax, base, abl, title in [
        (axes[0, 0], base_w, abl_w, "Width prediction"),
        (axes[1, 0], base_h, abl_h, "Height prediction"),
    ]:
        lo = float(min(base.min(), abl.min()))
        hi = float(max(base.max(), abl.max()))
        pad = (hi - lo) * 0.04 + 1e-6
        ax.scatter(base, abl, s=7, alpha=0.35, linewidths=0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1)
        ax.set_title(title)
        ax.set_xlabel("baseline pred")
        ax.set_ylabel("own image zeroed pred")

    bins_w = np.linspace(np.quantile(delta_w, 0.01), np.quantile(delta_w, 0.99), 50)
    bins_h = np.linspace(np.quantile(delta_h, 0.01), np.quantile(delta_h, 0.99), 50)
    axes[0, 1].hist(delta_w, bins=bins_w, color="#4C78A8", alpha=0.85)
    axes[0, 1].axvline(0, color="black", lw=1)
    axes[0, 1].set_title("Width delta")
    axes[0, 1].set_xlabel("ablated - baseline")
    axes[1, 1].hist(delta_h, bins=bins_h, color="#F58518", alpha=0.85)
    axes[1, 1].axvline(0, color="black", lw=1)
    axes[1, 1].set_title("Height delta")
    axes[1, 1].set_xlabel("ablated - baseline")

    bins_err_w = np.linspace(np.quantile(err_delta_w, 0.01), np.quantile(err_delta_w, 0.99), 50)
    bins_err_h = np.linspace(np.quantile(err_delta_h, 0.01), np.quantile(err_delta_h, 0.99), 50)
    axes[0, 2].hist(err_delta_w, bins=bins_err_w, color="#54A24B", alpha=0.85)
    axes[0, 2].axvline(0, color="black", lw=1)
    axes[0, 2].set_title("Width abs-error change")
    axes[0, 2].set_xlabel("ablated abs err - baseline abs err")
    axes[1, 2].hist(err_delta_h, bins=bins_err_h, color="#E45756", alpha=0.85)
    axes[1, 2].axvline(0, color="black", lw=1)
    axes[1, 2].set_title("Height abs-error change")
    axes[1, 2].set_xlabel("ablated abs err - baseline abs err")

    fig.suptitle("Effect of zeroing the masked slot's own image feature", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    tokenizer_order = ckpt.get("tokenizer_order", ["image", "text", "pos", "size", "angle", "opacity", "font"])
    if args.mask_attr not in tokenizer_order:
        raise ValueError(f"--mask-attr '{args.mask_attr}' not in tokenizer_order {tokenizer_order}")

    prefix = f"poster_input_{args.split}"
    base = args.base_dir
    x_all = np.load(os.path.join(base, f"{prefix}_X.npy"), mmap_mode="r")
    m_all = np.load(os.path.join(base, f"{prefix}_mask.npy"), mmap_mode="r")
    font_all = np.load(os.path.join(base, f"{prefix}_font_idx.npy"), mmap_mode="r")
    type_all = np.load(os.path.join(base, f"{prefix}_type_idx.npy"), mmap_mode="r")
    with open(os.path.join(base, f"{prefix}_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    fields = {f["name"]: f for f in schema["fields"]}
    poster_ids = load_poster_ids(os.path.join(base, f"{prefix}_index.csv"))

    slots = x_all.shape[1]
    num_fonts = schema.get("font", {}).get("num_fonts", int(font_all.max() + 1))
    if "attr_stage" in ckpt and "tokenizer.font_emb.weight" in ckpt["attr_stage"]:
        num_fonts = int(ckpt["attr_stage"]["tokenizer.font_emb.weight"].shape[0])
    num_roles = schema.get("type", {}).get("num_types", int(type_all.max() + 1))

    attr_stage, elem_stage = make_models(ckpt, schema, fields, num_fonts, num_roles, slots)
    decoder = make_decoder(256, fields[args.mask_attr]["dim"])
    attr_stage.load_state_dict(ckpt["attr_stage"])
    elem_stage.load_state_dict(ckpt["elem_stage"])
    load_decoder_state(decoder, ckpt["decoders"][args.mask_attr])

    attr_stage.to(device).eval()
    elem_stage.to(device).eval()
    decoder.to(device).eval()

    tok_idx = tokenizer_order.index(args.mask_attr)
    rows = []
    n = x_all.shape[0]
    bs = args.batch_size

    for start in range(0, n, bs):
        batch_indices = list(range(start, min(start + bs, n)))
        x = torch.from_numpy(np.array(x_all[batch_indices])).float().to(device)
        mask = torch.from_numpy(np.array(m_all[batch_indices])).to(device)
        font = torch.from_numpy(np.array(font_all[batch_indices])).long().to(device)
        typ = torch.from_numpy(np.array(type_all[batch_indices])).long().to(device)

        img = x[:, :, fields["image"]["offset"][0] : fields["image"]["offset"][1]]
        text = x[:, :, fields["text"]["offset"][0] : fields["text"]["offset"][1]]
        pos = x[:, :, fields["pos"]["offset"][0] : fields["pos"]["offset"][1]]
        size = x[:, :, fields["size"]["offset"][0] : fields["size"]["offset"][1]]
        angle = x[:, :, fields["angle"]["offset"][0] : fields["angle"]["offset"][1]]
        opacity = x[:, :, fields["opacity"]["offset"][0] : fields["opacity"]["offset"][1]]

        valid_mask = mask == 1
        if args.mask_attr in ("text", "font"):
            present_mask = font != 0
        elif args.mask_attr == "size" and not args.allow_non_text_size:
            present_mask = font != 0
        else:
            present_mask = torch.ones_like(valid_mask)

        sampled = torch.zeros((len(batch_indices), slots), dtype=torch.bool, device=device)
        for local_i, poster_idx in enumerate(batch_indices):
            valid_idxs = torch.nonzero(valid_mask[local_i] & present_mask[local_i], as_tuple=False).view(-1)
            if valid_idxs.numel() == 0:
                continue
            generator = torch.Generator(device=device)
            generator.manual_seed(int(args.seed) + poster_idx)
            perm = torch.randperm(valid_idxs.numel(), generator=generator, device=device)
            chosen = valid_idxs[perm[: min(args.mask_count, valid_idxs.numel())]]
            sampled[local_i, chosen] = True

        if not sampled.any():
            continue

        slot_attr_mask = torch.zeros((len(batch_indices), slots, len(tokenizer_order)), dtype=torch.bool, device=device)
        masked_attr_id = torch.zeros((len(batch_indices), slots), dtype=torch.long, device=device)
        slot_attr_mask[:, :, tok_idx] = sampled
        masked_attr_id[sampled] = tok_idx + 1

        input_mask = valid_mask.clone()
        if not args.keep_masked_slot_visible:
            input_mask[sampled] = 0

        img_ablated = img.clone()
        img_ablated[sampled] = 0.0

        with torch.no_grad():
            elem_base = attr_stage(img, text, pos, size, angle, opacity, font, slot_attr_mask=slot_attr_mask)
            ctx_base = elem_stage(elem_base, role_idx=typ, mask=input_mask, masked_attr_id=masked_attr_id)
            pred_base = decoder(ctx_base)

            elem_abl = attr_stage(img_ablated, text, pos, size, angle, opacity, font, slot_attr_mask=slot_attr_mask)
            ctx_abl = elem_stage(elem_abl, role_idx=typ, mask=input_mask, masked_attr_id=masked_attr_id)
            pred_abl = decoder(ctx_abl)

        for local_i, poster_idx in enumerate(batch_indices):
            for slot in torch.nonzero(sampled[local_i], as_tuple=False).view(-1).tolist():
                gt = size[local_i, slot].detach().cpu().numpy().astype(float)
                base_pred = pred_base[local_i, slot].detach().cpu().numpy().astype(float)
                abl_pred = pred_abl[local_i, slot].detach().cpu().numpy().astype(float)
                rec = {
                    "poster_idx": poster_idx,
                    "poster_id": poster_ids[poster_idx] if poster_idx < len(poster_ids) else "",
                    "slot": int(slot),
                    "font_idx": int(font[local_i, slot].item()),
                    "type_idx": int(typ[local_i, slot].item()),
                }
                for dim in range(fields[args.mask_attr]["dim"]):
                    rec[f"gt_{dim}"] = float(gt[dim])
                    rec[f"base_pred_{dim}"] = float(base_pred[dim])
                    rec[f"ablated_pred_{dim}"] = float(abl_pred[dim])
                    rec[f"delta_pred_{dim}"] = float(abl_pred[dim] - base_pred[dim])
                    rec[f"base_abs_err_{dim}"] = float(abs(base_pred[dim] - gt[dim]))
                    rec[f"ablated_abs_err_{dim}"] = float(abs(abl_pred[dim] - gt[dim]))
                    rec[f"delta_abs_err_{dim}"] = float(abs(abl_pred[dim] - gt[dim]) - abs(base_pred[dim] - gt[dim]))
                rows.append(rec)

    stem = f"own_image_ablation_{args.mask_attr}_{args.split}"
    records_csv = out_dir / f"{stem}.csv"
    summary_csv = out_dir / f"{stem}_summary.csv"
    plot_png = out_dir / f"{stem}.png"

    fieldnames = [
        "poster_idx",
        "poster_id",
        "slot",
        "font_idx",
        "type_idx",
        "gt_0",
        "gt_1",
        "base_pred_0",
        "base_pred_1",
        "ablated_pred_0",
        "ablated_pred_1",
        "delta_pred_0",
        "delta_pred_1",
        "base_abs_err_0",
        "base_abs_err_1",
        "ablated_abs_err_0",
        "ablated_abs_err_1",
        "delta_abs_err_0",
        "delta_abs_err_1",
    ]
    write_csv(records_csv, rows, fieldnames=fieldnames)
    summary_rows = summarize(rows)
    write_csv(summary_csv, summary_rows)
    make_plot(plot_png, rows)

    print(f"Wrote {len(rows)} ablation records to {records_csv}")
    print(f"Wrote summary to {summary_csv}")
    print(f"Wrote plot to {plot_png}")
    for row in summary_rows:
        print(
            "{dimension}: n={n} base_mae={base_mae:.6f} ablated_mae={ablated_mae:.6f} "
            "mean_abs_delta={mean_abs_delta_pred:.6f} p95_abs_delta={p95_abs_delta_pred:.6f}".format(**row)
        )


if __name__ == "__main__":
    main()
