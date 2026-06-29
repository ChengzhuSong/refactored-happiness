#!/usr/bin/env python3
"""Plot GT vs Pred from CSV with dot size = n_text_elements and color = slot_pos.

This script supports a 2D color mapping where hue=angle and value=radius and
draws a small color-wheel inset to explain the mapping when enabled.
"""
import csv
import math
from pathlib import Path
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print('Missing plotting dependencies:', e)
    print('Run this in your plotting env (conda activate test) and ensure numpy/matplotlib are installed')
    sys.exit(1)


def plot_from_csv(csv_path, out=None, pos_dim=0, pos_norm=False, size_scale=10.0, alpha=0.7, cmap='viridis', show=True,
                  pos_2d=False, pos_2d_dims=(0, 1), pos_2d_radius_scale=1.0, region_n=0):
    """Plot GT vs Pred from a CSV file.

    Parameters:
      csv_path: path-like to CSV produced by plot_scatter_eval.py
      out: if provided, save plot to this path; otherwise (default) call plt.show()
      pos_dim: which slot_pos column to use for color (0-based)
      pos_norm: if True, color by L2 norm of slot_pos vector
      pos_2d: if True, color by 2D (hue=angle, val=radius)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f'CSV not found: {path}')

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        header_idx = {n: i for i, n in enumerate(header)}
        gt_cols = [c for c in header if c.startswith('gt_')]
        pred_cols = [c for c in header if c.startswith('pred_')]
        pos_cols = [c for c in header if c.startswith('slot_pos_')]
        if len(gt_cols) == 0 or len(pred_cols) == 0:
            raise ValueError('CSV missing gt_/pred_ columns')
        if 'n_text_elements' not in header_idx:
            raise ValueError('CSV missing n_text_elements column')

        gt_i = header_idx[gt_cols[0]]
        pred_i = header_idx[pred_cols[0]]
        ntext_i = header_idx['n_text_elements']

        pos_idxs = [header_idx[c] for c in pos_cols]
        has_midpoint = 'midpoint_x' in header_idx and 'midpoint_y' in header_idx
        has_canvas = 'canvas_width' in header_idx and 'canvas_height' in header_idx

        gts = []
        preds = []
        ntexts = []
        pos_vecs = []
        midpoints_x = []
        midpoints_y = []
        canv_w = []
        canv_h = []

        for row in reader:
            try:
                gt = float(row[gt_i])
                pred = float(row[pred_i])
                ntext = float(row[ntext_i])
            except Exception:
                continue
            gts.append(gt)
            preds.append(pred)
            ntexts.append(ntext)
            # read slot_pos vector if present
            if pos_idxs:
                try:
                    pos_vec = [float(row[idx]) for idx in pos_idxs]
                except Exception:
                    pos_vec = [0.0 for _ in pos_idxs]
            else:
                pos_vec = []
            # read midpoints and canvas sizes if present
            if has_midpoint:
                try:
                    midx = float(row[header_idx['midpoint_x']])
                    midy = float(row[header_idx['midpoint_y']])
                except Exception:
                    midx = None
                    midy = None
            else:
                midx = None
                midy = None
            if has_canvas:
                try:
                    cwidth = float(row[header_idx['canvas_width']])
                    cheight = float(row[header_idx['canvas_height']])
                except Exception:
                    cwidth = None
                    cheight = None
            else:
                cwidth = None
                cheight = None
            pos_vecs.append(pos_vec)
            midpoints_x.append(midx)
            midpoints_y.append(midy)
            canv_w.append(cwidth)
            canv_h.append(cheight)

    if len(gts) == 0:
        raise RuntimeError('No data read from CSV')

    x = np.array(gts)
    y = np.array(preds)
    sizes = np.array(ntexts) * size_scale + 10.0

    # default scalar color is either pos_norm or a single pos_dim
    if pos_norm:
        pos_vals = np.array([math.sqrt(sum(v * v for v in pv)) if len(pv) > 0 else 0.0 for pv in pos_vecs])
        colors_arg = pos_vals
        cbar_label = 'slot_pos_norm'
    else:
        if pos_2d:
            # use two dims from pos_vecs
            if len(pos_vecs) == 0:
                raise ValueError('CSV missing slot_pos_* columns required for pos_2d')
            dx, dy = pos_2d_dims
            pos_x = np.array([pv[dx] if dx < len(pv) else 0.0 for pv in pos_vecs])
            pos_y = np.array([pv[dy] if dy < len(pv) else 0.0 for pv in pos_vecs])
            thetas = np.arctan2(pos_y, pos_x)
            hues = (thetas + math.pi) / (2 * math.pi)
            radii = np.sqrt(pos_x ** 2 + pos_y ** 2) * float(pos_2d_radius_scale)
            if len(radii) > 0:
                radii = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
            else:
                radii = np.zeros_like(radii)
            import matplotlib as mpl
            hsv = np.stack([hues, np.ones_like(hues) * 0.9, radii], axis=1)
            rgba = mpl.colors.hsv_to_rgb(hsv)
            colors_arg = rgba
            cbar_label = 'pos_2d(hue=angle,val=radius)'
        else:
            # scalar pos_dim color
            if len(pos_vecs) == 0:
                colors_arg = np.zeros(len(x))
            else:
                if pos_dim < 0 or pos_dim >= len(pos_vecs[0]):
                    raise ValueError(f'pos_dim {pos_dim} out of range')
                colors_arg = np.array([pv[pos_dim] for pv in pos_vecs])
            cbar_label = f'slot_pos_{pos_dim}'

    # region-based discrete coloring (e.g., 3x3 grid) using midpoints and canvas sizes
    if region_n and region_n > 1:
        # require midpoints and canvas sizes in CSV
        if not (has_midpoint and has_canvas):
            raise ValueError('region_n>1 requires midpoint_x/midpoint_y and canvas_width/canvas_height in CSV')
        # compute region id per point (row-major, top-left=0)
        mids_x = np.array([0.0 if v is None else v for v in midpoints_x])
        mids_y = np.array([0.0 if v is None else v for v in midpoints_y])
        cws = np.array([1.0 if v is None else v for v in canv_w])
        chs = np.array([1.0 if v is None else v for v in canv_h])
        # avoid zero division
        cws[cws == 0] = 1.0
        chs[chs == 0] = 1.0
        # compute bin indices
        bx = np.floor((mids_x / cws) * region_n).astype(int)
        by = np.floor((mids_y / chs) * region_n).astype(int)
        bx = np.clip(bx, 0, region_n - 1)
        by = np.clip(by, 0, region_n - 1)
        region_ids = by * region_n + bx
        colors_arg = region_ids
        cmap = 'tab10'
        # build legend patches for each region
        import matplotlib.patches as mpatches
        cmap_obj = plt.get_cmap('tab10')
        region_labels = []
        # names: rows top->bottom, cols left->right
        row_names = ['upper', 'middle', 'lower']
        col_names = ['left', 'middle', 'right']
        legend_patches = []
        for rid in range(region_n * region_n):
            ry = rid // region_n
            rx = rid % region_n
            rname = f"{row_names[ry]}-{col_names[rx]}" if region_n == 3 else f"r{rid}"
            color = cmap_obj(rid % cmap_obj.N)
            patch = mpatches.Patch(color=color, label=rname)
            legend_patches.append(patch)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, s=sizes, c=colors_arg, cmap=None if pos_2d else cmap, alpha=alpha, edgecolors='none')
    if region_n and region_n > 1:
        # add legend with patches
        plt.legend(handles=legend_patches, title='region', loc='upper right', fontsize=7)
    elif not pos_2d:
        plt.colorbar(sc, label=cbar_label)
    else:
        # add a small polar color wheel inset to explain hue=angle -> color and radius->value
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            import matplotlib as mpl
            ax_wheel = inset_axes(plt.gca(), width="18%", height="18%", loc='upper right')
            res = 200
            ys, xs = np.mgrid[-1:1:complex(0, res), -1:1:complex(0, res)]
            rs = np.sqrt(xs ** 2 + ys ** 2)
            thetas = np.arctan2(ys, xs)
            hues_w = (thetas + math.pi) / (2 * math.pi)
            vals = np.clip(rs, 0, 1)
            sats = np.ones_like(vals) * 0.9
            hsv_img = np.stack([hues_w, sats, vals], axis=2)
            rgb_img = mpl.colors.hsv_to_rgb(hsv_img)
            rgb_img[rs > 1] = 1.0
            ax_wheel.imshow(rgb_img, origin='lower', extent=[-1, 1, -1, 1])
            ax_wheel.set_xticks([])
            ax_wheel.set_yticks([])
            ax_wheel.set_title('pos: hue=angle, val=radius', fontsize=7)
            ax_wheel.set_frame_on(False)
        except Exception:
            pass

    # Fit a simple linear regression (predicted = m * gt + b)
    try:
        m, b = np.polyfit(x, y, 1)
    except Exception:
        m = 0.0
        b = float(y.mean()) if len(y) > 0 else 0.0

    y_reg = m * x + b
    residuals = y - y_reg
    sse = float((residuals ** 2).sum())
    rmse = float(np.sqrt(sse / len(y)))
    mae = float(np.mean(np.abs(residuals)))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst != 0 else float('nan')

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], color='red', linewidth=1, label='identity')
    plt.plot([lo, hi], [m * lo + b, m * hi + b], color='blue', linewidth=1, linestyle='--', label=f'reg: y={m:.3f}x+{b:.3f}')
    plt.xlabel('ground-truth')
    plt.ylabel('predicted')
    plt.title(f'Scatter GT vs Pred — size (n={len(x)})')
    plt.legend(loc='upper left')

    # Residual histogram inset
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(plt.gca(), width="30%", height="25%", loc='lower right')
        axins.hist(residuals, bins=40, color='gray', alpha=0.8)
        axins.axvline(0, color='red', linewidth=0.8)
        axins.set_title('residuals', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=7)
    except Exception:
        pass

    metrics = dict(slope=float(m), intercept=float(b), rmse=rmse, mae=mae, r2=r2, n=int(len(x)))
    print('Regression metrics:', metrics)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=200)
        print('Saved plot to', out)
    if show and not out:
        plt.show()
        print('Displayed plot')

    return metrics


if __name__ == '__main__':
    default_csv = Path('evaluations/eval_size_gt_pred_test_full_with_midpoints.csv')
    if default_csv.exists():
        plot_from_csv(default_csv, show=True,region_n=3)
    else:
        print('No default CSV found. Import plot_from_csv and call plot_from_csv(csv_path, ...)')
