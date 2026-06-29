#!/usr/bin/env python3
"""Export interactive Plotly HTML scatter plots from eval CSVs.

This script reads one or more evaluation CSVs (the ones created by
`plot_scatter_eval.py`), optionally computes a 3x3 region label from
midpoint_x/midpoint_y and canvas sizes, and writes an HTML file with an
interactive scatter where color=region and size=n_text_elements.

Usage:
  python3 scripts/plotly_export.py evaluations/eval_size_gt_pred_train_full_with_midpoints.csv
"""
import sys
from pathlib import Path
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
except Exception:
    print('Please install plotly (pip install plotly)')
    raise


REGION_NAMES = ['upper-left','upper-middle','upper-right',
                'middle-left','middle','middle-right',
                'lower-left','lower-middle','lower-right']

# fixed color palette for regions (one color per region, stable order)
REGION_COLORS = {
    'upper-left': '#1f77b4',
    'upper-middle': '#ff7f0e',
    'upper-right': '#2ca02c',
    'middle-left': '#d62728',
    'middle': '#9467bd',
    'middle-right': '#8c564b',
    'lower-left': '#e377c2',
    'lower-middle': '#7f7f7f',
    'lower-right': '#bcbd22',
}


def add_region_labels(df, region_n=3):
    if 'region' in df.columns:
        return df
    if not all(c in df.columns for c in ('midpoint_x', 'midpoint_y', 'canvas_width', 'canvas_height')):
        return df
    def label_row(r):
        try:
            cx = float(r['midpoint_x']) / float(r['canvas_width'])
            cy = float(r['midpoint_y']) / float(r['canvas_height'])
        except Exception:
            return 'unknown'
        cx = min(max(cx, 0.0), 0.9999)
        cy = min(max(cy, 0.0), 0.9999)
        col = int(cx * region_n)
        rowi = int(cy * region_n)
        rid = rowi * region_n + col
        if region_n == 3:
            return REGION_NAMES[rid]
        return f'region_{rid}'
    df['region'] = df.apply(label_row, axis=1)
    return df


def export_html(csv_path, out_html=None, region_n=3, dim=0):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(p)
    df = add_region_labels(df, region_n=region_n)
    # pick x and y columns for plotting (specified dim)
    gt_cols = [c for c in df.columns if c.startswith('gt_')]
    pred_cols = [c for c in df.columns if c.startswith('pred_')]
    if not gt_cols or not pred_cols:
        raise ValueError('CSV missing gt_/pred_ columns')
    if dim < 0 or dim >= len(gt_cols):
        raise ValueError(f'dim {dim} out of range (0..{len(gt_cols)-1})')
    xcol = gt_cols[dim]
    ycol = pred_cols[dim]

    # add all gt/pred/abs_err columns to hover
    hover_cols = ['poster_id','slot_idx','midpoint_x','midpoint_y'] if 'poster_id' in df.columns else []
    hover_cols += [c for c in df.columns if c.startswith('gt_') or c.startswith('pred_') or c.startswith('abs_err_')]

    # If region column exists, map regions to fixed colors so same-region
    # points share identical colors across plots.
    color_discrete_map = None
    regions_ordered = None
    if 'region' in df.columns:
        unique_regions = list(df['region'].unique())
        # ensure consistent ordering: prefer REGION_NAMES order first
        regions_ordered = [r for r in REGION_NAMES if r in unique_regions] + [r for r in unique_regions if r not in REGION_NAMES]
        # build color map for regions: use REGION_COLORS when possible
        color_discrete_map = {r: REGION_COLORS[r] for r in unique_regions if r in REGION_COLORS}

    # build base scatter (with or without region color)
    if color_discrete_map:
        scatter_fig = px.scatter(df, x=xcol, y=ycol,
                                 color='region',
                                 color_discrete_map=color_discrete_map,
                                 size='n_text_elements' if 'n_text_elements' in df.columns else None,
                                 hover_data=hover_cols if hover_cols else None,
                                 title=p.name)
    else:
        scatter_fig = px.scatter(df, x=xcol, y=ycol,
                                 size='n_text_elements' if 'n_text_elements' in df.columns else None,
                                 hover_data=hover_cols if hover_cols else None,
                                 title=p.name)

    # create subplots: scatter on top, histogram of ground-truth below
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05,
                        subplot_titles=(p.name, 'GT distribution'))
    # add scatter traces into top subplot
    for trace in scatter_fig.data:
        # ensure marker opacity is slightly reduced so lines are visible
        if 'marker' in trace and isinstance(trace['marker'], dict):
            trace['marker']['opacity'] = 0.8
        else:
            trace['marker'] = dict(opacity=0.8)
        fig.add_trace(trace, row=1, col=1)
    # add per-region histograms in bottom subplot (stacked when multiple regions)
    pal = px.colors.qualitative.Dark24
    if 'region' in df.columns:
        region_list = regions_ordered if regions_ordered is not None else sorted(list(df['region'].unique()))
        added_any = False
        for i, r in enumerate(region_list):
            try:
                vals = df.loc[df['region'] == r, xcol].to_numpy(dtype=float)
            except Exception:
                vals = []
            if len(vals) == 0:
                continue
            added_any = True
            color = REGION_COLORS.get(r, pal[i % len(pal)])
            fig.add_trace(go.Histogram(x=vals, nbinsx=40, marker_color=color, name=r, opacity=0.75), row=2, col=1)
        if added_any:
            fig.update_layout(barmode='stack')
        else:
            # fallback single histogram if all regions empty
            try:
                hist_vals = df[xcol].to_numpy(dtype=float)
            except Exception:
                hist_vals = []
            fig.add_trace(go.Histogram(x=hist_vals, nbinsx=40, marker_color='lightgray', showlegend=False), row=2, col=1)
    else:
        try:
            hist_vals = df[xcol].to_numpy(dtype=float)
        except Exception:
            hist_vals = []
        fig.add_trace(go.Histogram(x=hist_vals, nbinsx=40, marker_color='lightgray', showlegend=False), row=2, col=1)
    # axis labels
    fig.update_xaxes(title_text=xcol, row=2, col=1)
    fig.update_yaxes(title_text=ycol, row=1, col=1)
    fig.update_yaxes(title_text='count', row=2, col=1)
    fig.update_layout(title=p.name, height=700)

    # add identity line (y=x) and regression overlays inside a numeric-safe block
    try:
        xvals = df[xcol].to_numpy(dtype=float)
        yvals = df[ycol].to_numpy(dtype=float)
        if len(xvals) > 0:
            xmin, xmax = float(np.nanmin(xvals)), float(np.nanmax(xvals))
        else:
            xmin, xmax = 0.0, 1.0

        # identity line (thin black)
        fig.add_shape(type='line', x0=xmin, x1=xmax, y0=xmin, y1=xmax, line=dict(color='black', width=1))

        # only add regressions when we have at least 2 points
        if len(xvals) > 1:
            xr = np.array([xmin, xmax])
            # overall regression
            m_all, b_all = np.polyfit(xvals, yvals, 1)
            yr_all = m_all * xr + b_all
            residuals_all = yvals - (m_all * xvals + b_all)
            rmse_all = float(np.sqrt((residuals_all ** 2).mean()))
            ss_res_all = float((residuals_all ** 2).sum())
            ss_tot_all = float(((yvals - np.nanmean(yvals)) ** 2).sum())
            r2_all = float(1.0 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else float('nan')
            fig.add_trace(go.Scatter(x=xr, y=yr_all, mode='lines', name=f'overall (n={len(xvals)} rmse={rmse_all:.3f} r2={r2_all:.3f})', line=dict(color='blue', width=3)))

            # per-region regression lines
            regions = df['region'].unique() if 'region' in df.columns else ['all']
            reg_metrics = []
            loop_order = regions_ordered if regions_ordered is not None else sorted(list(regions))
            pal = px.colors.qualitative.Dark24
            for i, r in enumerate(loop_order):
                if r == 'all':
                    sub = df
                else:
                    sub = df[df['region'] == r]
                n_sub = len(sub)
                if n_sub < 2:
                    reg_metrics.append((r, n_sub, None, None, None))
                    continue
                xv = sub[xcol].to_numpy(dtype=float)
                yv = sub[ycol].to_numpy(dtype=float)
                m_r, b_r = np.polyfit(xv, yv, 1)
                yr = m_r * xr + b_r
                residuals = yv - (m_r * xv + b_r)
                rmse = float(np.sqrt((residuals ** 2).mean()))
                mae = float(np.mean(np.abs(residuals)))
                ss_res = float((residuals ** 2).sum())
                ss_tot = float(((yv - np.nanmean(yv)) ** 2).sum())
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
                # color for this region
                color = REGION_COLORS.get(r, pal[i % len(pal)])
                name = f'{r} (n={n_sub} rmse={rmse:.3f} r2={r2:.3f})'
                fig.add_trace(go.Scatter(x=xr, y=yr, mode='lines', name=name, legendgroup=r, line=dict(color=color, dash='dash', width=3)))
                reg_metrics.append((r, n_sub, rmse, mae, r2))

    except Exception:
        # numeric issue or empty data -> skip regression overlays
        pass

    # write HTML output
    if out_html is None:
        out_path = p.with_suffix('.html')
    else:
        out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))


def main(argv):
    if len(argv) < 2:
        print('Usage: plotly_export.py <csv1> [csv2 ...]')
        return 1
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('csvs', nargs='+')
    p.add_argument('--region-n', type=int, default=3)
    p.add_argument('--dim', type=int, default=0, help='which gt/pred dim to plot (0-based)')
    args = p.parse_args(argv[1:])
    for csv in args.csvs:
        p = Path(csv)
        out = Path('evaluations') / (p.stem + f'.dim{args.dim}.html')
        export_html(p, out_html=out, region_n=args.region_n, dim=args.dim)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
