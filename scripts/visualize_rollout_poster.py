#!/usr/bin/env python3
"""Create a one-poster HTML overlay for attention rollout records."""
import argparse
import base64
import csv
import html
import json
import mimetypes
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout", default="evaluations/rollout_size_current.jsonl")
    p.add_argument("--elements-csv", default="data/crello/crello_test_elements_per_image.csv")
    p.add_argument("--base-dir", default="data/crello")
    p.add_argument("--out", default=None)
    p.add_argument("--out-dir", default="evaluations/attention")
    p.add_argument("--poster-id", default=None)
    p.add_argument("--record-index", type=int, default=0)
    p.add_argument("--all", action="store_true", help="Write one HTML per rollout record")
    return p.parse_args()


ELEMENT_COLUMNS = [
    "poster_id",
    "element_index",
    "image_pos",
    "type",
    "left",
    "top",
    "width",
    "height",
    "font",
    "font_size",
    "text",
    "element_text",
    "preview",
    "image",
    "canvas_width",
    "canvas_height",
]


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def pct(value, total):
    if total <= 0:
        return "0%"
    return f"{100.0 * value / total:.6f}%"


def load_rollout_record(path, poster_id=None, record_index=0):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for rec in json.loads(line):
                records.append(rec)
                if poster_id is not None and str(rec.get("poster_id")) == poster_id:
                    return rec
    if not records:
        raise ValueError(f"No records found in {path}")
    if poster_id is not None:
        raise ValueError(f"Poster id {poster_id!r} not found in {path}")
    if record_index < 0 or record_index >= len(records):
        raise IndexError(f"--record-index {record_index} outside 0..{len(records)-1}")
    return records[record_index]


def load_rollout_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.extend(json.loads(line))
    return records


def load_elements(path, poster_id):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("poster_id") == poster_id:
                rows.append(row)
    rows.sort(key=lambda r: (inum(r.get("element_index")), inum(r.get("image_pos"))))
    if not rows:
        raise ValueError(f"No elements found for poster_id={poster_id!r} in {path}")
    return rows


def load_elements_many(path, poster_ids):
    wanted = set(str(pid) for pid in poster_ids)
    grouped = {pid: [] for pid in wanted}
    if not wanted:
        return grouped
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("poster_id")
            if pid not in wanted:
                continue
            grouped[pid].append({col: row.get(col, "") for col in ELEMENT_COLUMNS})
    for rows in grouped.values():
        rows.sort(key=lambda r: (inum(r.get("element_index")), inum(r.get("image_pos"))))
    return grouped


def safe_filename_part(value):
    value = str(value)
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def data_uri(path):
    if not path or not os.path.exists(path):
        return ""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def text_snippet(row, max_len=90):
    text = row.get("element_text") or row.get("text") or ""
    text = " ".join(str(text).split())
    if len(text) > max_len:
        return text[: max_len - 1] + "..."
    return text


def element_kind(row):
    if inum(row.get("font")) != 0:
        return "text"
    typ = row.get("type", "")
    return f"type {typ}" if typ != "" else "non-text"


def element_bbox(row):
    return {
        "left": fnum(row.get("left")),
        "top": fnum(row.get("top")),
        "width": fnum(row.get("width")),
        "height": fnum(row.get("height")),
    }


def fmt_vec(vec):
    if not isinstance(vec, list):
        return ""
    return ", ".join(f"{float(v):.4f}" for v in vec)


def render_html(record, rows, base_dir):
    poster_id = str(record.get("poster_id", ""))
    masked_slot = int(record.get("slot"))
    top = record.get("top", [])
    score_by_slot = {int(slot): float(score) for slot, score in top}
    rank_by_slot = {int(slot): rank + 1 for rank, (slot, _score) in enumerate(top)}
    max_score = max(score_by_slot.values(), default=1.0)

    canvas_w = fnum(rows[0].get("canvas_width"), 1000.0)
    canvas_h = fnum(rows[0].get("canvas_height"), 1000.0)
    preview_rel = rows[0].get("preview") or ""
    preview_path = os.path.join(base_dir, preview_rel)
    preview_uri = data_uri(preview_path)
    target = rows[masked_slot] if 0 <= masked_slot < len(rows) else {}
    target_box = element_bbox(target)

    rects = []
    for slot, row in enumerate(rows):
        box = element_bbox(row)
        is_masked = slot == masked_slot
        is_source = slot in score_by_slot
        kind = element_kind(row)
        label = f"{slot}"
        classes = ["box"]
        if kind == "text":
            classes.append("text-box")
        if is_source:
            classes.append("source-box")
            label = f"{slot} #{rank_by_slot[slot]}"
        if is_masked:
            classes.append("masked-box")
            label = f"{slot} target"
        alpha = 0.08
        if is_source and max_score > 0:
            alpha = min(0.36, 0.10 + 0.34 * score_by_slot[slot] / max_score)
        rects.append(
            f"""
            <div class="{' '.join(classes)}"
                 style="left:{pct(box['left'], canvas_w)}; top:{pct(box['top'], canvas_h)};
                        width:{pct(box['width'], canvas_w)}; height:{pct(box['height'], canvas_h)};
                        background:rgba(31, 119, 180, {alpha if is_source else 0.03});">
              <span>{html.escape(label)}</span>
            </div>
            """
        )

    predicted_box = ""
    pred = record.get("pred")
    if isinstance(pred, list) and len(pred) >= 2:
        pred_w = max(0.0, float(pred[0]) * 1000.0)
        pred_h = max(0.0, float(pred[1]) * 1000.0)
        predicted_box = f"""
            <div class="box predicted-box"
                 style="left:{pct(target_box['left'], canvas_w)}; top:{pct(target_box['top'], canvas_h)};
                        width:{pct(pred_w, canvas_w)}; height:{pct(pred_h, canvas_h)};">
              <span>pred</span>
            </div>
        """

    source_rows = []
    for rank, (slot_raw, score_raw) in enumerate(top, start=1):
        slot = int(slot_raw)
        score = float(score_raw)
        if 0 <= slot < len(rows):
            row = rows[slot]
            box = element_bbox(row)
            snippet = text_snippet(row)
            kind = element_kind(row)
            source_rows.append(
                f"""
                <tr>
                  <td>{rank}</td>
                  <td>{slot}</td>
                  <td>{score:.4f}</td>
                  <td>{html.escape(kind)}</td>
                  <td>{html.escape(snippet) if snippet else '<span class="muted">-</span>'}</td>
                  <td>{box['left']:.1f}, {box['top']:.1f}, {box['width']:.1f}, {box['height']:.1f}</td>
                </tr>
                """
            )

    target_text = text_snippet(target, max_len=180)
    if isinstance(pred, list) and len(pred) >= 2:
        pred_bbox_text = (
            f"{target_box['left']:.1f}, {target_box['top']:.1f}, "
            f"{float(pred[0]) * 1000.0:.1f}, {float(pred[1]) * 1000.0:.1f}"
        )
    else:
        pred_bbox_text = "-"
    title = f"Rollout Overlay - {poster_id}"
    image_markup = (
        f'<img class="poster-img" src="{preview_uri}" alt="poster preview">'
        if preview_uri
        else '<div class="missing-img">preview image not found</div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: #202124;
      background: #f6f7f9;
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 12px;
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 18px;
      font-size: 13px;
    }}
    .metric {{
      background: white;
      border: 1px solid #dde1e7;
      border-radius: 6px;
      padding: 10px 12px;
      min-width: 0;
    }}
    .metric b {{
      display: block;
      font-size: 11px;
      color: #667085;
      margin-bottom: 4px;
      font-weight: 600;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 1.25fr) minmax(360px, 0.75fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: white;
      border: 1px solid #dde1e7;
      border-radius: 6px;
      padding: 14px;
    }}
    .poster-wrap {{
      position: relative;
      width: 100%;
      aspect-ratio: {canvas_w:.6f} / {canvas_h:.6f};
      overflow: hidden;
      background: #e9edf2;
      border: 1px solid #c8ced8;
    }}
    .poster-img, .missing-img {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: fill;
    }}
    .missing-img {{
      display: grid;
      place-items: center;
      color: #667085;
      font-size: 13px;
    }}
    .box {{
      position: absolute;
      border: 1.5px solid rgba(25, 28, 33, 0.45);
      box-sizing: border-box;
      overflow: visible;
    }}
    .box span {{
      position: absolute;
      left: 3px;
      top: 3px;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 11px;
      line-height: 1.25;
      background: rgba(255, 255, 255, 0.86);
      color: #111827;
      white-space: nowrap;
    }}
    .text-box {{
      border-color: rgba(15, 118, 110, 0.65);
    }}
    .source-box {{
      border: 3px solid #1f77b4;
    }}
    .masked-box {{
      border: 4px solid #d62728;
      background: rgba(214, 39, 40, 0.14) !important;
      z-index: 10;
    }}
    .predicted-box {{
      border: 4px dashed #f97316;
      background: rgba(249, 115, 22, 0.08) !important;
      z-index: 12;
    }}
    .predicted-box span {{
      background: rgba(255, 247, 237, 0.92);
      color: #9a3412;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      margin-top: 10px;
      font-size: 12px;
      color: #4b5563;
    }}
    .swatch {{
      display: inline-block;
      width: 14px;
      height: 10px;
      margin-right: 5px;
      border: 2px solid #999;
      vertical-align: -1px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid #edf0f3;
      padding: 8px 6px;
    }}
    th {{
      color: #667085;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0;
      font-weight: 700;
    }}
    .target {{
      margin-bottom: 14px;
      font-size: 13px;
      line-height: 1.45;
    }}
    .target code {{
      background: #f1f3f6;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    .muted {{
      color: #8a94a3;
    }}
    @media (max-width: 920px) {{
      .layout, .summary {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{html.escape(title)}</h1>
    <section class="summary">
      <div class="metric"><b>Poster</b>{html.escape(poster_id)}</div>
      <div class="metric"><b>Masked Slot</b>{masked_slot} ({html.escape(element_kind(target))})</div>
      <div class="metric"><b>GT Size</b>{html.escape(fmt_vec(record.get("gt")))} normalized</div>
      <div class="metric"><b>Pred Size</b>{html.escape(fmt_vec(record.get("pred")))} normalized</div>
    </section>
    <section class="layout">
      <div class="panel">
        <div class="poster-wrap">
          {image_markup}
          {''.join(rects)}
          {predicted_box}
        </div>
        <div class="legend">
          <span><span class="swatch" style="border-color:#d62728;background:rgba(214,39,40,.14)"></span>masked text size target</span>
          <span><span class="swatch" style="border-color:#f97316;border-style:dashed;background:rgba(249,115,22,.08)"></span>predicted size</span>
          <span><span class="swatch" style="border-color:#1f77b4;background:rgba(31,119,180,.22)"></span>top rollout source</span>
          <span><span class="swatch" style="border-color:rgba(15,118,110,.65)"></span>text element</span>
        </div>
      </div>
      <div class="panel">
        <div class="target">
          <b>Target text</b><br>
          {html.escape(target_text) if target_text else '<span class="muted">No text captured</span>'}<br>
          <b>BBox</b> <code>{target_box['left']:.1f}, {target_box['top']:.1f}, {target_box['width']:.1f}, {target_box['height']:.1f}</code>
          <br><b>Pred bbox</b> <code>{html.escape(pred_bbox_text)}</code>
          <br><b>Abs error</b> {html.escape(fmt_vec(record.get("abs_err")))}
        </div>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Slot</th>
              <th>Score</th>
              <th>Kind</th>
              <th>Text</th>
              <th>BBox</th>
            </tr>
          </thead>
          <tbody>
            {''.join(source_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main():
    args = parse_args()
    if args.all:
        records = [
            rec for rec in load_rollout_records(args.rollout)
            if rec.get("poster_id") and rec.get("is_text", True)
        ]
        grouped = load_elements_many(args.elements_csv, [rec.get("poster_id") for rec in records])
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        skipped = 0
        for rec in records:
            poster_id = str(rec.get("poster_id"))
            rows = grouped.get(poster_id) or []
            if not rows:
                skipped += 1
                continue
            poster_idx = int(rec.get("poster_idx", written))
            out = out_dir / f"{poster_idx}_{safe_filename_part(poster_id)}.html"
            out.write_text(render_html(rec, rows, args.base_dir), encoding="utf-8")
            written += 1
        print(f"Wrote {written} HTML files to {out_dir}")
        if skipped:
            print(f"Skipped {skipped} records with missing element rows")
        return

    record = load_rollout_record(args.rollout, poster_id=args.poster_id, record_index=args.record_index)
    poster_id = str(record.get("poster_id"))
    rows = load_elements(args.elements_csv, poster_id)
    out = args.out or f"evaluations/rollout_overlay_{poster_id}.html"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render_html(record, rows, args.base_dir), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
