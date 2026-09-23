#!/usr/bin/env python
"""Contact sheet of held-out renders from h3dgs_eval_chunk.py outputs: for each chunk, the best / median / worst
of the SAVED views (ground truth | render, PSNR in the label).
  python h3dgs_render_strip.py <proj> <out.png> <chunk> [<chunk> ...]  [--n 3] [--width 640] [--eval_dir output/eval_chunk_{c}]
"""
import argparse, json, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("out"); ap.add_argument("chunks", nargs="+")
ap.add_argument("--n", type=int, default=3); ap.add_argument("--width", type=int, default=640); ap.add_argument("--eval_dir", default="output/eval_chunk_{c}")
a = ap.parse_args(); P = Path(a.proj); GT = P / "camera_calibration/rectified/images"
font = ImageFont.load_default()
rows = []
for c in a.chunks:
    ed = P / a.eval_dir.format(c=c); rdir = ed / "render_0"
    sc = {r["name"]: r for r in json.load(open(ed / "scores.json")) if r["tau"] == 0 and r["chunk"] == c}
    saved = sorted((n for n in sc if (rdir / n).exists()), key=lambda n: sc[n]["psnr"])
    if not saved: continue
    picks = [saved[-1], saved[len(saved) // 2], saved[0]] if a.n == 3 else saved[:: max(1, len(saved) // a.n)][: a.n]
    for tag, n in zip(["best", "median", "worst"] if a.n == 3 else [""] * a.n, picks):
        rows.append((c, tag, n, sc[n]["psnr"], sc[n].get("psnr_fg")))
if not rows: raise SystemExit("no saved renders found")
W = a.width; tiles = []
for c, tag, n, p, pfg in rows:
    g = Image.open(GT / n).convert("RGB"); r = Image.open(rows and (P / a.eval_dir.format(c=c) / "render_0" / n)).convert("RGB")
    H = int(g.height * W / g.width); g = g.resize((W, H)); r = r.resize((W, H))
    t = Image.new("RGB", (2 * W + 10, H + 22), "white"); t.paste(g, (0, 22)); t.paste(r, (W + 10, 22))
    d = ImageDraw.Draw(t); lab = f"chunk {c}  {tag}  {n}   PSNR {p:.2f} dB" + (f"  (FG {pfg:.2f})" if pfg else "")
    d.text((4, 4), lab + "      left: ground truth   right: render", fill="black", font=font); tiles.append(t)
sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles) + 6 * (len(tiles) - 1)), "white"); y = 0
for t in tiles: sheet.paste(t, (0, y)); y += t.height + 6
sheet.save(a.out); print(f"wrote {a.out}: {len(tiles)} rows, {sheet.size}")
for c, tag, n, p, pfg in rows: print(f"  {c} {tag:6s} {n} {p:.2f}")
