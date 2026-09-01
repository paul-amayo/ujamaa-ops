#!/usr/bin/env python3
"""Per-block FG-PSNR profile + prod-bar judgment (the QA-gate measurement).

Renders every keyframe of each block through the live render service
(:8004, block-pinned, stabilized), scores FG-PSNR vs scratch_sam3 GT under
the sky mask, and evaluates the prod bars:

    med >= 21   P5 >= 17   interior-min >= 16   worst-vs-med <= 4

(interior = not the first/last 2 frames; holdouts — every 10th frame,
ns-train eval-interval 10 — are excluded from bars, reported separately.)

Usage: judge_recipe.py <label> <block> [<block> ...]
Writes lab_notebook/data/psnr_profiles/<label>_profiles.json and prints
one bar row per block. Compare runs with two labels + diff_profiles.py.
"""
import asyncio, io, json, sys
from pathlib import Path

import numpy as np
import websockets
from PIL import Image

R = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/lio_row100")
GT = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3")
SKY = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/sky_masks")
OUT = Path("/home/paperspace/code/lab_notebook/data/psnr_profiles")

async def main():
    label = sys.argv[1]
    blocks = [int(b) for b in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    async with websockets.connect("ws://localhost:8004/ws", max_size=2**24) as ws:
        await ws.send(json.dumps({"t": "query"}))
        while True:
            m = await ws.recv()
            if isinstance(m, str) and json.loads(m).get("t") == "query_ack":
                break
        seq = 900000
        for b in blocks:
            tj = json.loads((R / f"block_{b:03d}/transforms.json").read_text())
            frames = sorted(tj["frames"], key=lambda f: f["file_path"])
            rows = []
            for idx, fr in enumerate(frames):
                kf = fr["file_path"].split("/")[-1].split(".")[0]
                c2w = np.asarray(fr["transform_matrix"], float)
                prev = None
                for k in range(30):
                    seq += 1
                    await ws.send(json.dumps({
                        "t": "pose", "seq": seq, "c2w": c2w.ravel().tolist(),
                        "w": 1280, "h": 720, "quality": 95, "block": b}))
                    while True:
                        m = await asyncio.wait_for(ws.recv(), timeout=300)
                        if isinstance(m, (bytes, bytearray)):
                            a = np.asarray(Image.open(io.BytesIO(bytes(m[20:]))).convert("RGB"), np.float32)
                            break
                    if a.std(axis=(0, 1)).max() < 6:
                        prev = None; await asyncio.sleep(1.5); continue
                    if prev is not None and np.abs(a - prev).mean() < 0.5:
                        break
                    prev = a; await asyncio.sleep(0.12)
                g, s = GT / f"{kf}.png", SKY / f"{kf}.png"
                if not (g.exists() and s.exists()):
                    continue
                gt = np.asarray(Image.open(g).convert("RGB").resize((1280, 720)), np.float64)
                mk = np.asarray(Image.open(s).convert("L").resize((1280, 720), Image.NEAREST)) < 127
                if mk.sum() < 1000:
                    continue
                fg = 10 * np.log10(255**2 / max(((a.astype(np.float64) - gt) ** 2)[mk].mean(), 1e-9))
                rows.append({"kf": kf, "idx": idx, "fg": round(fg, 2),
                             "holdout": idx % 10 == 0,
                             "edge": idx < 2 or idx >= len(frames) - 2})
            out[str(b)] = rows
            tr = [r["fg"] for r in rows if not r["holdout"]]
            inter = [r["fg"] for r in rows if not r["holdout"] and not r["edge"]]
            med = float(np.median(tr)); p5 = float(np.percentile(tr, 5))
            imin = min(inter); wvm = med - imin
            ho = float(np.median([r["fg"] for r in rows if r["holdout"]]))
            bars = [med >= 21, p5 >= 17, imin >= 16, wvm <= 4]
            print(f"[{label} b{b:03d}] med {med:.2f} P5 {p5:.2f} int-min {imin:.2f} "
                  f"wvm {wvm:.1f} ho {ho:.2f}  bars {'/'.join('P' if x else 'F' for x in bars)}",
                  flush=True)
    (OUT / f"{label}_profiles.json").write_text(json.dumps(out))
    print(f"[saved] {OUT}/{label}_profiles.json", flush=True)

asyncio.run(main())
