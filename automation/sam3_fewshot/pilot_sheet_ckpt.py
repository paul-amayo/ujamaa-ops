"""Stock vs another checkpoint on the same frames (few-shot pilot): tree crop, green = stock detections, yellow = the other model's,
cyan ring = confirmed 3D orange, white dot = any tracker object. Frames = the six with the most other-model detections that sit on
nothing (no confirmed orange, no tracker object within 12 px) - the candidates for false positives.
sam3/h3dgs env: SAM3_TAG=<tag> pilot_sheet_ckpt.py [workdir] -> <workdir>/fewshot/ckpt_<tag>_sheet.jpg"""
import json, os, sys, numpy as np
from PIL import Image, ImageDraw
W = sys.argv[1] if len(sys.argv) > 1 else "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/fruit3d_t72"; F = f"{W}/fewshot"; TAG = os.environ.get("SAM3_TAG", "dryrun60")
A = {r["name"]: r for r in json.load(open(f"{F}/ckpt_stock_refined_kfonly_thr6.json"))}; B = {r["name"]: r for r in json.load(open(f"{F}/ckpt_{TAG}_refined_kfonly_thr6.json"))}
tr = json.load(open(f"{F}/tracks.json")); tri = json.load(open(f"{F}/tri_refined_kfonly.json")); ok = lambda v: v["reproj_med_px"] <= 6 and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
TRK, CONF = {}, {}
for t, obs in tr.items():
    for o in obs:
        if o["in_tree"]: TRK.setdefault(o["name"], []).append(o); (CONF.setdefault(o["name"], []).append(o) if ok(tri.get(t, {"reproj_med_px": 99, "parallax_deg": 0, "depth_min": 0, "depth_max": 99, "dist_centroid": 99})) else None)
near = lambda x, pts: any(np.hypot(x["u"] - p["u"], x["v"] - p["v"]) <= 12 for p in pts)
nothing = {n: [x for x in B[n]["dets"] if not near(x, CONF.get(n, [])) and not near(x, TRK.get(n, []))] for n in B}; pick = sorted(nothing, key=lambda n: -len(nothing[n]))[:6]; tiles = []
for n in pick:
    im = Image.open(f"{W}/frames/{n}.png").convert("RGB"); t = np.array(Image.open(f"{W}/masks/{n}.png")) > 127; ys, xs = np.where(t); box = (max(0, xs.min() - 24), max(0, ys.min() - 24), min(1280, xs.max() + 24), min(720, ys.max() + 24)); d = ImageDraw.Draw(im)
    for o in TRK.get(n, []): d.ellipse([o["u"] - 2, o["v"] - 2, o["u"] + 2, o["v"] + 2], fill=(255, 255, 255))
    for o in CONF.get(n, []): r = max(7, np.sqrt(o["area"] / np.pi) + 4); d.ellipse([o["u"] - r, o["v"] - r, o["u"] + r, o["v"] + r], outline=(0, 230, 255), width=2)
    for x in A[n]["dets"]: r = max(4, np.sqrt(x["area"] / np.pi)); d.ellipse([x["u"] - r, x["v"] - r, x["u"] + r, x["v"] + r], outline=(40, 220, 40), width=2)
    for x in B[n]["dets"]: r = max(5, np.sqrt(x["area"] / np.pi) + 2); d.ellipse([x["u"] - r, x["v"] - r, x["u"] + r, x["v"] + r], outline=(255, 230, 0), width=2)
    c = im.crop(box).resize((int((box[2] - box[0]) * 1.5), int((box[3] - box[1]) * 1.5)), Image.LANCZOS); ImageDraw.Draw(c).text((6, 6), f"{n}: stock {len(A[n]['dets'])} (green), {TAG} {len(B[n]['dets'])} (yellow); on nothing {len(nothing[n])}", fill=(255, 255, 0)); tiles.append(c)
h = max(t.height for t in tiles); w = max(t.width for t in tiles); sheet = Image.new("RGB", (3 * w, 2 * h), (0, 0, 0))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 3) * w, (i // 3) * h))
sheet.save(f"{F}/ckpt_{TAG}_sheet.jpg", quality=90); print("[sheet]", [(n, len(nothing[n])) for n in pick], "->", f"{F}/ckpt_{TAG}_sheet.jpg")
