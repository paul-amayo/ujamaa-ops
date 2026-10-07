"""Check sheet for the few-shot pilot: frames with the most confirmed-orange views the stock detector missed. Tree crop; green = stock
detection, cyan ring = confirmed 3D orange seen by the tracker here, red ring = confirmed orange view with NO stock detection
(candidate miss). sam3/h3dgs env: pilot_sheet.py -> <workdir>/fewshot/misses_sheet.jpg"""
import json, os, numpy as np
from PIL import Image, ImageDraw
W = os.environ.get("FRUIT3D_WORK", "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3/fruit3d_t72"); F = f"{W}/fewshot"; THR = float(os.environ.get("REPROJ_PX", "6"))
tr = json.load(open(f"{F}/tracks.json")); tri = json.load(open(f"{F}/tri_refined_kfonly.json")); st = {r["name"]: r["detections"] for r in json.load(open(f"{W}/detections.json"))}
ok = lambda v: v["reproj_med_px"] <= THR and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
C = {t: [o for o in tr[t] if o["in_tree"]] for t, v in tri.items() if ok(v)}; by = {}
for t, obs in C.items():
    for o in obs: by.setdefault(o["name"], []).append(o)
near = lambda n, o: any(np.hypot(d["u"] - o["u"], d["v"] - o["v"]) <= 12 for d in st[n])
miss = {n: [o for o in obs if not near(n, o)] for n, obs in by.items()}; pick = sorted(miss, key=lambda n: -len(miss[n]))[:6]; tiles = []
for n in pick:
    im = Image.open(f"{W}/frames/{n}.png").convert("RGB"); t = np.array(Image.open(f"{W}/masks/{n}.png")) > 127; ys, xs = np.where(t); box = (max(0, xs.min() - 24), max(0, ys.min() - 24), min(1280, xs.max() + 24), min(720, ys.max() + 24))
    d = ImageDraw.Draw(im)
    for s in st[n]: r = max(4, np.sqrt(s["area"] / np.pi)); d.ellipse([s["u"] - r, s["v"] - r, s["u"] + r, s["v"] + r], outline=(40, 220, 40), width=2)
    for o in by[n]: r = max(6, np.sqrt(o["area"] / np.pi) + 3); d.ellipse([o["u"] - r, o["v"] - r, o["u"] + r, o["v"] + r], outline=(255, 40, 40) if not near(n, o) else (0, 230, 255), width=2)
    c = im.crop(box).resize((int((box[2] - box[0]) * 1.5), int((box[3] - box[1]) * 1.5)), Image.LANCZOS); ImageDraw.Draw(c).text((6, 6), f"{n}: stock {len(st[n])} dets, confirmed views {len(by[n])}, missed by stock {len(miss[n])}", fill=(255, 255, 0)); tiles.append(c)
h = max(t.height for t in tiles); w = max(t.width for t in tiles); sheet = Image.new("RGB", (3 * w, 2 * h), (0, 0, 0))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 3) * w, (i // 3) * h))
sheet.save(f"{F}/misses_sheet.jpg", quality=90); print("[sheet]", [(n, len(miss[n]), len(by[n])) for n in pick], "->", f"{F}/misses_sheet.jpg")
