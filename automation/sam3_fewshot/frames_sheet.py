"""Stock vs fine-tuned SAM3 on single frames for Paul (2026-10-07): per held-out tree the frame where the fine-tuned model found the most
confirmed oranges that stock missed (the gain) and the frame with its most detections on nothing (the cost). Green = stock detection,
yellow = fine-tuned detection, cyan ring = 3D-confirmed orange, small white dot = any tracked object. Tree crop at 2x.
sam3/h3dgs env: SAM3_TAG=<run> frames_sheet.py -> /home/paperspace/data/demo_video_v2/sam3_fewshot/frames_<tag>.jpg"""
import json, os, numpy as np
from PIL import Image, ImageDraw, ImageFont
TAG = os.environ.get("SAM3_TAG", "union40_ep1"); D = "/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3"; ok = lambda v: v["reproj_med_px"] <= 6 and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22); tiles = []
for t in (5, 72):
    W = f"{D}/fruit3d_t{t}"; A = {r["name"]: r for r in json.load(open(f"{W}/fewshot/ckpt_stock_refined_kfonly_thr6.json"))}; B = {r["name"]: r for r in json.load(open(f"{W}/fewshot/ckpt_{TAG}_refined_kfonly_thr6.json"))}
    tr = json.load(open(f"{W}/fewshot/tracks.json")); tri = json.load(open(f"{W}/fewshot/tri_refined_kfonly.json")); TRK, CONF = {}, {}
    for tid, obs in tr.items():
        for o in obs:
            if o["in_tree"]: TRK.setdefault(o["name"], []).append(o); CONF.setdefault(o["name"], []).append(o) if (tid in tri and ok(tri[tid])) else None
    near = lambda x, pts: any(np.hypot(x["u"] - p["u"], x["v"] - p["v"]) <= 12 for p in pts)
    gain = {n: sum(1 for c in CONF.get(n, []) if near(c, B[n]["dets"]) and not near(c, A[n]["dets"])) for n in B}; cost = {n: sum(1 for x in B[n]["dets"] if not near(x, CONF.get(n, [])) and not near(x, TRK.get(n, []))) for n in B}
    for n, lab in ((max(gain, key=gain.get), "gain"), (max(cost, key=cost.get), "cost")):
        im = Image.open(f"{W}/frames/{n}.png").convert("RGB"); tm = np.array(Image.open(f"{W}/masks/{n}.png")) > 127; ys, xs = np.where(tm); box = (max(0, xs.min() - 24), max(0, ys.min() - 24), min(1280, xs.max() + 24), min(720, ys.max() + 24))
        c = im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS); d = ImageDraw.Draw(c); X = lambda p: ((p["u"] - box[0]) * 2, (p["v"] - box[1]) * 2)
        for o in TRK.get(n, []): x, y = X(o); d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 255))
        for s in A[n]["dets"]: x, y = X(s); r = max(8, 2 * np.sqrt(s["area"] / np.pi)); d.ellipse([x - r, y - r, x + r, y + r], outline=(40, 230, 40), width=3)
        for s in B[n]["dets"]: x, y = X(s); r = max(11, 2 * np.sqrt(s["area"] / np.pi) + 4); d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 225, 0), width=3)
        for o in CONF.get(n, []): x, y = X(o); r = max(15, 2 * np.sqrt(o["area"] / np.pi) + 8); d.ellipse([x - r, y - r, x + r, y + r], outline=(0, 235, 255), width=3)
        cap = f"tree {t}, {n} ({lab}): stock {len(A[n]['dets'])} det, {TAG} {len(B[n]['dets'])} det; confirmed oranges here {len(CONF.get(n, []))}, found by stock {sum(near(c_, A[n]['dets']) for c_ in CONF.get(n, []))}, by {TAG} {sum(near(c_, B[n]['dets']) for c_ in CONF.get(n, []))}; on nothing {cost[n]}"
        d.rectangle([0, 0, d.textlength(cap, font=F) + 16, 34], fill=(0, 0, 0)); d.text((8, 5), cap, font=F, fill=(255, 255, 255)); tiles.append(c)
w = max(t.width for t in tiles); h = max(t.height for t in tiles); sheet = Image.new("RGB", (2 * w + 8, 2 * h + 8 + 40), (0, 0, 0)); ImageDraw.Draw(sheet).text((8, 8), "green = stock SAM3   yellow = fine-tuned   cyan = 3D-confirmed orange   white dot = tracked object", font=F, fill=(255, 255, 255))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 2) * (w + 8), 40 + (i // 2) * (h + 8)))
out = f"/home/paperspace/data/demo_video_v2/sam3_fewshot/frames_{TAG}.jpg"; os.makedirs(os.path.dirname(out), exist_ok=True); sheet.save(out, quality=90); print("[frames]", out, sheet.size)
