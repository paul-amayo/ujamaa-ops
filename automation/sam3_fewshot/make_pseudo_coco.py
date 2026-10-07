"""Pseudo-label training set for few-shot SAM3 (no hand labels): one COCO json over the fruit3d trees named on the command line,
images = the canonical in-tree crops (tree bbox + pad 24, 2x, the recipe SAM3 is run with) as JPEG, annotations = tracker masks
(tracks.json with RLE from `TRACK_MASKS=1 pilot_t72.py track`) of in-tree observations. LABELS=confirmed keeps only tracks the
3D check confirmed (tri_refined_kfonly.json, reproj <= 6 px: pure, sparse); LABELS=tracks (default) keeps every in-tree tracker
observation with prob >= MIN_PROB (complete enough that unlabelled real fruit does not teach the model to suppress fruit).
python3 make_pseudo_coco.py <out dir> <fruit3d workdir> [...]  -> <out>/images/*.jpg, <out>/_annotations.coco.json"""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mu
OUT = Path(sys.argv[1]); (OUT / "images").mkdir(parents=True, exist_ok=True); LABELS = os.environ.get("LABELS", "tracks"); MIN_PROB = float(os.environ.get("MIN_PROB", "0.5")); PAD, UP = 24, 2.0
images, anns = [], []; ok = lambda v: v["reproj_med_px"] <= 6 and v["parallax_deg"] >= 2 and v["depth_min"] > 0.5 and v["depth_max"] < 12 and v["dist_centroid"] < 6
for W in sys.argv[2:]:
    W = Path(W); tr = json.load(open(W / "fewshot/tracks.json")); tri = json.load(open(W / "fewshot/tri_refined_kfonly.json")) if (W / "fewshot/tri_refined_kfonly.json").exists() else {}
    keep = {t for t, v in tri.items() if ok(v)} if LABELS == "confirmed" else set(tr); by = {}
    for t, obs in tr.items():
        if t not in keep: continue
        for o in obs:
            if o["in_tree"] and o["prob"] >= MIN_PROB and "rle" in o: by.setdefault(o["name"], []).append((t, o))
    for n, obs in sorted(by.items()):
        tm = np.array(Image.open(W / "masks" / f"{n}.png")) > 127; ys, xs = np.where(tm); y0, y1 = max(0, ys.min() - PAD), min(720, ys.max() + PAD); x0, x1 = max(0, xs.min() - PAD), min(1280, xs.max() + PAD)
        cw, ch = int((x1 - x0) * UP), int((y1 - y0) * UP); fn = f"{W.name}_{n}.jpg"; Image.open(W / "frames" / f"{n}.png").convert("RGB").crop((x0, y0, x1, y1)).resize((cw, ch), Image.BILINEAR).save(OUT / "images" / fn, quality=95)
        iid = len(images) + 1; images.append({"id": iid, "file_name": fn, "width": cw, "height": ch, "tree": W.name, "frame": n})
        for t, o in obs:
            fm = mu.decode({"size": o["rle"]["size"], "counts": o["rle"]["counts"].encode()}).astype(bool)[y0:y1, x0:x1]
            m = np.array(Image.fromarray(fm.astype(np.uint8) * 255).resize((cw, ch), Image.BILINEAR)) > 127
            if m.sum() < 4: continue
            r = mu.encode(np.asfortranarray(m.astype(np.uint8))); yy, xx = np.where(m)
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "segmentation": {"size": r["size"], "counts": r["counts"].decode()}, "area": int(m.sum()), "bbox": [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)], "iscrowd": 0, "confirmed": t in {t_ for t_, v in tri.items() if ok(v)}, "prob": o["prob"], "track": t})
json.dump({"images": images, "annotations": anns, "categories": [{"id": 1, "name": "fruit", "supercategory": "fruit"}], "info": {"labels": LABELS, "min_prob": MIN_PROB, "source": "SAM3 video tracks (+ 3D check) on citrus fruit3d clips, make_pseudo_coco.py"}}, open(OUT / "_annotations.coco.json", "w"))
print(f"[coco] {len(images)} images, {len(anns)} annotations ({sum(a['confirmed'] for a in anns)} on 3D-confirmed tracks), labels={LABELS} -> {OUT}", flush=True)
