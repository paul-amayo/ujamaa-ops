"""Label-completeness test for the pseudo-label fine-tune: union the tracker labels of a COCO train set with stock SAM3's own per-frame
detections on the same crops (canonical recipe: text 'fruit' on the crop, parent gate 0.5 against the tree mask), adding a detection
where no existing annotation centroid lies within R px. sam3 env: add_stock_labels.py <coco dir in> <coco dir out>"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from pycocotools import mask as mu
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
IN, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True); R = 12.0; PAD, UP = 24, 2.0; D = Path("/home/paperspace/data/citrus_all/05_13D_Jackal/prod/scratch_sam3")
if not (OUT / "images").exists(): os.symlink(IN / "images", OUT / "images")
C = json.load(open(IN / "_annotations.coco.json")); by = {}
for a in C["annotations"]: by.setdefault(a["image_id"], []).append(a)
proc = Sam3Processor(build_sam3_image_model()); added = 0; aid = max(a["id"] for a in C["annotations"]) + 1
for im in C["images"]:
    img = Image.open(IN / "images" / im["file_name"]).convert("RGB"); tm = np.array(Image.open(D / im["tree"] / "masks" / f"{im['frame']}.png")) > 127; ys, xs = np.where(tm)
    y0, y1 = max(0, ys.min() - PAD), min(720, ys.max() + PAD); x0, x1 = max(0, xs.min() - PAD), min(1280, xs.max() + PAD); tcrop = np.array(Image.fromarray(tm[y0:y1, x0:x1].astype(np.uint8) * 255).resize((im["width"], im["height"]), Image.NEAREST)) > 127
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = proc.set_image(img); r = proc.set_text_prompt(state=st, prompt="fruit")
    masks = r.get("masks", []); have = [(a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2) for a in by.get(im["id"], [])]
    for m in masks:
        arr = (m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)).squeeze(); mm = np.array(Image.fromarray(arr.astype(np.uint8) * 255).resize((im["width"], im["height"]), Image.BILINEAR)) > 127
        if mm.sum() < 4 or (mm & tcrop).sum() / mm.sum() < 0.5: continue
        yy, xx = np.where(mm); cx, cy = xx.mean(), yy.mean()
        if any(np.hypot(cx - u, cy - v) <= R * UP for u, v in have): continue   # R in frame px -> crop px (2x)
        e = mu.encode(np.asfortranarray(mm.astype(np.uint8))); C["annotations"].append({"id": aid, "image_id": im["id"], "category_id": 1, "segmentation": {"size": e["size"], "counts": e["counts"].decode()}, "area": int(mm.sum()), "bbox": [int(xx.min()), int(yy.min()), int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)], "iscrowd": 0, "confirmed": False, "prob": None, "track": "stock"}); aid += 1; added += 1; have.append((cx, cy))
C["info"]["labels"] = C["info"].get("labels", "") + "+stock"; json.dump(C, open(OUT / "_annotations.coco.json", "w")); print(f"[union] {len(C['images'])} images: {len(C['annotations']) - added} tracker labels + {added} stock detections added -> {OUT}", flush=True)
