"""Stage B — canonical SAM3 in-tree fruit detection on stage-A native frames.

Same recipe as fruit_in_trees_ledger / crop_ab: tree-mask bbox + pad 24,
2.0x upscale, prompt 'fruit', parent-overlap gate 0.5. Adds per-detection
centroid px + median depth over the fruit mask (from the aligned 10 fps ZED
depth PNGs); ring fallback (front quartile of tree-masked neighborhood) where
ZED holes sit on the fruit — tagged depth_src so clustering can tier it.
Runs in the sam3 pixi env. Usage: fruit3d_detect.py <workdir> [workdir ...]
(one model load for the whole fleet). Output: <workdir>/detections.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

PAD, UPSCALE, OVERLAP, PROMPT = 24, 2.0, 0.5, "fruit"

model = build_sam3_image_model()
processor = Sam3Processor(model)
print(f"[det] SAM3 ready; {len(sys.argv)-1} workdirs", flush=True)


def run_workdir(work):
    meta = json.loads((work / "meta.json").read_text())
    out = []
    for rec in meta["frames"]:
        name = rec["name"]
        rgb = Image.open(work / "frames" / f"{name}.png").convert("RGB")
        tmask = np.array(Image.open(work / "masks" / f"{name}.png")) > 127
        depth = np.array(Image.open(work / "depth" / f"{name}.png"))
        if depth.ndim == 3:
            depth = depth[..., 0]
        H, W = tmask.shape
        ys, xs = np.where(tmask)
        y0, y1 = max(0, ys.min() - PAD), min(H, ys.max() + PAD)
        x0, x1 = max(0, xs.min() - PAD), min(W, xs.max() + PAD)
        cw, ch = x1 - x0, y1 - y0
        crop = rgb.crop((x0, y0, x1, y1)).resize(
            (int(cw * UPSCALE), int(ch * UPSCALE)), Image.BILINEAR)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(crop)
            res = processor.set_text_prompt(state=state, prompt=PROMPT)
        masks, scores = res.get("masks"), res.get("scores")
        if masks is None:
            continue
        sc = (scores.float().cpu().numpy().tolist()
              if hasattr(scores, "cpu") else list(scores or []))
        dets = []
        for oi, m in enumerate(masks):
            arr = (m.detach().cpu().numpy() if hasattr(m, "detach")
                   else np.asarray(m))
            if arr.ndim == 3:
                arr = arr.squeeze()
            small = np.array(Image.fromarray(arr.astype(np.uint8) * 255)
                             .resize((cw, ch), Image.BILINEAR)) > 127
            fm = np.zeros((H, W), bool)
            fm[y0:y1, x0:x1] = small
            area = int(fm.sum())
            if area == 0:
                continue
            if (fm & tmask).sum() / area < OVERLAP:
                continue
            fys, fxs = np.where(fm)
            dz = depth[fm]
            dz = dz[(dz > 0) & np.isfinite(dz)]
            dsrc = "fruit"
            if dz.size < 4:
                # ZED depth holes sit exactly on small fruit. Fallback: fruit
                # hangs on the OUTER canopy, so the front quartile of nearby
                # tree-masked depth is the right prior.
                ring = np.zeros((H, W), bool)
                cy, cx = int(fys.mean()), int(fxs.mean())
                ring[max(0, cy - 12):min(H, cy + 13),
                     max(0, cx - 12):min(W, cx + 13)] = True
                rz = depth[ring & tmask & ~fm]
                rz = rz[(rz > 0) & np.isfinite(rz)]
                if rz.size >= 8:
                    dz = np.array([np.percentile(rz, 25)])
                    dsrc = "ring"
            dets.append({"u": float(fxs.mean()), "v": float(fys.mean()),
                         "area": area,
                         "score": float(sc[oi]) if oi < len(sc) else 0.0,
                         "depth_med": float(np.median(dz)) if dz.size else None,
                         "depth_n": int(dz.size), "depth_src": dsrc})
        out.append({"name": name, "ts_ms": rec["ts_ms"],
                    "is_kf": rec["is_kf"], "donor_kf": rec["donor_kf"],
                    "detections": dets})
    (work / "detections.json").write_text(json.dumps(out, indent=1))
    tot = sum(len(r["detections"]) for r in out)
    kf_tot = sum(len(r["detections"]) for r in out if r["is_kf"])
    print(f"[det] {work.name}: {tot} detection-events on {len(out)} frames "
          f"({kf_tot} on keyframes)", flush=True)


for w in sys.argv[1:]:
    work = Path(w)
    if not (work / "meta.json").exists():
        print(f"[det] SKIP {work} (no meta.json)", flush=True)
        continue
    run_workdir(work)
print("[det] ALL DONE")
