#!/usr/bin/env python3
"""langsplat_nms_check.py (2026-11-11) — the released LangSplat mask_nms against the GPU-vectorised one of preprocess_fast.py on real SAM masks
of one image (downscaled so the released loop finishes): prints the kept indices of both and whether they are identical.
langsplat conda env, cwd = LangSplat:  python langsplat_nms_check.py <image> [--width 480]"""
import argparse, importlib.util, sys, time
import cv2, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--width', type=int, default=480); a = ap.parse_args()
def load(name):
    sys.argv = [name]; spec = importlib.util.spec_from_file_location(name.replace('.py', ''), name); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
orig, fast = load('preprocess.py'), load('preprocess_fast.py')
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry['vit_h'](checkpoint='ckpts/sam_vit_h_4b8939.pth').to('cuda')
gen = SamAutomaticMaskGenerator(model=sam, points_per_side=32, pred_iou_thresh=0.7, box_nms_thresh=0.7, stability_score_thresh=0.85, crop_n_layers=1, crop_n_points_downscale_factor=1, min_mask_region_area=100)
img = cv2.imread(a.image); h, w = img.shape[:2]; img = cv2.resize(img, (a.width, int(h * a.width / w))); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks_default, masks_s, masks_m, masks_l = gen.generate(img)
for lvl, ml in zip(('default', 's', 'm', 'l'), (masks_default, masks_s, masks_m, masks_l)):
    seg = torch.from_numpy(np.stack([m['segmentation'] for m in ml])); sc = torch.from_numpy(np.stack([m['predicted_iou'] for m in ml])) * torch.from_numpy(np.stack([m['stability_score'] for m in ml]))
    t0 = time.time(); k0 = orig.mask_nms(seg, sc, iou_thr=0.8, score_thr=0.7, inner_thr=0.5); t1 = time.time(); k1 = fast.mask_nms(seg, sc, iou_thr=0.8, score_thr=0.7, inner_thr=0.5); t2 = time.time()
    same = torch.equal(k0.view(-1).cpu(), k1.view(-1).cpu())
    print(f'[check] level {lvl}: {len(ml)} masks, kept released {k0.numel()} / fast {k1.numel()}, identical {same}; released {t1 - t0:.1f} s, fast {t2 - t1:.2f} s', flush=True)
