#!/usr/bin/env python3
"""masked_psnr_by_size.py — rendering quality INSIDE semantic masks by object size (the COCO small/medium/large idea for splats, 2026-10-13):
for a splatfacto / HiGH run, render the census (training) views with the trained bilateral grid applied (as the training loss saw them) and the
held-out views raw, and report PSNR over the pixels of every supervision instance grouped by the instance's size in the frame — the E1 strata
(>= 3 %, 1-3 %, 0.2-1 %, < 0.2 % of the frame) and the COCO pixel buckets at the render resolution (< 32², 32²-96², > 96² px, scaled by the
render scale) — plus the whole-frame PSNR. Fruit ids (>= 10000) are also reported on their own. Compare a stock cell with its LoD variant
on the same frames to measure what mask-driven level of detail buys the rendering.
nerf_new pixi env:  masked_psnr_by_size.py --config <run config.yml> --sup <supervision dir> [--frames f.png ...] [--n-train 30] [--out json]"""
import argparse, json, math, re
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--frames', nargs='*', default=[]); ap.add_argument('--n-train', type=int, default=30); ap.add_argument('--out', default='')
a = ap.parse_args(); UNLAB, FRUIT = 65535, 10000
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; dm = pipe.datamanager
sets = {'train': dm.train_dataset, 'eval': dm.eval_dataset}; names = {s: [Path(p).name for p in ds.image_filenames] for s, ds in sets.items()}
if a.frames: pick = {s: [i for i, n in enumerate(names[s]) if n in a.frames] for s in sets}
else:
    n = len(names['train']); pick = {'train': sorted(range(n), key=lambda i: int(re.sub(r'\D', '', names['train'][i]) or 0))[::max(1, n // a.n_train)][:a.n_train], 'eval': list(range(len(names['eval'])))}
STRATA = (('>=3%', 0.03, 1.01), ('1-3%', 0.01, 0.03), ('0.2-1%', 0.002, 0.01), ('<0.2%', 0.0, 0.002)); COCO = (('small<32^2', 0, 32 ** 2), ('medium', 32 ** 2, 96 ** 2), ('large>96^2', 96 ** 2, 10 ** 12))
acc = {s: {k: [0.0, 0] for k in ('frame', 'all_masks', 'fruit') + tuple(x[0] for x in STRATA) + tuple(x[0] for x in COCO)} for s in sets}; nfr = {s: 0 for s in sets}
with torch.no_grad():
    for s, ds in sets.items():
        for i in pick[s]:
            f = Path(a.sup) / names[s][i]
            if not f.exists(): continue
            cam = ds.cameras[i:i + 1].to(dev); rgb = model.get_outputs(cam)['rgb']; g = ds[i]['image'].to(dev)[..., :3]; H, W = rgb.shape[:2]
            if s == 'train' and getattr(model.config, 'use_bilateral_grid', False): rgb = model._apply_bilateral_grid(rgb[None], i, H, W)[0]
            se = ((rgb.clamp(0, 1) - g) ** 2).mean(-1)   # per-pixel squared error
            ids = np.array(Image.fromarray(np.array(Image.open(f), np.uint16)).resize((W, H), Image.NEAREST)); idt = torch.from_numpy(ids.astype(np.int64)).to(dev)
            scale_px = (H * W) / (1280 * 720)   # COCO buckets defined at 1280x720, scaled to the render size
            u, c = torch.unique(idt, return_counts=True); nfr[s] += 1; acc[s]['frame'][0] += se.sum().item(); acc[s]['frame'][1] += se.numel()
            for uid, cnt in zip(u.tolist(), c.tolist()):
                if uid == UNLAB: continue
                m = idt == uid; e = se[m].sum().item(); frac = cnt / (H * W)
                for key in ('all_masks',) + (('fruit',) if uid >= FRUIT else ()): acc[s][key][0] += e; acc[s][key][1] += cnt
                for lab, lo, hi in STRATA:
                    if lo <= frac < hi: acc[s][lab][0] += e; acc[s][lab][1] += cnt
                for lab, lo, hi in COCO:
                    if lo <= cnt / scale_px < hi: acc[s][lab][0] += e; acc[s][lab][1] += cnt
psnr = lambda v: (10 * math.log10(1 / max(v[0] / v[1], 1e-10))) if v[1] else float('nan')
out = {s: {k: {'psnr': psnr(v), 'px': v[1]} for k, v in acc[s].items()} | {'frames': nfr[s]} for s in sets}
print(f"[masked-psnr] {Path(a.config).parent.parent.name}: " + ' | '.join(f"{s} ({nfr[s]} fr): " + ', '.join(f"{k} {out[s][k]['psnr']:.2f}" for k in ('frame', 'all_masks', 'fruit', '>=3%', '1-3%', '0.2-1%', '<0.2%', 'small<32^2', 'medium', 'large>96^2') if out[s][k]['px']) for s in sets), flush=True)
if a.out: json.dump(out, open(a.out, 'w'), indent=1)
