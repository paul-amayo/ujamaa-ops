#!/usr/bin/env python3
"""Lindendhof identity, step 2 (2026-10-12): each SAM3 trunk mask (trunk_masks.py) -> one 3D trunk observation in the survey's LO
world frame. Depth = the lane's own v3 model rendered at that frame's training/held-out camera (eval mode, expected depth), so the
labels later painted from it agree with the geometry the census renders. Per mask: the pixels in the nearest half of its depth range
(the trunk's front surface; masks bleed onto the background) are back-projected and their median is the observation; the mask's
lowest 3D point is kept as a ground-contact estimate. Camera: nerfstudio c2w (OpenGL) -> world = the run's dataparser transform
inverted (p_world = R^T (p_ns / scale - t)).
nerf_new pixi env:  trunk_obs.py <survey root> <trunk mask root> [lanes=1,2,3,4,5] -> <trunk mask root>/lane<n>/trunk_obs.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
S, M = Path(sys.argv[1]), Path(sys.argv[2]); LANES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4,5').split(',')]
for n in LANES:
    cfg = S / f'experimental/v3/chunk_lane{n}/outputs/sf_60000_bilateral/splatfacto/run/config.yml'; t0 = time.time()
    _, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
    dp = json.load(open(cfg.parent / 'dataparser_transforms.json')); T = np.array(dp['transform'], np.float64); s = float(dp['scale']); RT, tT = T[:, :3], T[:, 3]
    cams = {}
    for ds in (pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset):
        for i, p in enumerate(ds.image_filenames): cams[Path(p).name] = (ds, i)
    obs = []; nf = 0
    for jf in sorted((M / f'lane{n}').glob('f_*.json')):
        inst = json.load(open(jf)); fname = jf.name.replace('.json', '.png')
        if not inst or fname not in cams: continue
        ds, i = cams[fname]; cam = ds.cameras[i:i + 1].to(dev)
        with torch.no_grad(): depth = model.get_outputs(cam)['depth'][..., 0].float().cpu().numpy()   # [H, W] z-depth, nerfstudio units
        lab = np.array(Image.open(jf.with_suffix('.png'))); H, W = lab.shape
        fx, fy, cx, cy = (float(v) for v in (cam.fx[0, 0], cam.fy[0, 0], cam.cx[0, 0], cam.cy[0, 0])); c2w = cam.camera_to_worlds[0].cpu().numpy().astype(np.float64)
        for it in inst:
            vv, uu = np.nonzero(lab == it['idx'])
            if len(uu) < 30: continue
            d = depth[vv, uu]; ok = np.isfinite(d) & (d > 0)
            if ok.sum() < 30: continue
            lo, mid = np.percentile(d[ok], [5, 50]); sel = ok & (d >= lo) & (d <= mid)
            u, v, z = uu[sel], vv[sel], d[sel]
            pc = np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z], 1)          # OpenGL camera coordinates
            pns = pc @ c2w[:3, :3].T + c2w[:3, 3]; pw = (pns / s - tT) @ RT        # nerfstudio -> survey LO world (metres)
            bot = np.argmax(v)                                                     # lowest mask pixel in the image -> ground contact
            obs.append({'frame': fname, 'idx': it['idx'], 'score': it['score'], 'area': it['area'], 'xyz': np.median(pw, 0).round(3).tolist(),
                        'z_low': round(float(np.percentile(pw[:, 2], 5)), 3), 'range_m': round(float(np.median(z) / s), 2), 'n_px': int(sel.sum())})
        nf += 1
    json.dump(obs, open(M / f'lane{n}' / 'trunk_obs.json', 'w'), indent=0)
    P = np.array([o['xyz'] for o in obs]) if obs else np.zeros((0, 3))
    print(f'[trunk-obs] lane {n}: {len(obs)} observations from {nf} frames in {time.time() - t0:.0f}s; range median {np.median([o["range_m"] for o in obs]):.1f} m; '
          f'xyz span {np.ptp(P, 0).round(1).tolist() if len(P) else []}', flush=True)
    del pipe, model; torch.cuda.empty_cache()
