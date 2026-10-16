#!/usr/bin/env python3
"""fuji_trunk_census.py --config <run config.yml> --survey <side root> [--tag plain30k --every 2] (2026-10-16) — trunk census of one Fuji-SfM side:
SAM3 'tree trunk' masks (prod/bateleur/sam3_trunk) rasterised through the splat exactly like the fruit census of fruit_registry_trunks.py
(trunk share of the labelled mass >= 0.5, mass >= 1) -> fruit_registry/trunk_census_<tag>.npz (P = trunk gaussian means, model frame; Wt).
nerf_new pixi env."""
import argparse, json, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--tag', default='plain30k'); ap.add_argument('--every', type=int, default=2); a = ap.parse_args()
S = a.survey.resolve(); R = S / 'fruit_registry'; R.mkdir(exist_ok=True); cache = R / f'trunk_census_{a.tag}.npz'
import torch
from gsplat import rasterization
from pycocotools import mask as mask_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
t0 = time.time(); _, pipe, _, step = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
tj = json.loads((S / 'transforms.json').read_text()); W0, H0 = tj['w'], tj['h']; masks = {}; nmask = 0
for cm in sorted((S / 'prod/bateleur/sam3_trunk').glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items():
        m = np.ones((H0, W0), np.uint8)
        for e in ents:
            mk = mask_utils.decode(e['rle']).astype(bool)
            if mk.shape == (H0, W0): m[mk] = 2; nmask += 1
        masks[f'image_{l2i[int(li)]}.png'] = m
Wt = torch.zeros(N, device=dev); Wn = torch.zeros(N, device=dev); nv = 0
for i in range(0, len(ds), a.every):
    nm = Path(ds.image_filenames[i]).name
    if nm not in masks: continue
    cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); W, H = int(cam.width.item()), int(cam.height.item())
    m = torch.from_numpy(masks[nm]).to(dev); col = torch.zeros(N, 2, device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gf, = torch.autograd.grad((out[0, ..., 0] * (m == 2).float()).sum() + (out[0, ..., 1] * (m == 1).float()).sum(), col); Wt += gf[:, 0].clamp(min=0); Wn += gf[:, 1].clamp(min=0); nv += 1
trunk = ((Wt / (Wt + Wn).clamp(min=1e-9)) >= 0.5) & (Wt >= 1.0); np.savez_compressed(cache, P=means[trunk].cpu().numpy(), Wt=Wt[trunk].cpu().numpy(), nviews=nv, N=N, trunk_masks=nmask, frames=len(masks))
print(f'[trunks] {S.name}: trunk census over {nv} views ({len(masks)} frames with trunk masks, {nmask} masks) in {time.time() - t0:.0f} s: {int(trunk.sum())} trunk gaussians of {N} -> {cache.name}', flush=True)
