#!/usr/bin/env python3
"""fruit_registry_maskgraph.py — stage 1 of the MASK-GRAPH registry (2026-10-15 16:5x, Paul: "we are going to bridge the gap, no shortcuts").
Why: the mask-linked registry's two under-count mechanisms are (1) single linkage bridging clusters of touching apples through gaussians
that pass a 15-cm depth test at a neighbour's mask (one component = a dozen apples), and (2) 26-29 % of the apple masks inside the tree
have no gaussian with census fruit share >= 0.5 (their painters are mixed primitives, share ~0.25). Both come from judging GAUSSIANS.
Here the SAM3 masks are the units: for every mask in every view, the exact alpha mass each gaussian contributes to that mask's pixels
(gradient of the rendered dummy channel wrt the gaussian's colour, one backward pass per mask, at --scale resolution). A gaussian behind
another apple contributes ~nothing to the front mask (transmittance), so no depth test is needed; no share threshold is needed either.
Output fruit_registry/maskgraph_<tag>.npz: sparse rows (mrow, gid, w) + the mask table (mview, mlab, mmass, marea) in the same (view, mask)
indexing as masklink_votes_<tag>.npz (views in dataset order among those with SAM3 masks, masks sorted by score ascending).
nerf_new pixi env:  fruit_registry_maskgraph.py --config <run config.yml> --survey <survey> --tag plain30k [--scale 0.5] [--min-w 0.002]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from pycocotools import mask as mask_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--tag', default='plain30k')
ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--min-w', type=float, default=0.002, help='keep contributions >= this fraction of the mask mass'); a = ap.parse_args()
S = a.survey.resolve(); R = S / 'fruit_registry'; t0 = time.time(); _, pipe, _, _ = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
labmaps = {}
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, ents in meta['frame_entries'].items():
        k = l2i[int(li)]; Lm = None
        for j, e in enumerate(sorted(ents, key=lambda e: e.get('score', 0))):
            m = mask_utils.decode(e['rle']).astype(bool)
            if Lm is None: Lm = np.full(m.shape, -1, np.int16)
            Lm[m] = j
        if Lm is not None: labmaps[f'image_{k}.png'] = Lm
MR, GID, WW, MV, ML, MM, MA, names = [], [], [], [], [], [], [], []; mrow = 0; vi = 0; tv = time.time()
for i in range(len(ds)):
    nm = Path(ds.image_filenames[i]).name
    if not (S / 'fruit_masks' / nm).exists() or nm not in labmaps: continue
    cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev).clone(); W, H = int(cam.width.item()), int(cam.height.item())
    s = a.scale; W2, H2 = int(W * s), int(H * s); K[0, 0, 0] *= s; K[0, 1, 1] *= s; K[0, 0, 2] *= s; K[0, 1, 2] *= s
    Lm_full = labmaps[nm]; Lm = torch.from_numpy(Lm_full[::int(round(1 / s)), ::int(round(1 / s))][:H2, :W2].copy()).to(dev); nmk = int(Lm_full.max()) + 1; areas = np.bincount(Lm_full[Lm_full >= 0].ravel(), minlength=nmk)
    dummy = torch.ones(N, 1, device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=dummy, viewmats=vm, Ks=K, width=W2, height=H2, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    img = out[0, ..., 0]
    for j in range(nmk):
        mk = (Lm == j)
        if not mk.any(): MV.append(vi); ML.append(j); MM.append(0.0); MA.append(int(areas[j])); mrow += 1; continue
        gr, = torch.autograd.grad((img * mk.float()).sum(), dummy, retain_graph=True); w = gr[:, 0]; tot = float(w.sum()); keep = torch.where(w >= a.min_w * tot)[0]
        MR.append(np.full(len(keep), mrow, np.int32)); GID.append(keep.cpu().numpy().astype(np.int32)); WW.append(w[keep].cpu().numpy().astype(np.float32)); MV.append(vi); ML.append(j); MM.append(tot); MA.append(int(areas[j])); mrow += 1
    names.append(nm); vi += 1
    if vi in (1, 5, 20) or vi % 100 == 0: print(f'[maskgraph] {S.name}: view {vi} ({nmk} masks) — {(time.time() - tv) / vi:.1f} s per view, {mrow} masks so far', flush=True)
np.savez_compressed(R / f'maskgraph_{a.tag}.npz', mrow=np.concatenate(MR), gid=np.concatenate(GID), w=np.concatenate(WW), mview=np.array(MV, np.int32), mlab=np.array(ML, np.int16), mmass=np.array(MM, np.float32), marea=np.array(MA, np.int32), names=np.array(names), N=N, scale=a.scale, min_w=a.min_w)
print(f'[maskgraph] {S.name} ({a.tag}): {mrow} masks over {vi} views, {sum(len(x) for x in GID)} (mask, gaussian) entries, {time.time() - t0:.0f} s -> maskgraph_{a.tag}.npz', flush=True)
