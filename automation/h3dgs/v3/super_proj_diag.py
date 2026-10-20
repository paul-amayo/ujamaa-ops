#!/usr/bin/env python3
"""super_proj_diag.py (2026-10-20) — which Gaussians render a trunk? Project every Gaussian of the cell into one verdict frame (workspace
camera, or the chunk's COLMAP pose through dataparser_transforms as splat_feature_render.py does), keep those whose centre falls inside the
SAM 3 trunk mask of tree k and whose depth lies within 5 % of the rendered depth (visible), and tabulate their seed labels under one or more
seeds. nerf_new pixi env:  super_proj_diag.py <identity> <parts dir with verdict/> <frame> <tree> <labels.npy>[,<labels.npy>...] [--sparse <chunk sparse/0>]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
ap = argparse.ArgumentParser(); ap.add_argument('identity'); ap.add_argument('parts'); ap.add_argument('frame'); ap.add_argument('tree', type=int); ap.add_argument('labels'); ap.add_argument('--sparse', default=''); a = ap.parse_args()
N, P = Path(a.identity), Path(a.parts); k = a.tree; f = a.frame; cfg = Path(json.load(open(N / 'W.json'))['config']); WS = N.parent
_, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; cam = None
for ds in (pipe.datamanager.eval_dataset, pipe.datamanager.train_dataset):
    names = [Path(p).name for p in ds.image_filenames]
    if f in names: cam = ds.cameras[names.index(f):names.index(f) + 1].to(dev); src = 'workspace'; break
if cam is None:
    from read_write_model import read_images_binary, qvec2rotmat
    sp = a.sparse or str(WS / 'sparse/0'); dp = json.load(open(cfg.parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); ws = json.load(open(WS / 'transforms.json'))
    im = {i.name: i for i in read_images_binary(os.path.join(sp, 'images.bin')).values()}[f]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    c2w = T @ (np.linalg.inv(w2c) @ np.diag([1.0, -1.0, -1.0, 1.0])); c2w[:3, 3] *= float(dp['scale'])
    cam = Cameras(camera_to_worlds=torch.tensor(c2w[:3], dtype=torch.float32)[None], fx=float(ws['fl_x']), fy=float(ws['fl_y']), cx=float(ws['cx']), cy=float(ws['cy']), width=int(ws['w']), height=int(ws['h'])).to(dev); src = 'colmap'
with torch.no_grad(): depth = model.get_outputs(cam)['depth'][..., 0]
H, W = depth.shape; vm = get_viewmat(cam.camera_to_worlds)[0]; K = cam.get_intrinsics_matrices()[0].to(dev); m = model.gauss_params['means'].detach(); p = (vm[:3, :3] @ m.T).T + vm[:3, 3]; z = p[:, 2]
u = K[0, 0] * p[:, 0] / z + K[0, 2]; v = K[1, 1] * p[:, 1] / z + K[1, 2]; inb = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
recs = json.load(open(P / 'verdict' / f.replace('.png', '.json'))); zp = np.load(P / 'verdict' / f.replace('.png', '.npz')); Hf, Wf = zp['shape']; M = np.unpackbits(zp['masks'], axis=1)[:, :Hf * Wf].reshape(-1, Hf, Wf).astype(bool)
trunk = np.zeros((Hf, Wf), bool)
for j, r in enumerate(recs):
    if r['prompt'] == 'tree trunk' and r['tree'] == k: trunk |= M[j]
trunk = torch.from_numpy(np.array(Image.fromarray(trunk).resize((W, H), Image.NEAREST))).to(dev)
idx = torch.where(inb)[0]; ui, vi = u[idx].long(), v[idx].long(); inm = trunk[vi, ui]; d = depth[vi, ui]; vis = (z[idx] - d).abs() <= 0.05 * d; sel = idx[inm & vis]; selall = idx[inm]
opa = torch.sigmoid(model.gauss_params['opacities'].detach()).squeeze(-1); w = opa[sel]
print(f'[proj] {f} ({src} camera, {W}x{H}): gaussians with centre inside the trunk mask of tree {k}: {len(selall)}; visible (depth within 5 %): {len(sel)}, opacity sum {float(w.sum()):.1f}')
for lp in a.labels.split(','):
    lab = torch.from_numpy(np.load(lp)).to(dev); l = lab[sel]; tn = 20000 + 4 * k; cats = {'trunk node': l == tn, 'branch node': l == tn + 1, 'leaves node': l == tn + 2, f'tree {k}': l == k, 'other tree': (l >= 0) & (l < 10000) & (l != k), 'other part': (l >= 20000) & ((l < tn) | (l > tn + 2)), 'void': l < 0}
    print(f'[proj] {Path(lp).parent.name}/{Path(lp).name}: ' + ', '.join(f'{n} {int(c.sum())} ({100 * float(w[c].sum() / w.sum()):.0f} % of opacity)' for n, c in cats.items() if c.any()))
