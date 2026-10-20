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
ap = argparse.ArgumentParser(); ap.add_argument('identity'); ap.add_argument('parts'); ap.add_argument('frame'); ap.add_argument('tree', type=int); ap.add_argument('labels'); ap.add_argument('--sparse', default=''); ap.add_argument('--all-heads', default='', help='a super_eval.json: run every (frame, tree) head and join the composed IoU; frame/tree arguments ignored'); ap.add_argument('--out', default=''); ap.add_argument('--config', default='', help='splat config.yml to read the Gaussians from (default: the identity dir W.json config)'); a = ap.parse_args()
N, P = Path(a.identity), Path(a.parts); k = a.tree; f = a.frame; cfg = Path(a.config) if a.config else Path(json.load(open(N / 'W.json'))['config']); WS = N.parent
_, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); dev = model.device
from read_write_model import read_images_binary, qvec2rotmat
sp = a.sparse or str(WS / 'sparse/0'); dp = json.load(open(cfg.parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); ws = json.load(open(WS / 'transforms.json')); ims = {i.name: i for i in read_images_binary(os.path.join(sp, 'images.bin')).values()}
def camera(f):
    for ds in (pipe.datamanager.eval_dataset, pipe.datamanager.train_dataset):
        names = [Path(p).name for p in ds.image_filenames]
        if f in names: return ds.cameras[names.index(f):names.index(f) + 1].to(dev), 'workspace'
    im = ims[f]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = T @ (np.linalg.inv(w2c) @ np.diag([1.0, -1.0, -1.0, 1.0])); c2w[:3, 3] *= float(dp['scale'])
    return Cameras(camera_to_worlds=torch.tensor(c2w[:3], dtype=torch.float32)[None], fx=float(ws['fl_x']), fy=float(ws['fl_y']), cx=float(ws['cx']), cy=float(ws['cy']), width=int(ws['w']), height=int(ws['h'])).to(dev), 'colmap'
m = model.gauss_params['means'].detach(); opa = torch.sigmoid(model.gauss_params['opacities'].detach()).squeeze(-1); labs = {lp: torch.from_numpy(np.load(lp)).to(dev) for lp in a.labels.split(',')}
def trunk_mask(f, k, W, H):
    recs = json.load(open(P / 'verdict' / f.replace('.png', '.json'))); zp = np.load(P / 'verdict' / f.replace('.png', '.npz')); Hf, Wf = zp['shape']; M = np.unpackbits(zp['masks'], axis=1)[:, :Hf * Wf].reshape(-1, Hf, Wf).astype(bool); t = np.zeros((Hf, Wf), bool)
    for j, r in enumerate(recs):
        if r['prompt'] == 'tree trunk' and r['tree'] == k: t |= M[j]
    return torch.from_numpy(np.array(Image.fromarray(t).resize((W, H), Image.NEAREST))).to(dev)
def head(f, k):
    cam, src = camera(f)
    with torch.no_grad(): depth = model.get_outputs(cam)['depth'][..., 0]
    H, W = depth.shape; vm = get_viewmat(cam.camera_to_worlds)[0]; K = cam.get_intrinsics_matrices()[0].to(dev); p = (vm[:3, :3] @ m.T).T + vm[:3, 3]; z = p[:, 2]
    u = K[0, 0] * p[:, 0] / z + K[0, 2]; v = K[1, 1] * p[:, 1] / z + K[1, 2]; inb = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H); trunk = trunk_mask(f, k, W, H)
    idx = torch.where(inb)[0]; ui, vi = u[idx].long(), v[idx].long(); inm = trunk[vi, ui]; d = depth[vi, ui]; vis = (z[idx] - d).abs() <= 0.05 * d; sel = idx[inm & vis]; w = opa[sel]; tn = 20000 + 4 * k
    out = {'frame': f, 'tree': k, 'camera': src, 'trunk_px': int(trunk.sum().item()), 'in_mask': int(inm.sum().item()), 'visible': int(len(sel)), 'opacity_sum': float(w.sum())}
    for lp, lab in labs.items():
        l = lab[sel]; out[Path(lp).parent.name + '/' + Path(lp).name] = {'trunk_node_n': int((l == tn).sum()), 'trunk_node_opacity_share': float(w[l == tn].sum() / w.sum()) if len(sel) else 0.0, 'tree_n': int((l == k).sum()), 'void_n': int((l < 0).sum()), 'other_part_n': int(((l >= 20000) & (l != tn)).sum())}
    return out
if a.all_heads:
    ev = json.load(open(a.all_heads)); rows = []
    for h in ev['per_head']:
        o = head(h['frame'], h['tree']); o['composed_iou'] = h['super_node'][0]; o['composed_hit'] = h['super_node'][3]; o['trunk_frac'] = h['trunk_frac']; rows.append(o)
        lp = list(labs)[0]; d = o[Path(lp).parent.name + '/' + Path(lp).name]; print(f"[proj] {o['frame']} tree {o['tree']:3d}: trunk {100 * o['trunk_frac']:.2f} % | in-mask {o['in_mask']:5d}, at trunk depth {o['visible']:3d} (opacity {o['opacity_sum']:5.1f}), trunk-node share of that opacity {d['trunk_node_opacity_share']:.2f} ({d['trunk_node_n']} gaussians; tree {d['tree_n']}, void {d['void_n']}, other part {d['other_part_n']}) | composed IoU {o['composed_iou']:.3f} hit {o['composed_hit']}")
    if a.out: json.dump(rows, open(a.out, 'w'), indent=1)
else:
    o = head(a.frame, a.tree); print(json.dumps(o, indent=1))
