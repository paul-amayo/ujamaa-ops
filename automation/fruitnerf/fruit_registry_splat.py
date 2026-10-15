#!/usr/bin/env python3
"""fruit_registry_splat.py — the splat-based fruit registry (2026-10-15): census the per-frame fruit masks (fruit_masks/, 2 fruit / 1 labelled
non-fruit) through a trained splat — per gaussian, alpha mass on fruit px and on non-fruit px over all training views, and the number of views in
which it paints fruit — then FRUIT gaussians (fruit share of the labelled mass >= SHARE, fruit mass >= MASS) are clustered in 3-D by single
linkage at EPS x the apple radius (scene-unit radius from the triangulated registry x the dataparser scale); an apple = a cluster whose
gaussians paint fruit in >= MIN_VIEWS views, counted inside the tree crop (0.5 x the median camera distance from the vertical axis through the
cameras' look-at point). Pre-registered: SHARE 0.5, MASS 1, EPS 1.0, MIN_VIEWS 3. Also reports fruit purity (share of the fruit px' rendered
mass from fruit gaussians).
nerf_new pixi env:  fruit_registry_splat.py --config <run config.yml> --survey <survey> --gt 179 [--tag name]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--gt', type=int, default=0)
ap.add_argument('--share', type=float, default=0.5); ap.add_argument('--mass', type=float, default=1.0); ap.add_argument('--eps', type=float, default=1.0); ap.add_argument('--min-views', type=int, default=3)
ap.add_argument('--crop', type=float, default=0.5); ap.add_argument('--scale', type=float, default=0.5); ap.add_argument('--tag', default='splat'); a = ap.parse_args()
S = a.survey.resolve(); _, pipe, _, step = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
dps = float(json.load(open(a.config.parent / 'dataparser_transforms.json'))['scale'])
rt = json.load(open(S / 'fruit_registry/registry.json')); R_model = rt['apple_radius_scene_median'] * dps
Wf = torch.zeros(N, device=dev); Wn = torch.zeros(N, device=dev); nfr = torch.zeros(N, device=dev); acc = [0.0, 0.0]
for i in range(len(ds)):
    nm = Path(ds.image_filenames[i]).name; fm = S / 'fruit_masks' / nm
    if not fm.exists(): continue
    cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev).clone(); W, H = int(cam.width.item() * a.scale), int(cam.height.item() * a.scale); K[:, :2] *= a.scale
    m = torch.from_numpy(np.array(Image.open(fm).resize((W, H), Image.NEAREST))).to(dev); f = (m == 2).float(); n_ = (m == 1).float()
    col = torch.zeros(N, 2, device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gf, = torch.autograd.grad((out[0, ..., 0] * f).sum() + (out[0, ..., 1] * n_).sum(), col)
    Wf += gf[:, 0].clamp(min=0); Wn += gf[:, 1].clamp(min=0); nfr += (gf[:, 0] > 0.05).float()
share = Wf / (Wf + Wn).clamp(min=1e-9); fruit = (share >= a.share) & (Wf >= a.mass)
P = means[fruit].cpu().numpy(); V = nfr[fruit].cpu().numpy(); M = Wf[fruit].cpu().numpy()
pairs = cKDTree(P).query_pairs(a.eps * R_model, output_type='ndarray') if len(P) > 1 else np.zeros((0, 2), int)
G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P))); k, lab = connected_components(G, directed=False)
cen = np.array([np.average(P[lab == c], axis=0, weights=M[lab == c]) for c in range(k)]); views = np.array([V[lab == c].max() for c in range(k)]); size = np.bincount(lab, minlength=k)
cams = torch.stack([ds.cameras.camera_to_worlds[i] for i in range(len(ds))]).numpy(); C = cams[:, :3, 3]; Fd = -cams[:, :3, 2]; UP = cams[:, :3, 1].mean(0); UP /= np.linalg.norm(UP)   # nerfstudio c2w is OpenGL: looks along -z, up +y
A_ = np.zeros((3, 3)); b_ = np.zeros(3)
for c, d in zip(C, Fd / np.linalg.norm(Fd, axis=1, keepdims=True)): Pm = np.eye(3) - np.outer(d, d); A_ += Pm; b_ += Pm @ c
LOOK = np.linalg.solve(A_, b_)
def hd(Y): dd = Y - LOOK; return np.linalg.norm(dd - np.outer(dd @ UP, UP), axis=1)
cam_r = float(np.median(hd(C))); inside = hd(cen) <= a.crop * cam_r if k else np.zeros(0, bool); apple = inside & (views >= a.min_views); n = int(apple.sum())
err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
sens = {f'share{sh}_eps{e}': None for sh in (0.3, 0.5) for e in (0.75, 1.0, 1.5)}
print(f"[splat-registry] {S.name} ({a.tag}, step {step}, {N} gaussians): fruit gaussians {int(fruit.sum())} (share >= {a.share}, mass >= {a.mass}); clusters {k}, apples (>= {a.min_views} views, in the tree): **{n}**{err}; "
      f"gaussians per apple median {np.median(size[apple]) if n else 0:.0f}, views per apple median {np.median(views[apple]) if n else 0:.0f}", flush=True)
json.dump({'tag': a.tag, 'step': int(step), 'apples': n, 'gt': a.gt or None, 'fruit_gaussians': int(fruit.sum()), 'clusters': int(k), 'centroids': cen[apple].tolist(), 'views': views[apple].tolist()},
          open(S / f'fruit_registry/registry_splat_{a.tag}.json', 'w'))
