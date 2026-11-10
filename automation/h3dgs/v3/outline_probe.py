#!/usr/bin/env python3
"""outline_probe.py (2026-11-10) — why do pixels on tree outlines answer a neighbouring tree with a high score? (HiGH Fig. 1, tree 36
on citrus 05 cell 1_0, held-out kf_002411.) The held-out frame is rasterised with one channel per pixel group (the census trick of
splat_census.py: d(sum of channel k over group k)/d(colours) = each Gaussian's blend weight into the group), the groups being the
argmax-rule decisions for tree 36 against the supervision:
  TP        tree 36 pixels answered 36
  FP_tree   pixels of another tree's mask answered 36 (the outline of the trees in front)
  FP_unlab  unlabelled pixels answered 36 (ground strip, sky edge)
For each group: the share of its blend weight carried by Gaussians seeded 36 / another tree / void, and for the Gaussians seeded 36 that
carry FP weight, their census evidence (W row of 36, of their own visible tree, of the unlabelled pixels, the total) against those
carrying TP weight; plus where they sit (distance of their mean to the tree-36 Gaussians' centroid, metres).
nerf_new pixi env:  outline_probe.py [--frame kf_002411.png] [--target 36]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--frame', default='kf_002411.png'); ap.add_argument('--target', type=int, default=36); a = ap.parse_args()
N_ = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); FD = Path('/home/paperspace/code/aru_sil_core/src/docs/paper/figures')
SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
cfg = json.load(open(N_ / 'W.json'))['config']; DPS = float(json.load(open(Path(cfg).parent / 'dataparser_transforms.json'))['scale'])
seed = np.load(N_ / 'features_B.bin.labels.npy'); Wz = np.load(json.load(open(N_ / 'features_B.bin.json'))['w_npz'], allow_pickle=True)
Wm = Wz['W']; wl = [int(x) for x in Wz['labels']]; print('[probe] W', Wm.shape, 'labels', len(wl), 'bg row' if 65535 in wl else 'no bg row')
# the argmax decision of the hero heat file (render of the same field at 640x360)
z = np.load(FD / 'hero_heat_05_1_0_kf002411.npz', allow_pickle=True); m = json.loads(str(z['meta'])); wt = {int(k): w for k, w in m['word_table'].items()}
ids = [t for t in sorted(wt) if t < 10000 and wt[t] in z.files]; void = z['void'].astype(bool); am = np.array(ids)[np.stack([z[wt[t]] for t in ids]).argmax(0)]; am[void] = -1
h, w = am.shape; I = np.array(Image.fromarray(np.array(Image.open(SUP / a.frame))).resize((w, h), Image.NEAREST)).astype(np.int64)
pred = am == a.target; groups = {'TP': pred & (I == a.target), 'FP_tree': pred & (I != a.target) & (I < 10000), 'FP_unlab': pred & (I == 65535)}
for k, g in groups.items(): print(f'[probe] {k}: {int(g.sum())} px')
_, pipe, _, step = eval_setup(Path(cfg), test_mode='test'); model = pipe.model; model.eval(); dev = model.device
cam = None
for ds in (pipe.datamanager.eval_dataset, pipe.datamanager.train_dataset):
    names = [Path(p).name for p in ds.image_filenames]
    if a.frame in names: cam = ds.cameras[names.index(a.frame):names.index(a.frame) + 1].to(dev); break
assert cam is not None, 'frame not found in the eval or train cameras'
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev).clone(); W0, H0 = int(cam.width.item()), int(cam.height.item()); s = w / W0; K[:, :2] *= s
onehot = torch.from_numpy(np.stack([groups[k] for k in groups], -1).astype(np.float32)).to(dev)
col = torch.zeros(N, len(groups), device=dev, requires_grad=True); torch.set_grad_enabled(True)
out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=viewmat, Ks=K, width=w, height=h, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
gw, = torch.autograd.grad((out[0] * onehot).sum(), col); gw = gw.cpu().numpy()   # [N, groups]
row = {u: i for i, u in enumerate(wl)}; Wt = Wm[row[a.target]] if a.target in row else np.zeros(N); Wbg = Wm[row[65535]] if 65535 in row else np.zeros(N); Wtot = Wm.sum(0)
tree_rows = [row[u] for u in wl if u < 10000]; Wbest = Wm[tree_rows].max(0); Wbest_id = np.array([wl[r] for r in tree_rows])[Wm[tree_rows].argmax(0)]
mu = means.cpu().numpy(); c36 = mu[seed == a.target].mean(0)
res = {}
for j, k in enumerate(groups):
    wk = gw[:, j]; tot = wk.sum(); s36 = wk[seed == a.target].sum() / tot; svoid = wk[seed < 0].sum() / tot; soth = 1 - s36 - svoid
    sel = (seed == a.target) & (wk > 0); ww = wk[sel] / wk[sel].sum()
    ev36 = float((ww * (Wt[sel] / np.maximum(Wtot[sel], 1e-9))).sum()); evbg = float((ww * (Wbg[sel] / np.maximum(Wtot[sel], 1e-9))).sum()); evtot = float((ww * Wtot[sel]).sum())
    dist = float((ww * np.linalg.norm(mu[sel] - c36, axis=1)).sum() / DPS)
    res[k] = dict(px=int(groups[k].sum()), share_weight_seeded_target=float(s36), share_seeded_other=float(soth), share_void=float(svoid), seeded_target_census_share_target=ev36, seeded_target_census_share_unlabelled=evbg, seeded_target_census_total=evtot, seeded_target_dist_to_target_centroid_m=dist)
    print(f'[probe] {k:9s} weight from Gaussians seeded {a.target}: {100 * s36:5.1f} %  seeded other: {100 * soth:5.1f} %  void: {100 * svoid:5.1f} %  | seeded-{a.target} carriers: census share {a.target} {ev36:.2f}, unlabelled {evbg:.2f}, census total {evtot:.1f} px, {dist:.2f} m from the tree-{a.target} centroid')
json.dump(res, open('/tmp/d36hero/outline_probe.json', 'w'), indent=1); print('[probe] wrote /tmp/d36hero/outline_probe.json')
