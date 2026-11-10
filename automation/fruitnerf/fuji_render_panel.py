#!/usr/bin/env python3
"""fuji_render_panel.py <out.npz> [--side west --image image_179.png --tree Tree07] (2026-11-10) — the Fuji-SfM panel of the HiGH cross-site figure
as a RENDER of our own splat (Paul, Fig. 3 v3: "a render with relevancy, not an image with masks"), matching the three IoU panels.
  splat     the side's splatfacto run the registry ran on (high/<side>/outputs/sf_30000_bilateral, the dataparser_transforms.json fuji_eval.py and
            fuji_tree_counts.py read), rendered from that side's training camera <image> (nerfstudio's undistorted pinhole).
  apples    the merged apples of merged_flip_r10.npz are rebuilt from fuji_eval.py's rule (masklink_components_plain30k_record of each side ->
            GT frame by align_flip, GT region + 0.3 m, greedy one-to-one east<->west within 0.10 m; asserted equal to the npz), so each merged
            apple knows its <side> registry component; the tree's TP / FP sets are fuji_tree_counts.py's (asserted against tree_counts_flip_d36.json).
  lighting  the census trick (splat_census.py / outline_probe.py): one colour channel per group, every Gaussian of the group's registry
            components (masklink_votes fid[comp == c]) at 1, the rest 0; the rasteriser is linear in the colours, so channel k of the render
            is each pixel's alpha-blend weight from group k. Apples of the tree with no <side> component (east-only singles) cannot be lit here.
  misses    the tree's GT apples with no TP: no Gaussians, so their centres are projected with the render camera; drawn when in frame and not
            behind the rendered surface (centre depth <= rendered depth + 0.10 m); ring radius = 4 cm at the centre's depth.
nerf_new pixi env (GPU, one frame):  fuji_render_panel.py /tmp/x/fuji_render.npz"""
import argparse, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--side', default='west'); ap.add_argument('--image', default='image_179.png'); ap.add_argument('--tree', default='Tree07')
ap.add_argument('--counts', default='/home/paperspace/data/public/fuji_sfm/tree_counts_flip_d36.json'); ap.add_argument('--merged', default='/home/paperspace/data/public/fuji_sfm/merged_flip_r10.npz')
ap.add_argument('--dump', default='plain30k_record'); ap.add_argument('--align', default='align_flip'); ap.add_argument('--radius', type=float, default=0.10); ap.add_argument('--margin', type=float, default=0.3); a = ap.parse_args()
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R = ROOT / 'high'; RUN = 'outputs/sf_30000_bilateral/splatfacto/run'
M = np.load(a.merged); P, pair, GT, GT_tree = M['P'], M['pair'], M['GT'], M['GT_tree']
def frames(side):
    S = R / side; z = np.load(S / f'fruit_registry/{a.align}.npz'); dpt = json.loads((S / RUN / 'dataparser_transforms.json').read_text()); return float(z['s']), z['R'], z['t'], np.array(dpt['transform']), float(dpt['scale'])
def model_to_gt(side, X):
    s, Rm, t, T, sc = frames(side); colmap = ((X / sc) - T[:, 3]) @ T[:, :3]; return (s * (Rm @ colmap.T)).T + t
def gt_to_model(side, X):
    s, Rm, t, T, sc = frames(side); colmap = (Rm.T @ ((X - t) / s).T).T; return sc * (colmap @ T[:, :3].T + T[:, 3])
# ---- rebuild the merge with provenance (fuji_eval.py, --align-file align_flip, one2one) ----
lo, hi = GT.min(0) - a.margin, GT.max(0) + a.margin; per = []
for side in ('east', 'west'):
    d_ = np.load(R / side / f'fruit_registry/masklink_components_{a.dump}.npz'); cid = np.where(d_['apple'])[0]; Pg = model_to_gt(side, d_['centroid'][cid]); ins = np.all((Pg >= lo) & (Pg <= hi), 1); per.append((Pg[ins], cid[ins]))
(A_, ca), (B_, cb) = per; d = np.linalg.norm(A_[:, None, :] - B_[None, :, :], axis=-1); ua = np.zeros(len(A_), bool); ub = np.zeros(len(B_), bool); mP, mE, mW = [], [], []
for k in np.argsort(d, axis=None):
    i, j = divmod(int(k), len(B_))
    if d[i, j] > a.radius: break
    if ua[i] or ub[j]: continue
    ua[i] = ub[j] = True; mP.append((A_[i] + B_[j]) / 2); mE.append(ca[i]); mW.append(cb[j])
P2 = np.concatenate([np.array(mP).reshape(-1, 3), A_[~ua], B_[~ub]]); comp_e = np.concatenate([mE, ca[~ua], np.full((~ub).sum(), -1)]).astype(int); comp_w = np.concatenate([mW, np.full((~ua).sum(), -1), cb[~ub]]).astype(int)
assert P2.shape == P.shape and np.allclose(P2, P, atol=1e-9), 'rebuilt merge differs from merged_flip_r10.npz'
COMP = {'east': comp_e, 'west': comp_w}[a.side]; print(f'[fuji-render] merge rebuilt with provenance: {len(mP)} pairs + {(~ua).sum()} east + {(~ub).sum()} west singles = {len(P2)} (equals {a.merged})')
# ---- the tree's TP / FP / FN (fuji_tree_counts.py) ----
J = json.load(open(a.counts)); row = next(r for r in J['rows'] if r['tree'] == a.tree)
c500 = np.load(ROOT / 'their_cloud_500k.npy').astype(np.float64); c0 = c500.mean(0); w_, v_ = np.linalg.eigh(np.cov((c500[:, :2] - c0[:2]).T)); row2 = v_[:, -1]; along = lambda X: (X[:, :2] - c0[:2]) @ row2
TR = np.array(J['trunks_along']); assign = np.argmin(np.abs(along(P)[:, None] - TR[None, :]), axis=1); trees = sorted(set(GT_tree)); gmed = np.array([np.median(along(GT[GT_tree == t_])) for t_ in trees])
dd = np.abs(gmed[:, None] - TR[None, :]); used = set(); t2k = {}
for k in np.argsort(dd, axis=None):
    i, j = divmod(int(k), len(TR))
    if dd[i, j] > 0.6: break
    if trees[i] in t2k or j in used: continue
    t2k[trees[i]] = j; used.add(j)
gi = np.where(GT_tree == a.tree)[0]; mine = assign == t2k[a.tree]; tp = mine & (pair >= 0) & np.isin(pair, gi); fp = mine & ~tp; fn = np.setdiff1d(gi, pair[tp])
assert (int(mine.sum()), int(tp.sum()), len(gi)) == (row['ours'], row['tp'], row['gt']), 'tree sets differ from the counts json'
lit = {'TP': COMP[tp], 'FP': COMP[fp]}; nolit = {k: int((v < 0).sum()) for k, v in lit.items()}
print(f"[fuji-render] {a.tree}: F1 {row['f1']:.3f} (json); TP {tp.sum()}, FP {fp.sum()}, FN {len(fn)}; with a {a.side} component (lightable): TP {int((lit['TP'] >= 0).sum())}, FP {int((lit['FP'] >= 0).sum())}; {a.side}-less: {nolit}")
# ---- render ----
import torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
S = R / a.side; _, pipe, _, step = eval_setup(S / RUN / 'config.yml', test_mode='inference'); model = pipe.model; model.eval(); dev = model.device
cam = None
for ds in (pipe.datamanager.train_dataset, pipe.datamanager.eval_dataset):
    names = [Path(p).name for p in ds.image_filenames]
    if a.image in names: i_ = names.index(a.image); cam = ds.cameras[i_:i_ + 1].to(dev); break
assert cam is not None, a.image
print(f'[fuji-render] camera {a.image}: {int(cam.width.item())}x{int(cam.height.item())}, fx {float(cam.fx.item()):.1f}, distortion {cam.distortion_params.cpu().numpy().round(4).tolist() if cam.distortion_params is not None else None}, step {step}')
with torch.no_grad(): o = model.get_outputs(cam); rgb = o['rgb'][0].clamp(0, 1).cpu().numpy() if o['rgb'].dim() == 4 else o['rgb'].clamp(0, 1).cpu().numpy(); depth = o['depth'].squeeze().cpu().numpy()
v = np.load(S / f'fruit_registry/masklink_votes_plain30k.npz'); fid = v['fid']; comp = np.load(S / f'fruit_registry/masklink_components_{a.dump}.npz')['comp']
g = model.gauss_params; N = g['means'].shape[0]; col = torch.zeros(N, 2, device=dev); nG = {}
for k_, (name, cs) in enumerate(lit.items()):
    gids = fid[np.isin(comp, cs[cs >= 0])]; col[torch.from_numpy(gids).to(dev), k_] = 1.0; nG[name] = int(len(gids))
viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); W0, H0 = int(cam.width.item()), int(cam.height.item())
with torch.no_grad():
    out, _, _ = rasterization(means=g['means'], quats=g['quats'] / g['quats'].norm(dim=-1, keepdim=True), scales=torch.exp(g['scales']), opacities=torch.sigmoid(g['opacities']).squeeze(-1), colors=col,
                              viewmats=viewmat, Ks=K, width=W0, height=H0, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
wgt = out[0].cpu().numpy(); print(f"[fuji-render] lit Gaussians TP {nG['TP']}, FP {nG['FP']}; pixels with blend weight >= 0.5: TP {int((wgt[..., 0] >= 0.5).sum())}, FP {int((wgt[..., 1] >= 0.5).sum())}")
# ---- misses: GT centres through the render camera ----
Xm = gt_to_model(a.side, GT[fn]); Vm = viewmat[0].cpu().numpy(); Kn = K[0].cpu().numpy(); pc = Xm @ Vm[:3, :3].T + Vm[:3, 3]; z = pc[:, 2]
u = Kn[0, 0] * pc[:, 0] / z + Kn[0, 2]; vv = Kn[1, 1] * pc[:, 1] / z + Kn[1, 2]; inf = (z > 0) & (u >= 0) & (u < W0) & (vv >= 0) & (vv < H0)
s_, _, _, _, sc = frames(a.side); tol = 0.10 / s_ * sc; Dm = np.full(len(fn), np.nan); Dm[inf] = depth[vv[inf].astype(int), u[inf].astype(int)]; vis = inf & (z <= Dm + tol)
rr = Kn[0, 0] * (0.04 / s_ * sc) / np.maximum(z, 1e-6)
print(f'[fuji-render] misses: {len(fn)}; in frame {int(inf.sum())}, not behind the rendered surface {int(vis.sum())} (drawn)')
np.savez_compressed(a.out, rgb=(rgb * 255).round().astype(np.uint8), w_tp=wgt[..., 0].astype(np.float16), w_fp=wgt[..., 1].astype(np.float16), miss_u=u[vis], miss_v=vv[vis], miss_r=rr[vis],
                    tree=a.tree, f1=row['f1'], p=row['p'], r=row['r'], counts=a.counts, side=a.side, image=a.image, n_tp=int(tp.sum()), n_fp=int(fp.sum()), n_fn=len(fn),
                    lit_tp=int((lit['TP'] >= 0).sum()), lit_fp=int((lit['FP'] >= 0).sum()), miss_drawn=int(vis.sum()), miss_inframe=int(inf.sum()), gauss_tp=nG['TP'], gauss_fp=nG['FP'], step=step)
print('[fuji-render] wrote', a.out)
