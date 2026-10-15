#!/usr/bin/env python3
"""fruit_registry_trunks.py — per-tree crop by TRUNKS (2026-10-15): FruitNeRF cropped its point clouds by hand to the tree of interest; our
stand-in is physical — an apple belongs to the nearest trunk. SAM3 'tree trunk' masks (prod/bateleur/sam3_trunk) are censused through the
splat like the fruit (trunk share of the labelled mass >= 0.5, mass >= 1); the trunk gaussians' horizontal positions are smoothed (KDE,
0.05 cam_r) and the peaks (one vote per gaussian; >= 10 % of the strongest, 0.3 cam_r apart) are the trunks. The target trunk = the peak nearest the look-point;
peaks within 0.4 cam_r of it (support posts, forks) count as the target. Apples = the mask-linked candidates (components seen in >= 3
views, from masklink_components_<tag>.npz) inside the camera ring whose nearest trunk is the target's. GT-blind; rules fixed before running.
nerf_new pixi env:  fruit_registry_trunks.py --config <run config.yml> --survey <survey> --gt 113 --tag plain30k"""
import argparse, json, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--gt', type=int, default=0)
ap.add_argument('--tag', default='plain30k'); ap.add_argument('--every', type=int, default=2); ap.add_argument('--rebuild', action='store_true'); a = ap.parse_args()
S = a.survey.resolve(); R = S / 'fruit_registry'; cache = R / f'trunk_census_{a.tag}.npz'
if a.rebuild or not cache.exists():
    import torch
    from gsplat import rasterization
    from pycocotools import mask as mask_utils
    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.models.splatfacto import get_viewmat
    t0 = time.time(); _, pipe, _, step = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
    g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
    tj = json.loads((S / 'transforms.json').read_text()); W0, H0 = tj['w'], tj['h']; masks = {}
    for cm in sorted((S / 'prod/bateleur/sam3_trunk').glob('clip_*/frame_entries.json')):
        meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
        for li, ents in meta['frame_entries'].items():
            m = np.ones((H0, W0), np.uint8)
            for e in ents:
                mk = mask_utils.decode(e['rle']).astype(bool)
                if mk.shape == (H0, W0): m[mk] = 2
            masks[f'image_{l2i[int(li)]}.png'] = m
    Wt = torch.zeros(N, device=dev); Wn = torch.zeros(N, device=dev); nv = 0
    for i in range(0, len(ds), a.every):
        nm = Path(ds.image_filenames[i]).name
        if nm not in masks: continue
        cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); W, H = int(cam.width.item()), int(cam.height.item())
        m = torch.from_numpy(masks[nm]).to(dev); col = torch.zeros(N, 2, device=dev, requires_grad=True)
        out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        gf, = torch.autograd.grad((out[0, ..., 0] * (m == 2).float()).sum() + (out[0, ..., 1] * (m == 1).float()).sum(), col); Wt += gf[:, 0].clamp(min=0); Wn += gf[:, 1].clamp(min=0); nv += 1
    trunk = ((Wt / (Wt + Wn).clamp(min=1e-9)) >= 0.5) & (Wt >= 1.0); np.savez_compressed(cache, P=means[trunk].cpu().numpy(), Wt=Wt[trunk].cpu().numpy(), nviews=nv, N=N, trunk_masks=len(masks))
    print(f'[trunks] {S.name}: trunk census over {nv} views ({len(masks)} trunk masks frames) in {time.time() - t0:.0f} s: {int(trunk.sum())} trunk gaussians of {N}', flush=True)
c = np.load(R / f'masklink_components_{a.tag}.npz'); cen, nvw, LOOK, UP, cam_r, mv = c['centroid'], c['nviews'], c['LOOK'], c['UP'], float(c['cam_r']), int(c['min_views'])
e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1); H2 = lambda Y: np.stack([(Y - LOOK) @ e1, (Y - LOOK) @ e2], 1) / cam_r
t = np.load(cache); Tp = H2(t['P']); Tw = t['Wt']; th = (t['P'] - LOOK) @ UP / cam_r
# trunk gaussians: keep the lower canopy only (below the median apple height) so branches in the crown do not vote
A = H2(cen); ah = (cen - LOOK) @ UP / cam_r; cand = nvw >= mv; ring = cand & (np.linalg.norm(A, axis=1) <= 1.0); hcut = np.median(ah[ring]); low = th <= hcut
Tp, Tw = Tp[low], Tw[low]; near = np.linalg.norm(Tp, axis=1) <= 1.3
gx = np.arange(-1.3, 1.31, 0.02); GX, GY = np.meshgrid(gx, gx, indexing='ij'); G = np.stack([GX.ravel(), GY.ravel()], 1); dens = np.zeros(len(G))
for s in range(0, len(G), 4000): dens[s:s + 4000] = np.exp(-0.5 * (((G[s:s + 4000, None, :] - Tp[None, near, :]) ** 2).sum(-1)) / 0.05 ** 2).sum(1)   # one vote per trunk gaussian: mass-weighting let the near, big-gaussian target trunk drown the neighbours' trunks (tree_02 map, 10:2x)
D = dens.reshape(GX.shape); from scipy.ndimage import maximum_filter
pk = (D == maximum_filter(D, size=31)) & (D >= 0.1 * D.max()); P = np.stack([GX[pk], GY[pk]], 1); Pv = D[pk]; o = np.argsort(-Pv); P, Pv = P[o], Pv[o]
keep = []
for i in range(len(P)):   # non-max suppression at 0.3 cam_r
    if all(np.linalg.norm(P[i] - P[j]) > 0.3 for j in keep): keep.append(i)
P, Pv = P[keep], Pv[keep]; dl = np.linalg.norm(P, axis=1); it = int(np.argmin(dl)); target = P[it]
same = np.linalg.norm(P - target, axis=1) <= 0.4   # posts / forks next to the target trunk count as the target
dA = np.linalg.norm(A[:, None, :] - P[None, :, :], axis=-1); nearest = np.argmin(dA, axis=1); mine = same[nearest]
apple = ring & mine; n = int(apple.sum()); err = f' vs GT {a.gt}: {100 * (n - a.gt) / a.gt:+.1f} %' if a.gt else ''
print(f"[trunks] {S.name} ({a.tag}): {int(low.sum())} trunk gaussians below the median apple height, {len(P)} trunk peaks at (x, y) cam_r: " + '; '.join(f'({p[0]:+.2f}, {p[1]:+.2f}) {v / Pv.max():.2f}' for p, v in zip(P, Pv))
      + f"; target = peak {it} at {dl[it]:.2f} cam_r from the look-point ({int(same.sum())} peaks merged into it); candidates in ring {int(ring.sum())}, assigned to the target trunk {n} -> apples **{n}**{err}; to neighbours: "
      + ', '.join(f'peak {j} {int((ring & (nearest == j)).sum())}' for j in range(len(P)) if not same[j]), flush=True)
json.dump({'tag': a.tag, 'apples': n, 'gt': a.gt or None, 'trunk_peaks': P.tolist(), 'target_peak': it, 'candidates_in_ring': int(ring.sum())}, open(R / f'registry_trunks_{a.tag}.json', 'w'))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7)); ax.scatter(Tp[near, 0], Tp[near, 1], s=2, c='0.75', label='trunk gaussians (lower canopy)')
    for j, p in enumerate(P): ax.scatter(*p, marker='*', s=220, c='k' if same[j] else 'tab:red', zorder=5)
    ax.scatter(A[ring & ~mine, 0], A[ring & ~mine, 1], s=8, c='tab:red', label=f'apples of neighbours ({int((ring & ~mine).sum())})'); ax.scatter(A[apple, 0], A[apple, 1], s=10, c='tab:green', label=f'apples of this tree ({n})')
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls='--')); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.legend(fontsize=8); ax.set_title(f'{S.name}: trunks (stars; black = target) and apple assignment, top view (cam_r units)')
    fig.savefig(R / f'registry_trunks_topdown_{a.tag}.png', dpi=110)
except Exception as ex: print('[trunks] figure skipped:', ex)
