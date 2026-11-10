#!/usr/bin/env python3
"""fruitnerf_render_panel.py <out.npz> [--numbers <paper numbers/fruit_registry.json>] (2026-11-10) — the FruitNeRF panel of the HiGH cross-site figure
(sites_strip.py --fruitnerf): a render of our splat of one FruitNeRF real tree with every apple the registry of record counted lit in one colour.
FruitNeRF's real trees have a hand count per tree only (no per-apple positions), so there is no TP / FP / miss here: the panel prints ours / hand.
  tree     chosen from the numbers json alone, before any image: the tree whose relative count error |ours_record - hand| / hand is the median of the three.
  counted  the registry of record (lab notebook 10-15 "Transitive-safe evidence merge"): masklink_components_plain30k_f095_evt_dom_ng.npz, apples =
           the dump's apple flag (min views, windfall layer out) inside the camera ring (hd <= cam_r) and between the row cuts stored by
           masklink_row_cut.py in registry_masklink_plain30k_f095_evt_dom_ng.json; asserted equal to ours_record.
  camera   the tree's own training camera that sees the most counted apples: centroid in frame and on the camera's side of the tree (nearer the
           camera than the ring's look-point) — registry geometry only, no image read.
  lighting census trick (splat_census.py / outline_probe.py): one colour channel, every Gaussian of a counted apple (masklink_votes fid[comp == c])
           at 1, the rest 0; the render of that channel is each pixel's alpha-blend weight from the counted apples.
nerf_new pixi env (GPU, one frame):  fruitnerf_render_panel.py /mnt/shared2/d36sites/fruitnerf_render.npz"""
import argparse, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--numbers', default='/home/paperspace/code/aru_sil_core/src/docs/paper/numbers/fruit_registry.json')
ap.add_argument('--tag', default='plain30k_f095_evt_dom_ng'); ap.add_argument('--votes', default='plain30k'); a = ap.parse_args()
J = json.load(open(a.numbers))['fruitnerf']; hand, ours = J['hand_counts'], J['ours_record']; trees = sorted(hand); err = {t: abs(ours[t] - hand[t]) / hand[t] for t in trees}
T = sorted(trees, key=lambda t: err[t])[len(trees) // 2]
print('[fruitnerf-render] rule: the tree whose |ours - hand| / hand is the median of the three; ' + ', '.join(f'{t} {ours[t]} / {hand[t]} = {100 * err[t]:.1f} %' for t in trees) + f' -> {T}')
S = Path(f'/home/paperspace/data/public/fruitnerf/high/{T}'); R = S / 'fruit_registry'; RUN = S / 'outputs/sf_30000_bilateral/splatfacto/run'
d = np.load(R / f'masklink_components_{a.tag}.npz'); cen, LOOK, UP, cam_r = d['centroid'], d['LOOK'], d['UP'], float(d['cam_r']); rc = json.loads((R / f'registry_masklink_{a.tag}.json').read_text())['rowcut']
e1 = np.cross(UP, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(UP, e1); X = np.stack([(cen - LOOK) @ e1, (cen - LOOK) @ e2], 1) / cam_r
ring = d['apple'] & (np.linalg.norm(X, axis=1) <= 1.0); w_, v_ = np.linalg.eigh(np.cov(X[ring].T)); t = X @ v_[:, 1]   # masklink_row_cut.py's row axis (eigvec of the ring's candidates)
sets = [np.where(ring & (s_ * t >= rc['left']) & (s_ * t <= rc['right']))[0] for s_ in (1, -1)]   # an eigenvector's sign is LAPACK's choice: take the sign that reproduces the record
ok = [s_ for s_ in sets if len(s_) == ours[T]]; assert ok and all(np.array_equal(ok[0], s_) for s_ in ok), f'row cut does not reproduce ours_record {ours[T]}: {[len(s_) for s_ in sets]}'
counted = ok[0]
print(f"[fruitnerf-render] {T}: registry of record {a.tag}: {int(d['apple'].sum())} apples, {int(ring.sum())} in the ring, {len(counted)} between the row cuts {rc['left']:+.2f} / {rc['right']:+.2f} (= ours_record {ours[T]}; hand {hand[T]})")
import torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
_, pipe, _, step = eval_setup(RUN / 'config.yml', test_mode='inference'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
cams = ds.cameras.to(dev); VM = get_viewmat(cams.camera_to_worlds).cpu().numpy(); KK = cams.get_intrinsics_matrices().cpu().numpy(); Wc = cams.width.cpu().numpy().ravel(); Hc = cams.height.cpu().numpy().ravel()
Cc = cen[counted]; score = []
for i in range(len(VM)):
    pc = Cc @ VM[i, :3, :3].T + VM[i, :3, 3]; z = pc[:, 2]; u = KK[i, 0, 0] * pc[:, 0] / np.maximum(z, 1e-6) + KK[i, 0, 2]; v = KK[i, 1, 1] * pc[:, 1] / np.maximum(z, 1e-6) + KK[i, 1, 2]
    camC = -VM[i, :3, :3].T @ VM[i, :3, 3]; near = np.linalg.norm(Cc - camC, axis=1) <= np.linalg.norm(LOOK - camC)
    score.append(int(((z > 0) & (u >= 0) & (u < Wc[i]) & (v >= 0) & (v < Hc[i]) & near).sum()))
ci = int(np.argmax(score)); name = Path(ds.image_filenames[ci]).name; cam = cams[ci:ci + 1]
print(f'[fruitnerf-render] camera {name}: {score[ci]} of {len(counted)} counted apples in frame on its side of the tree (max over {len(VM)} training cameras); {int(Wc[ci])}x{int(Hc[ci])}, step {step}')
with torch.no_grad(): o = model.get_outputs(cam); rgb = o['rgb'].squeeze(0).clamp(0, 1).cpu().numpy()
vo = np.load(R / f'masklink_votes_{a.votes}.npz'); fid = vo['fid']; comp = d['comp']; assert len(comp) == len(fid), 'components and votes differ in fruit-gaussian count'
g = model.gauss_params; N = g['means'].shape[0]; gids = fid[np.isin(comp, counted)]; col = torch.zeros(N, 1, device=dev); col[torch.from_numpy(gids).to(dev), 0] = 1.0
with torch.no_grad():
    out, _, _ = rasterization(means=g['means'], quats=g['quats'] / g['quats'].norm(dim=-1, keepdim=True), scales=torch.exp(g['scales']), opacities=torch.sigmoid(g['opacities']).squeeze(-1), colors=col,
                              viewmats=get_viewmat(cam.camera_to_worlds), Ks=cam.get_intrinsics_matrices().to(dev), width=int(Wc[ci]), height=int(Hc[ci]), sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10,
                              render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
w = out[0, ..., 0].cpu().numpy(); print(f'[fruitnerf-render] lit Gaussians {len(gids)} of {N}; pixels with blend weight >= 0.5: {int((w >= 0.5).sum())}')
np.savez_compressed(a.out, rgb=(rgb * 255).round().astype(np.uint8), w_count=w.astype(np.float16), tree=T, ours=ours[T], hand=hand[T], rel_err=err[T], image=name, in_view=score[ci], numbers=a.numbers, tag=a.tag, step=step, gauss=len(gids))
print('[fruitnerf-render] wrote', a.out)
