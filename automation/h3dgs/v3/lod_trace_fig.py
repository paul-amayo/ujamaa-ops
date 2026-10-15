#!/usr/bin/env python3
"""lod_trace_fig.py — trace the gaussians that paint ONE orange across the neighbouring training frames (Paul 2026-10-13: "we observe this
over multiple frames, why is the fruit coverage so large? show me a trace of these gaussian centres against masks"). The painters = the
gaussians carrying >= --min-share of the orange's rendered mass in the reference frame (grad of the masked render wrt dummy colours), each
with its census word, size and census fruit share (W.npz). Panels: the reference frame and up to --n-frames neighbours, cropped around the
projected painters, with that frame's fruit masks as white contours and every painter as a circle at its world size (orange = fruit word,
green = tree word, grey = other / none; filled when its centre lies inside a fruit mask of that frame). The table: per painter, in how many of
the frames its centre falls inside a fruit mask -> <out>.json.
nerf_new pixi env:  lod_trace_fig.py --config <config.yml> --identity <dir with features_B.bin.labels.npy + W.npz> --sup <id maps>
                    --frame kf.png --orange-rank 3 [--n-frames 6] [--span 12] [--min-share 0.02] --out <png>"""
import argparse, json, re
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--identity', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--frame', required=True)
ap.add_argument('--orange-rank', type=int, default=3, help='1 = largest orange of the frame'); ap.add_argument('--n-frames', type=int, default=6); ap.add_argument('--span', type=int, default=24)
ap.add_argument('--min-share', type=float, default=0.02); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device; ds = pipe.datamanager.train_dataset
names = [Path(p).name for p in ds.image_filenames]; order = sorted(range(len(names)), key=lambda i: int(re.sub(r'\D', '', names[i]) or 0)); pos = {i: k for k, i in enumerate(order)}
idx = names.index(a.frame); cam = ds.cameras[idx:idx + 1].to(dev); H, W = int(cam.height.item()), int(cam.width.item()); dps = float(json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))['scale'])
def idmap(nm): return np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / nm), np.uint16)).resize((W, H), Image.NEAREST))
ids = idmap(a.frame); mall = (ids >= FRUIT) & (ids < UNLAB); lab, nlab = ndimage.label(mall); sizes = np.bincount(lab.ravel())[1:]; k = np.argsort(-sizes)[a.orange_rank - 1]; m = lab == k + 1; fid = int(np.bincount(ids[m]).argmax())
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1); size_m = scales.max(dim=-1).values / dps
labels = np.load(Path(a.identity) / 'features_B.bin.labels.npy'); assert labels.shape[0] == N
Z = np.load(Path(a.identity) / 'W.npz'); Wm, wl = Z['W'], Z['labels']; fr = (wl >= FRUIT) & (wl < UNLAB); fshare = np.where(Wm.sum(0) > 0, Wm[fr].sum(0) / np.maximum(Wm.sum(0), 1e-6), 0.0)
# painters: per-gaussian alpha mass on the orange's pixels in the reference frame
viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); dummy = torch.ones(N, 3, device=dev, requires_grad=True)
feat, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=dummy, sh_degree=None, viewmats=viewmat, Ks=K, width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
S = (feat[0, ..., 0] * torch.from_numpy(m).to(dev).float()).sum(); mass = torch.autograd.grad(S, dummy)[0][:, 0].clamp(min=0).detach(); tot = float(mass.sum())
sel = torch.where(mass >= a.min_share * tot)[0]; sel = sel[torch.argsort(-mass[sel])]; P = sel.cpu().numpy()
word = lambda l: 'fruit' if l == fid else ('tree' if 0 <= l < FRUIT else ('other-fruit' if l >= FRUIT else 'none'))
tree_word = int(np.bincount(labels[P][(labels[P] >= 0) & (labels[P] < FRUIT)]).argmax()) if ((labels[P] >= 0) & (labels[P] < FRUIT)).any() else -1
painters = [{'g': int(i), 'word': word(int(labels[i])), 'label': int(labels[i]), 'size_cm': float(size_m[i] * 100), 'mass_share': float(mass[i] / tot), 'census_fruit_share': float(fshare[i]), 'inside': []} for i in P]
print(f'[trace] {a.frame} orange rank {a.orange_rank} (fruit word {fid}, {int(m.sum())} px, co-painting tree word {tree_word}): {len(P)} painters carry {100 * float(mass[sel].sum() / tot):.0f}% of its rendered mass; '
      + ', '.join(f"g{p['g']} {p['word']} {p['size_cm']:.1f} cm {100 * p['mass_share']:.0f}% (census fruit share {p['census_fruit_share']:.2f})" for p in painters[:12]), flush=True)
# frames: the reference + neighbours in capture order that have fruit masks
cands = [order[j] for j in range(max(pos[idx] - a.span, 0), min(pos[idx] + a.span + 1, len(order))) if order[j] != idx]
cands = [i for i in cands if (Path(a.sup) / names[i]).exists()]; cands.sort(key=lambda i: abs(pos[i] - pos[idx])); frames = [idx]
def project(i):
    c = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(c.camera_to_worlds); Ki = c.get_intrinsics_matrices().to(dev)
    Pc = (vm[0] @ torch.cat([means[sel], torch.ones(len(sel), 1, device=dev)], 1).T).T; z = Pc[:, 2].clamp(min=1e-4)
    u = (Ki[0, 0, 0] * Pc[:, 0] / z + Ki[0, 0, 2]).cpu().numpy(); v = (Ki[0, 1, 1] * Pc[:, 1] / z + Ki[0, 1, 2]).cpu().numpy(); r = (Ki[0, 0, 0] * scales[sel].max(dim=-1).values / z).cpu().numpy()
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (Pc[:, 2] > 0).cpu().numpy(); return u, v, r, inb
for i in cands:   # neighbours in capture order that have fruit masks AND see at least a third of the painters
    if len(frames) >= a.n_frames: break
    mm = idmap(names[i])
    if ((mm >= FRUIT) & (mm < UNLAB)).any() and project(i)[3].sum() >= max(1, len(sel) // 3): frames.append(i)
panels = []; isfruit = np.array([p['word'] == 'fruit' for p in painters]); wmass = mass[sel].cpu().numpy()
with torch.no_grad():
    for i in frames:
        u, v, r, inb = project(i)
        mi = idmap(names[i]); fm = (mi >= FRUIT) & (mi < UNLAB)
        inside = np.zeros(len(sel), bool); inside[inb] = fm[v[inb].astype(int), u[inb].astype(int)]
        dist = ndimage.distance_transform_edt(~fm) if fm.any() else np.full(fm.shape, 1e6); near = np.zeros(len(sel)); near[inb] = dist[v[inb].astype(int), u[inb].astype(int)]
        for p, ins, nb, d in zip(painters, inside, inb, near): p['inside'].append({'frame': names[i], 'in_image': bool(nb), 'inside_fruit_mask': bool(ins), 'px_to_nearest_fruit': float(d) if nb else None})
        gt = ds[i]['image'][..., :3].cpu().numpy(); wsel = inb & isfruit if (inb & isfruit).any() else inb; ww = wmass[wsel] / wmass[wsel].sum()   # crop around the mass-weighted centre of the fruit-word painters
        cu, cv = (ww * u[wsel]).sum(), (ww * v[wsel]).sum(); rad = max(4 * np.median(r[wsel]), 40); Y0, Y1, X0, X1 = int(max(cv - rad, 0)), int(min(cv + rad, H)), int(max(cu - rad, 0)), int(min(cu + rad, W))
        if i == idx: m_crop = m[Y0:Y1, X0:X1]
        panels.append((names[i], gt[Y0:Y1, X0:X1], fm[Y0:Y1, X0:X1], u - X0, v - Y0, r, inside, inb, int(inside.sum()), int(inb.sum())))
        print(f'[trace] {names[i]}: {int(inside.sum())} of {int(inb.sum())} painters inside a fruit mask ({int(fm.sum())} fruit px in the frame)', flush=True)
COL = {'fruit': '#e8730c', 'tree': '#2e8b3d', 'other-fruit': '#b05cff', 'none': '#888888'}; n = len(panels); fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 4.6), squeeze=False)
for ax, (nm, gt, fm, u, v, r, inside, inb, ni, nb) in zip(axs[0], panels):
    ax.imshow(gt, interpolation='nearest', alpha=0.6)
    if fm.any(): ax.contour(fm, levels=[0.5], colors='white', linewidths=1.2)
    if nm == a.frame: ax.contour(m[panels[0][1].shape[0] and slice(None) or slice(None)] if False else m_crop, levels=[0.5], colors='yellow', linewidths=1.6)
    cols = [COL[p['word']] for p in painters]; o_ = np.argsort(-r)
    ax.scatter(u[o_], v[o_], s=np.clip(np.pi * r[o_] ** 2, 8, 5000), facecolors=[cols[j] if inside[j] else 'none' for j in o_], edgecolors=[cols[j] for j in o_], linewidths=1.2, alpha=0.6)
    ax.scatter(u, v, s=8, c='black'); ax.set_xlim(0, gt.shape[1]); ax.set_ylim(gt.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{nm}{" (reference)" if nm == a.frame else ""}\n{ni} of {nb} painters inside a fruit mask', fontsize=10)
nf, nt = sum(p['word'] == 'fruit' for p in painters), sum(p['word'] == 'tree' for p in painters)
fig.suptitle(f'{len(painters)} gaussians paint >= {100 * a.min_share:.0f}% each of orange #{a.orange_rank} in {a.frame} ({nf} fruit-word, {nt} tree-word, {len(painters) - nf - nt} other); circles at world size, filled = centre inside a fruit mask of that frame; yellow = this orange', fontsize=11)
fig.tight_layout(); fig.savefig(a.out, dpi=150); json.dump({'frame': a.frame, 'fruit_word': fid, 'tree_word': tree_word, 'painters': painters}, open(Path(a.out).with_suffix('.json'), 'w'), indent=1)
for p in painters: p['frames_inside'] = sum(x['inside_fruit_mask'] for x in p['inside']); p['frames_seen'] = sum(x['in_image'] for x in p['inside'])
print('[trace] per painter (word, size, mass share of the orange, census fruit share, frames inside a fruit mask / frames in image): ' + '; '.join(f"g{p['g']} {p['word']} {p['size_cm']:.1f}cm {100 * p['mass_share']:.0f}% cs{p['census_fruit_share']:.2f} {p['frames_inside']}/{p['frames_seen']}" for p in painters), flush=True)
print(f'[trace] -> {a.out}', flush=True)
