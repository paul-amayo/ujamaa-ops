#!/usr/bin/env python3
"""lod_identity_fig.py — the identity side of the LoD picture (Paul 2026-10-13: "which gaussians are added together to give the specific fruit
32-d feature, which ones leak"): a rendered pixel's feature is the opacity-weighted sum of the words seeded on the gaussians stacked along its
ray, so for each orange (largest oranges of a training view) we rasterise WHO paints it — the census label of every gaussian, grouped as the
orange's own fruit word / its tree's word / another word / unassigned — before and after the LoD. Columns: ground truth | gaussians on the
orange coloured by word (orange = fruit word, green = tree word, grey = other / none) before | after | fruit-word share of the rendered mass
per pixel (the level rule lights fruit where that share reaches ~0.5: fruit-word norm 7.4 vs tree-word 5.2, depth midpoint 6.3) before | after.
Numbers per orange and run: mass composition over the mask, fruit-lit pixel fraction (share >= 0.5), centres on the orange by word.
nerf_new pixi env:  lod_identity_fig.py --before <config.yml>:<identity dir> --after <config.yml>:<identity dir> --sup <id maps>
                    --frames-json <frames_fruit.json> [--frame kf.png] [--n-oranges 3] [--pad 2.5] --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--before', required=True); ap.add_argument('--after', default=''); ap.add_argument('--sup', required=True)
ap.add_argument('--frames-json', default=''); ap.add_argument('--frame', default=''); ap.add_argument('--n-oranges', type=int, default=3); ap.add_argument('--pad', type=float, default=2.5)
ap.add_argument('--depth-band-m', type=float, default=0.5); ap.add_argument('--lit', type=float, default=0.5); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535
runs = []
for name, spec in (('before', a.before), ('after', a.after)):
    if spec: cfg, _, ident = spec.partition(':'); runs.append((name, Path(cfg), Path(ident)))
_, pipe, _, _ = eval_setup(runs[0][1], test_mode='test'); ds = pipe.datamanager.train_dataset; names = {Path(p).name: i for i, p in enumerate(ds.image_filenames)}
cand = [a.frame] if a.frame else json.load(open(a.frames_json))['frames']; best = None
for nm in cand:
    if nm not in names or not (Path(a.sup) / nm).exists(): continue
    ids = np.array(Image.open(Path(a.sup) / nm), np.uint16); n = int(((ids >= FRUIT) & (ids < UNLAB)).sum())
    if n and (best is None or n > best[0]): best = (n, nm)
frame = best[1]; idx = names[frame]; cam0 = ds.cameras[idx:idx + 1]; H, W = int(cam0.height.item()), int(cam0.width.item())
ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / frame), np.uint16)).resize((W, H), Image.NEAREST)); mall = (ids >= FRUIT) & (ids < UNLAB)
lab, nlab = ndimage.label(mall); sizes = np.bincount(lab.ravel())[1:]; order = np.argsort(-sizes)[:a.n_oranges]; oranges = []
for k in order:
    m = lab == k + 1; fid = int(np.bincount(ids[m]).argmax()); ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    r = max(max(y1 - y0, x1 - x0) / 2 * a.pad, 24); Y0, Y1, X0, X1 = int(max(cy - r, 0)), int(min(cy + r, H)), int(max(cx - r, 0)), int(min(cx + r, W))
    oranges.append({'id': int(k + 1), 'fruit_word': fid, 'tree_word': fid - FRUIT, 'px': int(sizes[k]), 'mask': m, 'crop': (Y0, Y1, X0, X1)})
gt = ds[idx]['image'][..., :3].cpu().numpy(); del pipe; torch.cuda.empty_cache()
print(f'[lod-id] {frame} (training view): oranges ' + ', '.join(f"#{o['id']} fruit word {o['fruit_word']}, {o['px']} px" for o in oranges), flush=True)
res = {}
for name, cfg, ident in runs:
    _, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
    ds = pipe.datamanager.train_dataset; cam = ds.cameras[idx:idx + 1].to(dev); dps = float(json.load(open(cfg.parent / 'dataparser_transforms.json'))['scale'])
    g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); scales = torch.exp(g['scales'].detach()); size_m = scales.max(dim=-1).values / dps
    labels = np.load(ident / 'features_B.bin.labels.npy'); assert labels.shape[0] == N, f'{ident}: labels {labels.shape[0]} vs checkpoint {N}'; labt = torch.from_numpy(labels.astype(np.int64)).to(dev)
    with torch.no_grad():
        viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); out = model.get_outputs(cam); depth = out['depth'].squeeze(-1).cpu().numpy()
    P = (viewmat[0] @ torch.cat([means, torch.ones(N, 1, device=dev)], 1).T).T; Z = P[:, 2]; front = Z > 1e-4
    u = (K[0, 0, 0] * P[:, 0] / Z.clamp(min=1e-4) + K[0, 0, 2]); v = (K[0, 1, 1] * P[:, 1] / Z.clamp(min=1e-4) + K[0, 1, 2]); rad_px = K[0, 0, 0] * scales.max(dim=-1).values / Z.clamp(min=1e-4)
    ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1); per = []
    uniq = np.unique(labels[labels >= 0]).tolist(); li = {l: j for j, l in enumerate(uniq)}; Lp = 64 if len(uniq) < 64 else 128
    oh = torch.zeros(N, Lp, device=dev); assigned = torch.from_numpy(labels >= 0).to(dev); oh[assigned, torch.from_numpy(np.array([li[l] for l in labels[labels >= 0]])).to(dev)] = 1.0
    with torch.no_grad():   # mass per census word at every pixel (one-hot rasterisation): names the word that co-paints each orange
        massw, _, _ = rasterization(means=means, quats=g['quats'].detach(), scales=scales, opacities=torch.sigmoid(g['opacities'].detach()).squeeze(-1), colors=oh, sh_degree=None,
                                    viewmats=viewmat, Ks=K, width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    massw = massw[0].cpu().numpy()
    for o in oranges:
        mw = massw[o['mask']].sum(0); nonfruit = [(mw[li[l]], l) for l in uniq if l < FRUIT]; top = sorted(nonfruit, reverse=True)[:3]
        if name == 'before' or 'tree_word' not in o: o['tree_word'] = int(top[0][1]) if top else -1
        o.setdefault('co_painters', {})[name] = [(int(l), float(mv / max(mw.sum(), 1e-6))) for mv, l in top]
        isf, ist = labt == o['fruit_word'], labt == o['tree_word']; oth = (labt >= 0) & ~isf & ~ist; none = labt < 0
        C = torch.stack([isf, ist, oth, none], -1).float()                                                   # who paints each pixel, by word group
        with torch.no_grad():
            feat, _, _ = rasterization(means=means, quats=g['quats'].detach(), scales=scales, opacities=torch.sigmoid(g['opacities'].detach()).squeeze(-1), colors=C, sh_degree=None,
                                       viewmats=viewmat, Ks=K, width=W, height=H, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
        comp = feat[0].cpu().numpy(); tot = comp.sum(-1, keepdims=True).clip(1e-6); share = comp / tot; m = o['mask']
        dz = float(np.median(depth[m])); on = front & torch.from_numpy(m).to(dev)[vi, ui] & ((Z - dz).abs() < a.depth_band_m * dps)
        grp = torch.where(isf, 0, torch.where(ist, 1, torch.where(oth, 2, 3)))
        per.append({'mass_share_over_mask': {k: float(share[m][:, j].mean()) for j, k in enumerate(('fruit_word', 'tree_word', 'other_word', 'unassigned'))},
                    'fruit_lit_frac': float((share[m][:, 0] >= a.lit).mean()), 'centres_on_orange': {k: int(((grp == j) & on).sum()) for j, k in enumerate(('fruit_word', 'tree_word', 'other_word', 'unassigned'))},
                    'share_map': share[..., 0][o['crop'][0]:o['crop'][1], o['crop'][2]:o['crop'][3]], 'u': (u[on] - o['crop'][2]).cpu().numpy(), 'v': (v[on] - o['crop'][0]).cpu().numpy(), 'r': rad_px[on].cpu().numpy(), 'g': grp[on].cpu().numpy()})
    res[name] = per
    print(f'[lod-id] {name} ({cfg.parent.parent.name}, {N} gaussians, {int((labels >= FRUIT).sum())} fruit-word gaussians): ' + '; '.join(f"orange #{o['id']} co-painted by words " + ', '.join(f'{l} ({100 * sh:.0f}%)' for l, sh in o['co_painters'][name]) for o in oranges) + ' | ' + '; '.join(
        f"orange #{o['id']}: mass fruit-word {p['mass_share_over_mask']['fruit_word']:.2f} / tree-word {p['mass_share_over_mask']['tree_word']:.2f} / other {p['mass_share_over_mask']['other_word']:.2f} / none {p['mass_share_over_mask']['unassigned']:.2f}, fruit lit {100 * p['fruit_lit_frac']:.0f}% of px, centres F/T/O/none {p['centres_on_orange']['fruit_word']}/{p['centres_on_orange']['tree_word']}/{p['centres_on_orange']['other_word']}/{p['centres_on_orange']['unassigned']}" for o, p in zip(oranges, per)), flush=True)
    del pipe, model; torch.cuda.empty_cache()
COL = np.array(['#e8730c', '#2e8b3d', '#888888', '#bbbbbb']); ncol = 1 + 2 * len(runs); fig, axs = plt.subplots(len(oranges), ncol, figsize=(3.9 * ncol, 3.9 * len(oranges)), squeeze=False)
for r, o in enumerate(oranges):
    Y0, Y1, X0, X1 = o['crop']; cs = (slice(Y0, Y1), slice(X0, X1)); mc = o['mask'][cs]
    axs[r, 0].imshow(gt[cs], interpolation='nearest'); axs[r, 0].contour(mc, levels=[0.5], colors='white', linewidths=1.0); axs[r, 0].set_title(f'orange #{o["id"]}: ground truth\n(fruit word {o["fruit_word"]}, co-painted by word {o["tree_word"]})', fontsize=10)
    for j, (name, _, _) in enumerate(runs):
        p = res[name][r]; c = p['centres_on_orange']; ax = axs[r, 1 + j]; ax.imshow(gt[cs], interpolation='nearest', alpha=0.45); ax.contour(mc, levels=[0.5], colors='white', linewidths=1.0)
        o_ = np.argsort(-p['r']); ax.scatter(p['u'][o_], p['v'][o_], s=np.clip(np.pi * p['r'][o_] ** 2, 6, 6000), facecolors=COL[p['g'][o_]], edgecolors='black', linewidths=0.5, alpha=0.45); ax.scatter(p['u'], p['v'], s=6, c='black')
        ax.set_xlim(0, X1 - X0); ax.set_ylim(Y1 - Y0, 0); ax.set_title(f'{name}: gaussians on the orange, by word\nfruit {c["fruit_word"]} · tree {c["tree_word"]} · other {c["other_word"]} · none {c["unassigned"]}', fontsize=10)
        s = p['mass_share_over_mask']; ax = axs[r, 1 + len(runs) + j]; im = ax.imshow(p['share_map'], vmin=0, vmax=1, cmap='Oranges', interpolation='nearest'); ax.contour(mc, levels=[0.5], colors='black', linewidths=1.0)
        if (p['share_map'] >= a.lit).any(): ax.contour(p['share_map'] >= a.lit, levels=[0.5], colors='#1f77b4', linewidths=1.2)
        ax.set_title(f'{name}: fruit-word share of the mass\nfruit {s["fruit_word"]:.2f} · tree {s["tree_word"]:.2f} · other {s["other_word"]:.2f} · none {s["unassigned"]:.2f} · lit {100 * p["fruit_lit_frac"]:.0f}%', fontsize=9.5)
    for c_ in range(ncol): axs[r, c_].set_xticks([]); axs[r, c_].set_yticks([])
fig.suptitle(f'{frame} (training view): who paints each orange — orange = its fruit word, green = the word that co-paints it (its tree), grey = other / unassigned; blue contour = fruit-word share >= {a.lit} (the level rule lights fruit about there)', fontsize=11); fig.tight_layout(); fig.savefig(a.out, dpi=160)
json.dump({'frame': frame, 'oranges': [{k: v for k, v in o.items() if k not in ('mask',)} for o in oranges], 'runs': {n: [{k: v for k, v in p.items() if k not in ('share_map', 'u', 'v', 'r', 'g')} for p in per] for n, per in res.items()}}, open(Path(a.out).with_suffix('.json'), 'w'), indent=1)
print(f'[lod-id] -> {a.out}', flush=True)
