#!/usr/bin/env python3
"""fruit_votes_fig.py — where one gaussian gets its census votes (2026-10-14): for a gaussian (e.g. the main carrier of an orange), its blend weight
in every census view split by the supervision label under it (its fruit word / its tree word / other trees / unlabelled), and the views that give
it the most weight drawn as crops: the photo with the supervision tinted (fruit = orange, its tree = green, other trees = blue, unlabelled = none),
the gaussian as a circle at its projected size, and the vote it casts there. Explains why a seeding rule gives or withholds its word.
nerf_new pixi env:  fruit_votes_fig.py --config <config.yml> --sup <id maps> --names-json <split_names.json> --gaussian <idx> --fruit-word 10001
                    --tree-word 5 [--n-views 8] --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--names-json', required=True)
ap.add_argument('--gaussian', type=int, required=True); ap.add_argument('--fruit-word', type=int, required=True); ap.add_argument('--tree-word', type=int, required=True)
ap.add_argument('--n-views', type=int, default=8); ap.add_argument('--out', required=True); a = ap.parse_args()
UNLAB = 65535
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); dev = model.device; tr = pipe.datamanager.train_dataset
idx = {Path(p).name: i for i, p in enumerate(tr.image_filenames)}; g = model.gauss_params; N = g['means'].shape[0]
means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1); G = a.gaussian
dps = float(json.load(open(Path(a.config).parent / 'dataparser_transforms.json'))['scale'])
names = json.load(open(a.names_json))['train']; rec = []
for nm in names:
    if nm not in idx or not (Path(a.sup) / nm).exists(): continue
    cam = tr.cameras[idx[nm]:idx[nm] + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev).clone(); Wf, Hf = int(cam.width.item()), int(cam.height.item())
    W, H = Wf // 2, Hf // 2; Kh = K.clone(); Kh[:, :2] *= 0.5                                       # census resolution
    ids = np.array(Image.fromarray(np.array(Image.open(Path(a.sup) / nm), np.uint16)).resize((W, H), Image.NEAREST)); lab = torch.from_numpy(ids.astype(np.int64)).to(dev)
    oh = torch.stack([lab == a.fruit_word, lab == a.tree_word, (lab != a.fruit_word) & (lab != a.tree_word) & (lab != UNLAB), lab == UNLAB], -1).float()
    col = torch.zeros(N, 4, device=dev, requires_grad=True)
    out, _, meta = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=Kh, width=W, height=H, sh_degree=None, packed=False,
                                 near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gc, = torch.autograd.grad((out[0] * oh).sum(), col); m = gc[G].detach().cpu().numpy()
    if m.sum() <= 1e-4: continue
    con = meta['conics'][0][G]; det = float((con[0] * con[2] - con[1] ** 2).clamp(min=1e-12)); foot = float(meta['opacities'][0][G]) * 2 * np.pi / np.sqrt(det)
    pc = vm[0] @ torch.cat([means[G], torch.ones(1, device=dev)]); u = float(K[0, 0, 0] * pc[0] / pc[2] + K[0, 0, 2]); v = float(K[0, 1, 1] * pc[1] / pc[2] + K[0, 1, 2]); r = float(K[0, 0, 0] * scales[G].max() / pc[2])
    rec.append({'frame': nm, 'm': m, 'vis': min(m.sum() / max(foot, 1e-9), 1.0), 'u': u, 'v': v, 'r': r, 'depth_m': float(pc[2]) / dps, 'W': Wf, 'H': Hf})
M = np.array([r_['m'] for r_ in rec]); print(f'[votes] g{G}: {len(rec)} census views give it weight; totals fruit {M[:, 0].sum():.1f} / tree {a.tree_word} {M[:, 1].sum():.1f} / other trees {M[:, 2].sum():.1f} / unlabelled {M[:, 3].sum():.1f}', flush=True)
top = sorted(rec, key=lambda r_: -r_['m'].sum())[:a.n_views]; top = sorted(top, key=lambda r_: r_['frame'])
ncol = min(4, len(top)); nrow = int(np.ceil(len(top) / ncol)); fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.5 * nrow), squeeze=False)
TINT = {'fruit': np.array([232, 115, 12]), 'tree': np.array([46, 139, 61]), 'other': np.array([31, 119, 180])}
for ax, r_ in zip(axs.ravel(), top):
    img = np.array(Image.open(Path(tr.image_filenames[idx[r_['frame']]])).convert('RGB')).astype(float); ids = np.array(Image.open(Path(a.sup) / r_['frame']), np.uint16)
    if ids.shape != img.shape[:2]: ids = np.array(Image.fromarray(ids).resize((img.shape[1], img.shape[0]), Image.NEAREST))
    for key, msk in (('fruit', ids == a.fruit_word), ('tree', ids == a.tree_word), ('other', (ids != a.fruit_word) & (ids != a.tree_word) & (ids != UNLAB))): img[msk] = 0.55 * img[msk] + 0.45 * TINT[key]
    R = max(6 * r_['r'], 60); cy, cx = r_['v'], r_['u']; Y0, Y1, X0, X1 = int(max(cy - R, 0)), int(min(cy + R, img.shape[0])), int(max(cx - R, 0)), int(min(cx + R, img.shape[1]))
    ax.imshow(img[Y0:Y1, X0:X1].astype(np.uint8), interpolation='nearest'); ax.add_patch(plt.Circle((cx - X0, cy - Y0), max(r_['r'], 2), fill=False, color='white', lw=2)); ax.plot(cx - X0, cy - Y0, 'w+', ms=10)
    m = r_['m']; vote = ['fruit', f'tree {a.tree_word}', 'other tree', 'unlabelled'][int(np.argmax(m))]
    ax.set_title(f"{r_['frame']} · {r_['depth_m']:.1f} m · visibility {r_['vis']:.2f}\nfruit {m[0]:.1f} · tree {m[1]:.1f} · other {m[2]:.1f} · unlabelled {m[3]:.1f} → {vote}", fontsize=9.5)
    if X0 == 0 or X1 == img.shape[1]: ax.set_xlabel('at the image edge', fontsize=9)
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
for ax in axs.ravel()[len(top):]: ax.axis('off')
fig.suptitle(f'Where gaussian g{G} gets its census votes (the {len(top)} views with the most weight of {len(rec)}): tint = supervision under it (orange = fruit {a.fruit_word}, green = tree {a.tree_word}, blue = other trees, none = unlabelled); circle = the gaussian', fontsize=10.5)
fig.tight_layout(); fig.savefig(a.out, dpi=140); print(f'[votes] -> {a.out}', flush=True)
