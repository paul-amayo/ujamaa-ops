#!/usr/bin/env python3
"""sites_contact_sheet.py (2026-11-11, Paul: "Klapmuts is still not good, Lindendhof is a regression") — every typical Fig. 3 candidate of a
survey on one sheet, drawn as the figure draws it, for Paul to choose by eye. Candidates: held-out TREE heads >= 3 % of the frame whose
encoder-path argmax IoU (protocol_tangent_vf) lies within --tol of the survey median, with the head's supervision not touching the frame
border (the plant wholly in view). Renders come from sites_pick_tangent.py's void-free maps (rendered on demand if missing). Each tile is
cropped to the scored pixels as in sites_strip --layout zoom, labelled with its index, cell, frame, plant id and IoU.
nerf_new pixi env:  sites_contact_sheet.py --survey <sv> --out <png> [--tol 0.03] [--max 24]"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--out', required=True); ap.add_argument('--tol', type=float, default=0.03)
ap.add_argument('--max', type=int, default=24); ap.add_argument('--tmp', default='/mnt/shared/voidfree_tmp/sites'); ap.add_argument('--render', default='/home/paperspace/logs/sec4/bin/splat_feature_render.py'); a = ap.parse_args()
dev = 'cuda'; inv = [json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')]; SUP = {c['identity']: Path(c['sup']) for c in inv}
heads = []
for c in [c for c in inv if c['survey'] == a.survey]:
    p = Path(c['identity']) / 'protocol_tangent_vf.jsonl'
    if p.exists():
        for l in open(p):
            h = json.loads(l)
            if h.get('split') == 'eval' and h.get('kind') == 'TREE' and h.get('gt_frac', 0) >= 0.03: h['identity'] = c['identity']; heads.append(h)
med = float(np.median([h['argmax']['iou'] for h in heads])); cands = sorted([h for h in heads if abs(h['argmax']['iou'] - med) <= a.tol], key=lambda h: (h['identity'], h['frame']))
TP, FP, FN = np.array([0.1, 0.8, 0.2]), np.array([1.0, 0.55, 0.0]), np.array([0.9, 0.1, 0.1]); tiles = []; emb = {}
for h in cands:
    N = Path(h['identity']); out = Path(a.tmp) / str(N).replace('/home/paperspace/data/', '').replace('/', '__'); f_npz = out / h['frame'].replace('.png', '.npz')
    if not f_npz.exists():
        cfg = json.load(open(N / 'W.json'))['config']; sp = N.parent / 'sparse' / '0'
        if not (sp / 'images.bin').exists(): sp = Path(cfg).parents[3] / 'sparse' / '0'
        subprocess.run(['/home/paperspace/.pixi/bin/pixi', 'run', 'python', a.render, '--config', cfg, '--features', str(N / 'features_B.bin'), '--frames', h['frame'], '--out-dir', str(out), '--colmap-sparse', str(sp), '--void-free', '0.5'], cwd='/home/paperspace/code/nerf_new', capture_output=True)
    z = np.load(f_npz); rgb = z['rgb']; H, W = rgb.shape[:2]
    t = np.array(Image.fromarray(np.array(Image.open(SUP[str(N)] / h['frame']))).resize((W, H), Image.NEAREST)) == int(h['id'])
    ys, xs = np.where(t)
    if ys.min() <= 1 or xs.min() <= 1 or ys.max() >= H - 2 or xs.max() >= W - 2: continue   # plant cut by the frame
    if str(N) not in emb:
        hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev); bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words'])
        wt = {int(k): w for k, w in json.loads(str(bank['meta']))['word_table'].items()}; ids = [k for k in sorted(wt) if k < 10000 and wt[k] in words]
        with torch.no_grad(): g = hyper.encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev)[[words.index(wt[k]) for k in ids]], project=False)
        emb[str(N)] = (ids, g / g.norm(dim=-1, keepdim=True))
    ids, g = emb[str(N)]; f = torch.tensor(z['features'], dtype=torch.float32, device=dev).reshape(-1, z['features'].shape[-1])
    with torch.no_grad(): am = np.array(ids)[((f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)) @ g.T).argmax(1).cpu().numpy()].reshape(H, W)
    pred = (am == int(h['id'])) & (z['features'].reshape(H, W, -1).__abs__().sum(-1) > 0) & (np.linalg.norm(z['features'], axis=-1) >= 0.5)
    u = pred | t; iou = (pred & t).sum() / max(u.sum(), 1); img = (rgb.astype(float) / 255) ** 0.85
    for m, col in ((pred & t, TP), (pred & ~t, FP), (~pred & t, FN)): img[m] = 0.25 * img[m] + 0.75 * col
    yy, xx = np.where(u); m_ = int(0.12 * max(yy.ptp(), xx.ptp())) + 8; y0, y1, x0, x1 = max(yy.min() - m_, 0), min(yy.max() + m_, H), max(xx.min() - m_, 0), min(xx.max() + m_, W)
    tiles.append((img[y0:y1, x0:x1], f"{len(tiles) + 1}: {N.parent.name} {h['frame'].replace('.png', '')} plant {h['id']}  IoU {iou:.2f}"))
    if len(tiles) >= a.max: break
nc = 4; nr = max(1, -(-len(tiles) // nc)); fig, axes = plt.subplots(nr, nc, figsize=(16, 3.2 * nr)); axes = np.atleast_1d(axes).ravel()
for ax in axes: ax.axis('off')
for ax, (im, lab) in zip(axes, tiles): ax.imshow(np.clip(im, 0, 1)); ax.set_title(lab, fontsize=9, loc='left')
fig.suptitle(f'{a.survey}: {len(cands)} held-out heads within {a.tol} of the median IoU {med:.3f} ({len(heads)} heads >= 3 %); {len(tiles)} shown with the plant wholly in frame', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(a.out, dpi=70); print(f'[sheet] {a.survey}: median {med:.3f}, {len(cands)} candidates, {len(tiles)} tiles -> {a.out}')
