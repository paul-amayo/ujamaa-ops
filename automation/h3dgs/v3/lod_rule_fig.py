#!/usr/bin/env python3
"""lod_rule_fig.py — what the fruit query LIGHTS, per splat and seeding rule (Paul 2026-10-14: "show some qualitative images like before"):
rows = the largest oranges of one training view, columns = ground truth | one panel per arm (a splat geometry + a seeding rule), each the arm's
own render with the pixels the fruit query lights under the level rule: orange = lit on a fruit pixel of that word, red = lit off it (leak),
white outline = the labelled fruit. Panel title: recall on this orange and leak pixels in the crop; column title: the frame's IoU for the word.
Needs per-pixel predictions from argmax_eval.py --dump-pred <arm>/pred_qual (fruit_level) and the arm's maps (rgb backdrop).
system python3:  lod_rule_fig.py --arms "stock, 10 %=<identity>:<pred dir>" ... --sup <id maps> --frame kf.png --images <ws>/images [--n-oranges 3] --out <png>"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--arms', nargs='+', required=True); ap.add_argument('--sup', required=True); ap.add_argument('--frame', required=True); ap.add_argument('--images', required=True)
ap.add_argument('--n-oranges', type=int, default=3); ap.add_argument('--pad', type=float, default=2.5); ap.add_argument('--rule', default='fruit_level'); ap.add_argument('--out', required=True); a = ap.parse_args()
FRUIT, UNLAB = 10000, 65535; stem = a.frame[:-4]
arms = []
for spec in a.arms:
    lab, rest = spec.rsplit('=', 1); ident, pdir = rest.split(':'); arms.append((lab, Path(ident), Path(pdir)))
z0 = np.load(next(p_ for p_ in (arms[0][1] / 'maps_fruit' / f'{stem}.npz', arms[0][1] / 'maps_s01' / f'{stem}.npz') if p_.exists())); h, w = z0['rgb'].shape[:2]
ids_full = np.array(Image.open(Path(a.sup) / a.frame), np.uint16); fr_full = (ids_full >= FRUIT) & (ids_full < UNLAB)
lab, n = ndimage.label(fr_full); sizes = np.bincount(lab.ravel())[1:]; order = np.argsort(-sizes)[:a.n_oranges]   # same numbering as lod_coverage_fig / lod_identity_fig (labelled at the map's native res)
ids = np.array(Image.fromarray(ids_full).resize((w, h), Image.NEAREST)); photo = np.array(Image.open(Path(a.images) / a.frame).convert('RGB').resize((w, h), Image.BILINEAR))
oranges = []
for k in order:
    m_full = lab == k + 1; fid = int(np.bincount(ids_full[m_full]).argmax()); m = np.array(Image.fromarray(m_full.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    ys, xs = np.where(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2; r = max(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * a.pad, 14)
    oranges.append({'id': int(k + 1), 'fid': fid, 'm': m, 'crop': (int(max(cy - r, 0)), int(min(cy + r, h)), int(max(cx - r, 0)), int(min(cx + r, w)))})
res = {}
for labn, ident, pdir in arms:
    p = np.load(pdir / f'{stem}.npz')[a.rule]; rgb = np.load(next(p_ for p_ in (ident / 'maps_fruit' / f'{stem}.npz', ident / 'maps_s01' / f'{stem}.npz', ident / 'maps_s05' / f'{stem}.npz', ident / 'maps_mv' / f'{stem}.npz') if p_.exists()))['rgb']; res[labn] = (p, rgb)
    gtw = {o['fid'] for o in oranges}; ious = []
    for f in sorted(gtw):
        pr, gt = p == f, ids == f; ious.append(f"{(pr & gt).sum() / max((pr | gt).sum(), 1):.2f}")
    print(f"[rule-fig] {labn}: frame IoU for fruit word(s) {sorted(gtw)} = {', '.join(ious)}; lit px {int((p >= 0).sum())}", flush=True)
ncol = 1 + len(arms); fig, axs = plt.subplots(len(oranges), ncol, figsize=(3.3 * ncol, 3.5 * len(oranges) + 0.6), squeeze=False)
out = {'frame': a.frame, 'oranges': []}
for r, o in enumerate(oranges):
    Y0, Y1, X0, X1 = o['crop']; cs = (slice(Y0, Y1), slice(X0, X1)); gtw = ids[cs] == o['fid']
    axs[r, 0].imshow(photo[cs], interpolation='nearest'); axs[r, 0].contour(gtw, levels=[0.5], colors='white', linewidths=1.2); axs[r, 0].set_title(f'orange #{o["id"]}: photo', fontsize=10)
    row = {'id': o['id'], 'fid': o['fid'], 'arms': {}}
    for j, (labn, _, _) in enumerate(arms):
        p, rgb = res[labn]; pc = p[cs] == o['fid']; on = pc & gtw; off = pc & ~gtw; rec = (p == o['fid'])[o['m']].mean()
        img = (rgb[cs] * 0.55).astype(np.uint8).copy(); img[on] = (232, 115, 12); img[off] = (214, 39, 40)
        ax = axs[r, 1 + j]; ax.imshow(img, interpolation='nearest'); ax.contour(gtw, levels=[0.5], colors='white', linewidths=1.2)
        ax.set_title(f'{labn}\nlit {100 * rec:.0f}% of the orange · leak {int(off.sum())} px', fontsize=10); row['arms'][labn] = {'recall': float(rec), 'leak_px_crop': int(off.sum())}
    out['oranges'].append(row)
    for c in range(ncol): axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
fig.suptitle(f'{a.frame} (training view): what "the fruit of this tree" lights — orange = on the fruit, red = off it (leak), white = labelled fruit', fontsize=11)
fig.tight_layout(); fig.savefig(a.out, dpi=150); json.dump(out, open(Path(a.out).with_suffix('.json'), 'w'), indent=1); print(f'[rule-fig] -> {a.out}', flush=True)
