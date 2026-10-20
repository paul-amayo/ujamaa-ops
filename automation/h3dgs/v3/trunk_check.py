#!/usr/bin/env python3
"""trunk_check.py (2026-10-20) — are the composed-query trunk predictions correct? Per held-out head (frame, tree k) of a super_eval run:
pred = super-field tree argmax k AND part argmax = k's trunk node; truth = SAM 3 trunk masks assigned to tree k (parts/verdict); all trunks =
every SAM 3 "tree trunk" on the whole frame (trunk_allmasks.py). Reports IoU / P / R, tolerant precision and recall at d px (map scale), the
trunk width, where the false-positive pixels fall (edge <= 2 px of the truth, on another detected trunk, near <= 15 px, far), whether the
largest predicted blob touches the truth, and whether the reference trunk agrees with a whole-frame detection. Writes <out>.json + a sheet.
system python3:  trunk_check.py <parts run dir (super_eval.json, super_dump/)> <parts dir (verdict/)> <nodes.npz> <trunk_all dir> <out.png>"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as nd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R, P, NZ, TA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), np.load(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
tn = {int(t): int(i) for i, t, ty in zip(NZ['ids'], NZ['tree'], NZ['type']) if int(ty) == 0}; ev = json.load(open(R / 'super_eval.json'))
def unpack(z):
    H, W = z['shape']; return np.unpackbits(z['masks'], axis=1)[:, :H * W].reshape(-1, H, W).astype(bool) if z['masks'].size else np.zeros((0, H, W), bool)
def rs(m, W, H): return np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
rows, tiles = [], []
for h in ev['per_head']:
    f, k = h['frame'], h['tree']; stem = f.replace('.png', ''); d = np.load(R / 'super_dump' / f'{stem}.npz'); H, W = d['tree_super'].shape
    pred = (d['tree_super'] == k) & d['live_super'] & (d['part_super'] == tn.get(k, -99))
    recs = json.load(open(P / 'verdict' / f'{stem}.json')); Mv = unpack(np.load(P / 'verdict' / f'{stem}.npz')); truth_f = np.zeros(Mv.shape[1:], bool)
    for i, r in enumerate(recs):
        if r['prompt'] == 'tree trunk' and r['tree'] == k: truth_f |= Mv[i]
    truth = rs(truth_f, W, H); Ma = unpack(np.load(TA / f'{stem}.npz')); alltr = np.zeros((H, W), bool)
    best_ref = max([(m & truth_f).sum() / (m | truth_f).sum() for m in Ma] or [0.0])
    for m in Ma: alltr |= rs(m, W, H)
    other = alltr & ~nd.binary_dilation(truth, iterations=2)
    dt_truth = nd.distance_transform_edt(~truth); dt_pred = nd.distance_transform_edt(~pred) if pred.any() else np.full((H, W), 1e9)
    tp = (pred & truth).sum(); u = (pred | truth).sum(); npred, ntr = int(pred.sum()), int(truth.sum())
    widths = [int(r.sum()) for r in truth if r.any()]; width = float(np.median(widths)) if widths else 0.0
    fp = pred & ~truth; edge = fp & (dt_truth <= 2); on_other = fp & ~edge & nd.binary_dilation(other, iterations=1); near = fp & ~edge & ~on_other & (dt_truth <= 15); far = fp & ~edge & ~on_other & ~near
    lab, n = nd.label(pred); big_hits = False
    if n: sizes = np.bincount(lab.ravel())[1:]; big = lab == (1 + int(np.argmax(sizes))); big_hits = bool((big & nd.binary_dilation(truth, iterations=2)).any())
    row = {'frame': f, 'tree': k, 'main': h['trunk_frac'] >= 0.002, 'truth_px': ntr, 'pred_px': npred, 'trunk_width_px': width, 'iou': float(tp / u) if u else 0.0, 'p': float(tp / npred) if npred else 0.0, 'r': float(tp / ntr) if ntr else 0.0,
           'p_2': float((dt_truth[pred] <= 2).mean()) if npred else 0.0, 'r_2': float((dt_pred[truth] <= 2).mean()) if ntr else 0.0, 'p_5': float((dt_truth[pred] <= 5).mean()) if npred else 0.0, 'r_5': float((dt_pred[truth] <= 5).mean()) if ntr else 0.0,
           'fp_px': int(fp.sum()), 'fp_edge': int(edge.sum()), 'fp_on_other_trunk': int(on_other.sum()), 'fp_near': int(near.sum()), 'fp_far': int(far.sum()), 'largest_blob_on_truth': big_hits, 'n_blobs': int(n), 'ref_vs_wholeframe_iou': round(float(best_ref), 3), 'n_wholeframe_trunks': int(len(Ma))}
    rows.append(row)
    ys, xs = np.where(truth | pred); cy, cx = int(np.median(np.where(truth)[0])) if ntr else int(ys.mean()), int(np.median(np.where(truth)[1])) if ntr else int(xs.mean()); s = 45
    img = d['rgb'].astype(float) / 255 * 0.55; img[pred & truth] = [0.1, 0.85, 0.2]; img[pred & ~truth] = [1.0, 0.55, 0.0]; img[~pred & truth] = [0.9, 0.1, 0.1]
    oe = other ^ nd.binary_erosion(other); img[oe] = [0.2, 0.9, 1.0]
    tiles.append((img[max(0, cy - s):cy + s, max(0, cx - s):cx + s], f"{stem[3:]} t{k}{'' if row['main'] else ' (<0.2%)'}\nIoU {row['iou']:.2f} | P5 {row['p_5']:.2f} R5 {row['r_5']:.2f}"))
json.dump(rows, open(OUT.replace('.png', '.json'), 'w'), indent=1)
for r in rows: print(f"[check] {r['frame']} t{r['tree']:2d}{' ' if r['main'] else '*'} truth {r['truth_px']:4d} px (width {r['trunk_width_px']:4.1f}) pred {r['pred_px']:4d} | IoU {r['iou']:.2f} P {r['p']:.2f} R {r['r']:.2f} | <=2px P {r['p_2']:.2f} R {r['r_2']:.2f} | <=5px P {r['p_5']:.2f} R {r['r_5']:.2f} | FP {r['fp_px']:4d}: edge {r['fp_edge']:3d} other-trunk {r['fp_on_other_trunk']:3d} near {r['fp_near']:3d} far {r['fp_far']:4d} | largest blob on truth {r['largest_blob_on_truth']} ({r['n_blobs']} blobs) | ref vs whole-frame IoU {r['ref_vs_wholeframe_iou']:.2f}")
M = [r for r in rows if r['main']]; med = lambda k, rr: float(np.median([r[k] for r in rr]))
for name, rr in (('main', M), ('all', rows)):
    fp = sum(r['fp_px'] for r in rr) or 1
    print(f"[check] {name} heads n={len(rr)}: median IoU {med('iou', rr):.3f} P {med('p', rr):.2f} R {med('r', rr):.2f} | <=2px P {med('p_2', rr):.2f} R {med('r_2', rr):.2f} | <=5px P {med('p_5', rr):.2f} R {med('r_5', rr):.2f} | trunk width median {med('trunk_width_px', rr):.1f} px | largest blob on the right trunk {sum(r['largest_blob_on_truth'] for r in rr)}/{len(rr)} | FP pixels: edge {sum(r['fp_edge'] for r in rr) / fp:.2f}, other trunk {sum(r['fp_on_other_trunk'] for r in rr) / fp:.2f}, near {sum(r['fp_near'] for r in rr) / fp:.2f}, far {sum(r['fp_far'] for r in rr) / fp:.2f} | reference vs whole-frame trunk IoU median {med('ref_vs_wholeframe_iou', rr):.2f}")
n = len(tiles); c = 6; fig, axes = plt.subplots((n + c - 1) // c, c, figsize=(c * 1.7, ((n + c - 1) // c) * 2.0), squeeze=False); fig.subplots_adjust(0.005, 0.005, 0.995, 0.97, 0.03, 0.3)
for ax in axes.flat: ax.axis('off')
for ax, (t, lab_) in zip(axes.flat, tiles): ax.imshow(np.clip(t, 0, 1)); ax.set_title(lab_, fontsize=6, pad=1)
fig.savefig(OUT, dpi=180); print('[check] wrote', OUT)
