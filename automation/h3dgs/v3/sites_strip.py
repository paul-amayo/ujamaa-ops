#!/usr/bin/env python3
"""sites_strip.py (2026-10-19, argmax version) — the HiGH paper's cross-site figure: one held-out frame per site with one identity query read
by the paper's protocol, the argmax rule (each non-void pixel goes to the address with the highest containment score among that level's
addresses), coloured true positive / false positive / miss against the supervision, with the argmax IoU of Table I. The head shown is chosen
without looking at the images: among the survey's held-out TREE heads covering >= 3 % of the frame, the one whose argmax IoU is closest to the
survey's median over those heads (protocol_v3.jsonl of every cell), so each panel is the typical case. Stage 1 runs hero_heat.py --words all
(nerf_new pixi env, GPU) for each chosen frame; stage 2 takes the argmax over the tree-level addresses and composes the strip. Portrait
(phone) frames are cropped to the strip's aspect around the head's centroid.  system python3:  sites_strip.py <paper dir> [--surveys a,b,c]"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('paper'); ap.add_argument('--surveys', default='01_13B_Jackal,dec_2025_ten_rows,nov_2025_a300,apr_2026_zed,IMG_7993_s0,IMG_7990_s1')
ap.add_argument('--out', default='fig_sites.png'); ap.add_argument('--stage', default='both', choices=['both', '1', '2']); a = ap.parse_args()
P = Path(a.paper); FIG = P / 'figures'; (FIG / 'sites').mkdir(exist_ok=True)
KIND = {'dec_2025_ten_rows': 'plant', 'IMG_7990_s1': 'plant', 'IMG_7993_s0': 'plant', 'apr_2026_zed': 'plant'}
SHORT = {'01_13B_Jackal': 'citrus 01', '02_13B_Jackal': 'citrus 02', '03_13B_Jackal': 'citrus 03', '04_13D_Jackal': 'citrus 04', '05_13D_Jackal': 'citrus 05', 'apr_2026_zed': 'Klapmuts Apr', 'dec_2025_ten_rows': 'Klapmuts Dec', 'nov_2025_a300': 'Lindendhof', 'IMG_7993_s0': 'cabbage', 'IMG_7990_s1': 'pepper'}
inv = [json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')]
picks = []
for sv in a.surveys.split(','):
    heads = []
    for c in [c for c in inv if c['survey'] == sv]:
        pth = Path(c['identity']) / 'protocol_v3.jsonl'
        if not pth.exists(): continue
        for l in open(pth):
            h = json.loads(l)
            if h.get('split') == 'eval' and h.get('kind') == 'TREE' and h.get('gt_frac', 0) >= 0.03: h['identity'] = c['identity']; heads.append(h)
    if not heads: print('[sites] no heads for', sv); continue
    med = float(np.median([h['argmax']['iou'] for h in heads])); h = min(heads, key=lambda h: abs(h['argmax']['iou'] - med))
    picks.append({'survey': sv, 'identity': h['identity'], 'frame': h['frame'], 'word': h['word'], 'id': int(h['id']), 'argmax_iou': h['argmax']['iou'], 'median': med, 'n': len(heads), 'gt_frac': h['gt_frac']})
    print(f"[sites] {sv}: {len(heads)} heads >= 3 %, median argmax IoU {med:.3f}; shown {h['frame']} id {h['id']} '{h['word']}' at {h['argmax']['iou']:.3f} ({100 * h['gt_frac']:.1f} % of the frame)")
json.dump(picks, open(FIG / 'sites' / 'picks.json', 'w'), indent=1)
if a.stage in ('both', '1'):
    for pk in picks:
        out = FIG / 'sites' / f"{pk['survey']}_all.npz"
        if out.exists(): continue
        cmd = f"cd /home/paperspace/code/nerf_new && pixi run python /home/paperspace/code/automation/h3dgs/v3/hero_heat.py --identity {pk['identity']} --frame {pk['frame']} --words all --out {out}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print('[sites] hero_heat', pk['survey'], 'rc', r.returncode, r.stderr.strip().splitlines()[-1:] if r.returncode else 'ok')
if a.stage == '1': raise SystemExit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ASPECT = 640 / 300
def argmax_mask(z, words, target_word):
    """pixels whose argmax over `words` (containment heats of one level) is target_word; void excluded"""
    heats = np.stack([z[w] for w in words]); best = heats.argmax(0); return (best == words.index(target_word)) & ~z['void'].astype(bool)
fig, axes = plt.subplots(1, len(picks), figsize=(7.2, 1.3)); fig.subplots_adjust(0.003, 0.02, 0.997, 0.98, 0.01, 0)
for ax, pk in zip(np.atleast_1d(axes), picks):
    z = np.load(FIG / 'sites' / f"{pk['survey']}_all.npz", allow_pickle=True); meta = json.loads(str(z['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}
    tree_words = [w for k, w in sorted(wt.items()) if k < 10000 and w in z.files]; rgb = z['rgb']; H, W = rgb.shape[:2]
    sup = Path(next(c['sup'] for c in inv if c['identity'] == pk['identity']))   # the inventory's supervision dir (text-bank manifests of the phone cells point at retired shims)
    ids = np.array(Image.fromarray(np.array(Image.open(sup / pk['frame']))).resize((W, H), Image.NEAREST)); truth = ids == pk['id']
    pred = argmax_mask(z, tree_words, pk['word']); u = (pred | truth).sum(); iou = float((pred & truth).sum() / u) if u else 0.0; pk['iou_argmax_fig'] = iou
    img = rgb.astype(float) / 255 * 0.5; img[pred & truth] = 0.3 * img[pred & truth] + 0.7 * np.array([0.1, 0.8, 0.2]); img[pred & ~truth] = 0.3 * img[pred & ~truth] + 0.7 * np.array([1.0, 0.55, 0.0]); img[~pred & truth] = 0.3 * img[~pred & truth] + 0.7 * np.array([0.9, 0.1, 0.1])
    if H > W:   # portrait phone frame: a band of the strip's aspect around the head
        ys = np.where(truth.any(1))[0]; cy = int(ys.mean()) if len(ys) else H // 2; bh = int(W / ASPECT); y0 = int(np.clip(cy - bh // 2, 0, H - bh)); crop = img[y0:y0 + bh]
    else: crop = img[:int(H * 0.83)]
    ax.imshow(np.clip(crop, 0, 1)); ax.axis('off')
    ax.text(0.015, 0.95, f"{SHORT[pk['survey']]}: \"{KIND.get(pk['survey'], 'tree')} {pk['id']}\"", fontsize=6.5, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    ax.text(0.985, 0.05, f'IoU {iou:.2f}', fontsize=6.5, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    print(f"[sites] {pk['survey']} {pk['frame']} id {pk['id']}: argmax IoU in the figure {iou:.3f} (protocol {pk['argmax_iou']:.3f}, survey median {pk['median']:.3f}); {'portrait' if H > W else 'landscape'}")
json.dump(picks, open(FIG / 'sites' / 'picks.json', 'w'), indent=1); fig.savefig(FIG / a.out, dpi=220); print('[sites] wrote', FIG / a.out)
