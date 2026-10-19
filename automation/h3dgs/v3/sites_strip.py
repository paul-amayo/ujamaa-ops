#!/usr/bin/env python3
"""sites_strip.py (2026-10-19) — the HiGH paper's cross-site figure: one held-out frame per site with one identity query read by containment
(threshold 0.85, as the hero figure) and its IoU against the supervision. The head shown is chosen without looking at the images: among the
survey's held-out TREE heads covering >= 3 % of the frame, the one whose argmax IoU is closest to the survey's median over those heads
(protocol_v3.jsonl of every cell), so the panel is the typical case, not the best one. Stage 1 runs hero_heat.py (nerf_new pixi env, GPU)
for each chosen frame; stage 2 composes the strip.  system python3:  sites_strip.py <paper dir> [--surveys a,b,c] [--thr 0.85]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('paper'); ap.add_argument('--surveys', default='01_13B_Jackal,dec_2025_ten_rows,nov_2025_a300,apr_2026_zed,IMG_7990_s1')
ap.add_argument('--thr', type=float, default=0.85); ap.add_argument('--out', default='fig_sites.png'); a = ap.parse_args()
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
    picks.append({'survey': sv, 'identity': h['identity'], 'frame': h['frame'], 'word': h['word'], 'id': int(h['id']), 'argmax_iou': h['argmax']['iou'], 'median': med, 'n': len(heads)})
    print(f"[sites] {sv}: {len(heads)} heads >= 3 %, median argmax IoU {med:.3f}; shown {h['frame']} id {h['id']} '{h['word']}' at {h['argmax']['iou']:.3f}")
json.dump(picks, open(FIG / 'sites' / 'picks.json', 'w'), indent=1)
for pk in picks:   # stage 1: the containment heat of the chosen word on the chosen frame
    out = FIG / 'sites' / f"{pk['survey']}.npz"
    if out.exists(): continue
    cmd = f"cd /home/paperspace/code/nerf_new && pixi run python /home/paperspace/code/automation/h3dgs/v3/hero_heat.py --identity {pk['identity']} --frame {pk['frame']} --words {pk['word']} --out {out}"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True); print('[sites] hero_heat', pk['survey'], 'rc', r.returncode, [l for l in r.stdout.splitlines() if l.startswith('[hero]')][-1:] , r.stderr.strip().splitlines()[-1:] if r.returncode else '')
# stage 2: the strip
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
CM = plt.get_cmap('inferno')
fig, axes = plt.subplots(1, len(picks), figsize=(7.2, 1.45)); fig.subplots_adjust(0.003, 0.02, 0.997, 0.98, 0.012, 0)
for ax, pk in zip(np.atleast_1d(axes), picks):
    z = np.load(FIG / 'sites' / f"{pk['survey']}.npz", allow_pickle=True); rgb = z['rgb']; h = z[pk['word']]
    meta = json.loads(str(np.load(Path(pk['identity']) / 'text_bank.npz', allow_pickle=True)['meta'])); sup = Path(meta['manifest']).parent; ids = np.array(Image.fromarray(np.array(Image.open(sup / pk['frame']))).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)); truth = ids == pk['id']
    m = h >= a.thr; u = (m | truth).sum(); iou = float((m & truth).sum() / u) if u else 0.0; pk['iou_thr'] = iou
    img = rgb.astype(float) / 255 * 0.45; hh = np.clip((h - a.thr) / max(1e-6, float(h.max()) - a.thr), 0, 1); col = CM(0.35 + 0.65 * hh)[..., :3]; img[m] = 0.25 * img[m] + 0.75 * col[m]
    crop = img[:int(rgb.shape[0] * 0.83)]; ax.imshow(np.clip(crop, 0, 1)); ax.axis('off')
    ax.text(0.015, 0.95, f"{SHORT[pk['survey']]}: \"{KIND.get(pk['survey'], 'tree')} {pk['id']}\"", fontsize=7, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    ax.text(0.985, 0.05, f'IoU {iou:.2f}', fontsize=7, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    print(f"[sites] {pk['survey']} {pk['frame']} id {pk['id']}: IoU at {a.thr} = {iou:.3f} (argmax {pk['argmax_iou']:.3f}, survey median {pk['median']:.3f})")
json.dump(picks, open(FIG / 'sites' / 'picks.json', 'w'), indent=1); fig.savefig(FIG / a.out, dpi=200); print('[sites] wrote', FIG / a.out)
