#!/usr/bin/env python3
"""MuST-C identity feasibility (JS-3): the supervision id map of a view over the image, for reading by eye (no score). Labelled
masks filled and outlined with their plant id; every projected census point drawn (on-row yellow, off-row magenta); unlabelled
SAM 3 masks that hold a census point outlined in red (ambiguous or off-row).
nerf_new pixi env:  mi_sup_overlay.py --dir <stop dir> --view nikon_5 --out x.jpg"""
import argparse, json
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True); a = ap.parse_args(); D = a.dir
C = json.load(open(f'{D}/cameras.json')); name = next(n for n, c in C['frames'].items() if c['label'] == a.view)
im = cv2.imread(f'{D}/views/{a.view}.png'); gt = cv2.imread(f'{D}/cell/supervision/trees_only/{name}', cv2.IMREAD_UNCHANGED); sam = cv2.imread(f'{D}/masks/{a.view}.png', cv2.IMREAD_UNCHANGED)
rs = np.random.RandomState(3); P = json.load(open(f'{D}/plants.json'))['plants']
for i in [int(u) for u in np.unique(gt) if u != 65535]:
    m = gt == i; col = rs.randint(60, 255, 3); im[m] = (0.55 * im[m] + 0.45 * col).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (255, 255, 255), 2)
seen = set()
for p in P:
    v = p['views'][name]
    if not v.get('in_image'): continue
    u_, v_ = int(v['u']), int(v['v']); k = int(sam[v_, u_])
    if k > 0 and gt[v_, u_] == 65535 and k not in seen:
        seen.add(k); cs, _ = cv2.findContours((sam == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
    c = (0, 255, 255) if p['on_row'] else (255, 0, 255); cv2.circle(im, (u_, v_), 7, c, -1); cv2.circle(im, (u_, v_), 7, (0, 0, 0), 1)
    cv2.putText(im, str(p['id']), (u_ + 9, v_ - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4); cv2.putText(im, str(p['id']), (u_ + 9, v_ - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
cv2.imwrite(a.out, im, [cv2.IMWRITE_JPEG_QUALITY, 85]); print('[sup-overlay]', a.out)
