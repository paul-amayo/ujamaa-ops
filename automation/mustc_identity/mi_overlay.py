#!/usr/bin/env python3
"""MuST-C identity feasibility: overlay for inspection (no score). Mask outlines (score >= --conf) and the projected census points
(on-row yellow with id, off-row magenta) on a dumped view.  nerf_new pixi env:  mi_overlay.py --dir <stop dir> --masks <mask dir> --view nikon_11 --out x.jpg [--conf 0.3]"""
import argparse, json
import cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--masks', required=True); ap.add_argument('--view', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--conf', type=float, default=0.3); ap.add_argument('--scale', type=float, default=1.0); a = ap.parse_args()
im = cv2.imread(f'{a.dir}/views/{a.view}.png'); lab = cv2.imread(f'{a.masks}/{a.view}.png', cv2.IMREAD_UNCHANGED); inst = json.load(open(f'{a.masks}/{a.view}.json'))
rs = np.random.RandomState(1)
for it in inst:
    if it['score'] < a.conf: continue
    m = (lab == it['idx']).astype(np.uint8); col = tuple(int(c) for c in rs.randint(60, 255, 3))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, col, 2)
P = json.load(open(f'{a.dir}/plants.json'))
for p in P['plants']:
    v = next((x for k, x in p['views'].items() if k.startswith(a.view + '.')), None)
    if not v or not v.get('in_image'): continue
    c = (0, 255, 255) if p['on_row'] else (255, 0, 255)
    cv2.circle(im, (int(v['u']), int(v['v'])), 7, c, -1); cv2.circle(im, (int(v['u']), int(v['v'])), 7, (0, 0, 0), 1)
    cv2.putText(im, str(p['id']), (int(v['u']) + 8, int(v['v']) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
if a.scale != 1.0: im = cv2.resize(im, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
cv2.imwrite(a.out, im, [cv2.IMWRITE_JPEG_QUALITY, 85]); print('[overlay]', a.out)
