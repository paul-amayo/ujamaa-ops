#!/usr/bin/env python3
"""SAM3 masks vs the field on "how many cabbages are there?" frames (UJAMAA 2026-10-04). Per frame: (1) best-containment field,
every pixel coloured by its best cabbage among the cabbages in view (as the cut lights them), (2) SAM3 masks (undistorted
into the training camera) in the SAME per-cabbage colours, (3) agreement: green = field and SAM3 name the same cabbage,
orange = field lights a cabbage where SAM3 has another / none, red = SAM3 cabbage the field misses; per-cabbage IoU.
usage: cabbage_sam3_vs_field.py image_N.png ...   (h3dgs env)"""
import json, math, sys
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FR_ = sys.argv[1:]; sys.argv = [sys.argv[0], '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root/experimental/h3dgs_native/chunk_lane_ud/features_census_v3g.bin', 'fig']   # the loader reads FEAT, TAG from argv
exec(open('/home/paperspace/code/automation/image_farm/cabbage_bc_score.py').read().split("res = []")[0])   # loader: CH, NI, IDS, WORDS, heats, ims, Cam, EX, SUP, W, H ...
import matplotlib.cm as cm
TREES = {int(k): np.array(v) for k, v in json.load(open('/home/paperspace/logs/demo_chunks/gwakungu_7993/demo_path.json'))['trees'].items()}
cmap = matplotlib.colormaps['tab20']; COL = {t: np.array(cmap(i % 20)[:3]) * 255 for i, t in enumerate(IDS)}
frames = FR_
fig, axs = plt.subplots(len(frames), 3, figsize=(13, 7.7 * len(frames)), dpi=80); axs = np.atleast_2d(axs)
for r, kf in enumerate(frames):
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; c2w = np.linalg.inv(w2c)
    cam = Cam(c2w, W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); img, _ = CH.render(cam, 3.0)
    if kf in EX: E_ = np.array(EX[kf], np.float32); img = torch.einsum('ij,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy().astype(np.float32)
    vis = []
    for t, x in TREES.items():
        p = w2c[:3, :3] @ x + w2c[:3, 3]
        if p[2] > 0:
            u, v = fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy
            if 0 <= u < W and 0 <= v < H: vis.append(t)
    hm = heats(NI.feature_pass(CH, cam, 3.0), WORDS); valid = (hm > -1).any(0); am = hm.argmax(0)
    best = torch.where(valid, torch.tensor(IDS, device=am.device)[am], torch.full_like(am, -1)).cpu().numpy().astype(np.int32)
    best = np.array(Image.fromarray(best).resize((W, H), Image.NEAREST)); best[~np.isin(best, vis)] = -1      # only cabbages in view light (as the cut)
    sup = np.array(Image.open(f'{SUP}/{kf}'), np.int64); sup[sup >= 10000] = -1
    a = rgb * 0.45; b = rgb * 0.45; c = rgb * 0.35
    for t in IDS:
        a[best == t] = a[best == t] * 0.3 + COL[t] * 0.7; b[sup == t] = b[sup == t] * 0.3 + COL[t] * 0.7
    ok = (best >= 0) & (best == sup); fl = (best >= 0) & (best != sup); ms = (sup >= 0) & (best != sup)
    c[ok] = (0, 200, 90); c[fl] = (255, 140, 0); c[ms & ~fl] = (230, 40, 40)
    ious = {t: (best == t)[sup == t].sum() / max(((best == t) | (sup == t)).sum(), 1) for t in np.unique(sup) if t >= 0 and (sup == t).sum() >= 200}
    for k, (arr, ttl) in enumerate([(a, f'{kf}: field, best cabbage per pixel ({len(set(best[best >= 0].tolist()))} cabbages lit)'),
                                    (b, f'SAM3 masks, undistorted ({len(ious)} cabbages labelled)'),
                                    (c, 'green same cabbage · orange field lights other/none · red missed\nIoU ' + ', '.join(f'{t}: {v:.2f}' for t, v in sorted(ious.items())))]):
        ax = axs[r, k]; ax.imshow(arr.astype(np.uint8)); ax.set_axis_off(); ax.set_title(ttl, fontsize=9, loc='left')
fig.tight_layout(); fn = '/home/paperspace/data/demo_video_v2/cabbage_cut/sam3_vs_field_how_many.png'; fig.savefig(fn); print('[fig]', fn)
