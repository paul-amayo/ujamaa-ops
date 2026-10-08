#!/usr/bin/env python3
"""Containment vs ground truth for one row on given keyframes (UJAMAA, 2026-10-03; Paul: "3112, why is IoU zero, show
containment and ground truth"). Panels per frame: render (own exposure) | SAM3 ground truth (the row's trees green,
other labelled trees grey) | containment heat for the row word (no-identity pixels black, the frame's split marked) |
result (correct green, false light orange, missed red)."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.argv = [sys.argv[0], sys.argv[1], '0', '0'] + sys.argv[2:]
exec(open('/home/paperspace/code/automation/demo/row_iou_frames.py').read().split("try: F = ImageFont")[0])   # shared loading + heat()
frames = [f'kf_{int(k):06d}.png' for k in sys.argv[4:]]
fig, axs = plt.subplots(len(frames), 4, figsize=(20, 3.3 * len(frames)), dpi=110); axs = np.atleast_2d(axs)
for r, kf in enumerate(frames):
    im = ims[kf]; w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    cam = Cam(np.linalg.inv(w2c), W, H, 2 * math.atan(H / (2 * fy)), cx / W, cy / H); img, _ = CH.render(cam, 3.0)
    if kf in EXPO:
        E_ = np.array(EXPO[kf], np.float32); img = torch.einsum('ji,jhw->ihw', torch.tensor(E_[:, :3]).cuda(), img) + torch.tensor(E_[:, 3]).cuda()[:, None, None]
    rgb = (img.clamp(0, 1).permute(1, 2, 0) * 255).byte().cpu().numpy()
    hm = heat(NI.feature_pass(CH, cam, 3.0)); a = NI.alpha(hm); lit = (a.float().cpu().numpy() > 0) if a is not None else np.zeros((H, W), bool)
    v = hm[hm > -1]; split = max(_otsu(v), float(torch.quantile(v, 0.5))); hm = hm.cpu().numpy()
    sup = np.array(Image.open(f'{NB}/supervision/trees_only/{kf}'), np.uint16); gt = np.isin(sup, ROW_TREES)
    gtv = rgb.astype(np.float32) * 0.35; gtv[(sup < 10000) & ~gt] = [150, 150, 150]; gtv[gt] = [0, 200, 90]
    hv = np.ma.masked_where(hm <= -1, hm)
    res = rgb.astype(np.float32) * 0.45; res[lit & gt] = [0, 200, 90]; res[lit & ~gt] = [255, 140, 0]; res[gt & ~lit] = [230, 40, 40]
    iou = (lit & gt).sum() / max((lit | gt).sum(), 1)
    for c, (arr, title) in enumerate([(rgb, f'{kf}: render'), (gtv.astype(np.uint8), f'SAM3 ground truth: row {ROW} = {int(gt.sum())} px (green), other trees grey'),
                                      (None, f'containment heat for "{word}" (row {ROW}); split {split:.3f}'), (res.astype(np.uint8), f'IoU {iou:.3f}: correct green, false orange, missed red')]):
        ax = axs[r, c]; ax.set_axis_off(); ax.set_title(title, fontsize=9, loc='left')
        if arr is not None: ax.imshow(arr)
        else:
            ax.set_facecolor('black'); ax.imshow(np.zeros((H, W, 3), np.uint8)); im_ = ax.imshow(hv, cmap='Blues', vmin=0.4, vmax=1.0)
            ax.contour(hm >= split, levels=[0.5], colors=['#eb6834'], linewidths=0.8); plt.colorbar(im_, ax=ax, fraction=0.025, pad=0.01)
fig.tight_layout(); fn = OUT.parent / f'row{ROW}_panels_{"_".join(k[3:9] for k in frames)}.png'; fig.savefig(fn); print(f'[panels] {fn}')
