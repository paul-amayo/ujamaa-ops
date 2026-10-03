#!/usr/bin/env python3
"""Per-frame PSNR across the phone cut's timeline (UJAMAA, 2026-10-03; Paul: "there are so many grey cuts, for each frame
give me the psnr graph for the video"). Each video frame is a render at a recorded keyframe pose; it is scored against
that keyframe's photo two ways:
  as shown      the plain render (reels: backdrop/d_NNNNN.png; walks: the captured live frame) vs the photo
  colour-matched the same after a per-frame 3x4 affine colour fit to the photo (the H3DGS exposure model) — removes the
                 camera's auto-exposure swings, so what remains is render quality (smear, grey, missing geometry)
FULL FRAME, sky included (Paul: "also we need the sky"); for citrus the sky alone is also scored (survey sky masks), colour-matched.
Segments and spans follow phone_cut_box.sh (v3 order, no globe/cards)."""
import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = Path('/home/paperspace/logs/demo_chunks'); C = '/home/paperspace/data/citrus_all'; OUT = Path('/home/paperspace/data/demo_video_v2/psnr'); OUT.mkdir(exist_ok=True)
def reel(rd, idx, gt, sky):
    P = json.load(open(D / rd / 'demo_path.json'))['frames']; P = P if idx is None else [P[i] for i in idx]
    return [(D / rd / 'backdrop' / f['name'], Path(gt) / f['src'], (Path(sky) / f['src']) if sky else None, f['src']) for f in P]
def walk(cap, root, n, gtdir):
    L = sorted(json.load(open(f'{root}/lio_image_poses.json')), key=lambda p: p['image_idx'])[:n]
    return [(Path(cap) / f'f_{k:05d}.jpg', Path(gtdir) / r['image_name'], None, r['image_name']) for k, r in enumerate(L)]
K = '/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/lane2'; KB = '/home/paperspace/data/image_farm/kendu_bay/2026-05-14/IMG_7975_s0'
SEG = [('Citrus B lane\n05 chunk 0_0', 0.0, 48.0, reel('05_0_0_720', None, f'{C}/05_13D_Jackal/prod/scratch_sam3', f'{C}/05_13D_Jackal/prod/tassili/sky_masks')),
       ('Fruit clip\n05 chunk 1_0', 48.0, 67.767, reel('05_1_0_720', list(range(13, 53)) + list(range(346, 361)), f'{C}/05_13D_Jackal/prod/scratch_sam3', f'{C}/05_13D_Jackal/prod/tassili/sky_masks')),
       ('Citrus A\n01 chunk 3_1', 67.767, 125.634, reel('01_3_1_720', None, f'{C}/01_13B_Jackal/prod/scratch_sam3', f'{C}/01_13B_Jackal/prod/tassili/sky_masks')),
       ('Klapmuts\nlane walk', 125.634, 160.634, walk('/home/paperspace/data/demo_video_v2/klapmuts-dec25', K, 280, f'{K}/h3dgs_e2/camera_calibration/rectified/images')),
       ('Gwakungu\ncabbage', 160.634, 179.034, reel('gwakungu_7993', None, '/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage/demo_root/prod/scratch_sam3', None)),
       ('Kendu Bay\nwalk', 179.034, 186.659, walk('/home/paperspace/data/demo_video_v2/kendu-0514-plants', KB, 61, f'{KB}/images'))]
def score(rp, gp, sp):
    r = np.asarray(Image.open(rp).convert('RGB')).astype(np.float32); g = Image.open(gp).convert('RGB')
    if g.size != (r.shape[1], r.shape[0]): g = g.resize((r.shape[1], r.shape[0]), Image.BILINEAR)
    g = np.asarray(g).astype(np.float32); R, G = r.reshape(-1, 3), g.reshape(-1, 3)
    shown = 10 * np.log10(255 ** 2 / max(((R - G) ** 2).mean(), 1e-9))
    X = np.c_[R[::3], np.ones(len(R[::3]))]; A, *_ = np.linalg.lstsq(X, G[::3], rcond=None)
    F = np.clip(np.c_[R, np.ones(len(R))] @ A, 0, 255); matched = 10 * np.log10(255 ** 2 / max(((F - G) ** 2).mean(), 1e-9))
    sky = float('nan')
    if sp is not None and sp.exists():
        m = (np.asarray(Image.open(sp).convert('L').resize((r.shape[1], r.shape[0]), Image.NEAREST)) >= 128).reshape(-1)
        if m.sum() > 500: sky = 10 * np.log10(255 ** 2 / max(((F[m] - G[m]) ** 2).mean(), 1e-9))
    return shown, matched, sky
rows, summ = [], []
for name, t0, t1, frames in SEG:
    vals = []
    for k, (rp, gp, sp, kf) in enumerate(frames):
        t = t0 + (k + 0.5) / len(frames) * (t1 - t0)
        if not rp.exists() or not gp.exists(): rows.append((name.replace('\n', ' '), k, kf, round(t, 2), '', '')); continue
        s, mt, sk = score(rp, gp, sp); vals.append((t, s, mt, sk)); rows.append((name.replace('\n', ' '), k, kf, round(t, 2), round(s, 2), round(mt, 2), '' if np.isnan(sk) else round(sk, 2)))
    v = np.array(vals); summ.append((name, t0, t1, v))
    sk = v[:, 3][~np.isnan(v[:, 3])]; print(f'[timeline] {name.replace(chr(10), " ")}: {len(v)}/{len(frames)} frames | as shown median {np.median(v[:, 1]):.1f} dB, < 25: {(v[:, 1] < 25).mean():.0%} | colour-matched median {np.median(v[:, 2]):.1f}, < 25: {(v[:, 2] < 25).mean():.0%}, min {v[:, 2].min():.1f}' + (f' | sky only (colour-matched) median {np.median(sk):.1f} on {len(sk)} frames' if len(sk) else ''), flush=True)
with open(OUT / 'phone_cut_psnr.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['segment', 'frame', 'keyframe', 'video_s', 'psnr_as_shown', 'psnr_colour_matched', 'psnr_sky_only_colour_matched']); w.writerows(rows)
TXT, TXT2, GRID, S1, S2 = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#eb6834'
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=170); fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
for i, (name, t0, t1, v) in enumerate(summ):
    if i % 2: ax.axvspan(t0, t1, color='#f0efeb', lw=0, zorder=0)
    ax.plot(v[:, 0], v[:, 1], color=S2, lw=1.2, zorder=2); ax.plot(v[:, 0], v[:, 2], color=S1, lw=1.6, zorder=3)
    ax.text((t0 + t1) / 2, 41.2, name, ha='center', va='top', fontsize=8.5, color=TXT2, linespacing=1.15)
    ax.text((t0 + t1) / 2, 7.4, f'median {np.median(v[:, 2]):.1f}', ha='center', va='bottom', fontsize=8, color=TXT2)
ax.axhline(25, color=TXT, lw=1, ls=(0, (4, 3)), zorder=4); ax.text(186.0, 25.4, '25 dB demo bar', ha='right', va='bottom', fontsize=8.5, color=TXT)
ax.set_xlim(0, 186.7); ax.set_ylim(7, 42); ax.set_xlabel('time in the phone cut (s)', color=TXT2, fontsize=9); ax.set_ylabel('PSNR vs the keyframe photo (dB)', color=TXT2, fontsize=9)
ax.grid(axis='y', color=GRID, lw=0.6, zorder=1); ax.tick_params(colors=TXT2, labelsize=8.5)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=S1, lw=1.8, label='colour-matched to the photo (render quality)'), Line2D([], [], color=S2, lw=1.4, label='as shown (mean exposure)')],
          loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9, labelcolor=TXT, borderaxespad=0.2)
ax.set_title('Per-frame PSNR of the demo phone cut, full frame with sky', loc='left', fontsize=12, color=TXT, pad=26)
fig.tight_layout(); fn = OUT / 'phone_cut_psnr.png'; fig.savefig(fn, facecolor=fig.get_facecolor()); print(f'[timeline] wrote {fn}')
