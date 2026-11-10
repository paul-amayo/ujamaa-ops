#!/usr/bin/env python3
"""sites_strip.py (2026-10-19, argmax version) — the HiGH paper's cross-site figure: one held-out frame per site with one identity query read
by the paper's protocol, the argmax rule (each non-void pixel goes to the address with the highest containment score among that level's
addresses), coloured true positive / false positive / miss against the supervision, with the argmax IoU of Table I. The head shown is chosen
without looking at the images: among the survey's held-out TREE heads covering >= 3 % of the frame, the one whose argmax IoU is closest to the
survey's median over those heads (protocol_v3.jsonl of every cell), so each panel is the typical case. Stage 1 runs hero_heat.py --words all
(nerf_new pixi env, GPU) for each chosen frame; stage 2 takes the argmax over the tree-level addresses and composes the strip. Portrait
(phone) frames are cropped to the strip's aspect around the head's centroid.  system python3:  sites_strip.py <paper dir> [--surveys a,b,c]
--layout zoom (draft 37, column-width Fig. 3): each panel is cropped to the bounding box of every scored pixel (prediction | supervision) plus a
margin, so nothing that enters the IoU leaves the view; the IoU printed is still the full-frame argmax IoU. Background levelled and desaturated,
site + query as a title above the panel, true positive / false positive / miss legend below. --bg natural (v2, default) keeps the photograph's
colour with a mild gamma lift; --bg grey is v1. --fuji <npz> (automation/fruitnerf/fuji_tree_panel.py) appends the fruit level: the apples our
registry assigns to the median-F1 Fuji-SfM tree, on the dataset's photograph, coloured against the GT apples, with that tree's F1."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('paper'); ap.add_argument('--surveys', default='01_13B_Jackal,dec_2025_ten_rows,nov_2025_a300,apr_2026_zed,IMG_7993_s0,IMG_7990_s1')
ap.add_argument('--out', default='fig_sites.png'); ap.add_argument('--cols', type=int, default=0, help='panels per row (0: one strip); draft 36 uses 2 at column width'); ap.add_argument('--width', type=float, default=7.2); ap.add_argument('--fs', type=float, default=6.5); ap.add_argument('--stage', default='both', choices=['both', '1', '2'])
ap.add_argument('--layout', default='frame', choices=['frame', 'zoom'], help="frame: full frames dimmed to 50 %% (drafts <= 36); zoom: each panel cropped to the bounding box of every scored pixel "
                "(prediction | supervision, so nothing that enters the IoU is cut off), background desaturated and brightened, title above, colour legend below")
ap.add_argument('--aspect', type=float, default=1.7, help='zoom layout: panel width / height'); ap.add_argument('--dpi', type=int, default=220)
ap.add_argument('--bg', default='natural', choices=['natural', 'grey'], help='zoom layout background: natural colour with a mild gamma lift (draft 37 v2), or levelled + 70 %% desaturated (draft 37 v1)')
ap.add_argument('--fuji', default=None, help='zoom layout: npz from automation/fruitnerf/fuji_tree_panel.py, appended as the last panel (the fruit level: apples of one tree vs the GT apples, F1 printed)'); a = ap.parse_args()
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
TP, FP, FN = np.array([0.1, 0.8, 0.2]), np.array([1.0, 0.55, 0.0]), np.array([0.9, 0.1, 0.1])
def background(rgb):
    """natural: the photograph with a mild gamma lift (0.85); grey: levelled (1-99 %), gamma 0.8, 70 % desaturated"""
    f = rgb.astype(float) / 255
    if a.bg == 'natural': return f ** 0.85
    LW = np.array([0.299, 0.587, 0.114]); lo, hi = np.percentile(f @ LW, [1, 99]); fl = np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1) ** 0.8; ll = (fl @ LW)[..., None]; return 0.12 + 0.8 * (ll + 0.3 * (fl - ll))
def zoom_box(u, H, W, asp, pad=0.12, box=None):
    """crop (y0, y1, x0, x1) of aspect asp that contains EVERY pixel of u (the scored pixels; or the given box) plus a margin, clamped to the frame"""
    if box is None: ys, xs = np.where(u); box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    y0, y1, x0, x1 = box; m = pad * max(y1 - y0, x1 - x0) + 8
    h = max(y1 - y0 + 2 * m, (x1 - x0 + 2 * m) / asp); w = h * asp
    k = min(1.0, W / w, H / h); w, h = w * k, h * k   # keep the aspect: shrink to the frame (an IoU panel then asserts nothing scored was cut)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; Y0 = int(round(np.clip(cy - h / 2, 0, H - h))); X0 = int(round(np.clip(cx - w / 2, 0, W - w)))
    return Y0, Y0 + int(round(h)), X0, X0 + int(round(w))
NP = len(picks) + (1 if a.fuji and a.layout == 'zoom' else 0); nc = a.cols or NP; nr = -(-NP // nc)
if a.layout == 'zoom':   # explicit inch layout: title band above each panel, legend band at the bottom
    gx, gy, tb, lb = 0.05, 0.03, a.fs / 72 * 1.45, a.fs / 72 * 2.2; pw = (a.width - gx * (nc - 1)) / nc; ph = pw / a.aspect; FH = lb + nr * (tb + ph) + gy * (nr - 1)
    fig = plt.figure(figsize=(a.width, FH)); axes = []
    for i in range(nr * nc):
        r, c = divmod(i, nc); top = FH - r * (tb + ph + gy) - tb; axes.append(fig.add_axes([c * (pw + gx) / a.width, (top - ph) / FH, pw / a.width, ph / FH]))
    print(f'[sites] zoom layout {a.width:.2f} x {FH:.2f} in, panels {pw:.2f} x {ph:.2f} in, text {a.fs} pt')
else:
    pw = a.width / nc; ph = pw / ASPECT * 1.0
    fig, axes = plt.subplots(nr, nc, figsize=(a.width, nr * ph * 1.02)); fig.subplots_adjust(0.003, 0.003, 0.997, 0.997, 0.015, 0.03)
axes = list(np.atleast_1d(axes).ravel())
for ax in axes[NP:]: ax.axis('off')
for ax, pk in zip(axes, picks):
    z = np.load(FIG / 'sites' / f"{pk['survey']}_all.npz", allow_pickle=True); meta = json.loads(str(z['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}
    tree_words = [w for k, w in sorted(wt.items()) if k < 10000 and w in z.files]; rgb = z['rgb']; H, W = rgb.shape[:2]
    sup = Path(next(c['sup'] for c in inv if c['identity'] == pk['identity']))   # the inventory's supervision dir (text-bank manifests of the phone cells point at retired shims)
    ids = np.array(Image.fromarray(np.array(Image.open(sup / pk['frame']))).resize((W, H), Image.NEAREST)); truth = ids == pk['id']
    pred = argmax_mask(z, tree_words, pk['word']); u = (pred | truth).sum(); iou = float((pred & truth).sum() / u) if u else 0.0; pk['iou_argmax_fig'] = iou   # full frame, never the crop
    label = f"{SHORT[pk['survey']]}: {KIND.get(pk['survey'], 'tree')} {pk['id']}"
    if a.layout == 'zoom':
        img = background(rgb)
        for m, col in ((pred & truth, TP), (pred & ~truth, FP), (~pred & truth, FN)): img[m] = 0.25 * img[m] + 0.75 * col
        y0, y1, x0, x1 = zoom_box(pred | truth, H, W, a.aspect); kept = (pred | truth)[y0:y1, x0:x1].sum() / max(u, 1); pk['zoom_box'] = [int(y0), int(y1), int(x0), int(x1)]; pk['zoom_scored_kept'] = float(kept)
        assert kept == 1.0, f"zoom crop drops scored pixels ({kept:.4f})"
        ax.imshow(np.clip(img[y0:y1, x0:x1], 0, 1), interpolation='lanczos', aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color('0.3')
        ax.text(0.0, 1.0 + 0.25 * tb / ph, label, fontsize=a.fs, va='bottom', ha='left', transform=ax.transAxes)
        ax.text(0.985, 0.035, f'IoU {iou:.2f}', fontsize=a.fs, color='white', weight='bold', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.65, pad=1.5, lw=0))
    else:
        img = rgb.astype(float) / 255 * 0.5; img[pred & truth] = 0.3 * img[pred & truth] + 0.7 * TP; img[pred & ~truth] = 0.3 * img[pred & ~truth] + 0.7 * FP; img[~pred & truth] = 0.3 * img[~pred & truth] + 0.7 * FN
        if H > W:   # portrait phone frame: a band of the strip's aspect around the head
            ys = np.where(truth.any(1))[0]; cy = int(ys.mean()) if len(ys) else H // 2; bh = int(W / ASPECT); y0 = int(np.clip(cy - bh // 2, 0, H - bh)); crop = img[y0:y0 + bh]
        else: crop = img[:int(H * 0.83)]
        ax.imshow(np.clip(crop, 0, 1)); ax.axis('off')
        ax.text(0.015, 0.95, label.replace(': ', ': "') + '"', fontsize=a.fs, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
        ax.text(0.985, 0.05, f'IoU {iou:.2f}', fontsize=a.fs, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=1.5, lw=0))
    print(f"[sites] {pk['survey']} {pk['frame']} id {pk['id']}: argmax IoU in the figure {iou:.3f} (protocol {pk['argmax_iou']:.3f}, survey median {pk['median']:.3f}); {'portrait' if H > W else 'landscape'}"
          + (f"; zoom crop {pk['zoom_box']} keeps {100 * pk['zoom_scored_kept']:.0f} % of scored pixels" if a.layout == 'zoom' else ''))
if a.layout == 'zoom' and a.fuji:   # the fruit level: our apples of one tree vs the GT apples, on the dataset's photograph (fuji_tree_panel.py)
    from matplotlib.patches import Circle
    fz = np.load(a.fuji, allow_pickle=True); ax = axes[NP - 1]; ph_ = fz['photo']; H, W = ph_.shape[:2]; u, v, cls, rp = fz['u'], fz['v'], fz['cls'], fz['rpx']
    y0, y1, x0, x1 = zoom_box(None, H, W, a.aspect, box=(int(np.floor((v - rp).min())), int(np.ceil((v + rp).max())), int(np.floor((u - rp).min())), int(np.ceil((u + rp).max()))))
    ax.imshow(np.clip(background(ph_)[y0:y1, x0:x1], 0, 1), interpolation='lanczos', aspect='auto', extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color('0.3')
    inc = (u >= x0) & (u < x1) & (v >= y0) & (v < y1); shown = [int((cls[inc] == c).sum()) for c in range(3)]   # a panel-aspect crop of a 3:2 photo can trim the top / bottom
    for o in np.argsort(-rp):   # far (small) discs last so they are not hidden under near ones
        ax.add_patch(Circle((u[o], v[o]), rp[o], facecolor=(*(TP, FP, FN)[cls[o]], 0.8), edgecolor='white', lw=0.35))
    ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5); tn = int(str(fz['tree']).replace('Tree', ''))
    ax.text(0.0, 1.0 + 0.25 * tb / ph, f'Fuji-SfM: apples of tree {tn}', fontsize=a.fs, va='bottom', ha='left', transform=ax.transAxes)
    ax.text(0.985, 0.035, f"F1 {float(fz['f1']):.2f}", fontsize=a.fs, color='white', weight='bold', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.65, pad=1.5, lw=0))
    print(f"[sites] Fuji-SfM {fz['tree']} on {fz['side']}/{fz['image']} ({fz['source']}): per-tree F1 {float(fz['f1']):.3f} from {fz['counts']} (P {float(fz['p']):.3f}, R {float(fz['r']):.3f}); "
          f"drawn TP/FP/FN {fz['drawn'].tolist()} of {fz['total'].tolist()} (the rest hidden from this view or out of frame), {shown} inside the panel's crop; crop {[y0, y1, x0, x1]}")
if a.layout == 'zoom':
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c, edgecolor='none') for c in (TP, FP, FN)], ['true positive', 'false positive', 'miss'], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=a.fs, frameon=False, handlelength=1.1, handleheight=0.9, handletextpad=0.4, columnspacing=1.4, borderaxespad=0.15)
json.dump(picks, open(FIG / 'sites' / 'picks.json', 'w'), indent=1); fig.savefig(FIG / a.out, dpi=a.dpi); print('[sites] wrote', FIG / a.out)
