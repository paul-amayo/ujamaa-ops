"""CONTAINMENT ladder eval — TESTING.md battery test 8 (added 2026-08-07 after
the block_003 replication shipped WITHOUT it; containment is the paper-highlight
test and must never again live only in scratch).

For one frame of a trained run: containment-score (own point + non-extrapolated
ancestor steps only — the OpenHype masks-inside-masks scoring) every requested
query word against its GT mask, best-IoU threshold sweep, 3-panel fig per query.

Queries are auto-derived from the frame's id map + hierarchy: every painted
tree, its fruit (if painted), and each tree's row. Words resolve through the
LIVE word table (get_word_for_id) — never literal words (the 2026-08-07
cassette/nectar stale-word false collapse).

Usage (pixi env):
  HIGH_EMBEDDER_CKPT=<embedder> pixi run python containment_eval.py \
    --config <run config.yml> --hyper-ckpt <embedder> \
    --hierarchy-json <hierarchy.json> --supervision-dir <id maps> \
    --frame kf_XXXXXX.png --kf-images <survey kf_images> --out <fig.png>
"""
import glob
import sys
FRUIT_ID_BASE = 10000   # must match compile_supervision.FRUIT_ID_BASE
import numpy as np
import torch
from PIL import Image
from pathlib import Path as _P
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import render_frame, build_hyper_embedder
from word_utils import get_word_for_id
from nerfstudio.utils.eval_utils import eval_setup
import lorentz as L
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dev = 'cuda'
torch.set_grad_enabled(False)
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--hyper-ckpt', required=True)
ap.add_argument('--hierarchy-json', required=True)
ap.add_argument('--supervision-dir', required=True)
ap.add_argument('--frame', required=True)
ap.add_argument('--kf-images', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
_, pipe, _, _step = eval_setup(_P(args.config))
model = pipe.model
model.eval()
# step=0 after eval_setup silently collapses SH to degree 0 (known gotcha)
# -> every fig backdrop rendered dusk-dark. Restore the trained step.
model.step = _step
ds = pipe.datamanager.train_dataset
hyper = build_hyper_embedder(args.hyper_ckpt, dev)
curv = hyper.curv.exp()
names = [_P(f).name for f in ds.image_filenames]
if args.frame not in names:
    # Supervision dirs can span a NEWER partition than this block's train
    # set (04's hygiene regen moved block boundaries; stage1-cached blocks
    # keep the old frame list) — the caller's most-painted pick may not be
    # a frame this model ever saw. Fall back to the most-painted
    # supervision frame the block DOES have instead of dying on .index.
    from PIL import Image as _Img
    have = set(names)
    best_n, best_f = -1, None
    for f in sorted(_P(args.supervision_dir).glob('kf_*.png')):
        if f.name not in have:
            continue
        n = int((np.array(_Img.open(f), np.uint16) != 65535).sum())
        if n > best_n:
            best_n, best_f = n, f.name
    if best_f is None:
        raise SystemExit(f'[containment] no supervision frame overlaps this '
                         f'block\'s train set ({args.frame} requested)')
    print(f'[containment] frame {args.frame} not in train set; '
          f'falling back to {best_f} ({best_n} painted px)')
    args.frame = best_f
i = names.index(args.frame)
cam = ds.cameras[i:i + 1].to(model.device)
cam.rescale_output_resolution(0.5)
rgb, alpha, feat = render_frame(model, cam, model.config.lang_field_dim)
print(f'[RGB PROBE] shape {rgb.shape} dtype {rgb.dtype} min {np.nanmin(rgb):.3f} '
      f'max {np.nanmax(rgb):.3f} mean {np.nanmean(rgb):.3f} NaN {np.isnan(rgb).sum()} '
      f'| alpha mean {np.nanmean(np.asarray(alpha)):.3f}')
# ALPHA NORMALISATION (2026-09-27, env CONTAIN_ALPHA_NORM=1): the rasterised feature is the alpha-weighted SUM of the
# visible gaussians' features, so wherever the accumulated alpha is below 1 (thin canopy, the denser H3DGS side-car
# geometry) the vector shrinks toward the hyperbolic origin and decodes as the coarser level (the row, not the tree).
# Dividing by the accumulation gives the alpha-weighted MEAN — the identity the pixel actually carries. Off by default
# (the battery test is unchanged); CONTAIN_DEBUG=1 prints per-label alpha / norm statistics either way.
import os as _os
_al = np.asarray(alpha, np.float32).reshape(feat.shape[0], feat.shape[1], -1)[..., 0]
_n0 = np.linalg.norm(np.asarray(feat), axis=-1)
if _os.environ.get('CONTAIN_ALPHA_NORM') == '1':
    feat = np.asarray(feat) / np.maximum(_al, 0.05)[..., None]
    print('[alpha norm] rendered features divided by the accumulated alpha (floor 0.05)')
Hh, Ww, _ = feat.shape
ft = torch.from_numpy(feat.reshape(-1, feat.shape[-1])).to(dev)
# NORM GATE (2026-08-19): a pixel whose rendered feature is ~zero carries NO
# identity — its gaussians were never assigned a census label (seed-only
# leaves ~2/3 of gaussians at zero) and it renders at the hyperbolic ORIGIN,
# which decodes closest to the ROW words (measured on 04: origin-decode sim
# pine 0.748 / oak 0.698 vs trees 0.28-0.48). Without this gate every
# unsupervised pixel answers the row query above threshold: the whole-frame
# row halo (pine prec 0.23 @ rec 1.0 on 04 and 05). "No identity" must score
# as nothing, not as the level that happens to live near the origin.
_norm = ft.norm(dim=-1)
_void = (_norm < 0.5).cpu().numpy().reshape(Hh, Ww)
print(f'[norm gate] {100.0 * _void.mean():.1f}% of pixels carry no identity '
      f'(rendered ||f|| < 0.5) — excluded from containment')
a = np.array(Image.open(_P(args.supervision_dir) / args.frame), np.uint16)
a = np.array(Image.fromarray(a).resize((Ww, Hh), Image.NEAREST))
if _os.environ.get('CONTAIN_DEBUG') == '1':
    for _u in [int(v) for v in np.unique(a) if v != 65535][:8]:
        _m = a == _u
        print(f'[debug] label {_u}: {int(_m.sum())} px, alpha mean {_al[_m].mean():.3f} p10 {np.percentile(_al[_m], 10):.3f}, '
              f'||f|| pre-norm mean {_n0[_m].mean():.2f} p10 {np.percentile(_n0[_m], 10):.2f}')
    _m = a == 65535
    print(f'[debug] unlabelled: {int(_m.sum())} px, alpha mean {_al[_m].mean():.3f}, ||f|| pre-norm mean {_n0[_m].mean():.2f}')


def containment_heat(word):
    t = model.image_encoder.tokenizer(word).to(dev)
    e = model.image_encoder.model.encode_text(t).float()
    e = e / e.norm(dim=-1, keepdim=True)
    out = []
    for j in range(0, ft.shape[0], 4096):
        sl = ft[j:j + 4096]
        h = L.exp_map0(sl, curv=curv)
        path, mask = L.get_interpolated_hyperbolic_features(
            h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
        d = hyper.decode_features(path, project=True)
        d = d / d.norm(dim=-1, keepdim=True)
        sm = (d @ e.T).view(sl.shape[0], 4)
        sm = sm.masked_fill(mask.to(sm.device).bool(), -1.0)
        d0 = hyper.decode_features(h, project=True)
        d0 = d0 / d0.norm(dim=-1, keepdim=True)
        own = (d0 @ e.T).squeeze(-1)
        out.append(torch.maximum(sm.max(1).values, own).cpu().numpy())
    h = np.concatenate(out).reshape(Hh, Ww)
    h[_void] = -1.0
    return h


def best_mask(heat, gt):
    if gt.sum() == 0:
        m = heat >= 0.99
        return m, 0.99, 0.0, 0.0, 0.0
    best = (0.0, None, None)
    # NOTE: if NO threshold overlaps GT, best stays None — report IoU 0.0
    # rather than crashing on (None & gt) (2026-08-17: masked 02's real
    # zero-overlap result behind a TypeError)
    for thr in np.concatenate([np.linspace(0.4, 0.95, 45),
                           np.linspace(0.955, 0.999, 12)]):
        m = heat >= thr
        inter = (m & gt).sum()
        union = (m | gt).sum()
        iou = inter / max(union, 1)
        if iou > best[0]:
            best = (iou, thr, m)
    iou, thr, m = best
    if m is None:
        # no threshold in the sweep overlapped GT at all: a real zero, not
        # an error — report it instead of crashing on (None & gt)
        m = np.zeros_like(gt, dtype=bool)
        thr = float('nan')
        return m, thr, 0.0, 0.0, 0.0
    tp = (m & gt).sum()
    return m, thr, iou, tp / max(m.sum(), 1), tp / max(gt.sum(), 1)


bg = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb
if np.nanmean(bg) > 0.97 or np.isnan(bg).any():
    bg = np.asarray(Image.open(
        _P(args.kf_images) / args.frame).convert('RGB').resize((Ww, Hh))).astype(np.float32) / 255.0
    print('[RGB PROBE] render unusable -> figs use the real keyframe')
rgb8 = (np.clip(bg, 0, 1) * 255).astype(np.uint8)

Hj = json.loads(_P(args.hierarchy_json).read_text())
row_of = {o['id']: o['row_id'] for o in Hj['objects']}
tree_of_fruit = {f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
# 0 <= u: id 0 is a REAL tree (the new census assigns global id 0 to each
# survey's first tree); the only void value in the supervision format is
# 65535. The old `0 < u` silently dropped id 0 from every query and every
# row union — kf_000007's 192k-px 'asus' canopy was invisible to the ladder
# while filling a fifth of the frame (Paul caught it, 2026-08-19).
uid = sorted(int(u) for u in np.unique(a) if 0 <= u < 65535)
trees = [u for u in uid if u < FRUIT_ID_BASE]
fruits = [u for u in uid if u >= FRUIT_ID_BASE]
cases = []
for t in trees:
    tf = [f for f in fruits if tree_of_fruit.get(f - FRUIT_ID_BASE) == t]
    gt_t = (a == t)
    for f in tf:
        gt_t = gt_t | (a == f)
    cases.append((get_word_for_id(t, 'mask'), f'TREE {t}', gt_t))
    for f in tf:
        cases.append((get_word_for_id(f - FRUIT_ID_BASE, 'fruit'), f'FRUIT of {t}', a == f))
for r in sorted({row_of[t] for t in trees if t in row_of}):
    gt_r = np.zeros_like(a, bool)
    for t in trees:
        if row_of.get(t) == r:
            gt_r |= (a == t)
            for f in fruits:
                if tree_of_fruit.get(f - FRUIT_ID_BASE) == t:
                    gt_r |= (a == f)
    cases.append((get_word_for_id(r, 'row'), f'ROW {r}', gt_r))
fig, ax = plt.subplots(len(cases), 3, figsize=(21, 4 * len(cases)), dpi=110)
for r, (word, kind, gt) in enumerate(cases):
    hm = containment_heat(word)
    m, thr, iou, prec, rec = best_mask(hm, gt)
    ov = rgb8.copy()
    ov[m & gt] = (0.2 * ov[m & gt] + 0.8 * np.array([0, 220, 90])).astype(np.uint8)
    ov[m & ~gt] = (0.2 * ov[m & ~gt] + 0.8 * np.array([255, 140, 0])).astype(np.uint8)
    ov[~m & gt] = (0.2 * ov[~m & gt] + 0.8 * np.array([230, 40, 40])).astype(np.uint8)
    ax[r, 0].imshow(m, cmap='gray')
    ax[r, 0].set_title(f'"{word}" ({kind}) — containment @ thr {thr:.2f}', loc='left', fontsize=11)
    gtv = np.zeros((Hh, Ww, 3), np.uint8)
    gtv[gt] = [0, 220, 90]
    ax[r, 1].imshow(gtv)
    ax[r, 1].set_title(f'GT ({int(gt.sum())} px)', loc='left', fontsize=11)
    ax[r, 2].imshow(ov)
    ax[r, 2].set_title(f'TP green / FP orange / FN red — IoU {iou:.3f} '
                       f'prec {prec:.3f} rec {rec:.3f}', loc='left', fontsize=11)
    print(f'{kind:18s} "{word}": thr {thr:.2f} IoU {iou:.3f} prec {prec:.3f} rec {rec:.3f}')
for aa in ax.ravel():
    aa.set_axis_off()
fig.suptitle(f'CONTAINMENT ladder — {args.frame}', x=0.05, ha='left', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(args.out, bbox_inches='tight')
print(f'SAVED {args.out}')
