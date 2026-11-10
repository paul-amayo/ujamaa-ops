#!/usr/bin/env python3
"""hero_grid.py (2026-11-10) — the HiGH letter's Fig. 1 as one block (Paul, draft 36): four queries as rows, the Euclidean
open-vocabulary field on the left (the lifted CLIP field of the open-vocabulary comparison: SAM 3 mask-crop CLIP features lifted onto
the Gaussians, PCA-32, no labels, read by LERF relevancy >= 0.5) and HiGH on the right (the containment field read by the argmax rule
of the protocol), on held-out frames of citrus 05 cell 1_0:
  row 1  "tree"                     kf_002411   left: relevancy of "tree"                  right: the tree level, every tree by its address
  row 2  "the tree on the right" /  kf_002411   left: relevancy of "the tree on the right"   right: the address of tree 36
         tree 36
  row 3  "row 4"                    kf_002411   left: relevancy of "row 4"                 right: the address of row 4
  row 4  "fruit on tree 5"          kf_000024   left: relevancy of "fruit on tree 5"       right: the fruit addresses under tree 5
IoU against the supervision of the frame (the trees_only identity maps) is printed on every panel and written to
<paper>/numbers/hero_grid.json.
Stage 1 (GPU, nerf_new pixi env via subprocess): splat_feature_render.py of features_lite.bin for the frames without a lite map, and
lite_relevancy.py for the prompts, both into --work (nothing is written into the identity directory).
Stage 2 (system python3): compose figures/fig_hero.pdf at column width (3.5 in) with 8 pt labels.
usage:  hero_grid.py <paper dir> --work /tmp/d36hero [--stage both|1|2] [--figdir <dir holding hero_heat_*.npz, hero_fruit_*.npz>]"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('paper'); ap.add_argument('--work', default='/tmp/d36hero'); ap.add_argument('--stage', default='both', choices=['both', '1', '2'])
ap.add_argument('--figdir', default='/home/paperspace/code/aru_sil_core/src/docs/paper/figures'); a = ap.parse_args()
P = Path(a.paper); W = Path(a.work); W.mkdir(parents=True, exist_ok=True); FD = Path(a.figdir)
N = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); A = N / 'arms'; V3 = Path('/home/paperspace/code/automation/h3dgs/v3'); NS = '/home/paperspace/code/nerf_new'
SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
HIER = '/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json'
F_T, F_F = 'kf_002411.png', 'kf_000024.png'
PROMPTS = {F_T: ['tree', 'the tree on the right', 'row 4'], F_F: ['fruit on tree 5']}
if a.stage in ('both', '1'):
    cfg = json.load(open(N / 'W.json'))['config']; ws = N.parent; sparse = ws / 'sparse/0'
    if not (sparse / 'images.bin').exists(): sparse = Path(cfg).parents[3] / 'sparse/0'
    (W / 'maps_lite').mkdir(exist_ok=True)
    for fr in PROMPTS:
        src = A / 'maps_lite' / fr.replace('.png', '.npz'); dst = W / 'maps_lite' / src.name
        if src.exists() and not dst.exists(): dst.symlink_to(src)
        if not dst.exists():
            r = subprocess.run(f'cd {NS} && pixi run python {V3}/splat_feature_render.py --config {cfg} --features {A}/features_lite.bin --frames {fr} --out-dir {W}/maps_lite --colmap-sparse {sparse}', shell=True, capture_output=True, text=True)
            print('[grid] render', fr, 'rc', r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:])
    # lite_relevancy.py reads <identity>/arms/maps_lite; run a copy whose map directory is the work directory
    src = (V3 / 'lite_relevancy.py').read_text().replace("z = np.load(N / 'arms/maps_lite' / a.frame.replace('.png', '.npz'))", f"z = np.load(Path('{W}/maps_lite') / a.frame.replace('.png', '.npz'))")
    assert f"Path('{W}/maps_lite')" in src; (W / 'lite_relevancy_work.py').write_text(src)
    for fr, pr in PROMPTS.items():
        out = W / f"lite_{fr.replace('.png', '')}.npz"
        r = subprocess.run(f'cd {NS} && pixi run python {W}/lite_relevancy_work.py --identity {N} --frame {fr} --prompts "{";".join(pr)}" --out {out}', shell=True, capture_output=True, text=True)
        print('[grid] lite', fr, 'rc', r.returncode, '\n  ' + '\n  '.join((r.stdout + r.stderr).strip().splitlines()[-4:]))
if a.stage == '1': raise SystemExit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
H = json.load(open(HIER)); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}; fruit_tree = {int(f['id']): int(f['tree_id']) for f in H['fruits']}
def idmap(frame, shape): return np.array(Image.fromarray(np.array(Image.open(SUP / frame))).resize((shape[1], shape[0]), Image.NEAREST))
def iou(p, t): p = p.astype(bool); t = t.astype(bool); u = (p | t).sum(); return float((p & t).sum() / u) if u else 0.0
def argmax_mask(z, words, targets):
    heats = np.stack([z[w] for w in words]); am = heats.argmax(0); return np.isin(am, [words.index(t) for t in targets]) & ~z['void'].astype(bool), am
zt = np.load(FD / 'hero_heat_05_1_0_kf002411.npz', allow_pickle=True); zf = np.load(FD / 'hero_fruit_05_1_0_kf_000024.npz', allow_pickle=True)
wt = {int(k): w for k, w in json.loads(str(zt['meta']))['word_table'].items()}; rw = {int(k): w for k, w in json.loads(str(zt['meta']))['row_words'].items()}
lt = np.load(W / 'lite_kf_002411.npz', allow_pickle=True); lf = np.load(W / 'lite_kf_000024.npz', allow_pickle=True)
TREE_WORDS = [wt[t] for t in sorted(wt) if t < 10000 and wt[t] in zt.files]; ROW_WORDS = [rw[r] for r in sorted(rw) if rw[r] in zt.files]
ID_T = idmap(F_T, zt['rgb'].shape[:2]); ID_F = idmap(F_F, zf['rgb'].shape[:2]); ID_TL = idmap(F_T, lt['rgb'].shape[:2]); ID_FL = idmap(F_F, lf['rgb'].shape[:2])
TREES = lambda I: (I < 10000)
ROW4 = lambda I: np.isin(I, [t for t, r in row_of.items() if r == 4])
F5_IDS = [10000 + f for f, t in fruit_tree.items() if t == 5]; F5 = lambda I: np.isin(I, F5_IDS)
# HiGH, argmax rule of the protocol
p1, am1 = argmax_mask(zt, TREE_WORDS, TREE_WORDS)                                   # the tree level: every non-void pixel to its tree
p2, _ = argmax_mask(zt, TREE_WORDS, [wt[36]]); p3, _ = argmax_mask(zt, ROW_WORDS, [rw[4]])
FW = [w for k, w in sorted(wt.items()) if w in zf.files]; f5w = [wt[k] for k in F5_IDS if k in wt and wt[k] in zf.files]
p4, _ = argmax_mask(zf, FW, f5w)                                                    # the mixed rule of the letter's fruit panel: argmax over every address
R = {'tree': dict(lite=iou(lt['tree'] >= 0.5, TREES(ID_TL)), high=iou(p1, TREES(ID_T))),
     'tree 36': dict(lite=iou(lt['the tree on the right'] >= 0.5, ID_TL == 36), high=iou(p2, ID_T == 36)),
     'row 4': dict(lite=iou(lt['row 4'] >= 0.5, ROW4(ID_TL)), high=iou(p3, ROW4(ID_T))),
     'fruit of tree 5': dict(lite=iou(lf['fruit on tree 5'] >= 0.5, F5(ID_FL)), high=iou(p4, F5(ID_F)), fruit_ids=F5_IDS, fruit_addressed=f5w)}
for k, v in R.items(): print(f'[grid] {k:16s} lifted CLIP IoU {v["lite"]:.3f}   HiGH IoU {v["high"]:.3f}')
json.dump(dict(cell='05_13D_Jackal chunk_1_0', frames=[F_T, F_F], rule='HiGH: argmax of the protocol (fruit: over every address); lifted CLIP: LERF relevancy >= 0.5',
               prompts_lifted={'tree': 'tree', 'tree 36': 'the tree on the right', 'row 4': 'row 4', 'fruit of tree 5': 'fruit on tree 5'}, iou=R), open(P / 'numbers' / 'hero_grid.json', 'w'), indent=1)
# compose: 4 rows x 2 columns at column width
CM = plt.get_cmap('inferno'); LIT = np.array([1.0, 0.78, 0.1]); PAL = plt.get_cmap('tab10')(np.arange(10))[:, :3]
def crop(img, fr): return img[: int(img.shape[0] * (0.83 if fr == F_T else 0.83))]
def lite_img(z, key):
    img = z['rgb'].astype(float) / 255; h = z[key]; m = h >= 0.5; out = img * 0.45; hh = np.clip((h - 0.5) / 0.5, 0, 1); col = CM(0.45 + 0.55 * hh)[..., :3]; out[m] = 0.25 * img[m] + 0.75 * col[m]; return out
def high_img(z, m, per_id=None):
    img = z['rgb'].astype(float) / 255; out = img * 0.45
    if per_id is None: out[m] = 0.25 * img[m] + 0.75 * LIT
    else:
        for j, k in enumerate(np.unique(per_id[m])): mm = m & (per_id == k); out[mm] = 0.25 * img[mm] + 0.75 * PAL[j % 10]
    return out
rows = [('"tree"', lite_img(lt, 'tree'), high_img(zt, p1, am1), F_T, '"tree"', 'tree level'),
        ('"tree 36"', lite_img(lt, 'the tree on the right'), high_img(zt, p2), F_T, '"the tree on the right"', 'tree 36'),
        ('"row 4"', lite_img(lt, 'row 4'), high_img(zt, p3), F_T, '"row 4"', 'row 4'),
        ('"fruit on tree 5"', lite_img(lf, 'fruit on tree 5'), high_img(zf, p4), F_F, '"fruit on tree 5"', 'fruit of tree 5')]
keys = ['tree', 'tree 36', 'row 4', 'fruit of tree 5']
plt.rcParams.update({'font.size': 8}); fig, axes = plt.subplots(4, 2, figsize=(3.5, 3.55)); fig.subplots_adjust(0.0, 0.0, 1.0, 0.94, 0.02, 0.06)
for i, (q, li, hi, fr, lq, hq) in enumerate(rows):
    for j, (img, lab, v) in enumerate(((li, lq, R[keys[i]]['lite']), (hi, hq, R[keys[i]]['high']))):
        ax = axes[i, j]; ax.imshow(np.clip(crop(img, fr), 0, 1)); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.text(0.02, 0.96, lab, fontsize=7.5, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, pad=1.2, lw=0))
        ax.text(0.98, 0.04, f'IoU {v:.2f}', fontsize=7.5, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.6, pad=1.2, lw=0))
fig.text(0.25, 0.985, 'Euclidean: lifted CLIP field', ha='center', va='top', fontsize=8.5, weight='bold'); fig.text(0.75, 0.985, 'hyperbolic: HiGH', ha='center', va='top', fontsize=8.5, weight='bold')
(P / 'figures').mkdir(exist_ok=True); fig.savefig(P / 'figures' / 'fig_hero.pdf', dpi=300); fig.savefig(P / 'figures' / 'fig_hero.png', dpi=300); print('[grid] wrote', P / 'figures' / 'fig_hero.pdf')
