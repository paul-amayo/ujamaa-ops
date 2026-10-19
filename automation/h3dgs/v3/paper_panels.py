#!/usr/bin/env python3
"""paper_panels.py (2026-10-17) — the panels of the HiGH paper's hero and method figures, one PNG each, from measured data of citrus 05 cell 1_0:
hero: image-feature field relevancy (LangSplat-lite) for three prompts and HiGH containment for three addresses on held-out frames;
method: the cell's scene graph, the radii of the addresses, a census frame's identity map against the seeded field, the containment walk of one pixel.
Inputs: figures/hero_heat_*.npz (hero_heat.py), lite_rel_*.npz (lite_relevancy.py), addresses_*.npz (address_embedding_dump.py), walk_probe_*.npz (walk_probe.py).
system python3:  paper_panels.py <figures dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
FIG = Path(sys.argv[1]); OUT = FIG / 'panels'; OUT.mkdir(exist_ok=True)
H = json.load(open('/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json')); obj_row = {int(o['id']): int(o['row_id']) for o in H['objects']}; fruits = H.get('fruits', [])
zt = np.load(FIG / 'hero_heat_05_1_0_kf002411.npz', allow_pickle=True); meta = json.loads(str(zt['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}; t_id = {w: k for k, w in wt.items()}
zf = np.load(FIG / 'hero_fruit_05_1_0_kf_000024.npz', allow_pickle=True); zl = np.load(FIG / 'lite_rel_05_1_0_kf002411.npz', allow_pickle=True); ad = np.load(FIG / 'addresses_05_1_0.npz', allow_pickle=True); wp = np.load(FIG / 'walk_probe_05_1_0_kf002411.npz', allow_pickle=True)
CM = plt.get_cmap('inferno')
SUP = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only')
def idmap(frame, size):
    a = np.array(Image.open(SUP / frame)); return np.array(Image.fromarray(a).resize(size, Image.NEAREST))
def iou(pred, truth):
    pred = pred.astype(bool); truth = truth.astype(bool); u = (pred | truth).sum(); return float((pred & truth).sum() / u) if u else 0.0
row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}
def overlay(rgb, h, floor, crop=None):
    img = rgb.astype(float) / 255.0; m = h >= floor; hh = np.clip((h - floor) / max(1e-6, float(h.max()) - floor), 0, 1); col = CM(0.35 + 0.65 * hh)[..., :3]; out = img * 0.45; out[m] = 0.25 * img[m] + 0.75 * col[m]
    y0, y1, x0, x1 = crop if crop else (0, 300, 0, rgb.shape[1]); return (out[y0:y1, x0:x1] * 255).astype(np.uint8), m
def save_panel(name, img, label, lit=None, w=5.2, h=2.45, metric='IoU'):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis('off')
    ax.text(0.012, 0.95, label, fontsize=24, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=3, lw=0))
    if lit is not None: ax.text(0.988, 0.05, f'{metric} {lit:.2f}', fontsize=20, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=2.5, lw=0))
    fig.savefig(OUT / f'{name}.png', dpi=150); plt.close(fig)
# ---- hero panels, top row: the photograph, an open query, the relevancy of every tree mask (CLIP mask-crop embedding, LERF relevancy)
mc = np.load(FIG / 'mask_clip_05_1_0_kf002411.npz', allow_pickle=True); photo = mc['photo']; masks = mc['masks']; prompts = list(mc['prompts']); rel = mc['relevancy']
ID_PHOTO = idmap('kf_002411.png', (photo.shape[1], photo.shape[0]))
def mask_panel(name, prompt, label, truth, floor=0.5):
    j = prompts.index(prompt); img = photo.astype(float) / 255.0; out = img * 0.5; litmask = np.zeros(photo.shape[:2], bool)
    for k in range(len(masks)):
        r = float(rel[k, j]); m = masks[k]
        if r >= floor: col = CM(0.35 + 0.65 * min(1.0, (r - floor) / 0.3))[:3]; out[m] = 0.3 * img[m] + 0.7 * np.array(col); litmask |= m
    nsel = int((rel[:, j] >= floor).sum()); lit = iou(litmask, truth)   # the masks ARE the supervision's geometry, so IoU is circular here: report the selection
    out = (out[:photo.shape[0] * 300 // 360] * 255).astype(np.uint8)
    fig = plt.figure(figsize=(5.2, 2.45)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(out); ax.axis('off')
    for k in range(len(masks)):
        ys, xs = np.where(masks[k][:out.shape[0]]); 
        if len(ys): ax.text(np.median(xs), np.median(ys), f'{rel[k, j]:.2f}', fontsize=19, color='white', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.55, pad=2, lw=0))
    ax.text(0.012, 0.95, label, fontsize=24, color='white', va='top', ha='left', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=3, lw=0))
    ax.text(0.988, 0.05, f'{nsel} of {len(masks)} trees selected', fontsize=20, color='white', va='bottom', ha='right', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.55, pad=2.5, lw=0)); fig.savefig(OUT / f'{name}.png', dpi=150); plt.close(fig); print(f'[sel] {name} {prompt}: {nsel} of {len(masks)} selected (IoU vs truth {lit:.3f}, circular)')
ALL_TREES = (ID_PHOTO != 65535) & (ID_PHOTO < 10000); T36 = ID_PHOTO == 36
mask_panel('hero_b', 'tree', 'query: "tree"', ALL_TREES); mask_panel('hero_c', 'the tree on the right', 'query: "the tree on the right"', T36); mask_panel('hero_d', 'cena', 'query: "tree 36" (its address)', T36)
def argmax_mask(z, words, target):
    heats = np.stack([z[w] for w in words]); return (heats.argmax(0) == words.index(target)) & ~z['void'].astype(bool)
TREE_WORDS = [wt[t] for t in sorted(wt) if t < 10000 and wt[t] in zt.files]; ROW_WORDS = [rw[r] for r in sorted(rw) if rw[r] in zt.files]
ID_T = idmap('kf_002411.png', (zt['rgb'].shape[1], zt['rgb'].shape[0])); ROW4 = np.isin(ID_T, [t for t, r in row_of.items() if r == 4]); T36m = ID_T == 36
for name, h, lab, truth, pred in (('hero_e', zt[rw[4]], 'query: "row 4"', ROW4, argmax_mask(zt, ROW_WORDS, rw[4])), ('hero_f', zt[wt[36]], 'query: "tree 36"', T36m, argmax_mask(zt, TREE_WORDS, wt[36]))):
    img, m = overlay(zt['rgb'], h, 0.85); v = iou(pred, truth); save_panel(name, img, lab, v); print(f'[iou] {name}: argmax {v:.3f} (heat >= 0.85 would give {iou(m, truth):.3f})')
ID_F = idmap('kf_000024.png', (zf['rgb'].shape[1], zf['rgb'].shape[0])); F1 = ID_F == 10001; cy, cx = 125, 136; FW = [wt[t] for t in sorted(wt) if (t >= 10000 or t < 10000) and wt[t] in zf.files]; predF = argmax_mask(zf, FW, wt[10001]); img, m = overlay(zf['rgb'], zf[wt[10001]], 0.85, crop=(max(0, cy - 70), cy + 70, max(0, cx - 150), cx + 150)); v = iou(predF, F1); save_panel('hero_g', img, 'query: "fruit 1"', v); print(f'[iou] hero_g fruit 1: {v:.3f} (fruit px in supervision {int(F1.sum())})')
save_panel('hero_a', (zt['rgb'][:300]).copy(), 'a held-out frame: which row, which tree, which fruit?', None)
# ---- method (a): the scene graph of the cell
cell_trees = sorted(k for k in wt if k < 10000); cell_fruit = sorted(k - 10000 for k in wt if k >= 10000); rows_present = sorted(set(obj_row[t] for t in cell_trees if t in obj_row and obj_row[t] >= 0)); unassigned = [t for t in cell_trees if obj_row.get(t, -1) < 0]
fig = plt.figure(figsize=(6.2, 3.0)); ax = fig.add_axes([0.01, 0.02, 0.98, 0.9]); tx = {}; x = 0.0; xr = {}
for r in rows_present:
    ts = [t for t in cell_trees if obj_row.get(t) == r]
    for t in ts: tx[t] = x; x += 1
    xr[r] = float(np.mean([tx[t] for t in ts])); x += 0.8
xc = float(np.mean(list(xr.values()))); ax.plot([xc], [3], 'k.', ms=8); ax.text(xc, 3.12, 'section', fontsize=8, ha='center')
for i, r in enumerate(rows_present): ax.plot([xc, xr[r]], [3, 2], '-', color='0.7', lw=0.6); ax.plot([xr[r]], [2], 's', color='#1b6ca8', ms=8); ax.text(xr[r], 2.16, f'row {r}', fontsize=7, ha='center')
for t, xx in tx.items(): ax.plot([xr[obj_row[t]], xx], [2, 1], '-', color='0.7', lw=0.6); ax.plot([xx], [1], 'o', color='#d98c3f', ms=5); ax.text(xx, 0.86, str(t), fontsize=5.5, ha='center', va='top', rotation=90)
fx = 0
for fi_ in cell_fruit:
    if fi_ < len(fruits) and int(fruits[fi_]['tree_id']) in tx:
        tid = int(fruits[fi_]['tree_id']); dx = (fx % 3 - 1) * 0.3; ax.plot([tx[tid], tx[tid] + dx], [1, 0], '-', color='0.7', lw=0.6); ax.plot([tx[tid] + dx], [0], '^', color='#2f7d32', ms=6); ax.text(tx[tid] + dx, -0.18, f'fruit {fi_}', fontsize=5.5, ha='center', va='top', rotation=90); fx += 1
ax.text(0.0, 3.45, f'{len(rows_present)} rows, {len(tx)} trees, {fx} fruit instances with an address' + (f'; {len(unassigned)} trees of this cell carry no row' if unassigned else ''), fontsize=7.5, ha='left'); ax.set_ylim(-1.3, 3.7); ax.set_xlim(-1, x); ax.axis('off'); fig.savefig(OUT / 'method_a.png', dpi=150); plt.close(fig)
# ---- method (b): radii
A = ad['tangent_raw']; lvl = ad['level']; G = zf['gaussian_tangent']; fig = plt.figure(figsize=(6.2, 3.0)); ax = fig.add_axes([0.12, 0.17, 0.85, 0.78]); bins = np.linspace(0, 9, 55)
for lv, col in (('row', '#1b6ca8'), ('tree', '#d98c3f'), ('fruit', '#2f7d32')): ax.hist(np.linalg.norm(A[lvl == lv], axis=1), bins=bins, color=col, alpha=0.9, label=f'{lv} addresses ({int((lvl == lv).sum())})')
ax.hist(np.linalg.norm(G, axis=1), bins=bins, color='0.55', alpha=0.35, weights=np.full(len(G), 40.0 / len(G)), label='seeded Gaussians (20k, scaled)')
for wpt in (2.77, 5.55, 8.32): ax.axvline(wpt, color='k', ls=':', lw=1.0)
ax.text(8.25, ax.get_ylim()[1] * 0.93, 'dotted: the waypoints of\nthe containment walk', fontsize=7, va='top', ha='right'); ax.set_xlabel('tangent norm  $\\|f\\|$  (= distance from the origin)', fontsize=8.5); ax.set_ylabel('number of addresses', fontsize=8.5); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7.5, frameon=False, loc='upper left'); fig.savefig(OUT / 'method_b.png', dpi=150); plt.close(fig)
# ---- method (c): census frame, identity map vs seeded field
zc = np.load(FIG / 'hero_heat_05_1_0_kf000025.npz', allow_pickle=True); sup = np.array(Image.open('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_native/chunk_1_0_sam3/supervision/trees_only/kf_000025.png')); sup = np.array(Image.fromarray(sup).resize((zc['rgb'].shape[1], zc['rgb'].shape[0]), Image.NEAREST))
twc = [w for k, w in sorted(wt.items()) if k < 10000 and w in zc.files]; stack = np.stack([zc[w] for w in twc], 0); amc = stack.argmax(0); amc[zc['void']] = -1
pal = plt.get_cmap('tab10')(np.arange(10))[:, :3]; base = zc['rgb'].astype(float) / 255.0; left = base * 0.55; right = base * 0.55; present = [k for k, w in enumerate(twc) if (sup == t_id[w]).sum() > 150]
for ci, k in enumerate(present):
    tid = t_id[twc[k]]; m1 = sup == tid; m2 = amc == k; left[m1] = 0.3 * base[m1] + 0.7 * pal[ci % 10]; right[m2] = 0.3 * base[m2] + 0.7 * pal[ci % 10]
for name, img, lab in (('method_c1', (left[:300] * 255).astype(np.uint8), 'census frame · identity map (supervision)'), ('method_c2', (right[:300] * 255).astype(np.uint8), 'census frame · the seeded field, read at the tree level')): save_panel(name, img, lab, None)
# ---- method (d): the walk of one pixel
wwords = list(wp['words']); i = 0; own = wp['cos_own'][i]; cp = np.asarray(wp['cos_path']).reshape(len(wp['pixels']), 4, -1)[i]; wn = wp['waypoint_norm'][i]; j_tree = wwords.index(wt[36]); r36 = obj_row[36]; j_row = wwords.index(rw[r36]); other = [wwords.index(rw[r]) for r in rw if r != r36 and rw[r] in wwords]
pts = [(float(wp['norm'][i]), float(own[j_tree]), float(own[j_row]), float(max(own[j] for j in other)), "the pixel's own point\n(tree depth)")] + [(float(wn[k]), float(cp[k, j_tree]), float(cp[k, j_row]), float(max(cp[k, j] for j in other)), 'the waypoint toward the origin\n(row depth)') for k in range(4) if wn[k] < wp['norm'][i] - 1e-3]
pts.sort(key=lambda t: -t[0]); xs = [p[0] for p in pts]; fig = plt.figure(figsize=(6.2, 3.0)); ax = fig.add_axes([0.13, 0.2, 0.84, 0.74])
ax.plot(xs, [p[1] for p in pts], 'o-', color='#d98c3f', ms=7, label='query: the address of tree 36 (the pixel\'s own tree)'); ax.plot(xs, [p[2] for p in pts], 's-', color='#1b6ca8', ms=7, label=f'query: the address of row {r36} (its row)'); ax.plot(xs, [p[3] for p in pts], 'x--', color='0.4', ms=7, label='the best other row')
for p in pts: ax.text(p[0], 0.30, p[4], fontsize=7, ha='center', va='bottom')
ax.invert_xaxis(); ax.set_xlabel("distance from the origin along the pixel's geodesic", fontsize=8.5); ax.set_ylabel('cosine to the query', fontsize=8.5); ax.set_ylim(0, 1.08); ax.set_xlim(4.9, 2.3); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7, frameon=False, loc='lower center'); fig.savefig(OUT / 'method_d.png', dpi=150); plt.close(fig)
print('panels written:', sorted(p.name for p in OUT.iterdir())); print('walk points:', [(round(p[0], 2), round(p[1], 2), round(p[2], 2), round(p[3], 2)) for p in pts])
