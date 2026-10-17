#!/usr/bin/env python3
"""method_fig.py (2026-10-17) — the HiGH paper's method figure in the layered scene-graph presentation of Osiris++ (its Fig. 3): (a) the scene
graph of one cell (citrus 05, cell 1_0) as layers over the cell's Gaussians, from the bottom: the Gaussians coloured by the address the census
gave them, the fruit instances, the trees at their scene-graph markers, the rows and the section, with the parent edges and the grounding edge
from each tree node to the centroid of its Gaussians; (b) the same nodes in the embedding: the measured tangent radius against the measured angle
to the parent's ray, with the rows laid out at equal angles (the one laid-out quantity). Inputs: figures/gaussians_05_1_0.npz (method_fig_dump.py,
Gaussians in the scene graph's frame), figures/addresses_05_1_0.npz (hero_heat.py --dump-addresses, raw CLIP text -> tangent), the scene graph
and the cell's text bank. system python3:  method_fig.py <paper dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
P = Path(sys.argv[1]); F = P / 'figures'
G = np.load(F / 'gaussians_05_1_0.npz'); A = np.load(F / 'addresses_05_1_0.npz', allow_pickle=True); TAN = {str(w): A['tangent_raw'][i] for i, w in enumerate(A['words'])}
H = json.load(open('/home/paperspace/data/citrus_all/05_13D_Jackal/prod/bateleur/scene_graph/marker_hierarchy.json'))
meta = json.loads(str(np.load('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity/text_bank.npz', allow_pickle=True)['meta']))
wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
objs = {int(o['id']): o for o in H['objects']}; fruits = {int(f['id']): f for f in H['fruits']}
cell_trees = sorted(k for k in wt if k < 10000 and k in objs); cell_fruit = sorted(k - 10000 for k in wt if k >= 10000 and (k - 10000) in fruits and int(fruits[k - 10000]['tree_id']) in wt)
row_of = {t: int(objs[t]['row_id']) for t in cell_trees}; rows = sorted({r for r in row_of.values() if r >= 0}); sec_of = {t: int(objs[t]['section_id']) for t in cell_trees}
xyz = G['xyz']; lab = G['label']
M = np.array([[objs[t]['xyz'][0], objs[t]['xyz'][2]] for t in cell_trees]); r0 = max(rows, key=lambda r: sum(1 for t in cell_trees if row_of[t] == r))
Mr = np.array([[objs[t]['xyz'][0], objs[t]['xyz'][2]] for t in cell_trees if row_of[t] == r0]); w_, v_ = np.linalg.eigh(np.cov((Mr - Mr.mean(0)).T)); d = v_[:, -1]; e = np.array([-d[1], d[0]]); c0 = M.mean(0)
def ground(p): q = np.asarray(p)[..., [0, 2]] - c0; return np.stack([q @ d, q @ e], -1)   # LIO x, z on the ground, rows along the first axis; y is down
gm = ground(xyz); up = -xyz[:, 1]; mk = {t: ground(np.array(objs[t]['xyz'])) for t in cell_trees}; gmk = np.array([mk[t] for t in cell_trees])
keep = (gm[:, 0] > gmk[:, 0].min() - 3) & (gm[:, 0] < gmk[:, 0].max() + 3) & (gm[:, 1] > gmk[:, 1].min() - 3) & (gm[:, 1] < gmk[:, 1].max() + 3) & (up > -1.0) & (up < 4.5)
gm, up, lab = gm[keep], up[keep], lab[keep]; print(f'[method] {int(keep.sum())} of {len(keep)} sampled gaussians in the cell crop; {len(cell_trees)} trees, {len(rows)} rows, {len(cell_fruit)} fruit instances, sections {sorted(set(sec_of.values()))}')
ROWCOL = {r: c for r, c in zip(rows, ['#1b6ca8', '#d98c3f', '#6a994e', '#9b4dca', '#c8102e', '#17becf', '#8c564b', '#e377c2', '#bcbd22', '#7f7f7f'])}; ORANGE = '#ff7f0e'; GREY = (0.80, 0.80, 0.80)
cols = np.tile(np.array(GREY), (len(lab), 1))
for t in cell_trees:
    if row_of[t] >= 0: cols[lab == t] = matplotlib.colors.to_rgb(ROWCOL[row_of[t]])
cols[lab >= 10000] = matplotlib.colors.to_rgb(ORANGE); print(f'[method] gaussians with a tree address {int(np.isin(lab, cell_trees).sum())}, fruit {int((lab >= 10000).sum())}, none {int((lab < 0).sum())}')
Z = {'gauss': 0.0, 'fruit': 1.0, 'tree': 2.0, 'row': 3.0, 'section': 4.0}
fig = plt.figure(figsize=(7.2, 3.6)); ax = fig.add_axes([0.12, -0.06, 0.52, 1.12], projection='3d'); ax.computed_zorder = False
order = np.argsort(lab >= 0)   # labelled gaussians drawn last
ax.scatter(gm[order, 0], gm[order, 1], np.zeros(len(gm)), s=0.25, c=cols[order], alpha=0.55, linewidths=0, depthshade=False, rasterized=True, zorder=1)
x0, x1 = gm[:, 0].min(), gm[:, 0].max(); y0, y1 = gm[:, 1].min(), gm[:, 1].max()
for name, z in Z.items():
    if name != 'gauss': ax.add_collection3d(Poly3DCollection([[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]], facecolor='0.5', alpha=0.08, edgecolor='0.6', lw=0.5, zorder=2 + 2 * z))
LABELS = {'gauss': 'Gaussians of the splat,\ncoloured by their\ncensus address', 'fruit': 'fruit instances', 'tree': 'trees', 'row': 'rows', 'section': 'section'}
pos = {}
for t in cell_trees:
    m = lab == t; pos[('tree', t)] = (mk[t][0], mk[t][1], Z['tree'])
    if m.sum() >= 5: cg = gm[m].mean(0); ax.plot([mk[t][0], cg[0]], [mk[t][1], cg[1]], [Z['tree'], 0], color='0.55', lw=0.35, ls=':', zorder=3)
for f in cell_fruit:
    t = int(fruits[f]['tree_id']); pos[('fruit', f)] = (mk[t][0], mk[t][1], Z['fruit']); ax.plot([mk[t][0], mk[t][0]], [mk[t][1], mk[t][1]], [Z['fruit'], Z['tree']], color='0.45', lw=0.5, zorder=4)
    m = lab == 10000 + f
    if m.sum() >= 3: cg = gm[m].mean(0); ax.plot([mk[t][0], cg[0]], [mk[t][1], cg[1]], [Z['fruit'], 0], color=ORANGE, lw=0.35, ls=':', zorder=3)
rpos = {}
for r in rows:
    ts = [t for t in cell_trees if row_of[t] == r]; q = np.mean([mk[t] for t in ts], 0); rpos[r] = (q[0], q[1], Z['row'])
    for t in ts: ax.plot([mk[t][0], q[0]], [mk[t][1], q[1]], [Z['tree'], Z['row']], color='0.45', lw=0.5, zorder=6)
secs = sorted(s for s in set(sec_of.values()) if s >= 0); spos = {}
for s in secs:
    rs = [r for r in rows if any(sec_of[t] == s and row_of[t] == r for t in cell_trees)]
    if not rs: continue
    q = np.mean([rpos[r][:2] for r in rs], 0); spos[s] = (q[0], q[1], Z['section'])
    for r in rs: ax.plot([rpos[r][0], q[0]], [rpos[r][1], q[1]], [Z['row'], Z['section']], color='0.45', lw=0.5, zorder=8)
for (kind, k), (x, y, z) in pos.items(): ax.scatter([x], [y], [z], s=14 if kind == 'tree' else 9, c=[ROWCOL[row_of[k]] if kind == 'tree' and row_of[k] >= 0 else (ORANGE if kind == 'fruit' else '0.4')], edgecolors='k', linewidths=0.3, depthshade=False, zorder=7 if kind == 'tree' else 5)
for r, (x, y, z) in rpos.items(): ax.scatter([x], [y], [z], s=30, marker='s', c=[ROWCOL[r]], edgecolors='k', linewidths=0.4, depthshade=False, zorder=9)
for s, (x, y, z) in spos.items(): ax.scatter([x], [y], [z], s=44, marker='D', c=['0.25'], edgecolors='k', linewidths=0.4, depthshade=False, zorder=11)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(0, 4.6); ax.set_box_aspect((1.0, (y1 - y0) / (x1 - x0), 0.75)); ax.view_init(elev=24, azim=-62); ax.set_axis_off()
ax.plot([x0, x0 + 10], [y0 - 1.5, y0 - 1.5], [0, 0], color='k', lw=1.0); ax.text(x0 + 5, y0 - 1.5, -0.35, '10 m', fontsize=7, ha='center', va='top')
from mpl_toolkits.mplot3d import proj3d
fig.canvas.draw()
for name, z in Z.items():   # each layer's label at the corner of its plane that lands leftmost on the page
    corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]; scr = [proj3d.proj_transform(*c, ax.get_proj())[:2] for c in corners]; i = int(np.argmin([sx for sx, sy in scr]))
    ax.annotate(LABELS[name], xy=scr[i], xycoords='data', xytext=(-4, 0), textcoords='offset points', fontsize=7.5, ha='right', va='center', annotation_clip=False)
fig.text(0.01, 0.95, '(a)', fontsize=9, weight='bold')
# (b) the embedding: measured radius, measured angle to the parent's ray; rows laid out at equal angles
axb = fig.add_axes([0.66, 0.02, 0.33, 0.86], projection='polar'); rng = np.random.default_rng(0); ref = rng.standard_normal(32)
def ang(child, parent, sign_ref):
    cp = np.clip(child @ parent / np.linalg.norm(child) / np.linalg.norm(parent), -1, 1); a = np.arccos(cp); u = sign_ref - (sign_ref @ parent) * parent / (parent @ parent); return a * (1 if child @ u >= 0 else -1)
rad = {'row': [], 'tree': [], 'fruit': []}; angs = {'tree': [], 'fruit': []}; rmax = 0
for i, r in enumerate(rows):
    th_r = 2 * np.pi * i / len(rows); Tr = TAN[rw[r]]; nr = np.linalg.norm(Tr); rad['row'].append(nr)
    axb.plot([th_r, th_r], [0, 9.0], color=ROWCOL[r], lw=0.6, ls=':', zorder=1); axb.scatter([th_r], [nr], s=30, marker='s', c=[ROWCOL[r]], edgecolors='k', linewidths=0.4, zorder=5)
    for t in [t for t in cell_trees if row_of[t] == r]:
        Tt = TAN[wt[t]]; nt = np.linalg.norm(Tt); a_t = ang(Tt, Tr, ref); rad['tree'].append(nt); angs['tree'].append(abs(a_t)); th_t = th_r + a_t
        axb.plot([th_r, th_t], [nr, nt], color=ROWCOL[r], lw=0.4, zorder=2); axb.scatter([th_t], [nt], s=11, c=[ROWCOL[r]], edgecolors='k', linewidths=0.3, zorder=6)
        for f in [f for f in cell_fruit if int(fruits[f]['tree_id']) == t]:
            Tf = TAN[wt[10000 + f]]; nf = np.linalg.norm(Tf); a_f = ang(Tf, Tt, ref); rad['fruit'].append(nf); angs['fruit'].append(abs(a_f)); th_f = th_t + a_f; rmax = max(rmax, nf)
            axb.plot([th_t, th_f], [nt, nf], color=ORANGE, lw=0.4, zorder=2); axb.scatter([th_f], [nf], s=9, c=[ORANGE], edgecolors='k', linewidths=0.3, zorder=6)
for wpt in (2.77, 5.55, 8.32): axb.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, wpt), color='0.6', lw=0.5, ls='--', zorder=0)
axb.set_rmax(max(9.0, rmax + 0.5)); axb.set_rticks([2, 4, 6, 8]); axb.set_rlabel_position(100); axb.tick_params(labelsize=6.5); axb.set_xticks([]); axb.grid(alpha=0.25)
for name, lab_ in (('row', 'rows'), ('tree', 'trees'), ('fruit', 'fruit')): axb.text(np.deg2rad(282), float(np.median(rad[name])), f"{lab_} {np.min(rad[name]):.1f}–{np.max(rad[name]):.1f}", fontsize=6.5, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, pad=1, lw=0), zorder=9)
axb.text(np.deg2rad(282), 2.77 + 0.0, '', fontsize=6)
fig.text(0.655, 0.95, '(b)', fontsize=9, weight='bold'); fig.text(0.84, 0.955, 'radius ‖v‖ and angle to the parent\'s ray, both measured;\nrow directions laid out at equal angles', fontsize=7, va='center', ha='center')
print(f"[method] radii rows {np.min(rad['row']):.2f}-{np.max(rad['row']):.2f} (median {np.median(rad['row']):.2f}), trees {np.min(rad['tree']):.2f}-{np.max(rad['tree']):.2f} ({np.median(rad['tree']):.2f}), fruit {np.min(rad['fruit']):.2f}-{np.max(rad['fruit']):.2f} ({np.median(rad['fruit']):.2f})")
print(f"[method] angle tree->row: median {np.degrees(np.median(angs['tree'])):.1f} deg, max {np.degrees(np.max(angs['tree'])):.1f}; fruit->tree median {np.degrees(np.median(angs['fruit'])):.2f} deg, max {np.degrees(np.max(angs['fruit'])):.2f}")
oth = [min(np.degrees(np.arccos(np.clip(TAN[wt[t]] @ TAN[rw[r]] / np.linalg.norm(TAN[wt[t]]) / np.linalg.norm(TAN[rw[r]]), -1, 1))) for r in rows if r != row_of[t]) for t in cell_trees if row_of[t] >= 0]
print(f"[method] angle tree->nearest other row: min {np.min(oth):.1f} deg, median {np.median(oth):.1f}")
fig.savefig(F / 'fig_method.pdf', dpi=300); fig.savefig(F / 'fig_method.png', dpi=130); print('[method] wrote', F / 'fig_method.pdf')
