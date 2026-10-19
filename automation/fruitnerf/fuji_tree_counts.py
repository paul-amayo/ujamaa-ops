#!/usr/bin/env python3
"""fuji_tree_counts.py [--merged <npz>] [--tag plain30k] [--fig <png>] (2026-10-16) — per-tree apple counts on Fuji-SfM through the hierarchy's tree
anchors, scored against the 11 GT trees. Rules (GT-blind, pre-registered in the notebook before the run): trunk gaussians from the SAM3
'tree trunk' census of each side, mapped to the dataset frame through align_flip; our ground = p2 of our splat means' height within the row;
trunk band 0.3-1.5 m above it; 1-D KDE along the row (one vote per gaussian, bandwidth 0.10 m, grid 0.02 m); peaks >= 10 % of the maximum,
non-max suppression 0.5 m; posts = peaks with >= 30 trunk gaussians above 4.5 m (a column that continues above the canopy = hail-net pole) dropped; both sides' peaks merged within 0.3 m;
each merged apple (fuji_eval.py --save-merged: one-to-one east<->west within 10 cm, GT region) -> the nearest trunk along the row.
Scoring: GT trees <-> our trunks by nearest median along-row position (one-to-one, table only); per tree our count vs GT count; on the
apples matched to a GT apple at 10 cm, the share whose trunk is the GT apple's tree; per-tree P / R / F1 under our labels. system python3."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
ap = argparse.ArgumentParser(); ap.add_argument('--merged', default='/home/paperspace/data/public/fuji_sfm/merged_flip_r10.npz'); ap.add_argument('--tag', default='plain30k'); ap.add_argument('--align', default='align_flip'); ap.add_argument('--keep-posts', action='store_true', help='keep pole columns as tree anchors (post-hoc variant, 10-16: the Tree11 trunk shares its column with a hail-net pole)'); ap.add_argument('--out-tag', default='record'); a = ap.parse_args(); FIG = f'/home/paperspace/data/public/fuji_sfm/tree_counts_flip_{a.out_tag}.png'; JS = f'/home/paperspace/data/public/fuji_sfm/tree_counts_flip_{a.out_tag}.json'
ROOT = Path('/home/paperspace/data/public/fuji_sfm'); R = ROOT / 'high'
cloud = np.load(ROOT / 'their_cloud_500k.npy').astype(np.float64); c0 = cloud.mean(0); w_, v_ = np.linalg.eigh(np.cov((cloud[:, :2] - c0[:2]).T)); row2 = v_[:, -1]; acr2 = np.array([-row2[1], row2[0]])
def rowframe(P): return np.stack([(P[:, :2] - c0[:2]) @ row2, (P[:, :2] - c0[:2]) @ acr2, P[:, 2]], 1)
def to_gt(side, X):
    S = R / side; z = np.load(S / f'fruit_registry/{a.align}.npz'); s, Rm, t = float(z['s']), z['R'], z['t']; dpt = json.loads((S / 'outputs/sf_30000_bilateral/splatfacto/run/dataparser_transforms.json').read_text()); T = np.array(dpt['transform']); sc = float(dpt['scale'])
    return (s * (Rm @ (((X / sc) - T[:, 3]) @ T[:, :3]).T)).T + t
gx = np.arange(-9, 9.001, 0.02); peaks_all = []
for side in ('east', 'west'):
    S = R / side; tc = np.load(S / f'fruit_registry/trunk_census_{a.tag}.npz'); Trf = rowframe(to_gt(side, tc['P'].astype(np.float64)))
    Mrf = rowframe(to_gt(side, np.load(S / 'fruit_registry/means_plain30k.npy').astype(np.float64))); ground = np.percentile(Mrf[np.abs(Mrf[:, 1]) < 1.5, 2], 2); h = Trf[:, 2] - ground
    band = (h >= 0.3) & (h <= 1.5) & (np.abs(Trf[:, 1]) < 1.5); x = Trf[band, 0]; dens = np.zeros(len(gx))
    for s0 in range(0, len(x), 20000): dens += np.exp(-0.5 * ((gx[:, None] - x[None, s0:s0 + 20000]) / 0.10) ** 2).sum(1)
    pk = (dens == maximum_filter1d(dens, size=51)) & (dens >= 0.1 * dens.max()); cand = gx[pk]; cv = dens[pk]; o = np.argsort(-cv); keep = []
    for i in o:
        if all(abs(cand[i] - cand[j]) > 0.5 for j in keep): keep.append(i)
    keep = sorted(keep, key=lambda i: cand[i]); peaks = []
    for i in keep:
        near = (np.abs(Trf[:, 0] - cand[i]) < 0.25) & (np.abs(Trf[:, 1]) < 0.6); top = np.percentile(h[near], 95) if near.sum() > 20 else 0; n_above = int((near & (h > 4.5)).sum()); post = n_above >= 30   # a 'trunk' column that continues above the canopy (> 4.5 m) is a hail-net pole
        peaks.append((cand[i], cv[i] / cv.max(), top, post))
    print(f"[trees] {side}: {len(tc['P'])} trunk gaussians ({int(band.sum())} in the 0.3-1.5 m band; our ground z {ground:.2f}); {len(peaks)} peaks: " + ' '.join(f"{p:+.2f}({'POST' if post else f'{top:.1f}m'})" for p, v, top, post in peaks))
    peaks_all += [(p, side) for p, v, top, post in peaks if (not post) or a.keep_posts]
# merge the sides' trunks within 0.3 m
ps = sorted(peaks_all); trunks = []
for p, side in ps:
    if trunks and abs(p - trunks[-1][0]) <= 0.3 and side not in trunks[-1][1]: trunks[-1] = ((trunks[-1][0] * len(trunks[-1][1]) + p) / (len(trunks[-1][1]) + 1), trunks[-1][1] + [side])
    else: trunks.append((p, [side]))
TR = np.array([t_[0] for t_ in trunks]); print(f"[trees] merged trunks ({len(TR)}): " + ' '.join(f"{p:+.2f}[{''.join(s_[0] for s_ in sides)}]" for p, sides in trunks))
M = np.load(a.merged); P = M['P']; pair = M['pair']; GT = M['GT']; GT_tree = M['GT_tree']; Prf = rowframe(P); GTrf = rowframe(GT)
assign = np.argmin(np.abs(Prf[:, 0][:, None] - TR[None, :]), axis=1)
trees = sorted(set(GT_tree)); gmed = np.array([np.median(GTrf[GT_tree == t_, 0]) for t_ in trees])
# one-to-one GT tree <-> trunk by nearest median (table only)
d = np.abs(gmed[:, None] - TR[None, :]); used_t = set(); t2k = {}
for k in np.argsort(d, axis=None):
    i, j = divmod(int(k), len(TR))
    if d[i, j] > 0.6: break   # a GT tree with no trunk within 0.6 m stays unmapped (count 0)
    if trees[i] in t2k or j in used_t: continue
    t2k[trees[i]] = j; used_t.add(j)
print(f"[trees] apples {len(P)} (merged, GT region), matched to a GT apple at 10 cm: {int((pair >= 0).sum())}")
print("tree    GT    our trunk(along)   GT median   ours   err     TP(right tree)  P      R      F1")
tp_all = 0; n_right = 0; n_matched = 0; rows = []
for i, t_ in enumerate(trees):
    j = t2k.get(t_); gi = np.where(GT_tree == t_)[0]; mine = assign == j if j is not None else np.zeros(len(P), bool); n_ours = int(mine.sum())
    right = mine & (pair >= 0) & np.isin(pair, gi); tp = int(right.sum()); fp = n_ours - tp; fn = len(gi) - tp; prec = tp / max(1, n_ours); rec = tp / len(gi); f1 = 2 * prec * rec / max(1e-9, prec + rec)
    mm = (pair >= 0) & np.isin(pair, gi); n_matched += int(mm.sum()); n_right += int((mm & mine).sum()); tp_all += tp
    rows.append((t_, len(gi), TR[j] if j is not None else np.nan, gmed[i], n_ours, tp, prec, rec, f1)); print(f"{t_}  {len(gi):4d}   {TR[j] if j is not None else float('nan'):+6.2f}            {gmed[i]:+6.2f}    {n_ours:4d}  {100 * (n_ours - len(gi)) / len(gi):+5.0f} %   {tp:4d}          {prec:.3f}  {rec:.3f}  {f1:.3f}")
ours_total = sum(r[4] for r in rows); print(f"[trees] ({a.out_tag}{', posts kept' if a.keep_posts else ''}) totals over the 11 trees: GT {len(GT)}, ours {ours_total}; apples matched to a GT apple whose trunk is the GT tree: {n_right} / {n_matched} ({100 * n_right / max(1, n_matched):.1f} %); per-tree |count error| median {np.median([abs(r[4] - r[1]) / r[1] for r in rows]) * 100:.0f} %, max {max(abs(r[4] - r[1]) / r[1] for r in rows) * 100:.0f} %; mean per-tree F1 {np.mean([r[8] for r in rows]):.3f}")
json.dump({'trunks_along': TR.tolist(), 'rows': [dict(zip(('tree', 'gt', 'trunk', 'gt_median', 'ours', 'tp', 'p', 'r', 'f1'), map(lambda v: v.item() if hasattr(v, 'item') else v, r))) for r in rows], 'right_tree': [n_right, n_matched]}, open(JS, 'w'), indent=1)
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True); cols = plt.cm.tab20(np.arange(len(TR)) % 20)   # column width: 6.5-7 pt text
    ax[0].scatter(Prf[:, 0], Prf[:, 2], s=1.2, c=cols[assign], linewidths=0); [ax[0].axvline(p, color='k', lw=0.6) for p in TR]; ax[0].set_ylabel('height [m]', fontsize=7)
    ax[0].text(0.008, 0.95, 'ours: apples by trunk anchor (lines)', fontsize=6.5, transform=ax[0].transAxes, va='top')
    ax[1].scatter(GTrf[:, 0], GTrf[:, 2], s=1.2, c=[plt.cm.tab20(trees.index(t_) % 20) for t_ in GT_tree], linewidths=0); [ax[1].axvline(m_, color='0.5', lw=0.6, ls='--') for m_ in gmed]; ax[1].set_ylabel('height [m]', fontsize=7); ax[1].set_xlabel('along the row [m]', fontsize=7)
    ax[1].text(0.008, 0.95, 'ground truth: apples by tree (dashed: tree medians)', fontsize=6.5, transform=ax[1].transAxes, va='top')
    for a_ in ax: a_.tick_params(labelsize=6); a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
    fig.tight_layout(h_pad=0.3); fig.savefig(FIG, dpi=220); print('[trees] figure', FIG)
except Exception as ex: print('[trees] figure skipped:', ex)
