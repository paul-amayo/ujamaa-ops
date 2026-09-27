"""Where and why do the two row outlines cross in the demo (Paul, 2026-09-27)? Uses the demo's per-block identity maps
(best tree / best row per pixel by margin across the side-cars, as composited) and the marker hierarchy (tree -> row):
  A  disagreement : pixels whose decoded ROW is not the hierarchy row of the TREE carrying them (incl. trees of other rows)
  B  row-only     : pixels with a row but no tree (the ground / trunk band that decodes at row level only)
  C  wrong side   : near-field row pixels on the opposite side of the lane from where that row's trees are (per frame,
                    from the tree centres and the camera's right vector — the drive is a loop, sides swap halfway)
For the worst frames by A: the trees involved and the owning block's supervision id map at those pixels — painted with a
tree of the decoded row (supervision cause) or unpainted (decode / blend cause). Figures <out>/row_cross_<kf>.png.
  python sidecar_row_cross_diag.py --survey 05_13D_Jackal --path demo_path.json --maps <demo>/maps --blocks 018 ... --backdrop-dir <dir> --out <dir>"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True); ap.add_argument('--maps', required=True); ap.add_argument('--blocks', nargs='+', required=True)
ap.add_argument('--backdrop-dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--top', type=int, default=6)
ap.add_argument('--row-pick', choices=('raw', 'margin'), default='margin', help='which row-word pick to score, matching sidecar_demo_overlay.py --row-pick')
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); pj = json.load(open(a.path)); FR = pj['frames']
allrow = sorted({w for v in pj['rows'].values() for w in v}); ROWID = {w: i for i, w in enumerate(allrow)}   # containment_eval prints 'ROW 0 "oak"', 'ROW 1 "pine"': word index = hierarchy row id here
H_ = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json')); tree_row = {t: r['id'] for r in H_['rows'] for t in r['object_ids']}; trees = {int(k): np.array(v) for k, v in pj['trees'].items()}
sup = {}
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/transforms.json'))):
    bd = Path(tj).parent; own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    for f in glob.glob(str(bd / 'supervision/trees_only/kf_*.png')):
        if os.path.basename(f) in own: sup[os.path.basename(f)] = f
def composite(name):
    best_t = None
    for b in a.blocks:
        p = Path(a.maps) / b / (name + '.npz')
        if not p.exists(): continue
        z = np.load(p); tmg = z['tmg'].astype(np.float32); rws = list(z['rows'])
        kr = 'rmg_raw' if (a.row_pick == 'raw' and 'rmg_raw' in z.files) else 'rmg'; rmg = z[kr].astype(np.float32)
        if best_t is None: best_t = np.full(tmg.shape, -1, np.int32); best_tm = np.full(tmg.shape, -9.0, np.float32); best_r = np.full(tmg.shape, -1, np.int32); best_rm = np.full(tmg.shape, -9.0, np.float32)
        m = tmg > best_tm; best_t[m] = z['tid'][m]; best_tm[m] = tmg[m]
        m = rmg > best_rm; best_r[m] = (np.array([ROWID[w] for w in rws], np.int32)[z['rid_raw' if kr == 'rmg_raw' else 'rid']])[m] if rws else -1; best_rm[m] = rmg[m]
    return best_t, best_tm, best_r, best_rm
tot = {'A': 0, 'B': 0, 'C': 0, 'rowpx': 0}; recs = []; A_trees = {}
for f in FR:
    bt, btm, br, brm = composite(f['name']); H, W = br.shape; ys, xx = np.mgrid[0:H, 0:W]; near = ys >= H // 2
    c2w = np.asarray(f['c2w_h']); cam = c2w[:3, 3]; right = c2w[:3, 0]; fwd = c2w[:3, 2]
    rowm = brm >= 0; treem = btm >= 0; tr_row = np.vectorize(lambda t: tree_row.get(int(t), -9))(bt)
    A = rowm & treem & (tr_row != br); B = rowm & ~treem
    side = {}
    for w, r in ROWID.items():
        offs = [trees[t] - cam for t in trees if tree_row.get(t) == r and abs(np.dot(trees[t] - cam, fwd)) < 30 and np.linalg.norm(trees[t] - cam) < 35]
        if offs: side[r] = 'right' if np.mean([np.dot(o, right) for o in offs]) > 0 else 'left'
    C = np.zeros_like(A)
    for r, sd in side.items():
        wrong = (xx < W * 0.45) if sd == 'right' else (xx > W * 0.55); C |= (br == r) & rowm & near & wrong
    rec = {'frame': f['name'], 'src': f.get('src', ''), 'A': int(A.sum()), 'B': int(B.sum()), 'C': int(C.sum()), 'rowpx': int(rowm.sum()), 'side': side}
    ids, cnt = np.unique(bt[A], return_counts=True)
    for t, c in zip(ids.tolist(), cnt.tolist()): A_trees[t] = A_trees.get(t, 0) + c
    rec['A_trees'] = [(int(t), int(c), tree_row.get(int(t)), [allrow[r] for r in np.unique(br[A & (bt == t)])]) for c, t in sorted(zip(cnt.tolist(), ids.tolist()), reverse=True)[:3]]
    recs.append(rec)
    for k in ('A', 'B', 'C', 'rowpx'): tot[k] += rec[k]
print(f"[cross] {len(FR)} frames; row pixels total {tot['rowpx']}: A tree/row disagreement {100*tot['A']/max(tot['rowpx'],1):.1f}%, B row-only (no tree) {100*tot['B']/max(tot['rowpx'],1):.1f}%, C wrong side {100*tot['C']/max(tot['rowpx'],1):.1f}%", flush=True)
top_trees = sorted(A_trees.items(), key=lambda kv: -kv[1])[:8]; print("[cross] trees carrying disagreement pixels (id: px, hierarchy row):", ', '.join(f"{t}: {c} (row {tree_row.get(t)})" for t, c in top_trees), flush=True)
recs.sort(key=lambda d: -d['A'])
for rec in recs[:a.top]:
    bt, btm, br, brm = composite(rec['frame']); H, W = br.shape; rowm = brm >= 0; treem = btm >= 0; tr_row = np.vectorize(lambda t: tree_row.get(int(t), -9))(bt); A = rowm & treem & (tr_row != br)
    msg = f"[cross] {rec['src']}: A {rec['A']} px (B {rec['B']}, C {rec['C']}, sides {rec['side']}): " + '; '.join(f"tree {t} (row {rw}) {c} px decoded as {'/'.join(ws)}" for t, c, rw, ws in rec['A_trees'])
    if rec['src'] in sup:
        a_ = np.array(Image.open(sup[rec['src']]), np.uint16); a_ = np.array(Image.fromarray(a_).resize((W, H), Image.NEAREST))
        ids, cnt = np.unique(a_[A], return_counts=True); pa = sorted(zip(cnt.tolist(), ids.tolist()), reverse=True)[:3]
        msg += " || supervision at A pixels: " + ', '.join(('unpainted' if i == 65535 else f"tree {i} (row {tree_row.get(int(i))})") + f" {c} px" for c, i in pa)
    print(msg, flush=True)
    img = cv2.imread(str(Path(a.backdrop_dir) / rec['frame'])); vis = img.copy()
    for r, col in ((0, (255, 220, 0)), (1, (0, 255, 255))):
        mm = (br == r) & rowm; cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, col, 1)
    vis[A] = (0.4 * vis[A] + 0.6 * np.array((0, 0, 255))).astype(np.uint8)
    if rec['src'] in sup:
        a_ = np.array(Image.open(sup[rec['src']]), np.uint16); a_ = np.array(Image.fromarray(a_).resize((W, H), Image.NEAREST)); sv = img.copy()
        for t in [int(v) for v in np.unique(a_) if v != 65535]: sv[a_ == t] = (0.5 * sv[a_ == t] + 0.5 * np.array((255, 220, 0) if tree_row.get(t) == 0 else (0, 255, 255) if tree_row.get(t) == 1 else (255, 0, 255))).astype(np.uint8)
        vis = np.concatenate([vis, sv], 1)
    cv2.putText(vis, f"{rec['src']}  red = decoded row != tree's row | right: supervision (yellow row 0, cyan row 1, magenta other row)", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"row_cross_{rec['src'] or rec['frame']}"), vis)
json.dump(recs[:60], open(OUT / 'row_cross_worst.json', 'w'), indent=1, default=str)
