"""Row crossing measured against the SUPERVISION, not against tree bookkeeping (Paul, 2026-09-27: "there are so many row
crossings, find the worst keyframe and diagnose why"). Every demo frame is a keyframe, so it has a supervision map; a
pixel's TRUE row is the hierarchy row of the tree painted there, and unpainted pixels have no row. For each frame, with
the row map the demo actually draws (best margin across models, --row-pick raw by default):
  CROSS : drawn row A on a pixel whose supervision says row B        <- the crossing you see
  SPILL : drawn row on a pixel supervision left unpainted            <- ground / far field / sky bleed
  MISS  : supervision says row r, nothing drawn
For the worst frames by CROSS, the raw per-word scores at those exact pixels are read back from the model that won them,
so the cause is measured: a genuine mis-ranking (the wrong word scores higher) or the gate admitting a pixel that carries
no identity at all. Figures <out>/gt_<kf>.png: drawn | supervision | cross in red.
  python sidecar_row_gt_diag.py --survey 05_13D_Jackal --path demo_path.json --maps <demo>/maps --models 018 ... \
      --backdrop-dir <dir> --out <dir> [--row-pick raw] [--top 4] [--modes row all rows]"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True); ap.add_argument('--maps', required=True)
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--backdrop-dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--row-pick', choices=('raw', 'margin'), default='raw'); ap.add_argument('--top', type=int, default=4)
ap.add_argument('--modes', nargs='+', default=['row', 'all', 'rows'], help='only frames whose demo mode draws rows')
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
pj = json.load(open(a.path)); FR = [f for f in pj['frames'] if f['mode'] in a.modes]
allrow = sorted({w for v in pj['rows'].values() for w in v}); ROWID = {w: i for i, w in enumerate(allrow)}
H_ = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json')); tree_row = {t: r['id'] for r in H_['rows'] for t in r['object_ids']}
sup = {}
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/transforms.json'))):
    bd = Path(tj).parent; own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    for f in glob.glob(str(bd / 'supervision/trees_only/kf_*.png')):
        if os.path.basename(f) in own: sup[os.path.basename(f)] = f
def composite(name):
    """the row map the demo draws: best margin across models, and which model won each pixel"""
    best_r = best_rm = best_m = None
    for b in a.models:
        p = Path(a.maps) / b / (name + '.npz')
        if not p.exists(): continue
        z = np.load(p); rws = list(z['rows'])
        if not rws: continue
        kr = 'rmg_raw' if (a.row_pick == 'raw' and 'rmg_raw' in z.files) else 'rmg'
        rmg = z[kr].astype(np.float32); rid = z['rid_raw' if kr == 'rmg_raw' else 'rid']
        if best_r is None: best_r = np.full(rmg.shape, -1, np.int32); best_rm = np.full(rmg.shape, -9.0, np.float32); best_m = np.full(rmg.shape, '', dtype=object)
        m = rmg > best_rm; best_r[m] = np.array([ROWID[w] for w in rws], np.int32)[rid][m]; best_rm[m] = rmg[m]; best_m[m] = b
    return best_r, best_rm, best_m
def gt_rows(name, shape):
    """true row per pixel from supervision: hierarchy row of the painted tree, -1 unpainted, -2 painted with a row we never draw"""
    g = np.full(shape, -1, np.int32)
    if name not in sup: return None
    s = np.array(Image.open(sup[name]), np.uint16); s = np.array(Image.fromarray(s).resize((shape[1], shape[0]), Image.NEAREST))
    for t in [int(v) for v in np.unique(s) if v != 65535]:
        r = tree_row.get(t, -3); g[s == t] = r if (r is not None and 0 <= r < len(allrow)) else -2
    return g
tot = dict(cross=0, spill=0, miss=0, drawn=0, gt=0); recs = []
for f in FR:
    br, brm, bm = composite(f['name'])
    if br is None: continue
    g = gt_rows(f.get('src', ''), br.shape)
    if g is None: continue
    drawn = brm >= 0
    cross = drawn & (g >= 0) & (br != g); spill = drawn & (g == -1); miss = (g >= 0) & ~drawn
    rec = dict(frame=f['name'], src=f['src'], mode=f['mode'], cross=int(cross.sum()), spill=int(spill.sum()), miss=int(miss.sum()),
               drawn=int(drawn.sum()), gt=int((g >= 0).sum()))
    recs.append(rec)
    for k in tot: tot[k] += rec[k]
print(f"[gt] {len(recs)} row-drawing frames; drawn {tot['drawn']} px, supervised row {tot['gt']} px: "
      f"CROSS {tot['cross']} ({100*tot['cross']/max(tot['drawn'],1):.1f}% of drawn), SPILL {tot['spill']} ({100*tot['spill']/max(tot['drawn'],1):.1f}%), "
      f"MISS {tot['miss']} ({100*tot['miss']/max(tot['gt'],1):.1f}% of supervised)", flush=True)
recs.sort(key=lambda d: -d['cross'])
for rec in recs[:a.top]:
    f = next(x for x in FR if x['name'] == rec['frame']); br, brm, bm = composite(rec['frame']); g = gt_rows(rec['src'], br.shape)
    drawn = brm >= 0; cross = drawn & (g >= 0) & (br != g)
    msg = f"[gt] {rec['src']} ({rec['mode']}, {rec['frame']}): CROSS {rec['cross']} px, SPILL {rec['spill']}, MISS {rec['miss']} (drawn {rec['drawn']})"
    # which model won the crossing pixels, and what did IT score for each row word there?
    for b in sorted({str(x) for x in np.unique(bm[cross]) if str(x)}):
        p = Path(a.maps) / b / (rec['frame'] + '.npz'); z = np.load(p); rws = list(z['rows']); sel = cross & (bm == b)
        kr = 'rmg_raw' if (a.row_pick == 'raw' and 'rmg_raw' in z.files) else 'rmg'
        rid = z['rid_raw' if kr == 'rmg_raw' else 'rid'][sel]; mg = z[kr].astype(np.float32)[sel]
        drew = [rws[i] for i in np.unique(rid)]; true = [allrow[i] for i in np.unique(g[sel]) if i >= 0]
        msg += f" || model {b}: {int(sel.sum())} px drawn {drew} where supervision says {true}, winning margin {mg.mean():+.3f} (max {mg.max():+.3f})"
    print(msg, flush=True)
    img = cv2.imread(str(Path(a.backdrop_dir) / rec['frame'])); H, W = br.shape
    img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
    dv = img.copy(); gv = img.copy(); cols = [(0, 220, 255), (255, 200, 0), (200, 0, 255)]
    for r in range(len(allrow)):
        m = (br == r) & drawn; dv[m] = (0.45 * dv[m] + 0.55 * np.array(cols[r % 3])).astype(np.uint8)
        m = g == r; gv[m] = (0.45 * gv[m] + 0.55 * np.array(cols[r % 3])).astype(np.uint8)
    gv[g == -2] = (0.45 * gv[g == -2] + 0.55 * np.array((255, 255, 255))).astype(np.uint8)
    cv = img.copy(); cv[cross] = (0.25 * cv[cross] + 0.75 * np.array((0, 0, 255))).astype(np.uint8)
    for im, t in ((dv, 'drawn'), (gv, 'supervision (white = row we never draw)'), (cv, f'CROSS {rec["cross"]} px')):
        cv2.putText(im, t, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"gt_{rec['src']}"), np.concatenate([dv, gv, cv], 1))
json.dump(recs[:80], open(OUT / 'gt_worst.json', 'w'), indent=1)
