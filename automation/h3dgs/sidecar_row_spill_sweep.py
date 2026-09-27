"""What the row mask draws that supervision never labelled, and what raising the gate would cost (Paul, 2026-09-27).
Same composite as the demo (best margin across models, raw pick) with the SKY MASK applied exactly as the demo applies it,
so these are the pixels actually on screen. Splits the drawn pixels against supervision:
  CROSS  drawn row A where supervision says row B       (a genuine mis-ranking)
  SPILL  drawn a row where supervision painted nothing  (ground / far field / canopy gap)
  MISS   supervision says row r, nothing drawn
and sweeps an extra offset on top of each model's own verdict threshold (the maps store the margin, so no re-render).
  python sidecar_row_spill_sweep.py --survey 05_13D_Jackal --path demo_path.json --maps <demo>/maps --models 018 ... """
import argparse, glob, json, os
from pathlib import Path
import numpy as np
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True); ap.add_argument('--maps', required=True)
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--modes', nargs='+', default=['row', 'all', 'rows'])
ap.add_argument('--offsets', type=float, nargs='+', default=[0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20])
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey; pj = json.load(open(a.path)); FR = [f for f in pj['frames'] if f['mode'] in a.modes]
allrow = sorted({w for v in pj['rows'].values() for w in v}); ROWID = {w: i for i, w in enumerate(allrow)}
H_ = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json')); tree_row = {t: r['id'] for r in H_['rows'] for t in r['object_ids']}
SKY = S / 'prod/tassili/sky_masks'
sup = {}
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/transforms.json'))):
    bd = Path(tj).parent; own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    for f in glob.glob(str(bd / 'supervision/trees_only/kf_*.png')):
        if os.path.basename(f) in own: sup[os.path.basename(f)] = f
acc = {d: dict(drawn=0, cross=0, spill=0, miss=0) for d in a.offsets}; gt_tot = 0; n = 0; sky_spill = 0; spill0 = 0
for f in FR:
    br = brm = None
    for b in a.models:
        p = Path(a.maps) / b / (f['name'] + '.npz')
        if not p.exists(): continue
        z = np.load(p); rws = list(z['rows'])
        if not rws: continue
        rmg = z['rmg_raw'].astype(np.float32); rid = z['rid_raw']
        if br is None: br = np.full(rmg.shape, -1, np.int32); brm = np.full(rmg.shape, -9.0, np.float32)
        m = rmg > brm; br[m] = np.array([ROWID[w] for w in rws], np.int32)[rid][m]; brm[m] = rmg[m]
    if br is None or f.get('src') not in sup: continue
    Hh, Ww = br.shape
    s = np.array(Image.open(sup[f['src']]), np.uint16); s = np.array(Image.fromarray(s).resize((Ww, Hh), Image.NEAREST))
    g = np.full(br.shape, -1, np.int32)
    for t in [int(v) for v in np.unique(s) if v != 65535]:
        r = tree_row.get(t, -3); g[s == t] = r if (r is not None and 0 <= r < len(allrow)) else -2
    skyp = SKY / f['src']
    sk = np.array(Image.open(skyp).convert('L').resize((Ww, Hh), Image.NEAREST)) > 127 if skyp.exists() else np.zeros(br.shape, bool)
    vis = ~sk                                            # the demo blacks out sky, so only these pixels are on screen
    sky_spill += int((sk & (brm >= 0) & (g == -1)).sum()); spill0 += int(((brm >= 0) & (g == -1)).sum())
    gt_tot += int(((g >= 0) & vis).sum()); n += 1
    for d in a.offsets:
        drawn = (brm >= d) & vis
        acc[d]['drawn'] += int(drawn.sum()); acc[d]['cross'] += int((drawn & (g >= 0) & (br != g)).sum())
        acc[d]['spill'] += int((drawn & (g == -1)).sum()); acc[d]['miss'] += int(((g >= 0) & vis & ~drawn).sum())
print(f"[spill] {n} row-drawing frames, sky masked. Of all spill at the current gate, {100*sky_spill/max(spill0,1):.0f}% is sky "
      f"(removed by the mask); supervised row pixels on screen: {gt_tot}", flush=True)
print(f"[spill] {'offset':>7} {'drawn':>10} {'CROSS':>9} {'SPILL':>11} {'spill%':>7} {'MISS':>10} {'miss%':>7}", flush=True)
for d in a.offsets:
    r = acc[d]
    print(f"[spill] {d:>7.2f} {r['drawn']:>10} {r['cross']:>9} {r['spill']:>11} {100*r['spill']/max(r['drawn'],1):>6.1f}% "
          f"{r['miss']:>10} {100*r['miss']/max(gt_tot,1):>6.1f}%", flush=True)

# --- what IS the spill? split it with the foreground (canopy) masks, which are independent of supervision ---
FG = S / 'prod/tassili/fg_masks'
sp = dict(fg=0, bg=0); miss_fg = 0; near_far = dict(fg_near_gt=0, fg_far=0)
for f in FR:
    br = brm = None
    for b in a.models:
        p = Path(a.maps) / b / (f['name'] + '.npz')
        if not p.exists(): continue
        z = np.load(p); rws = list(z['rows'])
        if not rws: continue
        rmg = z['rmg_raw'].astype(np.float32); rid = z['rid_raw']
        if br is None: br = np.full(rmg.shape, -1, np.int32); brm = np.full(rmg.shape, -9.0, np.float32)
        m = rmg > brm; br[m] = np.array([ROWID[w] for w in rws], np.int32)[rid][m]; brm[m] = rmg[m]
    if br is None or f.get('src') not in sup: continue
    Hh, Ww = br.shape
    s = np.array(Image.open(sup[f['src']]), np.uint16); s = np.array(Image.fromarray(s).resize((Ww, Hh), Image.NEAREST))
    g = np.full(br.shape, -1, np.int32)
    for t in [int(v) for v in np.unique(s) if v != 65535]:
        r = tree_row.get(t, -3); g[s == t] = r if (r is not None and 0 <= r < len(allrow)) else -2
    skyp = SKY / f['src']; sk = np.array(Image.open(skyp).convert('L').resize((Ww, Hh), Image.NEAREST)) > 127 if skyp.exists() else np.zeros(br.shape, bool)
    fgp = FG / f['src']
    if not fgp.exists(): continue
    fg = np.array(Image.open(fgp).convert('L').resize((Ww, Hh), Image.NEAREST)) > 127
    spill = (brm >= 0) & ~sk & (g == -1)
    sp['fg'] += int((spill & fg).sum()); sp['bg'] += int((spill & ~fg).sum())
    miss_fg += int(((g >= 0) & ~sk & (brm < 0) & fg).sum())
tt = sp['fg'] + sp['bg']
print(f"[spill] SPILL split by the foreground mask (canopy vs not, independent of supervision): "
      f"CANOPY {sp['fg']} ({100*sp['fg']/max(tt,1):.1f}%) = row identity on trees this frame's supervision never labelled; "
      f"NON-CANOPY {sp['bg']} ({100*sp['bg']/max(tt,1):.1f}%) = ground / structure", flush=True)
