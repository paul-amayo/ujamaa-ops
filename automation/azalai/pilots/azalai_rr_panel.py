#!/usr/bin/env python3
"""azalai_rr_panel.py (11 Nov 2026) — RSS Azalai pilot "row to relevancy without a keyframe": the pictures for the eye check.
  target sheets   one figure per Part 2 target, a row per datum (primary, oracle), four columns: the stop frame with its SAM 3
                  grow-bag masks (white outlines; m* yellow) and the revisit census projected at the stop frame (cyan dots; the
                  bag m* is named as = magenta ring; the target's partner = green cross) | the twin's colour render at the
                  believed pose with M_t (red) | the stop frame under M_t (the comparison rule R makes) | the same for the
                  bag-level twin. Titles carry the records of rr_targets<suffix>.jsonl.
                  --eye N: N targets per outcome of rule R (primary datum, canon, e = 0), seeded draw (seed 0).
  --example       rr_example.png: the right-plant target with the median IoU among those in lanes without a datum failure
                  whose naming the supervision confirms (a typical target, fixed rule); second row = the sweep (M_t at e = -0.90,
                  -0.45, 0, +0.45, +0.90 m over the same stop frame, with the mask picked and the outcome at each).
  --plan          rr_lane_frames<suffix>.png: plan view of the lane frames in both seasons (inner-row bags by side, camera
                  tracks, centre lines, row ends) and the revisit's inner-row bags carried into the twin by the primary lane
                  frames and by the site-wide fit.
  --part1 N       N seeded Part 1 targets: the free centre-line pose and the held-out keyframe, colour render with M_t.
nerf_new python3.10:  azalai_rr_panel.py <prep.json> <renders dir> <rr_targets.jsonl> <out dir> [--suffix _dec_apr1130] [--eye 3] [--example] [--plan] [--part1 4]"""
import argparse, json, glob, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths
from rig_calib import load_rig
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('renders'); ap.add_argument('targets'); ap.add_argument('out'); ap.add_argument('--suffix', default=''); ap.add_argument('--eye', type=int, default=0)
ap.add_argument('--example', action='store_true'); ap.add_argument('--plan', action='store_true'); ap.add_argument('--part1', type=int, default=0); ap.add_argument('--list', default=''); ap.add_argument('--dpi', type=int, default=100); a = ap.parse_args()
P = json.load(open(a.prep)); RD = Path(a.renders); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); ES = P['summary']['sweep_m']; ROOT = Path(P['rev_root'])
E = json.load(open(P['stpg']))['experiences']; Cr = np.array(E[P['rev']]['c2w'], float); Ct = np.array(E[P['twin']]['c2w'], float); Br = np.load(P['census_rev'])['bags_xyz'].astype(float); Bt = np.load(P['census_twin'])['bags_xyz'].astype(float)
KI = load_rig(survey_paths.monos(str(ROOT)))['intrinsics']; L = [json.loads(l) for l in open(a.targets)]; ev = {r['twin_bag']: r for r in L if r['part'] == 2 and 'skip' not in r}; PT = {p['twin_bag']: p for p in P['targets']}
ENT = {}
for f in sorted(glob.glob(str(ROOT / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
    c = json.load(open(f))
    for li, fr in enumerate(c['frames']): ENT.setdefault(int(fr['kf_idx']), [x['rle'] for x in c['frame_entries'].get(str(li), [])])
def live(k):
    ms = []
    for rle in ENT.get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool)[::2, ::2])
    return ms
def frame(k):
    f = ROOT / f'prod/scratch_sam3/kf_{k:06d}.png'; img = np.array(Image.open(f).convert('RGB').resize((640, 360), Image.BILINEAR)).astype(float)
    return np.clip(img * (110.0 / max(img.mean(), 1.0)) if img.mean() < 90 else img, 0, 255).astype(np.uint8)        # display gain only, for the dark passes
def edge(m, w=1): return m & ~ndimage.binary_erosion(m, iterations=w)
def tint(img, m, col, alpha=0.45): o = img.astype(float).copy(); o[m] = (1 - alpha) * o[m] + alpha * np.array(col); o[edge(m)] = col; return o.astype(np.uint8)
def census_px(k):
    w2c = np.linalg.inv(Cr[k]); q = (w2c[:3, :3] @ Br.T).T + w2c[:3, 3]; ok = np.isfinite(q).all(1) & (q[:, 2] > 0); u = np.full(len(q), np.nan); v = np.full(len(q), np.nan)
    u[ok] = KI['fx'] * q[ok, 0] / q[ok, 2] + KI['cx']; v[ok] = KI['fy'] * q[ok, 1] / q[ok, 2] + KI['cy']; ok &= (u >= 0) & (u < 1280) & (v >= 0) & (v < 720) & (np.linalg.norm(q, axis=1) < 8); return u / 2, v / 2, ok
MT = {}
def mask(cell, job, tw):
    if cell not in MT: z = np.load(RD / f'mt_{cell}.npz'); MT[cell] = (dict((str(n), i) for i, n in enumerate(z['jobs'])), z, tuple(int(x) for x in z['shape']))
    ix, z, (h, w) = MT[cell]; return np.unpackbits(z[tw][ix[job]], count=h * w).reshape(h, w).astype(bool) if tw in z.files else np.zeros((h, w), bool)
def live_panel(ax, k, x, partner, title):
    img = frame(k); ms = live(k)
    for m in ms: img[edge(m)] = 255
    if x.get('mstar') is not None: img = tint(img, ms[x['mstar']], (255, 220, 0), 0.4); img[edge(ms[x['mstar']], 2)] = (255, 220, 0)
    u, v, ok = census_px(k); ax.imshow(img); ax.scatter(u[ok], v[ok], s=10, c='cyan', edgecolors='k', linewidths=0.3)
    if ok[partner]: ax.scatter([u[partner]], [v[partner]], s=110, c='lime', marker='x', linewidths=2)
    if x.get('name') is not None: ax.scatter([u[x['name']]], [v[x['name']]], s=130, facecolors='none', edgecolors='magenta', linewidths=1.8)
    ax.set_xlim(0, 640); ax.set_ylim(360, 0); ax.set_title(title, fontsize=8.5); ax.axis('off')
def say(x): return f"{x['outcome'].upper()} (IoU {x['iou']:.3f}, named {x['name']}, {x['n_inside']} census bags in m*)" if x['outcome'] != 'no pick' else f"NO PICK ({x.get('reason')})"
def figure(t, path, rows=('primary', 'oracle')):
    r = ev[t]; p = PT[t]; tws = [tw for tw in ('canon', 'bag') if tw in r['R'][rows[0]]]; fig, ax = plt.subplots(len(rows), 2 + len(tws), figsize=((2 + len(tws)) * 6.4, len(rows) * 3.95), squeeze=False)
    for i, dn in enumerate(rows):
        d = r['R'][dn]; k = d['stop_k']; x = d['canon'][0]; z = np.load(RD / f'p2_{t}_{dn}_0.npz'); ms = live(k)
        live_panel(ax[i, 0], k, x, r['partner'], f"{P['rev']} stop frame kf {k} ({dn} datum; lane {r['lane']}, row end {P['lanes'][r['lane'] - 1]['row_end_minus_oracle_m'][dn] if dn != 'sitewide' else 0:+.2f} m from the oracle): {len(ms)} masks, m* yellow, partner {r['partner']} green x")
        ax[i, 1].imshow(tint(z['rgb'], z['canon'] == r['twin_id'], (230, 50, 40))); ax[i, 1].set_title(f"twin at the believed pose: target bag {t} = address {r['twin_id']} (M_t red, {x['mt_px']} px); naming {r['naming_vote']}, growth {r['growth']}", fontsize=8.5); ax[i, 1].axis('off')
        for j, tw in enumerate(tws):
            y = d[tw][0]; im = tint(frame(k), z[tw] == r['twin_id'], (230, 50, 40), 0.4)
            if y.get('mstar') is not None: im[edge(ms[y['mstar']], 2)] = (255, 220, 0)
            ax[i, 2 + j].imshow(im); ax[i, 2 + j].set_title(f"rule R, {'twin of the letter' if tw == 'canon' else 'bag-level twin'}: {say(y)}", fontsize=8.5); ax[i, 2 + j].axis('off')
    fig.subplots_adjust(left=0.003, right=0.997, top=0.955, bottom=0.005, wspace=0.01, hspace=0.09); fig.savefig(path, dpi=a.dpi, **({'pil_kwargs': {'quality': 82}} if str(path).endswith('jpg') else {})); plt.close(fig); print('[panel]', path, flush=True)
todo = []
if a.eye:
    rng = np.random.RandomState(0)
    for oc in ('right plant', 'wrong plant', 'unnamed', 'no pick'):
        C = sorted(t for t, r in ev.items() if r['R']['primary']['canon'][0]['outcome'] == oc)
        for t in (rng.choice(C, min(a.eye, len(C)), replace=False) if C else []): todo.append((int(t), f"eye_{oc.replace(' ', '_')}_{int(t)}"))
for s in [x for x in a.list.split(',') if x]: todo.append((int(s), f'target_{s}'))
for t, name in todo: figure(t, OUT / f'{name}.jpg')
if a.example:
    fail = set(P['summary']['datum_failures']); C = sorted([r for r in ev.values() if r['lane'] not in fail and r['naming_vote'] == 'agree' and r['R']['primary']['canon'][0]['outcome'] == 'right plant'], key=lambda r: (r['R']['primary']['canon'][0]['iou'], r['twin_bag'])); r = C[len(C) // 2]; t = r['twin_bag']
    json.dump(dict(rule='right-plant target (rule R, primary datum, canon twin, e = 0) with the median IoU among those in lanes without a datum failure whose naming the supervision confirms', n_candidates=len(C), twin_bag=t, record=r), open(OUT / f'rr_example{a.suffix}.json', 'w'), indent=1)
    d = r['R']['primary']; k = d['stop_k']; z = np.load(RD / f'p2_{t}_primary_0.npz'); ms = live(k); fig = plt.figure(figsize=(19.2, 7.6)); x = d['canon'][0]
    ax = fig.add_axes([0.003, 0.50, 0.33, 0.455]); live_panel(ax, k, x, r['partner'], f"{P['rev']} stop frame kf {k} (lane {r['lane']}, chosen from the registry and the lane frame alone): SAM 3 masks, m* yellow, census cyan, partner green x")
    ax = fig.add_axes([0.3365, 0.50, 0.33, 0.455]); ax.imshow(tint(z['rgb'], z['canon'] == r['twin_id'], (230, 50, 40))); ax.set_title(f"December twin rendered at the believed pose (no keyframe): relevancy of address {r['twin_id']} (M_t red)", fontsize=8.5); ax.axis('off')
    im = tint(frame(k), z['canon'] == r['twin_id'], (230, 50, 40), 0.4); im[edge(ms[x['mstar']], 2)] = (255, 220, 0); ax = fig.add_axes([0.67, 0.50, 0.33, 0.455]); ax.imshow(im); ax.set_title(f"rule R: {say(x)}; partner {r['partner']}", fontsize=8.5); ax.axis('off')
    for j, e in enumerate((-0.90, -0.45, 0.0, 0.45, 0.90)):
        n_e = ES.index(e); y = d['canon'][n_e]; im = tint(frame(k), mask(r['cell'], PT[t]['datum']['primary']['jobs'][n_e], 'canon'), (230, 50, 40), 0.4)
        if y.get('mstar') is not None: im[edge(ms[y['mstar']], 2)] = (255, 220, 0)
        ax = fig.add_axes([0.003 + j * 0.1995, 0.01, 0.197, 0.44]); ax.imshow(im); ax.set_title(f"believed pose off by e = {e:+.2f} m: {y['outcome']}" + (f" (named {y['name']})" if y.get('name') is not None else ''), fontsize=8.5); ax.axis('off')
    fig.savefig(OUT / f'rr_example{a.suffix}.png', dpi=110); plt.close(fig); print('[panel] example', t, flush=True)
if a.plan:
    G = np.array(P['G_rev_to_twin']); fig, ax = plt.subplots(1, 3, figsize=(27, 13)); tf = lambda T, p: (T[:3, :3] @ np.asarray(p, float).T).T + T[:3, 3]
    for j, (nm, B, C, inner, passes, key) in enumerate(((P['twin'], Bt, Ct, P['twin_inner'], P['twin_passes'], 'F_twin'), (P['rev'], Br, Cr, P['rev_inner'], P['rev_passes'], 'F_rev'))):
        ax[j].scatter(B[:, 0], B[:, 2], s=4, c='0.75')
        for i, l in enumerate(P['lanes']):
            k0, k1, sn = passes[i]; ax[j].plot(C[k0:k1 + 1, 0, 3], C[k0:k1 + 1, 2, 3], '-', c='0.3', lw=0.8); ib = np.array(inner[str(i + 1)]['bags']); lf = np.array(inner[str(i + 1)]['left'] or [], bool)
            if 'skip' in l or not len(ib): continue
            ax[j].scatter(B[ib[lf], 0], B[ib[lf], 2], s=9, c='tab:blue'); ax[j].scatter(B[ib[~lf], 0], B[ib[~lf], 2], s=9, c='tab:orange'); F = np.array(l[key] if key == 'F_twin' else l[key]['primary']); o, ax_ = F[:3, 3], F[:3, 2]; ln = l[f"{'twin' if key == 'F_twin' else 'rev'}_inner"]['length_m'] * sn
            ax[j].plot([o[0], o[0] + ln * ax_[0]], [o[2], o[2] + ln * ax_[2]], 'k--', lw=0.7); ax[j].scatter([o[0]], [o[2]], s=70, c='red', marker='*', zorder=5); ax[j].annotate(f"{i + 1}{'>' if sn > 0 else '<'}", (o[0], o[2] - 1.2 * sn), ha='center', va='center', fontsize=10)
        ax[j].set_aspect('equal'); ax[j].invert_xaxis(); ax[j].set_title(f"{nm}, its own graph frame: census (grey), inner rows (blue = left of +z, orange = right), camera tracks, lane centre lines (dashed), row ends = primary datum (red stars)", fontsize=8.5); ax[j].set_xlabel('x (m)'); ax[j].set_ylabel('z (m), rows along z')
    ax[2].scatter(Bt[:, 0], Bt[:, 2], s=5, c='0.8')
    for i, l in enumerate(P['lanes']):
        if 'skip' in l: continue
        ib = np.array(P['rev_inner'][str(i + 1)]['bags']); it = np.array(P['twin_inner'][str(i + 1)]['bags']); T = np.array(l['F_twin']) @ np.linalg.inv(np.array(l['F_rev']['primary'])); a_, b_ = tf(T, Br[ib]), tf(G, Br[ib])
        ax[2].scatter(Bt[it, 0], Bt[it, 2], s=16, c='k', label='twin inner-row bags' if i == 0 else None); ax[2].scatter(b_[:, 0], b_[:, 2], s=22, facecolors='none', edgecolors='tab:blue', linewidths=0.8, label='revisit inner-row bags, site-wide fit' if i == 0 else None)
        ax[2].scatter(a_[:, 0], a_[:, 2], s=14, c='tab:red', marker='+', linewidths=0.8, label='revisit inner-row bags, primary lane frames' if i == 0 else None); o = np.array(l['F_twin'])[:3, 3]
        ax[2].annotate(f"lane {i + 1}: {l['row_end_minus_oracle_m']['primary']:+.2f} m" + (' FAIL' if l['datum_failure'] else ''), (o[0], o[2] - 2.0 * l['sense']), ha='center', va='center', fontsize=8, color='tab:red' if l['datum_failure'] else 'k')
    ax[2].set_aspect('equal'); ax[2].invert_xaxis(); ax[2].legend(loc='upper center', fontsize=8, ncol=3); ax[2].set_title(f"{P['rev']} inner-row bags carried into the {P['twin']} frame; label = primary row end minus the oracle's", fontsize=8.5); ax[2].set_xlabel('x (m)')
    fig.tight_layout(); fig.savefig(OUT / f'rr_lane_frames{a.suffix}.png', dpi=80); plt.close(fig); print('[panel] plan', flush=True)
if a.part1:
    P1 = [r for r in L if r['part'] == 1]; rng = np.random.RandomState(0); sel = rng.choice(len(P1), min(a.part1, len(P1)), replace=False); pj = {(p['cell'], p['twin_id']): p for p in P['part1']}
    for i in sel:
        r = P1[int(i)]; p = pj[(r['cell'], r['twin_id'])]; fig, ax = plt.subplots(1, 3, figsize=(19.2, 3.95))
        for j, tag in enumerate(('C', 'H', 'T')):
            if tag not in p['poses']: ax[j].axis('off'); continue
            z = np.load(RD / f"{p['poses'][tag]['job']}.npz"); x = r['poses'][tag]['canon']; dep = z['depth'].astype(float); base = z['rgb'] if 'rgb' in z.files else (np.clip(1.0 / np.clip(dep, 0.3, 20), 0, 1)[..., None] * 255 * np.ones(3)).astype(np.uint8)
            ax[j].imshow(tint(base, z['canon'] == r['twin_id'], (230, 50, 40))); ax[j].axis('off')
            ax[j].set_title(f"{dict(C='free centre-line pose', H='held-out keyframe', T='training keyframe')[tag]} ({r['cell']}, address {r['twin_id']}): " + (f"lit {x['area_px']} px, lands on {x['nearest_anchor']} ({'target' if x['lands'] else 'NOT the target'}), error {x['err_m']:.2f} m, range {x['range_m']:.2f} m" if x['lit'] else 'not lit') + (' [inverse depth shown]' if 'rgb' not in z.files else ''), fontsize=8.5)
        fig.subplots_adjust(left=0.003, right=0.997, top=0.92, bottom=0.005, wspace=0.01); fig.savefig(OUT / f"part1_{r['cell']}_{r['twin_id']}.jpg", dpi=a.dpi, pil_kwargs={'quality': 82}); plt.close(fig); print('[panel] part 1', r['cell'], r['twin_id'], flush=True)
