#!/usr/bin/env python3
"""azalai_h3_twin_panel.py (11 Nov 2026) — RSS Azalai Stage 1 pilot H3 with the twin render: pictures of scored pairs, for the eye
check and for the figure-ready example. One figure per pair, two rows (correct arrival = the goal keyframe; wrong arrival = the
neighbour's goal keyframe), four columns: the live frame with its SAM3 grow-bag masks (white outlines; the mask under test m* in
yellow) | the twin's colour render at the believed pose with the identity masks of the target (red), previous (blue) and next
(green) row neighbour | the live frame under those masks (the comparison rule A makes on the twin of the letter) | the live frame
under the bag-level twin's masks. Titles carry the scores of h3_twin_pairs.jsonl. Picks:
  --example   the April pair whose correct-arrival target IoU (rule A, canon) is the median among the evaluable pairs whose wrong bag
              is a candidate and whose naming is confirmed (a typical pair, fixed rule);
  --eye N     N pairs per outcome cell of rule A canon (correct confirmed / rejected x wrong confirmed / caught), seeded draw;
  --pairs apr:12,dec:40   named pairs.
nerf_new python3.10:  azalai_h3_twin_panel.py <prep.json> <renders dir> <pairs.jsonl> <apr root> <dec root> <out dir> [--example] [--eye 4]"""
import argparse, json, glob, numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mu
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('renders'); ap.add_argument('pairs'); ap.add_argument('apr'); ap.add_argument('dec'); ap.add_argument('out')
ap.add_argument('--example', action='store_true'); ap.add_argument('--eye', type=int, default=0); ap.add_argument('--pairs-list', default=''); ap.add_argument('--dpi', type=int, default=110); ap.add_argument('--ext', default='jpg'); a = ap.parse_args()
RD = Path(a.renders); ROOT = dict(apr=Path(a.apr), dec=Path(a.dec)); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
R = [json.loads(l) for l in open(a.pairs)]; ev = {(r['season'], r['dec']): r for r in R if 'skip' not in r}
def entries(root):
    e = {}
    for f in sorted(glob.glob(str(root / 'prod/bateleur/sam3_growbag/clip_*/frame_entries.json'))):
        c = json.load(open(f))
        for li, fr in enumerate(c['frames']): e.setdefault(int(fr['kf_idx']), [x['rle'] for x in c['frame_entries'].get(str(li), [])])
    return e
ENT = {}
def live(season, k):
    if season not in ENT: ENT[season] = entries(ROOT[season])
    ms = []
    for rle in ENT[season].get(k, []): rr = dict(rle); rr['counts'] = rr['counts'].encode(); ms.append(mu.decode(rr).astype(bool)[::2, ::2])
    return ms
def iou(x, y):
    u = (x | y).sum(); return float((x & y).sum() / u) if u else 0.0
def edge(m, w=1): return m & ~ndimage.binary_erosion(m, iterations=w)
COL = dict(t=(230, 50, 40), prev=(40, 110, 230), next=(40, 170, 80))
def tint(img, ids, cand, alpha=0.5):
    o = img.astype(float).copy()
    for c, col in COL.items():
        if cand.get(c) is None: continue
        m = ids == cand[c]; o[m] = (1 - alpha) * o[m] + alpha * np.array(col); o[edge(m)] = col
    return o.astype(np.uint8)
def figure(season, t, path):
    r = ev[(season, t)]; z = np.load(RD / f'{season}_{t}_V0.npz'); cand = r['twin']; fig, ax = plt.subplots(2, 4, figsize=(4 * 6.4, 2 * 3.9))
    for row, case in enumerate(('correct', 'wrong')):
        k = r['goal_k'] if case == 'correct' else r['wrong_k']; img = np.array(Image.open(ROOT[season] / f'prod/scratch_sam3/kf_{k:06d}.png').convert('RGB').resize((640, 360), Image.BILINEAR)); ms = live(season, k)
        if season == 'apr': img = np.clip(img.astype(float) * 1.6, 0, 255).astype(np.uint8)   # the April frames are dark; display gain only
        def under_test(ids):
            Rt = ids == cand['t']
            if not ms or not Rt.any(): return None
            i = int(np.argmax([iou(m, Rt) for m in ms])); return ms[i] if iou(ms[i], Rt) > 0 else None
        mc, mb = under_test(z['canon']), under_test(z['bag'])
        p0 = img.copy()
        for m in ms: p0[edge(m)] = 255
        if mc is not None: p0[mc] = (0.55 * p0[mc] + 0.45 * np.array([255, 220, 0])).astype(np.uint8); p0[edge(mc, 2)] = (255, 220, 0)
        p1 = tint(z['rgb'], z['canon'], cand); p2 = tint(img, z['canon'], cand, 0.4); p3 = tint(img, z['bag'], cand, 0.4)
        if mc is not None: p2[edge(mc, 2)] = (255, 220, 0)
        if mb is not None: p3[edge(mb, 2)] = (255, 220, 0)
        A, Ab, B = r[case]['A_canon'], r[case]['A_bag'], r[case]['B_bag']; say = lambda x: 'CONFIRMED' if x['arrived'] else 'rejected'
        tt = [f"{'December held-out' if season == 'dec' else 'April'} live frame kf {k} ({case} arrival): {len(ms)} SAM 3 grow-bag masks, m* yellow",
              f"twin at the believed pose (bag {t}, {r['lane']}): target {cand['t']} red, prev {cand['prev']} blue, next {cand['next']} green",
              f"rule A, twin of the letter: {say(A)}  IoU target {A['iou_t']:.3f} vs neighbour {A['iou_n']:.3f}",
              f"rule A, bag-level twin: {say(Ab)}  {Ab['iou_t']:.3f} vs {Ab['iou_n']:.3f} | rule B: {say(B)}" + (f"  {B['s_t']:.2f} / {B['s_prev'] if B['s_prev'] is None else round(B['s_prev'], 2)} / {B['s_next'] if B['s_next'] is None else round(B['s_next'], 2)}" if 's_t' in B else '')]
        for c, (im, ti) in enumerate(zip((p0, p1, p2, p3), tt)): ax[row, c].imshow(im); ax[row, c].set_title(ti, fontsize=8.5); ax[row, c].axis('off')
    fig.subplots_adjust(left=0.003, right=0.997, top=0.955, bottom=0.005, wspace=0.01, hspace=0.09); fig.savefig(path, dpi=a.dpi, **({'pil_kwargs': {'quality': 82}} if str(path).endswith('jpg') else {})); plt.close(fig); print('[panel]', path, flush=True)
todo = []
if a.example:
    C = sorted([r for r in ev.values() if r['season'] == 'apr' and r['n1_is_candidate'] and r['naming_vote'] == 'agree'], key=lambda r: (r['correct']['A_canon']['iou_t'], r['dec'])); r = C[len(C) // 2]
    json.dump(dict(rule='median correct-arrival target IoU (rule A, canon) among evaluable April pairs whose wrong bag is a candidate and whose naming is confirmed', n_candidates=len(C), dec=r['dec'], record=r), open(OUT / 'h3_twin_example.json', 'w'), indent=1)
    figure('apr', r['dec'], OUT / 'h3_twin_example.png')
if a.eye:
    rng = np.random.RandomState(0)
    for season in ('apr', 'dec'):
        for cc in (True, False):
            for wc in (True, False):
                C = sorted(r['dec'] for r in ev.values() if r['season'] == season and r['correct']['A_canon']['arrived'] == cc and r['wrong']['A_canon']['arrived'] == wc)
                for t in (rng.choice(C, min(a.eye if season == 'apr' else max(1, a.eye // 2), len(C)), replace=False) if C else []): todo.append((season, int(t), f"eye_{season}_c{'C' if cc else 'r'}_w{'C' if wc else 'r'}_{int(t)}"))
for s in [x for x in a.pairs_list.split(',') if x]: todo.append((s.split(':')[0], int(s.split(':')[1]), f"pair_{s.replace(':', '_')}"))
for season, t, name in todo: figure(season, t, OUT / f'{name}.{a.ext}')
