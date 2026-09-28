"""Build the demo's question script from the identity maps, so every question is asked where its answer is on screen
(Paul, 2026-09-28: "questions are too vague ... think questions a farmer would ask visualized").

A farmer names a tree by its position in a row, not by an internal id, so this also derives the ORDINAL of every tree:
the hierarchy's rows give the membership, and the trees are ordered along the row's own axis (first principal direction
of their centres). Row ids are presented 1-based. Fruit ranking is measured, not asserted: fruit pixels per parent tree
over the whole drive, from the fruit side-cars' maps.

Writes demo_script.json: {rows: {row_label: [tree ids in order]}, label: {tree: "tree N of row M"},
fruit_rank: [[tree, px], ...], segments: [{lo, hi, mode, question, answer, focus, hold_at, hold}]}
  python sidecar_demo_script.py --survey 05_13D_Jackal --path demo_05/demo_path.json --maps demo_05/maps
      --fruit-maps demo_05/maps_fruit --models 018 ... --fruit-models 018 019 021 --out demo_05/demo_script.json"""
import argparse, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True)
ap.add_argument('--maps', required=True); ap.add_argument('--fruit-maps', required=True)
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--fruit-models', nargs='*', default=[])
ap.add_argument('--fruit-max-dist', type=float, default=15.0); ap.add_argument('--out', required=True); ap.add_argument('--hold', type=int, default=10, help='frames to freeze on an answer (10 = 2 s at 5 fps)')
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey
pj = json.load(open(a.path)); FR = pj['frames']; trees = {int(k): np.array(v) for k, v in pj['trees'].items()}
H = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))
# ---- tree ordinals: order each row's trees along the row's own axis, present rows and positions 1-based
rows_out = {}; label = {}; row_of = {}
for r in sorted(H['rows'], key=lambda r: r['id']):
    ids = [t for t in r['object_ids'] if t in trees]
    if len(ids) < 3: continue
    P = np.array([trees[t][:2] for t in ids]); d = P - P.mean(0)
    axis = np.linalg.svd(d, full_matrices=False)[2][0]
    seq = [ids[i] for i in np.argsort(d @ axis)]
    lab = f"row {r['id'] + 1}"; rows_out[lab] = seq
    for k, t in enumerate(seq): label[t] = f"tree {k + 1} of {lab}"; row_of[t] = lab
# ---- per-frame tree pixels (for "which tree am I looking at", and for asking where the answer is visible)
def tree_counts():
    per = {}
    for i, f in enumerate(FR):
        bt = btm = None
        for b in a.models:
            q = Path(a.maps) / b / (f['name'] + '.npz')
            if not q.exists(): continue
            z = np.load(q); tmg = z['tmg'].astype(np.float32)
            if bt is None: bt = np.full(tmg.shape, -1, np.int32); btm = np.full(tmg.shape, -9.0, np.float32)
            m = tmg > btm; bt[m] = z['tid'][m]; btm[m] = tmg[m]
        if bt is None: continue
        u, c = np.unique(bt[btm >= 0], return_counts=True)
        for u_, c_ in zip(u.tolist(), c.tolist()): per.setdefault(int(u_), np.zeros(len(FR), int))[i] = int(c_)
    return per
per = tree_counts()
# ---- fruit per parent tree, measured over the whole drive
meta = json.load(open(S / 'experimental/h3dgs/export_meta.json'))
R_W = np.asarray(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], float)[:3, :3]
FCEN = {b: (np.array([np.asarray(x['transform_matrix'], float)[:3, 3]
                      for x in json.load(open(S / 'experimental/h3dgs_sidecar' / f'block_{b}' / 'transforms.json'))['frames']]) @ R_W.T).mean(0)
        for b in a.fruit_models}
fruit = {}; fruit_per = {}
for i, f in enumerate(FR):
    cam = np.asarray(f['c2w_h'])[:3, 3]
    for b in a.fruit_models:
        p = Path(a.fruit_maps) / b / (f['name'] + '.npz')
        if not p.exists() or np.linalg.norm(cam - FCEN[b]) > a.fruit_max_dist: continue
        z = np.load(p); m = z['fmg'].astype(np.float32) >= 0
        if not m.any(): continue
        par = z['ftree'][z['fid'][m]]
        for t in np.unique(par):
            n = int((par == t).sum()); fruit[int(t)] = fruit.get(int(t), 0) + n
            fruit_per.setdefault(int(t), np.zeros(len(FR), int))[i] += n
rank = sorted(fruit.items(), key=lambda kv: -kv[1])
best_fruit = rank[0][0] if rank else None
def window(tree, lo, hi):
    """the frame in [lo, hi] where this tree is biggest, and whether it is worth asking there"""
    v = per.get(tree)
    if v is None: return None, 0
    seg = v[lo:hi + 1]; k = int(np.argmax(seg)); return lo + k, int(seg[k])
# ---- the script. Each question is placed on a stretch where its own answer peaks.
def seg(lo, hi, mode, q, ans, focus=None, hold_at=None):
    return dict(lo=lo, hi=hi, mode=mode, question=q, answer=ans, focus=focus,
                hold_at=hold_at if hold_at is not None else (lo + hi) // 2, hold=a.hold)
N = len(FR)
t9 = rows_out.get('row 2', [None] * 9)[8] if len(rows_out.get('row 2', [])) > 8 else None
segments = []
segments.append(seg(0, 38, 'plain', '', ''))
if t9 is not None:
    f_, _ = window(t9, 39, 60); segments.append(seg(39, 60, 'tree', f'show me {label[t9]}', label[t9].capitalize(), t9, f_))
cand = max(((t, window(t, 61, 90)[1]) for t in per if row_of.get(t) == 'row 2'), key=lambda kv: kv[1], default=(None, 0))
if cand[0] is not None:
    f_, _ = window(cand[0], 61, 90); segments.append(seg(61, 90, 'tree', 'which tree am I passing?', label[cand[0]].capitalize(), cand[0], f_))
nrow = len(rows_out.get('row 1', []))
segments.append(seg(91, 125, 'row', 'how many trees in this row?', f'{nrow} trees in row 1'))
if best_fruit is not None:
    # place this one where the winner's FRUIT is visible, not where its canopy is biggest
    v = fruit_per[best_fruit]; w = 30
    conv = np.convolve(v, np.ones(w, int), 'valid'); st = int(np.argmax(conv))
    pk = st + int(np.argmax(v[st:st + w]))
    segments.append(seg(st, st + w - 1, 'fruit', 'which tree has the most fruit?',
                        f'{label.get(best_fruit, "tree " + str(best_fruit)).capitalize()}', best_fruit, pk))
    print(f'[script] fruit window {st}-{st + w - 1} chosen on tree {best_fruit} fruit pixels (peak {pk}, {int(v[pk])} px)', flush=True)
cand2 = max(((t, window(t, 156, 185)[1]) for t in per), key=lambda kv: kv[1], default=(None, 0))
if cand2[0] is not None:
    f_, _ = window(cand2[0], 156, 185); segments.append(seg(156, 185, 'tree', 'which tree am I looking at?', label.get(cand2[0], '').capitalize(), cand2[0], f_))
segments.append(seg(186, 357, 'all', 'show me every tree', ''))
segments.append(seg(358, 436, 'rows', 'group them by row', ''))
segments.append(seg(437, N - 1, 'orchard', 'the whole block', ''))
# a question placed by where its answer is visible can land inside a broad segment. Lay the broad ones down first,
# then let the specific ones (the ones with an answer) cut into them, and re-emit contiguous runs.
slot = [None] * N
for sg_ in segments:
    if sg_['answer']: continue
    for i in range(sg_['lo'], min(sg_['hi'] + 1, N)): slot[i] = sg_
for sg_ in segments:
    if not sg_['answer']: continue
    for i in range(sg_['lo'], min(sg_['hi'] + 1, N)): slot[i] = sg_
for i in range(1, N):            # a question that moved leaves a hole: carry the previous segment across it
    if slot[i] is None: slot[i] = slot[i - 1]
for i in range(N - 2, -1, -1):
    if slot[i] is None: slot[i] = slot[i + 1]
segments = []; i = 0; seen_q = set()
while i < N:
    sg_ = slot[i]
    if sg_ is None: i += 1; continue
    j = i
    while j + 1 < N and slot[j + 1] is sg_: j += 1
    out_ = dict(sg_); out_['lo'] = i; out_['hi'] = j
    out_['retype'] = out_['question'] not in seen_q; seen_q.add(out_['question'])
    if not (i <= out_['hold_at'] <= j): out_['hold_at'] = -1          # its peak fell in a slice it no longer owns
    segments.append(out_); i = j + 1
out = dict(rows=rows_out, label={str(k): v for k, v in label.items()}, fruit_rank=rank, segments=segments)
json.dump(out, open(a.out, 'w'), indent=1)
print(f'[script] rows: ' + '; '.join(f'{k} = {v}' for k, v in rows_out.items()), flush=True)
print(f'[script] fruit rank (parent tree, px): {rank[:5]}', flush=True)
for s in segments:
    print(f"[script] {s['lo']:>3}-{s['hi']:>3} {s['mode']:<8} q={s['question']!r:<34} a={s['answer']!r:<22} focus={s['focus']} hold@{s['hold_at']}", flush=True)
