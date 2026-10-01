"""Question script for a CHUNK demo drive (2026-10-01, the Tuesday-demo chunks), from the identity maps — the chunk version
of sidecar_demo_script.py: the drive is several ~110-frame row passes of a 30 m cell (jump cuts between them), so the segments
are laid out PROPORTIONALLY to the drive length instead of the 09-28 fixed frame numbers, and the questions are the demo's:
  plain -> "show me this row" -> "which tree is this?" (the largest tree in view on that stretch) -> "show me every tree"
  -> "group them by row" -> [fruit: the bilingual question on the fruit-ranked tree, only when --fruit-models is given]
  -> "the whole orchard".  --end-of-row N: a 'tree' segment on the LAST tree of hierarchy row N ("Show me the trees at the
  end of row N", the 01 question).  Tree ordinals ("tree 3 of row 2") come from the hierarchy rows ordered along their axis.
Writes demo_script.json in the format sidecar_demo_overlay.py --script reads.
  python sidecar_demo_script_chunk.py --survey 05_13D_Jackal --path <demo_path.json> --maps <out>/maps --fruit-maps <out>/maps_fruit
      --models chunk_1_0_expo [--fruit-models chunk_1_0_expo] [--fruit-count 61] [--end-of-row 2] --out <demo_script.json>"""
import argparse, json
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--survey', required=True); ap.add_argument('--path', required=True)
ap.add_argument('--maps', required=True); ap.add_argument('--fruit-maps', default='')
ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--fruit-models', nargs='*', default=[])
ap.add_argument('--fruit-max-dist', type=float, default=30.0); ap.add_argument('--out', required=True); ap.add_argument('--hold', type=int, default=16, help='frames to freeze on an answer (16 = 2 s at 8 fps)')
ap.add_argument('--fruit-count', type=int, default=0, help='confirmed fruit count to state in the answer (from the registry, e.g. 61 for 05 tree 5); 0 = count not stated')
ap.add_argument('--fruit-tree', type=int, default=-1, help='force the fruit answer tree (default: the tree with the most fruit pixels over the drive)')
ap.add_argument('--end-of-row', type=int, default=-1, help="ask 'Show me the trees at the end of row N' (1-based hierarchy row id as presented) instead of 'which tree is this?'")
a = ap.parse_args()
S = Path('/home/paperspace/data/citrus_all') / a.survey
pj = json.load(open(a.path)); FR = pj['frames']; trees = {int(k): np.array(v) for k, v in pj['trees'].items()}; N = len(FR)
H = json.load(open(S / 'prod/bateleur/scene_graph/marker_hierarchy.json'))
rows_out = {}; label = {}; row_of = {}
for r in sorted(H['rows'], key=lambda r: r['id']):
    ids = [t for t in r['object_ids'] if t in trees]
    if len(ids) < 2: continue
    P = np.array([trees[t][:2] for t in ids]); d = P - P.mean(0); axis = np.linalg.svd(d, full_matrices=False)[2][0]
    seq = [ids[i] for i in np.argsort(d @ axis)]; lab = f"row {r['id'] + 1}"; rows_out[lab] = seq
    for k, t in enumerate(seq): label[t] = f"tree {k + 1} of {lab}"; row_of[t] = lab
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
        for u_, c_ in zip(u.tolist(), c.tolist()): per.setdefault(int(u_), np.zeros(N, int))[i] = int(c_)
    return per
per = tree_counts()
fruit = {}; fruit_per = {}
if a.fruit_models and a.fruit_maps:
    for i, f in enumerate(FR):
        for b in a.fruit_models:
            p = Path(a.fruit_maps) / b / (f['name'] + '.npz')
            if not p.exists(): continue
            z = np.load(p); m = z['fmg'].astype(np.float32) >= 0
            if not m.any(): continue
            par = z['ftree'][z['fid'][m]]
            for t in np.unique(par):
                n = int((par == t).sum()); fruit[int(t)] = fruit.get(int(t), 0) + n; fruit_per.setdefault(int(t), np.zeros(N, int))[i] += n
rank = sorted(fruit.items(), key=lambda kv: -kv[1])
best_fruit = a.fruit_tree if a.fruit_tree >= 0 else (rank[0][0] if rank else None)
def window(tree, lo, hi):
    v = per.get(tree)
    if v is None: return None, 0
    seg = v[lo:hi + 1]; k = int(np.argmax(seg)); return lo + k, int(seg[k])
def seg(lo, hi, mode, q, ans, focus=None, hold_at=None):
    return dict(lo=lo, hi=hi, mode=mode, question=q, answer=ans, focus=focus, hold_at=hold_at if hold_at is not None else (lo + hi) // 2, hold=a.hold)
F = lambda frac: int(round(frac * (N - 1)))
segments = [seg(0, F(0.08), 'plain', '', ''), seg(F(0.08) + 1, F(0.24), 'row', 'show me this row', '')]
lo, hi = F(0.24) + 1, F(0.36)
if a.end_of_row > 0 and f'row {a.end_of_row}' in rows_out:
    last = rows_out[f'row {a.end_of_row}'][-1]; f_, n_ = window(last, 0, N - 1)
    if f_ is not None and n_ > 0:
        lo2, hi2 = max(0, f_ - 12), min(N - 1, f_ + 28)
        segments.append(seg(lo2, hi2, 'tree', f'show me the trees at the end of row {a.end_of_row}', label[last].capitalize(), last, f_))
else:
    cand = max(((t, window(t, lo, hi)[1]) for t in per), key=lambda kv: kv[1], default=(None, 0))
    if cand[0] is not None and cand[1] > 0:
        f_, _ = window(cand[0], lo, hi); segments.append(seg(lo, hi, 'tree', 'which tree is this?', label.get(cand[0], f'tree {cand[0]}').capitalize(), cand[0], f_))
segments.append(seg(F(0.36) + 1, F(0.58), 'all', 'show me every tree', ''))
segments.append(seg(F(0.58) + 1, F(0.74), 'rows', 'group them by row', ''))
if best_fruit is not None and best_fruit in fruit_per:
    v = fruit_per[best_fruit]; w = min(40, N); conv = np.convolve(v, np.ones(w, int), 'valid'); st = int(np.argmax(conv)); pk = st + int(np.argmax(v[st:st + w]))
    cnt = f' — {a.fruit_count} oranges' if a.fruit_count else ''
    segments.append(seg(st, min(N - 1, st + w - 1), 'fruit', 'Ni mti gani wenye machungwa mengi zaidi? Nionyeshe.  /  Which tree has the most oranges? Show me.',
                        f'{label.get(best_fruit, "tree " + str(best_fruit)).capitalize()}{cnt}', best_fruit, pk))
    print(f'[script] fruit window {st}-{st + w - 1} on tree {best_fruit} (peak frame {pk}, {int(v[pk])} fruit px; drive total {fruit.get(best_fruit, 0)} px)', flush=True)
elif a.fruit_models: print('[script] fruit requested but no fruit pixels on the drive — no fruit segment', flush=True)
segments.append(seg(F(0.74) + 1, N - 1, 'orchard', 'the whole orchard', ''))
slot = [None] * N
for sg_ in segments:
    if sg_['answer']: continue
    for i in range(sg_['lo'], min(sg_['hi'] + 1, N)): slot[i] = sg_
for sg_ in segments:
    if not sg_['answer']: continue
    for i in range(sg_['lo'], min(sg_['hi'] + 1, N)): slot[i] = sg_
for i in range(1, N):
    if slot[i] is None: slot[i] = slot[i - 1]
for i in range(N - 2, -1, -1):
    if slot[i] is None: slot[i] = slot[i + 1]
segments = []; i = 0; seen_q = set()
while i < N:
    sg_ = slot[i]; j = i
    while j + 1 < N and slot[j + 1] is sg_: j += 1
    out_ = dict(sg_); out_['lo'] = i; out_['hi'] = j; out_['retype'] = out_['question'] not in seen_q; seen_q.add(out_['question'])
    if not (i <= out_['hold_at'] <= j): out_['hold_at'] = -1
    segments.append(out_); i = j + 1
json.dump(dict(rows=rows_out, label={str(k): v for k, v in label.items()}, fruit_rank=rank, segments=segments), open(a.out, 'w'), indent=1)
print(f'[script] {N} frames; rows: ' + '; '.join(f'{k} = {v}' for k, v in rows_out.items()), flush=True)
for s_ in segments: print(f"[script] {s_['lo']:>3}-{s_['hi']:>3} {s_['mode']:<8} q={s_['question'][:40]!r:<42} a={s_['answer']!r:<28} focus={s_['focus']} hold@{s_['hold_at']}", flush=True)
