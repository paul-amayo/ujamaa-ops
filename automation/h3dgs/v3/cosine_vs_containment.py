#!/usr/bin/env python3
"""cosine_vs_containment.py (2026-11-10, Paul: "if the encoder and decoder are trained well, there is no reason why this is not just a
cosine similarity ... find evidence that I'm right or I'm wrong"). On the void-free feature maps of a cell's protocol frames, three
readouts of the SAME features, each scored by the protocol's argmax rule over the level's addresses present in the cell:
  decoded-cosine   cos(h_phi(f(p)), t_q): the pixel decoded at its own feature, compared with the query's anchor (= arm D, own point)
  tangent-cosine   cos(f(p), g_theta(t_q)): the directions of the pixel's tangent vector and of the address's tangent vector, no decoding
  containment      the letter's s_q: max of the decoded cosine at the pixel and at the inner waypoints of its geodesic
Held-out heads (frame x node covering >= 0.2 % of the frame) are scored by IoU against the supervision; medians per level.
nerf_new pixi env:  cosine_vs_containment.py --identity <cell identity> --maps <dir of void-free maps> [--json out]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--maps', required=True); ap.add_argument('--json', default=''); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'
inv = {json.loads(l)['identity']: json.loads(l) for l in open('/home/paperspace/logs/identity_inventory_v2.jsonl')}; SUP = Path(inv[str(N)]['sup'])
hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); meta = json.loads(str(bank['meta']))
raw = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); anc = raw / raw.norm(dim=-1, keepdim=True)
with torch.no_grad(): tang = hyper.encode_features(raw, project=False)   # the addresses' tangent vectors (raw CLIP text in, as the census seeds)
tdir = tang / tang.norm(dim=-1, keepdim=True)
H = json.load(open(meta['hierarchy_json'])); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}
wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}
frames = json.load(open(N / 'frames.json'))['frames']; ev = set(json.load(open(N / 'split_names.json'))['eval'])
def readouts(ft, idx):   # [P, D] -> three [P, len(idx)] score matrices
    out = {k: [] for k in ('decoded-cosine', 'tangent-cosine', 'containment')}
    Qa, Qt = anc[idx], tdir[idx]
    for c0 in range(0, ft.shape[0], 32768):
        f = ft[c0:c0 + 32768]; h = L.exp_map0(f, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
        d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
        own = d0 @ Qa.T; sm = (d.view(f.shape[0], 4, -1) @ Qa.T).masked_fill(mask.to(dev).bool().view(f.shape[0], 4)[..., None], -1.0).max(1).values
        out['decoded-cosine'].append(own); out['containment'].append(torch.maximum(own, sm)); out['tangent-cosine'].append((f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)) @ Qt.T)
    return {k: torch.cat(v) for k, v in out.items()}
res = {lvl: {k: [] for k in ('decoded-cosine', 'tangent-cosine', 'containment')} for lvl in ('TREE', 'ROW')}
for fr in frames:
    if fr not in ev or not (Path(a.maps) / fr.replace('.png', '.npz')).exists(): continue
    z = np.load(Path(a.maps) / fr.replace('.png', '.npz')); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev); Hh, Ww, D = feat.shape; ft = feat.reshape(-1, D)
    void = (ft.norm(dim=-1) < 0.5).cpu().numpy(); I = np.array(Image.fromarray(np.array(Image.open(SUP / fr))).resize((Ww, Hh), Image.NEAREST)).astype(np.int64).reshape(-1)
    for lvl, ids, word in (('TREE', [t for t in sorted(wt) if t < 10000 and wt[t] in words], lambda k: wt[k]), ('ROW', [r for r in sorted(rw) if rw[r] in words], lambda k: rw[k])):
        if not ids: continue
        idx = [words.index(word(k)) for k in ids]
        with torch.no_grad(): sc = readouts(ft, idx)
        truths = {k: (I == k) if lvl == 'TREE' else np.isin(I, [t for t, r in row_of.items() if r == k]) for k in ids}
        for name, S in sc.items():
            am = np.array(ids)[S.argmax(1).cpu().numpy()]; am[void] = -1
            for k, t in truths.items():
                if t.mean() < 0.002: continue
                p = am == k; u = (p | t).sum(); res[lvl][name].append(float((p & t).sum() / u) if u else 0.0)
summary = {lvl: {name: (float(np.median(v)) if v else None, len(v)) for name, v in d.items()} for lvl, d in res.items()}
print(f'[cos] {N.parent.name}: ' + '; '.join(f'{lvl} ' + ', '.join(f'{n} {m:.3f} (n={c})' for n, (m, c) in d.items() if m is not None) for lvl, d in summary.items()))
if a.json: json.dump(dict(cell=str(N), summary=summary), open(a.json, 'w'), indent=1)
