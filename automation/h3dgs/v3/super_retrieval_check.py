#!/usr/bin/env python3
"""super_retrieval_check.py (2026-10-20) — the embedder check extended to the part level: radii of every node (tree addresses, fruit, part
image anchors) under the super embedder, each part's nearest tree node in geodesic distance (must be its own tree), each part's angle to
its tree's ray, and each tree's nearest row (the production check). nerf_new pixi env:  super_retrieval_check.py <ckpt> <text_bank.npz> <nodes.npz> <hierarchy json>"""
import json, sys
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
ck, bank_p, nodes_p, hj = sys.argv[1:5]; dev = 'cuda'; torch.set_grad_enabled(False); hyper = build_hyper_embedder(ck, dev); curv = hyper.curv.exp()
bank = np.load(bank_p, allow_pickle=True); meta = json.loads(str(bank['meta'])); wt = {int(k): w for k, w in meta['word_table'].items()}; rw = {int(k): w for k, w in meta['row_words'].items()}; words = [str(w) for w in bank['words']]
H = json.load(open(hj)); row_of = {int(o['id']): int(o['row_id']) for o in H['objects']}; nz = np.load(nodes_p); nid = [int(i) for i in nz['ids']]; ntree = [int(t) for t in nz['tree']]; ntype = [int(t) for t in nz['type']]
tree_ids = [k for k in sorted(wt) if k < 10000 and wt[k] in words and k in row_of]; row_ids = [r for r in sorted(rw) if rw[r] in words]
E = torch.tensor(bank['emb'], dtype=torch.float32, device=dev)
def enc(x, nt): return hyper.encode_features(x, project=True, node_types=torch.full((x.shape[0],), nt, device=dev, dtype=torch.long))
Xt = enc(E[[words.index(wt[k]) for k in tree_ids]], 0); Xr = enc(E[[words.index(rw[r]) for r in row_ids]], 1); Xp = enc(torch.tensor(np.asarray(nz['emb'], np.float32), device=dev), 5)
tan = lambda X: L.log_map0(X, curv=curv)
rt, rr, rp = tan(Xt).norm(dim=-1), tan(Xr).norm(dim=-1), tan(Xp).norm(dim=-1)
print(f'[check] radii: rows {rr.min():.2f}-{rr.max():.2f} (median {rr.median():.2f}), trees {rt.min():.2f}-{rt.max():.2f} ({rt.median():.2f}), parts {rp.min():.2f}-{rp.max():.2f} ({rp.median():.2f})')
D = L.pairwise_dist(Xt, Xr, curv); ok = sum(1 for i, k in enumerate(tree_ids) if row_ids[int(D[i].argmin())] == row_of[k]); print(f'[check] tree -> nearest row: {ok}/{len(tree_ids)} = {100 * ok / len(tree_ids):.1f} %')
Dp = L.pairwise_dist(Xp, Xt, curv); okp = sum(1 for j in range(len(nid)) if tree_ids[int(Dp[j].argmin())] == ntree[j]); print(f'[check] part -> nearest tree: {okp}/{len(nid)} = {100 * okp / len(nid):.1f} %')
tp, tt = tan(Xp), tan(Xt); ang = []
for j in range(len(nid)):
    i = tree_ids.index(ntree[j]); c = (tp[j] @ tt[i]) / (tp[j].norm() * tt[i].norm()); ang.append(float(torch.rad2deg(torch.arccos(c.clamp(-1, 1)))))
print(f'[check] part -> own tree ray angle: median {np.median(ang):.1f} deg, max {np.max(ang):.1f}; parts beyond their tree radius: {int(sum(1 for j in range(len(nid)) if rp[j] > rt[tree_ids.index(ntree[j])]))}/{len(nid)}')
for ty, name in ((0, 'trunk'), (1, 'branch'), (2, 'leaves')):
    sel = [j for j in range(len(nid)) if ntype[j] == ty]
    if sel: print(f'[check]   {name}: n {len(sel)}, radius median {rp[sel].median():.2f}, nearest-tree ok {sum(1 for j in sel if tree_ids[int(Dp[j].argmin())] == ntree[j])}/{len(sel)}')
