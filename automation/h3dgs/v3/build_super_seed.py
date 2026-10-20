#!/usr/bin/env python3
"""build_super_seed.py (2026-10-20, CVPR super hierarchy, step 5a) — census-init for the super hierarchy: build_census_init.py's assignment
rules (floors, background competes, fruit share-assign) over a census W whose labels are tree ids (< 10000), fruit ids (10000+) and part-type
node ids (20000+), with the anchors of trees and fruit the RAW CLIP-text embeddings of their addresses (as production) and the anchors of
parts their RAW CLIP-image node embeddings (nodes.npz), all encoded by the SUPER embedder. Writes features [N, 32] float32 (tangent vectors,
0 = void) and <out>.labels.npy (assigned label per Gaussian, -1 = none) + <out>.json.
nerf_new pixi env:  build_super_seed.py --w-npz <W.npz> --embedder <ckpt> --nodes-npz <nodes.npz> --out-features <bin>
                      [--tree-floor 1.0] [--bg-competes] [--bg-ratio 2] [--fruit-floor 0.01] [--fruit-share-assign 0.1]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, open_clip
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import lorentz as L
ap = argparse.ArgumentParser(); ap.add_argument('--w-npz', required=True); ap.add_argument('--embedder', required=True); ap.add_argument('--nodes-npz', required=True); ap.add_argument('--out-features', required=True, type=Path)
ap.add_argument('--tree-floor', type=float, default=1.0); ap.add_argument('--bg-competes', action='store_true'); ap.add_argument('--bg-ratio', type=float, default=2.0); ap.add_argument('--fruit-floor', type=float, default=0.01); ap.add_argument('--fruit-share-assign', type=float, default=0.1); ap.add_argument('--part-floor', type=float, default=None, help='mass floor for part-majority gaussians (default = tree floor; pre-registered 10-20 for the parts-as-fruit rule: 0.01, the fruit floor)'); ap.add_argument('--bg-vs-subtree', action='store_true', help='background competes against the mass of the majority node\'s tree subtree (tree + its parts) instead of the single best label (parts-as-fruit rule)'); ap.add_argument('--part-share-assign', type=float, default=0.0, help='seed a gaussian with the part of its majority tree whose mass is >= this share of the tree subtree mass (tree + its parts); 0 = majority only. Pre-registered 10-20: 0.1, the production fruit constant')
a = ap.parse_args(); dev = 'cuda'; torch.set_grad_enabled(False); FRUIT, PART, UNLAB = 10000, 20000, 65535
z = np.load(a.w_npz); W, labels = z['W'], [int(u) for u in z['labels']]; nz = np.load(a.nodes_npz); nid = [int(i) for i in nz['ids']]; nemb = np.asarray(nz['emb'], np.float32)
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16'); hyper = build_hyper_embedder(a.embedder, dev); curv = hyper.curv.exp()
bg_mass = None
if UNLAB in labels:
    bi = labels.index(UNLAB); bg_mass = W[bi]; keep = [i for i in range(len(labels)) if i != bi]; W = W[keep]; labels = [labels[i] for i in keep]
kind = np.array([2 if u >= PART else (1 if u >= FRUIT else 0) for u in labels])   # 0 tree, 1 fruit, 2 part
anchors = np.zeros((len(labels), 512), np.float32); words = {}
tx = [u for u in labels if u < PART]
if tx:
    ws = [(get_word_for_id(u - FRUIT, 'fruit') if u >= FRUIT else get_word_for_id(u, 'mask')) for u in tx]; ce = m.encode_text(tok(ws).to(dev)).float().cpu().numpy()
    for u, w, e in zip(tx, ws, ce): anchors[labels.index(u)] = e; words[u] = w
missing = []
for u in labels:
    if u >= PART:
        if u in nid: anchors[labels.index(u)] = nemb[nid.index(u)]
        else: missing.append(u)
nt = torch.tensor([4 if k == 1 else (5 if k == 2 else 0) for k in kind], device=dev)
hf = hyper.encode_features(torch.from_numpy(anchors).to(dev), project=True, node_types=nt); tl = L.log_map0(hf, curv=curv).cpu().numpy()
if missing: print(f'[super-seed] {len(missing)} part labels in W without a node (below min masks): their rows are dropped'); keep = [i for i, u in enumerate(labels) if u not in missing]; W = W[keep]; labels = [labels[i] for i in keep]; tl = tl[keep]; kind = kind[keep]
norms = np.linalg.norm(tl, axis=1); print('[super-seed] target norms: trees %.2f-%.2f, fruit %s, parts %s' % (norms[kind == 0].min(), norms[kind == 0].max(), f'{norms[kind == 1].min():.2f}-{norms[kind == 1].max():.2f}' if (kind == 1).any() else '-', f'{norms[kind == 2].min():.2f}-{norms[kind == 2].max():.2f}' if (kind == 2).any() else '-'))
tot = W.sum(0); maj = W.argmax(0); is_fruit = (kind == 1)[maj]; is_part = (kind == 2)[maj]; part_floor = a.tree_floor if a.part_floor is None else a.part_floor
floor = np.where(is_fruit, a.fruit_floor, np.where(is_part, part_floor, a.tree_floor)); assign = tot > floor
lab_arr = np.array(labels); tree_row = {u: i for i, u in enumerate(labels) if u < FRUIT}
sub_mass = {}   # per tree with parts: mass of the subtree (tree + its parts) per gaussian
for i, u in enumerate(labels):
    if u >= PART: t = (u - PART) // 4; sub_mass.setdefault(t, (W[tree_row[t]].copy() if t in tree_row else np.zeros(W.shape[1]))); sub_mass[t] += W[i]
if a.bg_competes and bg_mass is not None:
    best = W.max(0).copy()
    if a.bg_vs_subtree:
        for t, sm in sub_mass.items(): sel = is_part & (((lab_arr[maj] - PART) // 4) == t); best[sel] = np.maximum(best[sel], sm[sel])
    withheld = assign & (bg_mass >= a.bg_ratio * best); assign &= ~withheld; print(f'[super-seed] background competes ({"vs subtree" if a.bg_vs_subtree else "vs best label"}): {int(withheld.sum())} withheld; part floor {part_floor:g}')
if a.fruit_share_assign > 0 and (kind == 1).any():
    Wf = W.copy(); Wf[kind != 1] = 0.0; fshare = np.zeros_like(tot); sup_ = tot > 0; fshare[sup_] = Wf.sum(0)[sup_] / tot[sup_]; fmaj = np.where(Wf.sum(0) > 0, Wf.argmax(0), maj); promote = sup_ & (fshare > a.fruit_share_assign); maj = np.where(promote, fmaj, maj); assign = assign | promote
    print(f'[super-seed] share-assign: {int(promote.sum())} gaussians seeded with their fruit')
if a.part_share_assign > 0 and (kind == 2).any():   # the part level is finer than the tree: a gaussian rendering into a part of its tree with a share of the subtree mass >= tau carries the part
    lab_arr = np.array(labels); tree_of_part = {u: (u - PART) // 4 for u in labels if u >= PART}; promoted = 0
    for ti, t in enumerate(labels):
        if kind[ti] != 0: continue
        prow = [i for i, u in enumerate(labels) if u >= PART and tree_of_part[u] == t]
        if not prow: continue
        sel = np.where(assign & (lab_arr[maj] == t))[0]
        if not len(sel): continue
        sub = W[ti][sel] + W[prow][:, sel].sum(0); best = np.array(prow)[W[prow][:, sel].argmax(0)]; share = W[best, sel] / np.maximum(sub, 1e-9); go = share >= a.part_share_assign
        maj[sel[go]] = best[go]; promoted += int(go.sum())
    print(f'[super-seed] part share-assign (>= {a.part_share_assign:g} of the subtree mass): {promoted} gaussians seeded with a part of their tree')
feats = np.zeros((W.shape[1], tl.shape[1]), np.float32); feats[assign] = tl[maj[assign]]; lab = np.where(assign, np.array(labels)[maj], -1).astype(np.int32)
feats.tofile(a.out_features); np.save(str(a.out_features) + '.labels.npy', lab)
cnt = {'tree': int(((kind == 0)[maj] & assign).sum()), 'fruit': int(((kind == 1)[maj] & assign).sum()), 'part': int(((kind == 2)[maj] & assign).sum()), 'void': int((~assign).sum())}
json.dump({'n': int(W.shape[1]), 'dim': int(tl.shape[1]), 'w_npz': a.w_npz, 'embedder': a.embedder, 'nodes_npz': a.nodes_npz, 'assigned': cnt, 'labels': len(labels), 'tree_floor': a.tree_floor, 'bg_competes': a.bg_competes, 'bg_ratio': a.bg_ratio, 'fruit_floor': a.fruit_floor, 'fruit_share_assign': a.fruit_share_assign, 'part_share_assign': a.part_share_assign, 'part_floor': part_floor, 'bg_vs_subtree': a.bg_vs_subtree}, open(str(a.out_features) + '.json', 'w'), indent=1)
print(f"[super-seed] {W.shape[1]} gaussians: assigned {cnt} over {len(labels)} labels -> {a.out_features}")
