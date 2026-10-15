#!/usr/bin/env python3
"""build_flat_seed.py — paper ablation arm C (2026-10-13): the LERF-duplication analogue of the census seed. Every gaussian the census
assigned (build_census_init.py's <features>.labels.npy: label id or -1) gets THREE flat 32-d fields — its tree word, its row word and its
fruit word (zero when it carries none) — each the PCA-32 projection of the word's raw CLIP text embedding (basis fitted on the survey's
whole vocabulary: all tree, row and fruit words of the hierarchy). 96 floats per gaussian instead of 32; rows work by construction;
no hierarchy in the geometry. Scored by argmax_eval.py --space flat --basis <out>.basis.npz (decode = inverse PCA, cosine to CLIP text).
nerf_new pixi env:  build_flat_seed.py --labels <features_X.bin.labels.npy> --hierarchy-json <hj> --out <features_flat96.bin>"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from word_utils import get_word_for_id
import open_clip
FRUIT_ID_BASE = 10000
ap = argparse.ArgumentParser(); ap.add_argument('--labels', required=True); ap.add_argument('--hierarchy-json', required=True); ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--k', type=int, default=32, help='PCA dims per block (32 = arm C; 10 = the equal-storage arm, 3 x 10 = 30 floats <= the 32-d hyperbolic field)')
a = ap.parse_args(); dev = 'cuda'; torch.set_grad_enabled(False); K = a.k
lab = np.load(a.labels); Hj = json.load(open(a.hierarchy_json)); row_of = {o['id']: o['row_id'] for o in Hj['objects']}; tree_of_fruit = {f['id']: f['tree_id'] for f in Hj.get('fruits', [])}
# survey vocabulary -> PCA basis (raw CLIP, never unit-normalised, as the embedder sees it)
vocab = {('TREE', o['id']): get_word_for_id(o['id'], 'mask') for o in Hj['objects']}
vocab.update({('ROW', r['id']): get_word_for_id(r['id'], 'row') for r in Hj.get('rows', [])}); vocab.update({('ROW', r): get_word_for_id(r, 'row') for r in set(row_of.values()) if ('ROW', r) not in vocab})
vocab.update({('FRUIT', f['id']): get_word_for_id(f['id'], 'fruit') for f in Hj.get('fruits', [])})
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
keys = list(vocab); words = [vocab[k] for k in keys]
E = torch.cat([m.encode_text(tok(words[i:i + 256]).to(dev)).float() for i in range(0, len(words), 256)])
mean = E.mean(0); U, S_, V = torch.pca_lowrank(E - mean, q=K, center=False); comp = V[:, :K].T.contiguous()   # [K, 512]
Z = (E - mean) @ comp.T; zi = {k: i for i, k in enumerate(keys)}
expl = float((S_[:K] ** 2).sum() / ((E - mean) ** 2).sum())
N = len(lab); F = np.zeros((N, 3 * K), np.float32); Zc = Z.cpu().numpy()
t_of = np.full(N, -1); f_of = np.full(N, -1)
is_f = lab >= FRUIT_ID_BASE; t_of[~is_f & (lab >= 0)] = lab[~is_f & (lab >= 0)]
for g in np.flatnonzero(is_f): t_of[g] = tree_of_fruit.get(int(lab[g]) - FRUIT_ID_BASE, -1); f_of[g] = int(lab[g]) - FRUIT_ID_BASE
n_t = n_r = n_f = 0
for g in np.flatnonzero(t_of >= 0):
    t = int(t_of[g]); k = ('TREE', t)
    if k in zi: F[g, :K] = Zc[zi[k]]; n_t += 1
    r = row_of.get(t)
    if r is not None and ('ROW', r) in zi: F[g, K:2 * K] = Zc[zi[('ROW', r)]]; n_r += 1
    if f_of[g] >= 0 and ('FRUIT', int(f_of[g])) in zi: F[g, 2 * K:] = Zc[zi[('FRUIT', int(f_of[g]))]]; n_f += 1
a.out.parent.mkdir(parents=True, exist_ok=True); F.tofile(a.out)
np.savez(str(a.out) + '.basis.npz', components=comp.cpu().numpy(), mean=mean.cpu().numpy(), words=np.array(words), kinds=np.array([k[0] for k in keys]), ids=np.array([k[1] for k in keys]))
json.dump({'n': N, 'dim': 3 * K, 'blocks': ['tree', 'row', 'fruit'], 'assigned': int((lab >= 0).sum()), 'seeded_tree': n_t, 'seeded_row': n_r, 'seeded_fruit': n_f, 'labels': a.labels,
           'hierarchy_json': a.hierarchy_json, 'vocab': len(words), 'pca_explained': expl}, open(str(a.out) + '.json', 'w'), indent=1)
print(f'[flat-seed] {N} gaussians: tree {n_t}, row {n_r}, fruit {n_f} seeded; vocab {len(words)} words, PCA-{K} explains {100 * expl:.1f}% -> {a.out}', flush=True)
