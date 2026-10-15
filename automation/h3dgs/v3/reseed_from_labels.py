#!/usr/bin/env python3
"""reseed_from_labels.py — seed per-gaussian features from an existing census ASSIGNMENT (build_census_init.py's <features>.labels.npy)
with another embedder's targets (paper ablation arms B/N, 2026-10-13). The assignment (majority label, floors, bg-competes, fruit share)
depends only on the census W, never on the embedder, so the ablation arms reuse it instead of re-running the W algebra (Dec lanes: a
772 x 1.47 M matrix, half an hour per pass). Targets exactly as build_census_init: RAW CLIP text of the live word -> encode_features
with node_types (fruit = 4) -> tangent (log_map0) for a hyperbolic embedder, the embedding itself for a Euclidean one.
nerf_new pixi env:  reseed_from_labels.py --labels <labels.npy> --embedder <ckpt> --out <features.bin>"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import lorentz as L
import open_clip
FRUIT_ID_BASE = 10000
ap = argparse.ArgumentParser(); ap.add_argument('--labels', required=True); ap.add_argument('--embedder', required=True); ap.add_argument('--out', required=True, type=Path); a = ap.parse_args()
dev = 'cuda'; torch.set_grad_enabled(False); lab = np.load(a.labels); ids = sorted(int(u) for u in np.unique(lab) if u >= 0)
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(a.embedder, dev); curv = hyper.curv.exp()
words = [get_word_for_id(u - FRUIT_ID_BASE, 'fruit') if u >= FRUIT_ID_BASE else get_word_for_id(u, 'mask') for u in ids]
nt = torch.tensor([4 if u >= FRUIT_ID_BASE else 0 for u in ids], device=dev)
ce = torch.cat([m.encode_text(tok(words[i:i + 256]).to(dev)).float() for i in range(0, len(words), 256)])   # RAW — never unit-normalise
hf = hyper.encode_features(ce, project=True, node_types=nt); tl = (hf if getattr(hyper.config, 'euclidean', False) else L.log_map0(hf, curv=curv)).cpu().numpy()
feats = np.zeros((len(lab), tl.shape[1]), np.float32); idx = {u: i for i, u in enumerate(ids)}
sel = lab >= 0; feats[sel] = tl[np.vectorize(idx.get)(lab[sel])]
a.out.parent.mkdir(parents=True, exist_ok=True); feats.tofile(a.out)
json.dump({'n': int(len(lab)), 'dim': int(tl.shape[1]), 'assigned': int(sel.sum()), 'labels': a.labels, 'embedder': a.embedder, 'space': 'euclidean' if getattr(hyper.config, 'euclidean', False) else 'hyperbolic'}, open(str(a.out) + '.json', 'w'), indent=1)
print(f'[reseed] {sel.sum()}/{len(lab)} gaussians ({100 * sel.mean():.1f}%) from {len(ids)} labels, target norms {np.linalg.norm(tl, axis=1).min():.2f}-{np.linalg.norm(tl, axis=1).max():.2f} -> {a.out}', flush=True)
