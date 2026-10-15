#!/usr/bin/env python3
"""labels_from_seed.py — recover the census ASSIGNMENT (label id per gaussian, -1 = none) from an existing seed (features_B.bin) without
re-running the census algebra (2026-10-13). build_census_init writes each assigned gaussian exactly its label's target vector, so the label
is the target that matches the row (float equality after the same embedder pass); the targets come from the cell's text-bank ids and the
seed's own embedder (features_B.bin.json). Writes <features>.labels.npy like build_census_init --out-features does today.
nerf_new pixi env:  labels_from_seed.py --identity <ws>/identity [--features features_B.bin]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import lorentz as L
import open_clip
FRUIT_ID_BASE = 10000
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--features', default='features_B.bin'); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'; torch.set_grad_enabled(False); fb = json.load(open(N / (a.features + '.json')))
F = torch.from_numpy(np.fromfile(N / a.features, dtype=np.float32).reshape(fb['n'], fb['dim'])).to(dev)
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); ids = sorted(int(k) for k in meta['word_table'])
m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(fb['embedder'], dev); curv = hyper.curv.exp()
words = [get_word_for_id(u - FRUIT_ID_BASE, 'fruit') if u >= FRUIT_ID_BASE else get_word_for_id(u, 'mask') for u in ids]
nt = torch.tensor([4 if u >= FRUIT_ID_BASE else 0 for u in ids], device=dev)
ce = torch.cat([m.encode_text(tok(words[i:i + 256]).to(dev)).float() for i in range(0, len(words), 256)])
hf = hyper.encode_features(ce, project=True, node_types=nt); T = hf if getattr(hyper.config, 'euclidean', False) else L.log_map0(hf, curv=curv)
lab = np.full(F.shape[0], -1, np.int32); nz = (F.norm(dim=1) > 0).nonzero().squeeze(1); worst = 0.0
for j in range(0, len(nz), 262144):
    sel = nz[j:j + 262144]; d = torch.cdist(F[sel], T.float()); mn, am = d.min(1); worst = max(worst, mn.max().item()); lab[sel.cpu().numpy()] = np.array(ids)[am.cpu().numpy()]
np.save(str(N / a.features) + '.labels.npy', lab)
print(f'[labels] {len(nz)}/{F.shape[0]} assigned gaussians matched to {len(ids)} targets, worst distance {worst:.2e} (0 = exact) -> {N / a.features}.labels.npy', flush=True)
if worst > 1e-3: sys.exit('[labels] a seed row matches no target — the seed was not built against this embedder')
