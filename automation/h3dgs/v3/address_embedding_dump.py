#!/usr/bin/env python3
"""address_embedding_dump.py --identity <cell> --out <npz> (2026-10-17) — the cell's addresses as the EMBEDDER places them: raw (un-normalised)
CLIP-text embeddings of every bank word, exactly as build_census_init.py feeds them, encoded to tangent vectors (project=False); also the
normalised variant for comparison. Saves words, level per word, tangent vectors and norms. nerf_new pixi env."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/high')
from high_splat_hierarchy_accuracy import build_hyper_embedder
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'; EMB = json.load(open(N / 'features_B.bin.json'))['embedder']; hyper = build_hyper_embedder(EMB, dev)
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); meta = json.loads(str(bank['meta'])); wt = {w: int(k) for k, w in meta['word_table'].items()}; rw = {w: int(k) for k, w in meta['row_words'].items()}
import open_clip
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
with torch.no_grad():
    raw = model.encode_text(tok(words).to(dev)).float(); nrm = raw / raw.norm(dim=-1, keepdim=True)
    t_raw = hyper.encode_features(raw, project=False).cpu().numpy(); t_nrm = hyper.encode_features(nrm, project=False).cpu().numpy()
level = np.array(['row' if w in rw else ('fruit' if wt.get(w, 0) >= 10000 else 'tree') for w in words]); ids = np.array([rw[w] if w in rw else wt.get(w, -1) for w in words])
for lv in ('row', 'tree', 'fruit'):
    m = level == lv; nr = np.linalg.norm(t_raw[m], axis=1); nn = np.linalg.norm(t_nrm[m], axis=1); print(f"[addr] {lv:5s} n {m.sum():2d}: raw-CLIP tangent norm median {np.median(nr):.2f} ({nr.min():.2f}-{nr.max():.2f}); normalised-CLIP {np.median(nn):.2f} ({nn.min():.2f}-{nn.max():.2f}); raw CLIP text norm median {raw[torch.tensor(m)].norm(dim=-1).median().item():.2f}")
np.savez_compressed(a.out, words=np.array(words), level=level, ids=ids, tangent_raw=t_raw, tangent_nrm=t_nrm); print('[addr] wrote', a.out)
