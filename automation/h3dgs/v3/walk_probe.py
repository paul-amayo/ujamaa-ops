#!/usr/bin/env python3
"""walk_probe.py --identity <cell> --frame kf_002411.png --pixels y,x;y,x --out <npz> (2026-10-17) — the containment walk of a few real pixels for the
method figure: for each pixel, the cosine between the decode of its own point and of each of the 4 geodesic waypoints toward the origin (extrapolated
waypoints masked) and the CLIP-text embedding of EVERY address of the cell; also the pixel's tangent norm. nerf_new pixi env."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--frame', required=True); ap.add_argument('--pixels', required=True); ap.add_argument('--maps', default='maps_v3'); ap.add_argument('--out', required=True); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'; EMB = json.load(open(N / 'features_B.bin.json'))['embedder']; hyper = build_hyper_embedder(EMB, dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); emb = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); emb = emb / emb.norm(dim=-1, keepdim=True); meta = json.loads(str(bank['meta']))
z = np.load(N / a.maps / a.frame.replace('.png', '.npz')); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev)
pix = [tuple(int(v) for v in p.split(',')) for p in a.pixels.split(';')]; out = {'words': np.array(words), 'meta': json.dumps({'word_table': meta['word_table'], 'row_words': meta['row_words']}), 'pixels': np.array(pix)}
with torch.no_grad():
    ft = torch.stack([feat[y, x] for y, x in pix]); h = L.exp_map0(ft, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
    d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    cos_path = (d @ emb.T).cpu().numpy()                 # [n_pix, 4, n_words]
    cos_own = (d0 @ emb.T).cpu().numpy()                 # [n_pix, n_words]
    out['cos_path'] = cos_path; out['cos_own'] = cos_own; out['extrapolated'] = mask.cpu().numpy().astype(bool).reshape(len(pix), 4) if mask.numel() == len(pix) * 4 else mask.cpu().numpy(); out['norm'] = ft.norm(dim=-1).cpu().numpy(); out['waypoint_norm'] = L.log_map0(path.reshape(-1, path.shape[-1]).to(torch.float64), curv).norm(dim=-1).reshape(len(pix), 4).cpu().numpy()
    for i, (y, x) in enumerate(pix):
        top = np.argsort(-cos_own[i])[:3]; print(f"[walk] pixel ({y},{x}) norm {out['norm'][i]:.2f}: own point top addresses " + ', '.join(f"{words[j]}={cos_own[i, j]:.2f}" for j in top) + '; waypoint norms ' + ' '.join(f"{v:.2f}" for v in out['waypoint_norm'][i]) + ' extrapolated ' + str(out['extrapolated'][i].tolist()))
np.savez_compressed(a.out, **out); print('[walk] wrote', a.out)
