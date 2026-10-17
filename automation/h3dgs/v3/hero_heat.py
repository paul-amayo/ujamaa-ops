#!/usr/bin/env python3
"""hero_heat.py --identity <cell identity dir> --frame kf_002411.png --words birch,cena,bulge --out <npz> (2026-10-17) — containment heat
(containment_eval.py's decode: exp_map0 -> 4-step walk toward the origin, extrapolated steps masked, own point, cosine to the CLIP-text word,
max; void where ||f|| < 0.5) of a few words on one rendered feature map (maps_v3/<frame>.npz) for the paper's hero figure. nerf_new pixi env."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
import lorentz as L
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--frame', required=True); ap.add_argument('--words', required=True); ap.add_argument('--out', required=True); ap.add_argument('--maps', default='maps_v3'); ap.add_argument('--dump-addresses', action='store_true', help='also save every bank address as its tangent vector (the real embedding) and a sample of the cell per-Gaussian features'); ap.add_argument('--gaussians', type=int, default=20000); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'; EMB = json.load(open(N / 'features_B.bin.json'))['embedder']; hyper = build_hyper_embedder(EMB, dev); curv = hyper.curv.exp()
bank = np.load(N / 'text_bank.npz', allow_pickle=True); words = list(bank['words']); emb = torch.tensor(bank['emb'], dtype=torch.float32, device=dev); emb = emb / emb.norm(dim=-1, keepdim=True)
z = np.load(N / a.maps / a.frame.replace('.png', '.npz')); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev); H, W, D = feat.shape; ft = feat.reshape(-1, D); void = (ft.norm(dim=-1) < 0.5)
meta = json.loads(str(bank['meta'])); out = {'rgb': z['rgb'], 'alpha': z['alpha'], 'void': void.cpu().numpy().reshape(H, W), 'meta': json.dumps({'word_table': meta['word_table'], 'row_words': meta['row_words']})}
with torch.no_grad():
    h = L.exp_map0(ft, curv=curv); path, mask = L.get_interpolated_hyperbolic_features(h, steps=4, curv=curv, max_dist=11.1, return_mask=True)
    d = hyper.decode_features(path, project=True); d = d / d.norm(dim=-1, keepdim=True); d0 = hyper.decode_features(h, project=True); d0 = d0 / d0.norm(dim=-1, keepdim=True)
    sel = words if a.words == 'all' else a.words.split(',')
    for w in sel:
        e = emb[words.index(w)]; sm = (d @ e).view(ft.shape[0], 4).masked_fill(mask.to(dev).bool(), -1.0); own = d0 @ e; heat = torch.maximum(sm.max(1).values, own); heat[void] = -1.0; out[w] = heat.cpu().numpy().reshape(H, W)
        print(f"[hero] {w}: heat max {out[w].max():.3f}, pixels >= 0.6: {100 * (out[w] >= 0.6).mean():.1f} %")
if a.dump_addresses:
    with torch.no_grad():
        tang = hyper.encode_features(torch.tensor(bank['emb'], dtype=torch.float32, device=dev), project=False)   # RAW CLIP text, as build_census_init feeds it (normalising first shrinks every radius)
    if tang is None:
        import inspect; print('[hero] encoder methods:', [m for m in dir(hyper) if 'enc' in m.lower()][:10])
    else:
        out['address_words'] = np.array(words); out['address_tangent'] = tang.detach().cpu().numpy(); n = json.load(open(N / 'features_B.bin.json'))['n']; F = np.fromfile(N / 'features_B.bin', dtype=np.float32).reshape(n, 32); nz = np.where(np.linalg.norm(F, axis=1) > 0.5)[0]; rng = np.random.default_rng(0); sel = rng.choice(nz, min(a.gaussians, len(nz)), replace=False); out['gaussian_tangent'] = F[sel]; print(f"[hero] addresses {len(words)}: tangent norms rows/trees/fruit by id from meta; gaussians sampled {len(sel)} of {len(nz)} non-void ({n} total)")
np.savez_compressed(a.out, **out); print('[hero] wrote', a.out)
