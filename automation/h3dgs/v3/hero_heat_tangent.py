#!/usr/bin/env python3
"""hero_heat_tangent.py (2026-11-11, letter v39 review item 9) — Fig. 1's HiGH heats under the letter's readout, the encoder path:
for every word of an existing hero heat npz (same vocabulary, meta, rgb, alpha), the cosine of the pixel's void-free feature with g(t_word)
(raw CLIP text through the cell embedder's encoder, as the census seeds); void = ||f|| < 0.5. Written in hero_heat's format so hero_grid.py
reads it with --heat-t / --heat-f unchanged.
nerf_new pixi env:  hero_heat_tangent.py --heat <old heat npz> --maps <void-free feature map npz> --out <npz>"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, open_clip
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from high_splat_hierarchy_accuracy import build_hyper_embedder
ap = argparse.ArgumentParser(); ap.add_argument('--heat', required=True); ap.add_argument('--maps', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
N = Path('/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/v3/chunk_1_0/identity'); dev = 'cuda'
old = np.load(a.heat, allow_pickle=True); words = [w for w in old.files if w not in ('rgb', 'alpha', 'void', 'meta')]
hyper = build_hyper_embedder(json.load(open(N / 'features_B.bin.json'))['embedder'], dev)
clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
bank = np.load(N / 'text_bank.npz', allow_pickle=True); bw = list(bank['words'])
with torch.no_grad():
    raw = torch.stack([torch.tensor(bank['emb'][bw.index(w)], dtype=torch.float32) if w in bw else clip.encode_text(tok([w]).to(dev)).float()[0].cpu() for w in words]).to(dev)
    g = hyper.encode_features(raw, project=False); g = g / g.norm(dim=-1, keepdim=True)
    z = np.load(a.maps); f = torch.tensor(z['features'], dtype=torch.float32, device=dev); H, W, D = f.shape; f = f.reshape(-1, D)
    S = (f / f.norm(dim=-1, keepdim=True).clamp(min=1e-6)) @ g.T
void = (f.norm(dim=-1) < 0.5).view(H, W).cpu().numpy()
out = {w: S[:, i].view(H, W).cpu().numpy().astype(np.float32) for i, w in enumerate(words)}
np.savez_compressed(a.out, rgb=z['rgb'] if 'rgb' in z.files else old['rgb'], alpha=old['alpha'], void=void, meta=old['meta'], **out)
print(f'[hero-tangent] {len(words)} words, {H}x{W}, void {100 * void.mean():.1f} % -> {a.out}')
