#!/usr/bin/env python3
"""lite_relevancy.py --identity <cell> --frame kf_002411.png --prompts "tree;orange;..." --out <npz> (2026-10-17) — LERF relevancy of open prompts on the
LangSplat-lite field (arms/maps_lite, PCA-32 codes of SAM3 mask-crop CLIP image features, no labels): inverse PCA to 512-d, cosine to the prompt and to
the canonical negatives (object, things, stuff, texture), relevancy = min over negatives of softmax(10*[pos, neg])[pos]. The challenge panel of the hero
figure: on self-similar trees an image-feature field answers the category and cannot tell one tree from another. nerf_new pixi env."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, open_clip
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--frame', required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
N = Path(a.identity); dev = 'cuda'; b = np.load(N / 'arms/lite_targets/basis.npz'); C = torch.tensor(b['components'], dtype=torch.float32, device=dev); mu = torch.tensor(b['mean'], dtype=torch.float32, device=dev)
z = np.load(N / 'arms/maps_lite' / a.frame.replace('.png', '.npz')); feat = torch.tensor(z['features'], dtype=torch.float32, device=dev); H, W, D = feat.shape; ft = feat.reshape(-1, D); void = ft.norm(dim=-1) < 0.5
# No coverage mask: the panel must show the field's own output. (A held-out frame has no lite target, and the supervision's masks are not the baseline's to use.)
model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
prompts = a.prompts.split(';'); negs = ['object', 'things', 'stuff', 'texture']
with torch.no_grad():
    E = model.encode_text(tok(prompts + negs).to(dev)).float(); E = E / E.norm(dim=-1, keepdim=True); Ep, En = E[:len(prompts)], E[len(prompts):]
    dec = ft @ C + mu; dec = dec / dec.norm(dim=-1, keepdim=True).clamp(min=1e-8); cp = dec @ Ep.T; cn = dec @ En.T   # [P, n_prompts], [P, 4]
    out = {'rgb': z['rgb'], 'void': void.cpu().numpy().reshape(H, W)}
    for i, p in enumerate(prompts):
        logits = torch.stack([cp[:, i:i + 1].expand(-1, 4), cn], -1) * 10.0; rel = torch.softmax(logits, -1)[..., 0].min(-1).values; rel[void] = 0; out[p] = rel.cpu().numpy().reshape(H, W); print(f"[lite] '{p}': relevancy >= 0.5 on {100 * (out[p] >= 0.5).mean():.1f} % of px, max {out[p].max():.2f}")
np.savez_compressed(a.out, **out); print('[lite] wrote', a.out)
