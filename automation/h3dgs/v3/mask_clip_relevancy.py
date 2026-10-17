#!/usr/bin/env python3
"""mask_clip_relevancy.py --image <photo> --entries <frame_entries.json> --kf 2411 --prompts "tree;..." --out <npz> (2026-10-17) — the challenge in two
dimensions, no Gaussians: every SAM3 'tree' mask of one keyframe is cropped from the photograph (LangSplat's mask crop: the mask's bounding box with
the outside of the mask blacked out), encoded with CLIP ViT-B/16, and scored against each prompt with LERF relevancy (min over the canonical negatives
object / things / stuff / texture of softmax(10*[pos, neg])[pos]). Writes the photo, each mask, its relevancy per prompt and the pairwise cosines
between the mask embeddings. nerf_new pixi env (pycocotools, open_clip)."""
import argparse, json
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image
from pycocotools import mask as mask_utils
ap = argparse.ArgumentParser(); ap.add_argument('--image', required=True); ap.add_argument('--entries', required=True); ap.add_argument('--kf', type=int, required=True); ap.add_argument('--prompts', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
dev = 'cuda'; meta = json.load(open(a.entries)); li = [f['local_idx'] for f in meta['frames'] if int(f.get('kf_idx', f.get('image_idx', -1))) == a.kf][0]; ents = meta['frame_entries'][str(li)]
photo = np.asarray(Image.open(a.image).convert('RGB')); H, W = photo.shape[:2]; masks = []
for e in ents:
    m = mask_utils.decode(e['rle']).astype(bool)
    if m.shape != (H, W): m = np.array(Image.fromarray(m.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(bool)
    masks.append(m)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16'); prompts = a.prompts.split(';'); negs = ['object', 'things', 'stuff', 'texture']
crops = []
for m in masks:
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; c = photo[y0:y1, x0:x1].copy(); c[~m[y0:y1, x0:x1]] = 0; crops.append(preprocess(Image.fromarray(c)))
with torch.no_grad():
    I = model.encode_image(torch.stack(crops).to(dev)).float(); I = I / I.norm(dim=-1, keepdim=True); T = model.encode_text(tok(prompts + negs).to(dev)).float(); T = T / T.norm(dim=-1, keepdim=True); Tp, Tn = T[:len(prompts)], T[len(prompts):]
    cp = I @ Tp.T; cn = I @ Tn.T; rel = torch.stack([torch.softmax(torch.stack([cp[:, i:i + 1].expand(-1, 4), cn], -1) * 10.0, -1)[..., 0].min(-1).values for i in range(len(prompts))], 1)   # [n_masks, n_prompts]
    pair = (I @ I.T).cpu().numpy()
out = {'photo': photo, 'masks': np.stack(masks), 'prompts': np.array(prompts), 'relevancy': rel.cpu().numpy(), 'cosine': cp.cpu().numpy(), 'pairwise': pair, 'areas': np.array([int(m.sum()) for m in masks])}
for i, p in enumerate(prompts): print(f"[masks] '{p}': relevancy per mask " + ' '.join(f"{v:.2f}" for v in out['relevancy'][:, i]) + f" (cosine {out['cosine'][:, i].min():.3f}-{out['cosine'][:, i].max():.3f})")
iu = np.triu_indices(len(masks), 1); print(f"[masks] {len(masks)} tree masks; pairwise cosine between their CLIP embeddings: min {pair[iu].min():.3f} median {np.median(pair[iu]):.3f} max {pair[iu].max():.3f}")
np.savez_compressed(a.out, **out); print('[masks] wrote', a.out)
