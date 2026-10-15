#!/usr/bin/env python3
"""sam3_clip_targets.py — LangSplat-lite, step 1 (paper E3 fallback, 2026-10-13): per-pixel CLIP IMAGE features for a cell's census frames,
with no labels and no hierarchy. For every SAM3 instance mask of the frame (prod/bateleur/sam3_v2, the 'tree' prompt; --fruit adds
sam3_fruit) the two-level mask crop is embedded with CLIP ViT-B-16 exactly as the HiGH MaskCropEncoder does (masked crop + bbox context,
mean), all mask features of the cell are reduced to K dims by PCA (LangSplat compresses with an autoencoder to 3-d; we keep 32, the same
budget as the hyperbolic field), and each pixel gets the code of the smallest mask covering it. One npz per frame at the scoring size:
target [h, w, K] float32, covered [h, w] bool; basis.npz (components [K, 512], mean [512]) for the scorer's inverse-PCA decode.
Deviation from LangSplat, documented: prompted SAM3 masks at one granularity instead of automatic SAM masks at three.
nerf_new pixi env:  sam3_clip_targets.py --identity <ws>/identity --out-dir <dir> [--kf-dir ...] [--sam3-dir ...] [--render-scale 0.5] [--k 32]"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from pycocotools import mask as mask_utils
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from clip_image_embedder import MaskCropEncoder, MaskCropEncoderConfig
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--out-dir', required=True, type=Path); ap.add_argument('--kf-dir', default=''); ap.add_argument('--sam3-dir', default='')
ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--k', type=int, default=32); ap.add_argument('--min-area', type=int, default=400); ap.add_argument('--fruit', action='store_true', help='also lift sam3_fruit masks')
a = ap.parse_args(); N = Path(a.identity); root = str(N).split('/experimental/')[0]; dev = 'cuda'; torch.set_grad_enabled(False)
KF = Path(a.kf_dir or f'{root}/prod/scratch_sam3'); SAM = [Path(a.sam3_dir or f'{root}/prod/bateleur/sam3_v2')] + ([Path(f'{root}/prod/bateleur/sam3_fruit')] if a.fruit else [])
frames = [f for f in json.load(open(N / 'split_names.json'))['train'] if (KF / f).exists()]
# frame name -> mask records (kf_idx = clip start + local idx)
recs = {}
for sd in SAM:
    for cj in sorted(glob.glob(f'{sd}/clip_*/frame_entries.json')):
        d = json.load(open(cj))
        for li, ms in d['frame_entries'].items():
            name = f"kf_{d['start_idx'] + int(li):06d}.png"
            if name in frames: recs.setdefault(name, []).extend(ms)
enc = MaskCropEncoder(MaskCropEncoderConfig(), device=dev); feats, owners = [], []   # owners: (frame, mask index)
masks_by_frame = {}; t0 = time.time()
for f in frames:
    ms = [m for m in recs.get(f, []) if m['area'] >= a.min_area]
    if not ms: continue
    img = torch.from_numpy(np.array(Image.open(KF / f).convert('RGB'))).permute(2, 0, 1)   # uint8 [3, H, W]
    M = []
    for m in ms:
        rle = dict(m['rle']); rle['counts'] = rle['counts'].encode('utf-8'); M.append(torch.from_numpy(mask_utils.decode(rle).astype(np.uint8)))
    M = torch.stack(M)
    if M.shape[-2:] != img.shape[-2:]: M = torch.nn.functional.interpolate(M[None].float(), size=img.shape[-2:], mode='nearest')[0].to(torch.uint8)
    crops = torch.stack([enc._get_multi_level_crops(mm, img, num_levels=2, pad=True) for mm in M])   # [m, 2, 3, 224, 224], preprocessed as the HiGH encoder does (masked crop + bbox context)
    e = enc.model.encode_image(crops.flatten(0, 1).to(dev), normalize=False).float().view(len(ms), 2, -1).mean(1)
    feats.append(e.cpu()); owners += [(f, i) for i in range(len(ms))]; masks_by_frame[f] = (M, [m['area'] for m in ms])
E = torch.cat(feats); mean = E.mean(0); U, S, V = torch.pca_lowrank(E - mean, q=a.k, center=False); comp = V[:, :a.k].T.contiguous(); Z = ((E - mean) @ comp.T).numpy()
expl = float((S[:a.k] ** 2).sum() / ((E - mean) ** 2).sum()); a.out_dir.mkdir(parents=True, exist_ok=True)
np.savez(a.out_dir / 'basis.npz', components=comp.numpy(), mean=mean.numpy()); zi = {o: i for i, o in enumerate(owners)}
for f, (M, areas) in masks_by_frame.items():
    H, W = M.shape[-2:]; h, w = int(round(H * a.render_scale)), int(round(W * a.render_scale))
    Ms = torch.nn.functional.interpolate(M[None].float(), size=(h, w), mode='nearest')[0].bool().numpy()
    tgt = np.zeros((h, w, a.k), np.float32); cov = np.zeros((h, w), bool)
    for i in np.argsort(areas)[::-1]:   # largest first; smaller masks overwrite -> the smallest covering mask wins
        tgt[Ms[i]] = Z[zi[(f, i)]]; cov |= Ms[i]
    np.savez_compressed(a.out_dir / f.replace('.png', '.npz'), target=tgt, covered=cov)
json.dump({'frames': len(masks_by_frame), 'masks': len(owners), 'k': a.k, 'pca_explained': expl, 'kf_dir': str(KF), 'sam3': [str(s) for s in SAM], 'min_area': a.min_area, 'render_scale': a.render_scale, 'seconds': time.time() - t0}, open(a.out_dir / 'targets.json', 'w'), indent=1)
print(f'[targets] {N.parent.name}: {len(owners)} masks over {len(masks_by_frame)} of {len(frames)} census frames, CLIP crops -> PCA-{a.k} explains {100 * expl:.1f}%, {time.time() - t0:.0f}s -> {a.out_dir}', flush=True)
