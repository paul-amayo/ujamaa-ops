#!/usr/bin/env python3
"""projection_vote_seed.py — paper ablation E2b arm P (2026-10-13): the naive lift — project every gaussian CENTRE into each census frame and
vote for the supervision label at that pixel (no alpha weights, no occlusion, the Semantic-Gaussians-style projection), majority label per
gaussian -> <out>.labels.npy; the features follow with reseed_from_labels.py. Contrast: the census (build_census_init) weights each
gaussian by its alpha-blend contribution to the labelled pixels.
nerf_new pixi env:  projection_vote_seed.py --identity <ws>/identity --out <labels.npy> [--render-scale 0.5] [--min-votes 1]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
UNLAB = 65535
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--min-votes', type=int, default=1)
a = ap.parse_args(); N = Path(a.identity); dev = 'cuda'; torch.set_grad_enabled(False)
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); SUP = Path(meta['manifest']).parent; ids = sorted(int(k) for k in meta['word_table'])
cfg = Path(json.load(open(N / 'W.json'))['config']); split = json.load(open(N / 'split_names.json'))
_, pipe, _, _ = eval_setup(cfg, test_mode='test'); means = pipe.model.gauss_params['means'].detach(); NG = means.shape[0]; ds = pipe.datamanager.train_dataset
cams = {Path(p).name: i for i, p in enumerate(ds.image_filenames)}; frames = [f for f in split['train'] if f in cams and (SUP / f).exists()]
lut = torch.full((65536,), -1, dtype=torch.long, device=dev); lut[torch.tensor(ids, device=dev)] = torch.arange(len(ids), device=dev)
votes = torch.zeros(NG, len(ids), dtype=torch.int32, device=dev); hom = torch.cat([means, torch.ones(NG, 1, device=dev)], 1)
for f in frames:
    cam = ds.cameras[cams[f]:cams[f] + 1].to(dev); cam.rescale_output_resolution(a.render_scale); W_, H_ = int(cam.width.item()), int(cam.height.item())
    lab = torch.from_numpy(np.array(Image.fromarray(np.array(Image.open(SUP / f), np.uint16)).resize((W_, H_), Image.NEAREST)).astype(np.int64)).to(dev)
    pc = (get_viewmat(cam.camera_to_worlds)[0] @ hom.T).T; z = pc[:, 2]; K = cam.get_intrinsics_matrices()[0].to(dev)
    u = (K[0, 0] * pc[:, 0] / z + K[0, 2]).round().long(); v = (K[1, 1] * pc[:, 1] / z + K[1, 2]).round().long()
    ok = (z > 0.01) & (u >= 0) & (u < W_) & (v >= 0) & (v < H_); li = torch.full((NG,), -1, dtype=torch.long, device=dev); li[ok] = lut[lab[v[ok], u[ok]]]
    sel = li >= 0; votes[sel.nonzero().squeeze(1), li[sel]] += 1
n = votes.sum(1); maj = votes.argmax(1); lab_out = torch.where(n >= a.min_votes, torch.tensor(ids, device=dev)[maj], torch.full_like(maj, -1))
a.out.parent.mkdir(parents=True, exist_ok=True); np.save(a.out, lab_out.cpu().numpy().astype(np.int32))
print(f'[projvote] {N.parent.name}: {len(frames)} census frames, {int((n > 0).sum())} gaussians with a labelled projection, {int((lab_out >= 0).sum())} assigned (>= {a.min_votes} votes) of {NG} -> {a.out}', flush=True)
