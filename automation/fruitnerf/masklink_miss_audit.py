#!/usr/bin/env python3
"""masklink_miss_audit.py --config <splat config> --survey <tree> [--tag plain30k] [--every 8] — why do 26-29 % of the apple masks inside the
tree have no fruit gaussian? Renders, per sampled view, the alpha-weighted FRUIT SHARE of the gaussians painting each pixel (census share
Wf/(Wf+Wn) as a dummy colour) and reports, for the masks with no vote vs the explained ones: the painters' mean share, their alpha
coverage, and whether fruit gaussians (share >= 0.5) exist behind the rendered surface (depth-rejected). nerf_new pixi env."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from pycocotools import mask as mask_utils
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True); ap.add_argument('--survey', type=Path, required=True); ap.add_argument('--tag', default='plain30k'); ap.add_argument('--every', type=int, default=8); a = ap.parse_args()
S = a.survey.resolve(); R = S / 'fruit_registry'; v = np.load(R / f'masklink_votes_{a.tag}.npz'); c = np.load(R / f'masklink_components_{a.tag}.npz'); names = list(v['names']); comp, apple = c['comp'], c['apple']
_, pipe, _, _ = eval_setup(a.config, test_mode='test'); model = pipe.model; model.eval(); dev = model.device; ds = pipe.datamanager.train_dataset
g = model.gauss_params; N = g['means'].shape[0]; means = g['means'].detach(); quats = g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
share = torch.zeros(N, device=dev); fid = torch.from_numpy(v['fid']).to(dev); Wf = torch.from_numpy(v['Wf']).to(dev); Wn = torch.from_numpy(v['Wn']).to(dev)
# census share of ALL gaussians: recompute Wf/Wn quickly for the sampled views? No — the cache has it for fruit gaussians only; recompute for all over the sampled views
Wf_all = torch.zeros(N, device=dev); Wn_all = torch.zeros(N, device=dev)
sam = S / 'prod/bateleur/sam3_v2'
if not sam.exists(): sam = S / 'sam3_v2'
ents_by = {}
for cm in sorted(sam.glob('clip_*/frame_entries.json')):
    meta = json.loads(cm.read_text()); l2i = {f['local_idx']: int(f.get('kf_idx', f.get('image_idx'))) for f in meta['frames']}
    for li, e in meta['frame_entries'].items(): ents_by[f"image_{l2i[int(li)]}.png"] = sorted(e, key=lambda e: e.get('score', 0))
idx_views = [i for i in range(len(ds)) if Path(ds.image_filenames[i]).name in names]; samp = idx_views[::a.every]
cams = {}
for i in samp:
    nm = Path(ds.image_filenames[i]).name; fm = S / 'fruit_masks' / nm; cam = ds.cameras[i:i + 1].to(dev); vm = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); W, H = int(cam.width.item()), int(cam.height.item())
    m = torch.from_numpy(np.array(Image.open(fm))).to(dev); col = torch.zeros(N, 2, device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gf, = torch.autograd.grad((out[0, ..., 0] * (m == 2).float()).sum() + (out[0, ..., 1] * (m == 1).float()).sum(), col); Wf_all += gf[:, 0].clamp(min=0); Wn_all += gf[:, 1].clamp(min=0); cams[i] = (nm, vm, K, W, H)
share = Wf_all / (Wf_all + Wn_all).clamp(min=1e-9); isfruit = torch.zeros(N, dtype=torch.bool, device=dev); isfruit[fid] = True
key = v['vi'].astype(np.int64) * 100000 + v['lab'].astype(np.int64); voted = set(np.unique(key).tolist())
rows = []   # status, mean painter share, alpha coverage, fruit-gaussian mass behind (share>=0.5 painters' alpha), area
for i, (nm, vm, K, W, H) in cams.items():
    vi_ = names.index(nm); col = torch.stack([share, torch.ones(N, device=dev), isfruit.float()], 1)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=vm, Ks=K, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    sh_img = out[0, ..., 0].cpu().numpy(); cov = out[0, ..., 1].cpu().numpy(); fr_img = out[0, ..., 2].cpu().numpy()
    for j, e in enumerate(ents_by.get(nm, [])):
        mk = mask_utils.decode(e['rle']).astype(bool)
        if mk.shape != (H, W): continue
        k = vi_ * 100000 + j; st = 'none' if k not in voted else 'voted'; cv_ = cov[mk]; rows.append((st, float((sh_img[mk] / np.clip(cv_, 1e-6, None)).mean()), float(cv_.mean()), float((fr_img[mk] / np.clip(cv_, 1e-6, None)).mean()), int(mk.sum())))
rows = np.array(rows, dtype=object); 
for st in ('none', 'voted'):
    r = rows[rows[:, 0] == st]; sh = r[:, 1].astype(float); cv_ = r[:, 2].astype(float); fr = r[:, 3].astype(float)
    print(f"[miss-audit] {S.name} ({a.tag}, {len(cams)} views): {st} masks n={len(r)}: painters' fruit share median {np.median(sh):.2f} (p25 {np.percentile(sh, 25):.2f}, p75 {np.percentile(sh, 75):.2f}); share < 0.1 in {100 * np.mean(sh < 0.1):.0f} %, 0.1-0.5 in {100 * np.mean((sh >= 0.1) & (sh < 0.5)):.0f} %, >= 0.5 in {100 * np.mean(sh >= 0.5):.0f} %; alpha coverage median {np.median(cv_):.2f}; fraction of the paint from census-fruit gaussians median {np.median(fr):.2f}; area median {np.median(r[:, 4].astype(int)):.0f} px", flush=True)
