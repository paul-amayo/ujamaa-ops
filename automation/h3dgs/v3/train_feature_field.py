#!/usr/bin/env python3
"""train_feature_field.py — paper ablation E2b (2026-10-13): learn the per-gaussian identity features of a v3 splat by GRADIENT DESCENT against
the same per-pixel targets the census seed uses — stage 2's recipe of record: geodesic loss on the hyperboloid between the rendered tangent
vector and the pixel's label target, labelled pixels only, frozen geometry — from ZERO (--init zero: how LangSplat / stage 2 learn a feature
field) or from the census seed (--init <features_B.bin>: refine). Trains on the cell's census (training-split) frames that carry a supervision
map, at the scoring resolution; writes <out> [N, 32] float32 + json; render and score it like any arm (splat_feature_render.py, argmax_eval.py).
nerf_new pixi env:  train_feature_field.py --identity <ws>/identity --init zero|<bin> --out <bin> [--steps 3000] [--lr 0.01] [--render-scale 0.5]"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH')
from high_splat_hierarchy_accuracy import build_hyper_embedder
from word_utils import get_word_for_id
import lorentz as L
import open_clip
FRUIT_ID_BASE, UNLAB = 10000, 65535
ap = argparse.ArgumentParser(); ap.add_argument('--identity', required=True); ap.add_argument('--init', required=True, help="'zero' or a features bin to refine")
ap.add_argument('--out', required=True, type=Path); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--lr', type=float, default=0.01); ap.add_argument('--render-scale', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--log-every', type=int, default=500); a = ap.parse_args(); N = Path(a.identity); dev = 'cuda'; random.seed(a.seed); torch.manual_seed(a.seed)
meta = json.loads(str(np.load(N / 'text_bank.npz', allow_pickle=True)['meta'])); SUP = Path(meta['manifest']).parent; fb = json.load(open(N / 'features_B.bin.json'))
cfg = Path(json.load(open(N / 'W.json'))['config']); split = json.load(open(N / 'split_names.json'))
_, pipe, _, step = eval_setup(cfg, test_mode='test'); model = pipe.model; model.eval(); g = model.gauss_params
means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1); NG = means.shape[0]
cams = {Path(p).name: i for i, p in enumerate(pipe.datamanager.train_dataset.image_filenames)}; ds = pipe.datamanager.train_dataset
frames = [f for f in split['train'] if f in cams and (SUP / f).exists()]
# --- targets exactly as build_census_init: RAW CLIP text -> encode_features(node_types) -> tangent ---
ids = sorted(int(k) for k in meta['word_table']); m, _, _ = open_clip.create_model_and_transforms('ViT-B-16', 'laion2b_s34b_b88k', device=dev); tok = open_clip.get_tokenizer('ViT-B-16')
hyper = build_hyper_embedder(fb['embedder'], dev); curv = hyper.curv.exp()
words = [get_word_for_id(u - FRUIT_ID_BASE, 'fruit') if u >= FRUIT_ID_BASE else get_word_for_id(u, 'mask') for u in ids]
with torch.no_grad():
    ce = torch.cat([m.encode_text(tok(words[i:i + 256]).to(dev)).float() for i in range(0, len(words), 256)])
    hf = hyper.encode_features(ce, project=True, node_types=torch.tensor([4 if u >= FRUIT_ID_BASE else 0 for u in ids], device=dev)); T = L.log_map0(hf, curv=curv).float()
del m, hyper; lut = torch.full((65536,), -1, dtype=torch.long, device=dev); lut[torch.tensor(ids, device=dev)] = torch.arange(len(ids), device=dev)
# --- the census frames' cameras and supervision at the scoring resolution ---
cam0 = ds.cameras[cams[frames[0]]:cams[frames[0]] + 1].to(dev); cam0.rescale_output_resolution(a.render_scale); W_, H_ = int(cam0.width.item()), int(cam0.height.item())
maps = {f: torch.from_numpy(np.array(Image.fromarray(np.array(Image.open(SUP / f), np.uint16)).resize((W_, H_), Image.NEAREST)).astype(np.int64)).to(dev) for f in frames}
F = torch.zeros(NG, T.shape[1], device=dev) if a.init == 'zero' else torch.from_numpy(np.fromfile(a.init, dtype=np.float32).reshape(NG, -1)).to(dev)
F = torch.nn.Parameter(F.clone()); opt = torch.optim.Adam([F], lr=a.lr); F0 = F.detach().clone()
print(f'[train] {N.parent.name}: {NG} gaussians, {len(frames)} census frames at {W_}x{H_}, {len(ids)} labels, init {a.init if a.init == "zero" else Path(a.init).name}, {a.steps} steps lr {a.lr}', flush=True)
t0 = time.time(); run = 0.0
for it in range(1, a.steps + 1):
    f = random.choice(frames); i = cams[f]; cam = ds.cameras[i:i + 1].to(dev); cam.rescale_output_resolution(a.render_scale)
    feat, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=F, viewmats=get_viewmat(cam.camera_to_worlds), Ks=cam.get_intrinsics_matrices().to(dev),
                                   width=W_, height=H_, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', sh_degree=None, rasterize_mode=model.config.rasterize_mode)
    li = lut[maps[f]]; sel = li >= 0
    if sel.sum() == 0: continue
    pred = feat[0][sel].clamp(-12, 12); tgt = T[li[sel]].clamp(-12, 12)
    x = L.exp_map0(pred, curv=1.0); y = L.exp_map0(tgt, curv=1.0); dist = torch.acosh((-L.rowwise_inner(x, y)).clamp(min=1 + 1e-7))   # stage 2's safe_geodesic_loss, labelled px only
    loss = dist.mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
    if it % a.log_every == 0 or it == a.steps:
        with torch.no_grad(): nz = (F.norm(dim=1) > 0.5).float().mean().item(); mv = (F - F0).norm(dim=1).mean().item()
        print(f'[train] step {it}: geodesic {run:.3f} | ||f||>0.5 on {100 * nz:.1f}% of gaussians, mean move from init {mv:.3f} | {time.time() - t0:.0f}s', flush=True)
a.out.parent.mkdir(parents=True, exist_ok=True); F.detach().cpu().numpy().astype(np.float32).tofile(a.out)
json.dump({'n': NG, 'dim': int(T.shape[1]), 'init': a.init, 'steps': a.steps, 'lr': a.lr, 'frames': len(frames), 'labels': len(ids), 'embedder': fb['embedder'], 'final_geodesic_ema': run,
           'seconds': time.time() - t0}, open(str(a.out) + '.json', 'w'), indent=1)
print(f'[train] done -> {a.out} ({time.time() - t0:.0f}s)', flush=True)
