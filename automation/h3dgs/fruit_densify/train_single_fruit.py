#!/usr/bin/env python3
"""Fruit-aware densification of a TRAINED H3DGS chunk (UJAMAA 2026-10-04; Paul: "densify the chunk then").
Fine-tunes the chunk's leaf gaussians (pretrained done_*.pt) with the stock photometric loss and H3DGS densification, plus
a FRUIT term (the HiGH side-car recipe, metres): every view, gaussians whose projected centre lands in a dilated SAM3 fruit
pixel (chunk supervision ids >= 10000) get their densification gradient multiplied by --fruit_boost and are tagged; at each
densification step tagged carriers with max scale in (--fruit_split_m, --fruit_max_m) are SPLIT (N=2) regardless of the
gradient score (capped per step). Scaffold prefix (pc_info.txt) never densified; trained exposures loaded and frozen.
Run from the hierarchical-3d-gaussians repo (h3dgs env). Saves point_cloud/iteration_<N> + exposure.json in --model_path."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import torch, numpy as np
from torch import nn
from PIL import Image
from scipy import ndimage
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, build_rotation
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import DataLoader

p = ArgumentParser(); lp = ModelParams(p); op = OptimizationParams(p); pp = PipelineParams(p)
p.add_argument('--chunk_trained', required=True); p.add_argument('--load_iter', default='60000')
p.add_argument('--fruit_sup', required=True); p.add_argument('--fruit_boost', type=float, default=5.0)
p.add_argument('--fruit_split_m', type=float, default=0.04); p.add_argument('--fruit_max_m', type=float, default=0.5)
p.add_argument('--fruit_dilate', type=int, default=2); p.add_argument('--fruit_split_cap', type=int, default=300000)
p.add_argument('--max_gaussians', type=int, default=0)
args = p.parse_args(sys.argv[1:]); dataset = lp.extract(args); opt = op.extract(args); pipe = pp.extract(args)
safe_state(False); torch.autograd.set_detect_anomaly(False)
PRE = f'{args.chunk_trained}/point_cloud/iteration_{args.load_iter}'; dataset.pretrained = PRE
os.makedirs(dataset.model_path, exist_ok=True)
gaussians = GaussianModel(dataset.sh_degree); scene = Scene(dataset, gaussians)
gaussians.active_sh_degree = gaussians.max_sh_degree
SCAF = int(open(f'{PRE}/pc_info.txt').read().split()[0]); gaussians.scaffold_points = SCAF
names = [ci.image_name for ci in scene.train_cameras[1.0].list_cam_infos] + [ci.image_name for ci in scene.test_cameras[1.0].list_cam_infos]
EX = json.load(open(f'{args.chunk_trained}/exposure.json'))
gaussians.exposure_mapping = {n: i for i, n in enumerate(names)}
gaussians._exposure = nn.Parameter(torch.stack([torch.tensor(EX[n], dtype=torch.float32) if n in EX else torch.eye(3, 4) for n in names]).cuda().requires_grad_(True))
gaussians.training_setup(opt)
for g in gaussians.exposure_optimizer.param_groups: g['lr'] = 0.0       # trained exposures stay as they are
gaussians.update_learning_rate = (lambda f: (lambda it: [pg.__setitem__('lr', f(it)) for pg in gaussians.optimizer.param_groups if pg['name'] == 'xyz']))(gaussians.xyz_scheduler_args)
print(f'[fruit-ft] loaded {gaussians.get_xyz.shape[0]} gaussians ({SCAF} scaffold) from {PRE}; extent {scene.cameras_extent:.2f}; {len(names)} cameras, {sum(1 for n in names if n in EX)} with trained exposure', flush=True)
# fruit masks per image (chunk supervision id maps)
FR = {}
for n in names:
    f = os.path.join(args.fruit_sup, n + ('' if n.endswith('.png') else '.png'))
    if os.path.exists(f):
        a = np.array(Image.open(f), np.uint16); m = (a >= 10000) & (a != 65535)
        if m.any(): FR[n] = torch.from_numpy(ndimage.binary_dilation(m, iterations=args.fruit_dilate)).cuda()
print(f'[fruit-ft] fruit masks on {len(FR)} images ({sum(int(m.sum()) for m in FR.values())} px after {args.fruit_dilate}-px dilation)', flush=True)

def clone_mask(sel):
    g = gaussians
    g.densification_postfix(g._xyz[sel], g._features_dc[sel], g._features_rest[sel], g._opacity[sel], g._scaling[sel], g._rotation[sel])
def split_mask(sel, N=2):
    g = gaussians
    stds = g.get_scaling[sel].repeat(N, 1); samples = torch.normal(mean=torch.zeros((stds.size(0), 3), device='cuda'), std=stds)
    rots = build_rotation(g._rotation[sel]).repeat(N, 1, 1)
    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + g.get_xyz[sel].repeat(N, 1)
    g.densification_postfix(new_xyz, g._features_dc[sel].repeat(N, 1, 1), g._features_rest[sel].repeat(N, 1, 1), g._opacity[sel].repeat(N, 1),
                            g.scaling_inverse_activation(g.get_scaling[sel].repeat(N, 1) / (0.8 * N)), g._rotation[sel].repeat(N, 1))
    g.prune_points(torch.cat((sel, torch.zeros(N * int(sel.sum()), device='cuda', dtype=bool))))

def densify(fruit_acc, thr, extent):
    g = gaussians; grads = g.xyz_gradient_accum; grads[grads.isnan()] = 0.0
    n = grads.shape[0]; op_ = g.get_opacity.flatten(); maxs = g.get_scaling.max(dim=1).values
    base = (grads.squeeze(1) * g.max_radii2D * op_.pow(0.2) >= thr) & (op_ > 0.15); base[:SCAF] = False
    fr = (fruit_acc > 0) & (maxs > args.fruit_split_m) & (maxs < args.fruit_max_m); fr[:SCAF] = False
    if int(fr.sum()) > args.fruit_split_cap:
        keep = torch.topk(fruit_acc * fr, args.fruit_split_cap).indices; fr2 = torch.zeros_like(fr); fr2[keep] = True; fr = fr2
    over = args.max_gaussians > 0 and n >= args.max_gaussians
    clone = base & (maxs <= g.percent_dense * extent) & ~fr if not over else torch.zeros_like(base)
    split = (base & (maxs > g.percent_dense * extent) if not over else torch.zeros_like(base)) | fr
    nc, ns, nf = int(clone.sum()), int(split.sum()), int(fr.sum())
    clone_mask(clone)                                                  # appends; resets the accumulators
    split_mask(torch.cat((split, torch.zeros(nc, device='cuda', dtype=bool))))
    prune = (g.get_opacity < 0.005).squeeze(); prune[:SCAF] = False; g.prune_points(prune)
    g.max_radii2D = torch.zeros((g.get_xyz.shape[0]), device='cuda'); torch.cuda.empty_cache()
    return nc, ns, nf, int(prune.sum())

bg = torch.zeros(3, device='cuda'); gen = DataLoader(scene.getTrainCameras(), num_workers=8, prefetch_factor=1, persistent_workers=True, collate_fn=lambda x: x)
fruit_acc = torch.zeros(gaussians.get_xyz.shape[0], device='cuda'); it = 1; boosted = 0
while it <= opt.iterations:
    for batch in gen:
        for cam in batch:
            if it > opt.iterations: break
            for a in ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'): setattr(cam, a, getattr(cam, a).cuda())
            gaussians.update_learning_rate(it)
            bg = torch.rand(3, device='cuda')
            pkg = render(cam, gaussians, pipe, bg, use_trained_exp=True)
            image, vpt, vis, radii = pkg['render'], pkg['viewspace_points'], pkg['visibility_filter'], pkg['radii']
            gt = cam.original_image.cuda()
            if cam.alpha_mask is not None: image = image * cam.alpha_mask.cuda()
            loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1.0 - ssim(image, gt))
            loss.backward()
            with torch.no_grad():
                m = FR.get(cam.image_name)
                if m is not None and vpt.grad is not None:
                    idx = vis; xyz = gaussians.get_xyz[idx]          # H3DGS render returns visible INDICES (and radii for them)
                    hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1) @ cam.full_proj_transform
                    ndc = hom[:, :2] / hom[:, 3:4].clamp_min(1e-6); Hh, Ww = m.shape
                    px = ((ndc[:, 0] + 1) * Ww - 1) * 0.5; py = ((ndc[:, 1] + 1) * Hh - 1) * 0.5
                    ins = (px >= 0) & (px < Ww) & (py >= 0) & (py < Hh) & (hom[:, 3] > 0)
                    tag = idx[ins][m[py[ins].long(), px[ins].long()]]
                    if tag.numel(): vpt.grad[tag, :2] *= args.fruit_boost; fruit_acc[tag] += 1; boosted += int(tag.numel())
                if it < opt.densify_until_iter:
                    gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii)
                    gaussians.add_densification_stats(vpt, vis)
                    if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                        nb = gaussians.get_xyz.shape[0]; nc, ns, nf, npr = densify(fruit_acc, opt.densify_grad_threshold, scene.cameras_extent)
                        fruit_acc = torch.zeros(gaussians.get_xyz.shape[0], device='cuda')
                        print(f'[fruit-ft] it {it}: {nb} -> {gaussians.get_xyz.shape[0]} gaussians (clone {nc}, split {ns} incl. fruit {nf}, pruned {npr}); fruit-tagged hits since last step {boosted}; loss {loss.item():.4f}', flush=True); boosted = 0
                if gaussians._opacity.grad is not None:
                    rel = (gaussians._opacity.grad.flatten() != 0).nonzero().flatten().long()
                    gaussians.optimizer.step(rel); gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if not args.skip_scale_big_gauss:
                    vals = gaussians.get_scaling.max(dim=1).values; viol = vals > scene.cameras_extent * 0.02; viol[:SCAF] = False
                    gaussians._scaling[viol] = gaussians.scaling_inverse_activation(gaussians.get_scaling[viol] * 0.8)
            it += 1
        if it > opt.iterations: break
scene.save(opt.iterations)
print(f'[fruit-ft] saved {gaussians.get_xyz.shape[0]} gaussians -> {dataset.model_path}/point_cloud/iteration_{opt.iterations}', flush=True)
