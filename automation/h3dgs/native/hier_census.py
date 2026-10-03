#!/usr/bin/env python3
"""Interaction census on an H3DGS hierarchy's own nodes (UJAMAA, 2026-10-03) — the blocks' stage-2 census
(aru_sil_core scripts/gaussian_interaction_census.py) without nerfstudio and without a side-car.

Per census view: expand the hierarchy to the serving cut (tau px at --cut-width, as hier_render_service does), render a
C-channel map whose per-gaussian colours are a zero leaf tensor, and take the gradient of sum_label(sum over the label's
pixels of the label's channel). Rasterised colours are linear, so that gradient IS each rendered gaussian's alpha-blend
weight into each label's pixels. A rendered gaussian is the interpolation w*node + (1-w)*parent, so its weight goes to
the node (x w) and to the parent (x (1-w)). Summed over views (and averaged over --taus) -> W[label, node].
Supervision = the side-car's uint16 id maps (65535 = unlabelled; --with-bg censuses it as row 65535 = CENSUS_WITH_BG=1).
Views = the side-car run's TRAIN list (sidecar_split_names.py), so both censuses see the same frames.
Resolution: --render-scale 0.5 of the training images = the 0.5 rescale the nerfstudio census uses (floors stay calibrated).
Output npz: W [L, N_nodes] float32 + labels, the format build_census_init.py reads.

h3dgs env:  CUDA_HOME=/home/paperspace/code/_cuda12 PATH=$CUDA_HOME/bin:$PATH python hier_census.py \
   -s <chunk src> -m <trained chunk> --hierarchy <hierarchy.hier_opt> -i ../../rectified/images --eval --data_device cpu \
   --supervision-dir <id maps> --names-json <split_names.json> [--taus 3] [--cut-width 1280] [--with-bg] --out-npz <W.npz>"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hier_common as hc
import numpy as np
import torch
from PIL import Image
from gsplat import rasterization

UNLAB = 65535
ap = argparse.ArgumentParser()
lp, pp = hc.add_scene_args(ap)
ap.add_argument('--supervision-dir', required=True)
ap.add_argument('--names-json', required=True)
ap.add_argument('--taus', type=float, nargs='+', default=[3.0])
ap.add_argument('--cut-width', type=int, default=1280)
ap.add_argument('--render-scale', type=float, default=0.5)
ap.add_argument('--with-bg', action='store_true')
ap.add_argument('--out-npz', required=True)
args = ap.parse_args()

dataset, scene, g, cams, K0, bufs = hc.load(lp, args)
N = g._xyz.size(0)
names = json.load(open(args.names_json))['train']
print(f'[census] hierarchy {dataset.hierarchy}: {N} nodes, skybox {getattr(g, "skybox_points", 0)}; '
      f'{len(cams)} cameras; {len(names)} census views requested; taus {args.taus} at {args.cut_width} px; '
      f'render scale {args.render_scale}; bg row {"on" if args.with_bg else "off"}', flush=True)
rows, done, skipped, t0, n_exp = {}, 0, [], time.time(), []
torch.set_grad_enabled(True)
for name in names:
    sf = os.path.join(args.supervision_dir, name)
    if name not in cams or not os.path.exists(sf):
        skipped.append(name); continue
    cam = cams[name]; hc.to_cuda(cam)
    viewmat, Ks, Wr, Hr = hc.gsplat_cam(cam, K0, args.render_scale)
    a = np.array(Image.open(sf), np.uint16)
    a = np.array(Image.fromarray(a).resize((Wr, Hr), Image.NEAREST))
    uids = [int(u) for u in np.unique(a) if u != UNLAB]
    if args.with_bg and (a == UNLAB).any():
        uids.append(UNLAB)
    if not uids:
        skipped.append(name); continue
    lab = torch.from_numpy(a.astype(np.int64)).cuda()
    onehot = torch.stack([(lab == u) for u in uids], -1).float()          # [H, W, C]
    for tau in args.taus:
        thr = hc.cut_threshold(cam, tau, args.cut_width)
        means, scales, rots, opac, ri, pi, w = hc.expand(g, cam, thr, bufs)
        means, scales, rots, opac, sky = hc.with_skybox(g, means, scales, rots, opac)
        col = torch.zeros(ri.numel(), len(uids), device='cuda', requires_grad=True)
        colors = torch.cat([col, torch.zeros(sky, len(uids), device='cuda')]) if sky else col
        out, _, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=colors,
                                  viewmats=viewmat, Ks=Ks, width=Wr, height=Hr, sh_degree=None, packed=False,
                                  near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode='classic')
        gcol, = torch.autograd.grad((out[0] * onehot).sum(), col)
        wv = w[:, 0]
        for k, u in enumerate(uids):
            r = rows.get(u)
            if r is None:
                r = rows[u] = torch.zeros(N, device='cuda')
            gk = gcol[:, k] / len(args.taus)
            r.index_add_(0, ri, wv * gk)
            r.index_add_(0, pi, (1 - wv) * gk)
        n_exp.append(ri.numel())
        del out, col, colors, gcol
    done += 1
    if done % 50 == 0 or done == 1:
        print(f'[census] {done} views ({time.time() - t0:.0f}s), {len(rows)} labels so far, last cut {n_exp[-1]} gaussians', flush=True)
labels = sorted(rows)
W = torch.stack([rows[u] for u in labels]).cpu().numpy().astype(np.float32)
os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
np.savez_compressed(args.out_npz, W=W, labels=np.array(labels))
lab_rows = [i for i, u in enumerate(labels) if u != UNLAB]
tot = W[lab_rows].sum(0)
stats = {'nodes': int(N), 'views': done, 'skipped': len(skipped), 'skipped_first': skipped[:5], 'labels': len(labels),
         'taus': args.taus, 'cut_width': args.cut_width, 'render_scale': args.render_scale,
         'cut_gaussians_median': int(np.median(n_exp)) if n_exp else 0,
         'nodes_with_label_weight': int((tot > 0).sum()), 'nodes_label_weight_gt1': int((tot > 1.0).sum()),
         'minutes': round((time.time() - t0) / 60, 2)}
json.dump(stats, open(args.out_npz.replace('.npz', '.json'), 'w'), indent=1)
print(f'[census] saved W {W.shape} -> {args.out_npz}; {json.dumps(stats)}', flush=True)
