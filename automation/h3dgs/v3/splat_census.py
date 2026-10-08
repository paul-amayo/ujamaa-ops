#!/usr/bin/env python3
"""Flat-gaussian supervision census on a splatfacto checkpoint (recipe v3 identity, 2026-10-08) = hier_census.py without the cut.
For every census view that has a supervision PNG: rasterise the model's own geometry (means, exp(scales), quats, sigmoid(opacities),
the viewmat / K splatfacto's get_outputs builds) with a ZERO colour tensor of one channel per label in the frame and take
d(sum of channel k over label k's pixels)/d(colours): the rasteriser is linear in the colours, so that gradient is each gaussian's
total alpha-blend weight into label k's pixels. Accumulated over the views -> W [labels, N] + labels, the npz build_census_init.py
reads (--with-bg adds the unlabelled pixels as row 65535 for --bg-competes).
nerf_new pixi env:  splat_census.py --config <run config.yml> --supervision-dir <dir> --names-json <split_names.json>
                        --out-npz <W.npz> [--render-scale 0.5] [--with-bg]"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
UNLAB = 65535
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True); ap.add_argument('--supervision-dir', required=True); ap.add_argument('--names-json', required=True)
ap.add_argument('--out-npz', required=True); ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--with-bg', action='store_true')
args = ap.parse_args()
_, pipe, _, step = eval_setup(Path(args.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
tr = pipe.datamanager.train_dataset; idx = {Path(p).name: i for i, p in enumerate(tr.image_filenames)}
g = model.gauss_params; N = g['means'].shape[0]
means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
names = json.load(open(args.names_json))['train']
print(f'[census] {Path(args.config).parent.parent.name}: {N} gaussians (step {step}); {len(tr)} training cameras; {len(names)} census views requested; '
      f'render scale {args.render_scale}; bg row {"on" if args.with_bg else "off"}', flush=True)
rows, done, skipped, t0 = {}, 0, [], time.time()
torch.set_grad_enabled(True)
for name in names:
    sf = os.path.join(args.supervision_dir, name)
    if name not in idx or not os.path.exists(sf):
        skipped.append(name); continue
    cam = tr.cameras[idx[name]:idx[name] + 1].to(dev)
    viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev).clone(); W, H = int(cam.width.item()), int(cam.height.item())
    Wr, Hr = max(1, int(W * args.render_scale)), max(1, int(H * args.render_scale)); K[:, :2] *= args.render_scale
    a = np.array(Image.open(sf), np.uint16); a = np.array(Image.fromarray(a).resize((Wr, Hr), Image.NEAREST))
    uids = [int(u) for u in np.unique(a) if u != UNLAB]
    if args.with_bg and (a == UNLAB).any():
        uids.append(UNLAB)
    if not uids:
        skipped.append(name); continue
    lab = torch.from_numpy(a.astype(np.int64)).to(dev); onehot = torch.stack([(lab == u) for u in uids], -1).float()   # [h, w, C]
    col = torch.zeros(N, len(uids), device=dev, requires_grad=True)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=col, viewmats=viewmat, Ks=K, width=Wr, height=Hr,
                              sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    gcol, = torch.autograd.grad((out[0] * onehot).sum(), col)
    for k, u in enumerate(uids):
        r = rows.get(u)
        if r is None:
            r = rows[u] = torch.zeros(N, device=dev)
        r += gcol[:, k]
    done += 1
    if done % 50 == 0 or done == 1:
        print(f'[census] {done} views ({time.time() - t0:.0f}s), {len(rows)} labels so far', flush=True)
    del out, col, gcol
labels = sorted(rows)
Wm = torch.stack([rows[u] for u in labels]).cpu().numpy().astype(np.float32)
os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
np.savez_compressed(args.out_npz, W=Wm, labels=np.array(labels))
tot = Wm[[i for i, u in enumerate(labels) if u != UNLAB]].sum(0)
stats = {'gaussians': int(N), 'views': done, 'skipped': len(skipped), 'skipped_first': skipped[:5], 'labels': len(labels), 'render_scale': args.render_scale,
         'config': args.config, 'gaussians_with_label_weight': int((tot > 0).sum()), 'gaussians_label_weight_gt1': int((tot > 1.0).sum()), 'minutes': round((time.time() - t0) / 60, 2)}
json.dump(stats, open(args.out_npz.replace('.npz', '.json'), 'w'), indent=1)
print(f'[census] saved W {Wm.shape} -> {args.out_npz}; {json.dumps(stats)}', flush=True)
