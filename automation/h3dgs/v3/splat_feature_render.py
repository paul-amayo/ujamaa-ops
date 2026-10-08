#!/usr/bin/env python3
"""Render per-gaussian identity features (features.bin, float32 [N, D]) through a splatfacto checkpoint's own geometry (recipe v3
identity, 2026-10-08) = hier_feature_render.py without the cut. One npz per frame: features [h, w, D] float32, alpha [h, w] float32,
rgb [h, w, 3] uint8 (the model's colour render at the same size, figure backdrop), nodes = N — the layout containment_eval.py
--features-npz reads. Frames absent from the workspace (e.g. verdict frames held out of both splits) get their camera from the chunk's
COLMAP pose through the run's dataparser_transforms.json (--colmap-sparse); the mapping is self-checked against a workspace camera.
nerf_new pixi env:  splat_feature_render.py --config <config.yml> --features <bin> --frames kf_a.png ... [--render-scale 0.5]
                        --out-dir <dir> [--colmap-sparse <chunk sparse/0>]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True); ap.add_argument('--features', required=True); ap.add_argument('--frames', nargs='+', required=True)
ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--out-dir', required=True); ap.add_argument('--colmap-sparse', default='')
args = ap.parse_args()
_, pipe, _, step = eval_setup(Path(args.config), test_mode='test'); model = pipe.model; model.eval(); model.step = step; dev = model.device
dm = pipe.datamanager; cams = {}
for ds in (dm.train_dataset, dm.eval_dataset):
    for i, p in enumerate(ds.image_filenames):
        cams[Path(p).name] = (ds, i)
g = model.gauss_params; N = g['means'].shape[0]
means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
F = torch.from_numpy(np.fromfile(args.features, dtype=np.float32).reshape(N, -1)).to(dev)
os.makedirs(args.out_dir, exist_ok=True)
print(f'[render] {Path(args.config).parent.parent.name}: {N} gaussians, features {tuple(F.shape)} ({(F.norm(dim=1) > 0).float().mean().item() * 100:.1f}% non-zero); '
      f'{len(cams)} workspace cameras', flush=True)
colmap = None
def colmap_camera(name):
    """Camera for a frame outside the workspace: chunk COLMAP pose -> OpenGL -> the run's dataparser transform + scale."""
    global colmap
    if colmap is None:
        from read_write_model import read_images_binary, qvec2rotmat
        dp = json.load(open(Path(args.config).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform'])
        tj = json.load(open(Path(args.config).parent / 'config.yml'.replace('config.yml', '')).parent / 'transforms.json') if False else None
        ims = {im.name: im for im in read_images_binary(os.path.join(args.colmap_sparse, 'images.bin')).values()}
        ws = json.load(open(Path(dm.train_dataset.image_filenames[0]).parent.parent / 'transforms.json'))
        colmap = (ims, T, float(dp['scale']), ws, qvec2rotmat)
    ims, T, s, ws, qvec2rotmat = colmap
    im = ims.get(name)
    if im is None:
        return None
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    c2w = T @ (np.linalg.inv(w2c) @ np.diag([1.0, -1.0, -1.0, 1.0])); c2w[:3, 3] *= s
    return Cameras(camera_to_worlds=torch.tensor(c2w[:3], dtype=torch.float32)[None], fx=float(ws['fl_x']), fy=float(ws['fl_y']), cx=float(ws['cx']), cy=float(ws['cy']),
                   width=int(ws['w']), height=int(ws['h']), camera_type=CameraType.PERSPECTIVE).to(dev)
if args.colmap_sparse:   # self-check of the COLMAP -> nerfstudio mapping on the first workspace frame
    n0 = next(iter(cams)); ds, i = cams[n0]; a = ds.cameras[i:i + 1].to(dev).camera_to_worlds; b = colmap_camera(n0).camera_to_worlds
    print(f'[render] colmap->workspace camera check on {n0}: max |diff| {float((a - b).abs().max()):.2e} (translation {float((a[..., 3] - b[..., 3]).abs().max()):.2e})', flush=True)
torch.set_grad_enabled(False)
for name in args.frames:
    t0 = time.time()
    if name in cams:
        ds, i = cams[name]; cam = ds.cameras[i:i + 1].to(dev); src = 'workspace'
    elif args.colmap_sparse:
        cam = colmap_camera(name); src = 'colmap'
        if cam is None:
            print(f'[render] {name}: not a camera of the chunk — skipped', flush=True); continue
    else:
        print(f'[render] {name}: not a workspace camera (pass --colmap-sparse) — skipped', flush=True); continue
    cam.rescale_output_resolution(args.render_scale)
    viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
    kw = dict(means=means, quats=quats, scales=scales, opacities=opac, viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10,
              render_mode='RGB', rasterize_mode=model.config.rasterize_mode)
    feat, alpha, _ = rasterization(colors=F, sh_degree=None, **kw)
    rgb = model.get_outputs(cam)['rgb']
    out = os.path.join(args.out_dir, f'{name[:-4]}.npz')
    np.savez_compressed(out, features=feat[0].float().cpu().numpy(), alpha=alpha[0, ..., 0].float().cpu().numpy(), rgb=(rgb.clamp(0, 1) * 255).byte().cpu().numpy(), nodes=int(N))
    fn = feat[0].norm(dim=-1)
    print(f'[render] {name} ({src}): {w}x{h}, ||f||>0.5 on {(fn > 0.5).float().mean().item() * 100:.1f}% of px, {time.time() - t0:.1f}s -> {out}', flush=True)
