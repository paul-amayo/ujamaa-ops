#!/usr/bin/env python3
"""lerf_relevancy.py — render LERF relevancy maps for an identity cell's verdict frames (ovbench, 2026-11-10).
Loads a trained LERF run (kerrj/lerf db08d57 under PYTHONPATH) with nerfstudio's eval_setup, sets the query list as LERF positives
and calls the model's own get_outputs_for_camera: per query, the best of LERF's 30 scales (0..1.5) is picked over the image and
the map re-rendered at that scale; relevancy = min over the canonical negatives (object, things, stuff, texture) of
softmax(10 [pos, neg])[pos] — LERF's published readout, untouched. Frames outside the run's datasets (held out of both splits) get
their camera from the chunk COLMAP model through the run's dataparser_transforms.json, the mapping splat_feature_render.py uses for
HiGH, self-checked against a workspace camera. Render size = HiGH's (render_scale 0.5 -> 640 x 360).
Writes <out-dir>/<frame>.npz: S [H, W, Q] float16 relevancy, rgb [H, W, 3] uint8, queries [Q], best_scale_note.
Run from the training cwd (LERF's CLIP/DINO caches resolve to ./outputs/<data name>), nerf_new pixi env with
PYTHONPATH=<lerf src>:  lerf_relevancy.py --config <run>/config.yml --identity <identity> --out-dir <dir> [--queries-json q.json]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True); ap.add_argument('--identity', required=True); ap.add_argument('--out-dir', required=True)
ap.add_argument('--queries-json', required=True, help='{"queries": [...]} — the same list for every frame (cell vocabulary)')
ap.add_argument('--frames', nargs='*', default=None, help='default: <identity>/frames.json'); ap.add_argument('--render-scale', type=float, default=0.5)
ap.add_argument('--colmap-sparse', default='', help="default: parsed from <identity>/split_names.json 'source'")
a = ap.parse_args()
N = Path(a.identity); frames = a.frames or json.load(open(N / 'frames.json'))['frames']
Q = json.load(open(a.queries_json))['queries']
sparse = a.colmap_sparse or json.load(open(N / 'split_names.json'))['source'].split(' ')[0]
_, pipe, _, step = eval_setup(Path(a.config), test_mode='test'); model = pipe.model; model.eval(); dev = model.device
dm = pipe.datamanager; cams = {}
for ds in (dm.train_dataset, dm.eval_dataset):
    for i, p in enumerate(ds.image_filenames):
        cams[Path(p).name] = (ds, i)
train_names = {Path(p).name for p in dm.train_dataset.image_filenames}
ws = json.load(open(Path(dm.train_dataset.image_filenames[0]).parent.parent / 'transforms.json'))
dp = json.load(open(Path(a.config).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale'])
from read_write_model import read_images_binary, qvec2rotmat
ims = {im.name: im for im in read_images_binary(os.path.join(sparse, 'images.bin')).values()}
def colmap_camera(name):
    im = ims.get(name)
    if im is None: return None
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
    c2w = T @ (np.linalg.inv(w2c) @ np.diag([1.0, -1.0, -1.0, 1.0])); c2w[:3, 3] *= s
    return Cameras(camera_to_worlds=torch.tensor(c2w[:3], dtype=torch.float32)[None], fx=float(ws['fl_x']), fy=float(ws['fl_y']), cx=float(ws['cx']), cy=float(ws['cy']),
                   width=int(ws['w']), height=int(ws['h']), camera_type=CameraType.PERSPECTIVE).to(dev)
n0 = next(iter(cams)); ds0, i0 = cams[n0]; ca = ds0.cameras[i0:i0 + 1].to(dev).camera_to_worlds; cb = colmap_camera(n0).camera_to_worlds
print(f'[lerf] step {step} | {len(Q)} queries | colmap->workspace camera check on {n0}: max |diff| {float((ca - cb).abs().max()):.2e}', flush=True)
leak = [f for f in frames if f in train_names and json.load(open(N / 'frames.json'))['info'][f]['split'] == 'eval']
if leak: sys.exit(f'[lerf] held-out frames found in the LERF training set: {leak}')
os.makedirs(a.out_dir, exist_ok=True)
model.image_encoder.set_positives(Q)
for name in frames:
    t0 = time.time()
    if name in cams: ds, i = cams[name]; cam = ds.cameras[i:i + 1].to(dev); src = 'workspace-' + ('train' if name in train_names else 'eval')
    else:
        cam = colmap_camera(name); src = 'colmap'
        if cam is None: print(f'[lerf] {name}: no camera — skipped', flush=True); continue
    cam.rescale_output_resolution(a.render_scale)
    with torch.no_grad(): out = model.get_outputs_for_camera(cam)
    S = torch.cat([out[f'relevancy_{q}'] for q in range(len(Q))], -1).float().cpu().numpy()
    rgb = (out['rgb'].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    np.savez_compressed(Path(a.out_dir) / f'{name[:-4]}.npz', S=S.astype(np.float16), rgb=rgb, queries=np.array(Q), src=src, step=step)
    print(f'[lerf] {name} ({src}): {S.shape[1]}x{S.shape[0]} | max rel tree {S[..., 0].max():.3f} | {time.time() - t0:.0f}s', flush=True)
print('[lerf] done', flush=True)
