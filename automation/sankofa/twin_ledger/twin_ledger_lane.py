#!/usr/bin/env python3
"""twin_ledger_lane.py (11 Nov 2026, A100 session) - JIRS test JS-1, addendum A100-2: the twin's lane floor exactly as the
design writes it. At every fifth frame of each cell (the frames twin_ledger_render.py used, n % 5 == 0 over prep's frame list),
the expected depth of the full splat (gsplat RGB+ED, the render's call with the colours replaced by a constant, which leaves
the depth channel unchanged); the pixels 2 to 4 m ahead with |x| <= 0.3 m in the camera frame; the 90th percentile of camera
height minus point height along the pass's levelling direction, at least 15 points. Written twice per frame: without the
opacity cut (the design; primary) and with the render's accumulated-opacity >= 0.5 cut (the deviation line; must equal the
render's lane_h). No score. Waits while the GPU has less than --min-free-gb free. nerf_new pixi env:
  twin_ledger_lane.py <prep.json> <out dir> [--season dec,apr]"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from gsplat import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.cameras.cameras import Cameras, CameraType
ap = argparse.ArgumentParser(); ap.add_argument('prep'); ap.add_argument('out'); ap.add_argument('--season', default='dec,apr'); ap.add_argument('--render-scale', type=float, default=0.5); ap.add_argument('--min-free-gb', type=float, default=12.0); a = ap.parse_args()
torch.set_num_threads(3); torch.set_grad_enabled(False)
P = json.load(open(a.prep)); OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); GL = np.diag([1.0, -1.0, -1.0, 1.0])
for season in a.season.split(','):
    S = P[season]; Rw = np.array(S['R']); up = np.array(S['up'])
    for cell in sorted(S['cells']):
        f_out = OUT / f'{season}_{cell}.npz'
        if f_out.exists(): print(f'[lane] {season} {cell}: done already', flush=True); continue
        C = S['cells'][cell]; frames = C['frames']
        while torch.cuda.mem_get_info()[0] / 2**30 < a.min_free_gb: print(f'[lane] GPU has {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB free; waiting', flush=True); time.sleep(120)
        t0 = time.time(); _, pipe, _, step = eval_setup(Path(C['config']), test_mode='test'); model = pipe.model; model.eval(); dev = model.device
        g = model.gauss_params; N = g['means'].shape[0]; means, quats = g['means'].detach(), g['quats'].detach(); scales = torch.exp(g['scales'].detach()); opac = torch.sigmoid(g['opacities'].detach()).squeeze(-1)
        dp = json.load(open(Path(C['config']).parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform']); s = float(dp['scale']); fx, fy, cx, cy = C['K']; W_, H_ = C['wh']
        def camera(c2w_cv):
            c = T @ (np.asarray(c2w_cv, float) @ GL); c[:3, 3] *= s
            return Cameras(camera_to_worlds=torch.tensor(c[:3], dtype=torch.float32)[None], fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), width=int(W_), height=int(H_), camera_type=CameraType.PERSPECTIVE).to(dev)
        ds = pipe.datamanager.train_dataset; n0 = Path(ds.image_filenames[0]).name; tj = json.load(open(Path(C['ws']) / 'transforms.json'))
        p0 = next(np.array(f['transform_matrix'], float) @ GL for f in tj['frames'] if Path(f['file_path']).name == n0); d0 = float((ds.cameras[0:1].to(dev).camera_to_worlds - camera(p0).camera_to_worlds).abs().max())
        assert d0 < 1e-3, f'{cell}: camera mapping check failed on {n0}: {d0}'
        ones = torch.ones((N, 1), device=dev); upt = torch.tensor(up, dtype=torch.float32, device=dev); out = {k: [] for k in ('k_nocut', 'h_nocut', 'n_nocut', 'k_cut', 'h_cut', 'n_cut', 'alpha_band_median')}
        for n, fr in enumerate(frames):
            if n % 5: continue
            k = int(fr['k']); c2w = np.array(fr['c2w'], float); cam = camera(c2w); cam.rescale_output_resolution(a.render_scale); viewmat = get_viewmat(cam.camera_to_worlds); K = cam.get_intrinsics_matrices().to(dev); w, h = int(cam.width.item()), int(cam.height.item())
            fa, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, colors=ones, viewmats=viewmat, Ks=K, width=w, height=h, packed=False, near_plane=0.01, far_plane=1e10, render_mode='RGB+ED', sh_degree=None, rasterize_mode=model.config.rasterize_mode)
            z = fa[0, ..., 1] / s; al = alpha[0, ..., 0]
            vv, uu = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij'); kx, ky, kcx, kcy = K[0, 0, 0], K[0, 1, 1], K[0, 0, 2], K[0, 1, 2]
            pc = torch.stack([(uu + 0.5 - kcx) / kx * z, (vv + 0.5 - kcy) / ky * z, z], -1); M = torch.tensor(Rw.T @ c2w[:3, :3], dtype=torch.float32, device=dev)
            band = (z >= 2.0) & (z <= 4.0) & (pc[..., 0].abs() <= 0.3)
            for tag, ln in (('nocut', band), ('cut', band & (al >= 0.5))):
                if int(ln.sum()) >= 15: out[f'k_{tag}'].append(k); out[f'h_{tag}'].append(float(torch.quantile(-((pc[ln] @ M.T) @ upt), 0.9))); out[f'n_{tag}'].append(int(ln.sum()))
            out['alpha_band_median'].append(float(al[band].median()) if int(band.sum()) else float('nan'))
        np.savez(f_out, **{k: np.array(v) for k, v in out.items()})
        print(f"[lane] {season} {cell}: {len(out['h_nocut'])} frames without the cut (median {np.median(out['h_nocut']) if out['h_nocut'] else float('nan'):.4f} m), {len(out['h_cut'])} with it (median {np.median(out['h_cut']) if out['h_cut'] else float('nan'):.4f} m); median alpha in the band {np.nanmedian(out['alpha_band_median']):.3f}; {time.time() - t0:.0f}s", flush=True)
        del model, pipe, means, quats, scales, opac, ones; torch.cuda.empty_cache()
print('[lane] done', flush=True); (OUT / f'DONE_{a.season.replace(",", "_")}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
