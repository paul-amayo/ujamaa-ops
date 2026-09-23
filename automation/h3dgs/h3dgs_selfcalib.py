#!/usr/bin/env python
"""Pose-FIXED camera self-calibration from COLMAP features: with every camera pose held at its LiDAR-odometry
value (so scale/baseline are known and focal cannot trade against depth), alternate (a) re-triangulating every
track with the current intrinsics and (b) a robust least-squares fit of the intrinsics (fx fy cx cy [k1 k2]) to
all reprojections. Input: a COLMAP model from point_triangulator (images with xys/point3D_ids, points3D tracks).
  python h3dgs_selfcalib.py <model_dir> [--model PINHOLE|OPENCV] [--iters 6] [--fix_pp] [--fix_f]
"""
import argparse, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); from read_write_model import read_model, qvec2rotmat
ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("--model", default="OPENCV", choices=["PINHOLE", "OPENCV"])
ap.add_argument("--iters", type=int, default=6); ap.add_argument("--fix_pp", action="store_true"); ap.add_argument("--fix_f", action="store_true"); ap.add_argument("--max_obs", type=int, default=800000)
a = ap.parse_args()
cams, ims, pts = read_model(a.model_dir, ".bin"); cam = cams[1]; W, H = cam.width, cam.height
R = {i: qvec2rotmat(im.qvec) for i, im in ims.items()}; t = {i: im.tvec for i, im in ims.items()}
# observations: (image id, point id, xy)
obs_i, obs_p, obs_xy = [], [], []
for i, im in ims.items():
    m = im.point3D_ids >= 0
    obs_i.append(np.full(m.sum(), i)); obs_p.append(im.point3D_ids[m]); obs_xy.append(im.xys[m])
obs_i = np.concatenate(obs_i); obs_p = np.concatenate(obs_p); obs_xy = np.concatenate(obs_xy)
pid = {p: k for k, p in enumerate(sorted(pts))}; obs_k = np.array([pid[p] for p in obs_p])
Rs = np.stack([R[i] for i in obs_i]); ts = np.stack([t[i] for i in obs_i])
if len(obs_xy) > a.max_obs:
    sel = np.random.default_rng(0).choice(len(obs_xy), a.max_obs, replace=False); obs_i, obs_k, obs_xy, Rs, ts = obs_i[sel], obs_k[sel], obs_xy[sel], Rs[sel], ts[sel]
print(f"[selfcalib] {len(ims)} fixed cameras, {len(pts)} points, {len(obs_xy)} observations, start {cam.model} {np.round(cam.params, 3).tolist()}", flush=True)
X = np.stack([pts[p].xyz for p in sorted(pts)])
def project(params, Xk):
    fx, fy, cx, cy = params[:4]; k1, k2 = (params[4], params[5]) if len(params) > 4 else (0.0, 0.0)
    Pc = np.einsum("nij,nj->ni", Rs, Xk) + ts; z = Pc[:, 2]; x, y = Pc[:, 0] / z, Pc[:, 1] / z
    r2 = x * x + y * y; d = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([fx * x * d + cx, fy * y * d + cy], 1), z
def retriangulate(params):
    """linear DLT per track with the current intrinsics (undistorting the observations first)."""
    fx, fy, cx, cy = params[:4]; k1, k2 = (params[4], params[5]) if len(params) > 4 else (0.0, 0.0)
    xn = (obs_xy[:, 0] - cx) / fx; yn = (obs_xy[:, 1] - cy) / fy
    for _ in range(4 if (k1 or k2) else 0):   # invert the distortion by fixed-point iteration
        r2 = xn * xn + yn * yn; d = 1 + k1 * r2 + k2 * r2 * r2; xn = (obs_xy[:, 0] - cx) / fx / d; yn = (obs_xy[:, 1] - cy) / fy / d
    P = np.concatenate([Rs, ts[:, :, None]], 2)   # 3x4 per observation
    A1 = xn[:, None] * P[:, 2] - P[:, 0]; A2 = yn[:, None] * P[:, 2] - P[:, 1]
    order = np.argsort(obs_k, kind="stable"); ks = obs_k[order]; A1, A2 = A1[order], A2[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1, len(ks)]
    Xn = X.copy()
    for s, e in zip(starts[:-1], starts[1:]):
        if e - s < 2: continue
        A = np.concatenate([A1[s:e], A2[s:e]]); _, _, vt = np.linalg.svd(A, full_matrices=False); v = vt[-1]
        if abs(v[3]) > 1e-9: Xn[ks[s]] = v[:3] / v[3]
    return Xn
params = np.array(list(cam.params[:4]) + ([0.0, 0.0] if a.model == "OPENCV" else []), float)
if a.model == "OPENCV" and cam.model == "OPENCV": params[4:6] = cam.params[4:6]
for it in range(a.iters):
    X = retriangulate(params)
    uv, z = project(params, X[obs_k]); ok = z > 0.3; res = np.linalg.norm(uv - obs_xy, axis=1)
    keep = ok & (res < np.percentile(res[ok], 95))   # trim the worst 5 % (wrong matches / bad poses)
    def fun(p):
        q = params.copy(); q[free] = p; uvp, _ = project(q, X[obs_k][keep]); return (uvp - obs_xy[keep]).ravel()
    free = [j for j in range(len(params)) if not (a.fix_f and j in (0, 1)) and not (a.fix_pp and j in (2, 3))]
    sol = least_squares(fun, params[free], loss="huber", f_scale=1.0, max_nfev=50); params[free] = sol.x
    uv, _ = project(params, X[obs_k]); res2 = np.linalg.norm(uv - obs_xy, axis=1)
    print(f"[selfcalib] iter {it}: rms {np.sqrt(np.mean(res[keep]**2)):.3f} -> {np.sqrt(np.mean(res2[keep]**2)):.3f} px (median {np.median(res2[keep]):.3f}); params {np.round(params, 4).tolist()}", flush=True)
nom = cam.params[:4]
print(f"[selfcalib] RESULT {a.model}: fx {params[0]:.2f} fy {params[1]:.2f} cx {params[2]:.2f} cy {params[3]:.2f}" + (f" k1 {params[4]:.4f} k2 {params[5]:.4f}" if len(params) > 4 else "") + f"  | vs start: fx {100*(params[0]/nom[0]-1):+.2f}% fy {100*(params[1]/nom[1]-1):+.2f}% cx {params[2]-nom[2]:+.1f}px cy {params[3]-nom[3]:+.1f}px", flush=True)
