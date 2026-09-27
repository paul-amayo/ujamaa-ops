"""H3DGS hierarchy leaves -> HiGH stage-1 checkpoint for a block (the containment SIDE-CAR, Paul 2026-09-27: "prioritize
side car for containment, make sure we can match block style IoUs for row, plant and fruit"). H3DGS renders RGB only, so
the side-car is a HiGH model whose gaussians ARE the H3DGS leaves inside the block's region, converted into the block's
nerfstudio frame with the RGB SH carried over; the census-init stage-2 recipe then trains only the 32-d feature branch
(geometry frozen) and containment_eval / the render service run unchanged on the H3DGS geometry.

Frame map (analytic, no fitting): the block's transforms.json poses are OpenGL c2w in the LIO world; nerfstudio applies
its dataparser transform p_ns = s * (T p_lio + t); the H3DGS export rotated the same LIO world by R_W. So
p_ns = s * T R_W^T p_h + s t. Means, log-scales (+log s), quaternions (left-multiplied by the rotation), opacities
(hierarchy alpha is the opacity itself -> logit), SH (rotated with a numerically fitted 16x16 matrix on the 3DGS basis).
The template checkpoint provides the dict layout; Adam states are re-created (zeros) for the new gaussian count.
  python (h3dgs env) h3dgs_to_stage1.py --block <block_dir> --h3dgs <proj> --hier <.hier|.hier_opt> --template <ckpt>
         --dataparser <dataparser_transforms.json> --out <ckpt> [--margin 8] [--check-chunk 1_1]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians"); sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
from gaussian_hierarchy._C import load_hierarchy
ap = argparse.ArgumentParser(); ap.add_argument("--block", required=True); ap.add_argument("--h3dgs", required=True); ap.add_argument("--hier", required=True)
ap.add_argument("--template", required=True); ap.add_argument("--dataparser", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=float, default=8.0, help="metres around the block's camera bbox (H3DGS frame) to keep leaves from")
ap.add_argument("--margin-z", type=float, default=6.0); ap.add_argument("--check-chunk", default="", help="chunk whose images.bin gives the chunk-BA camera centres for a residual check")
ap.add_argument("--cell", default="", help="chunk dir (center.txt/extent.txt): keep leaves in the chunk cell + margin instead of the camera bbox (chunk side-cars)")
a = ap.parse_args(); t0 = time.time()
meta = json.load(open(Path(a.h3dgs) / "export_meta.json")); R_W = np.asarray(meta.get("world_rotation_to_zup") or meta["world_rotation_lio_to_h3dgs"], np.float64)[:3, :3]
dp = json.load(open(a.dataparser)); T = np.asarray(dp["transform"], np.float64); s = float(dp["scale"])
R = T[:, :3] @ R_W.T; t = T[:, 3]                      # p_ns = s * (R p_h + t)
assert abs(np.linalg.det(R) - 1) < 1e-4 and np.allclose(R @ R.T, np.eye(3), atol=1e-4), "frame map is not a rotation"
tj = json.load(open(Path(a.block) / "transforms.json")); assert str(tj.get("pose_convention", "")).startswith("opengl"), tj.get("pose_convention")
C_lio = np.array([np.asarray(f["transform_matrix"], np.float64)[:3, 3] for f in tj["frames"]]); C_h = C_lio @ R_W.T
lo, hi = C_h.min(0) - [a.margin, a.margin, a.margin_z], C_h.max(0) + [a.margin, a.margin, a.margin_z]
if a.cell:   # chunk side-car: the region is the chunk CELL (+ margin) rather than the camera bbox (chunk cameras reach into the neighbouring cells)
    ctr = np.loadtxt(Path(a.cell) / "center.txt"); ext = np.loadtxt(Path(a.cell) / "extent.txt")
    lo[:2] = ctr[:2] - ext[:2] / 2 - a.margin; hi[:2] = ctr[:2] + ext[:2] / 2 + a.margin
if a.check_chunk:
    from read_write_model import read_images_binary, qvec2rotmat
    ims = {im.name: im for im in read_images_binary(str(Path(a.h3dgs) / "camera_calibration/chunks" / a.check_chunk / "sparse/0/images.bin")).values()}; res = []
    for f in tj["frames"]:
        n = Path(f["file_path"]).name
        if n not in ims: continue
        w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(ims[n].qvec); w2c[:3, 3] = ims[n].tvec; ch = np.linalg.inv(w2c)[:3, 3]
        res.append(np.linalg.norm(s * (R @ ch + t) - s * (T[:, :3] @ np.asarray(f["transform_matrix"], np.float64)[:3, 3] + t)) / s)
    print(f"[sidecar] frame-map check vs chunk {a.check_chunk}: {len(res)} keyframes, camera-centre residual median {np.median(res):.3f} m max {np.max(res):.3f} m (chunk-BA shift)", flush=True)
xyz, shs, alpha, scales, rots, nodes, boxes = load_hierarchy(a.hier); n = nodes.numpy()
leaf_idx = n[n[:, 6] == 0, 2]; pos = xyz.numpy()[leaf_idx]; keep = np.all((pos >= lo) & (pos <= hi), axis=1); sel = leaf_idx[keep]
print(f"[sidecar] {a.hier}: {nodes.shape[0]} nodes, {len(leaf_idx)} leaves, {len(sel)} inside the block region (bbox {lo.round(1).tolist()}..{hi.round(1).tolist()} in the H3DGS frame)", flush=True)
means = (pos[keep] @ R.T + t) * s
lsc = scales.numpy()[sel] + np.log(s)
q = rots.numpy()[sel]; q = q / np.linalg.norm(q, axis=1, keepdims=True)
def quat_from_R(M):
    w = np.sqrt(max(0.0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2; x = (M[2, 1] - M[1, 2]) / (4 * w); y = (M[0, 2] - M[2, 0]) / (4 * w); z = (M[1, 0] - M[0, 1]) / (4 * w); return np.array([w, x, y, z])
qr = quat_from_R(R); w1, x1, y1, z1 = qr; w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
quats = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)
al = np.clip(np.abs(alpha.numpy()[sel].reshape(-1)), 1e-4, 1 - 1e-4); opac = np.log(al / (1 - al))[:, None]
# SH rotation: 3DGS real-SH basis (degree 3); new coefficients satisfy Y(d') sh' = Y(R^T d') sh for all directions d'
C0 = 0.28209479177387814; C1 = 0.4886025119029199; C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
def basis(D):
    x, y, z = D[:, 0], D[:, 1], D[:, 2]; xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    return np.stack([np.full_like(x, C0), -C1 * y, C1 * z, -C1 * x, C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy),
                     C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy), C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy), C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], 1)
rng = np.random.default_rng(0); Dn = rng.normal(size=(4000, 3)); Dn /= np.linalg.norm(Dn, axis=1, keepdims=True)
M = np.linalg.pinv(basis(Dn)) @ basis(Dn @ R)            # (16,16): sh' = M sh
sh = shs.numpy()[sel]                                     # (n,16,3)
sh_rot = np.einsum("ij,njc->nic", M, sh).astype(np.float32)
chk = np.abs(basis(Dn[:200]) @ sh_rot[:50].transpose(1, 0, 2).reshape(16, -1) - basis(Dn[:200] @ R) @ sh[:50].transpose(1, 0, 2).reshape(16, -1)).max()
ck = torch.load(a.template, map_location="cpu", weights_only=False); pre = [k for k in ck["pipeline"] if k.endswith("gauss_params.means")][0].rsplit("means", 1)[0]
ck["pipeline"].pop(pre + "high_features", None)   # a seed checkpoint may serve as the template: the stage-1 output carries no features (the chain adds zeros)
new = {"means": means, "scales": lsc, "quats": quats, "features_dc": sh_rot[:, 0, :], "features_rest": sh_rot[:, 1:, :], "opacities": opac}
old_n = ck["pipeline"][pre + "means"].shape[0]
for k, v in new.items(): ck["pipeline"][pre + k] = torch.from_numpy(np.ascontiguousarray(v)).float()
ck["optimizers"] = {}; ck["schedulers"] = {}   # no Adam moments: nerfstudio's load_optimizers iterates the loaded dict (empty = no-op) and eval_setup never reads them; halves the file (fleet of 43 blocks, 2026-09-27)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); torch.save(ck, a.out)
print(f"[sidecar] wrote {a.out}: {len(sel)} gaussians (template had {old_n}); scale {s:.4f}; SH rotation max error {chk:.2e}; opacity p50 {np.median(al):.3f}; log-scale p50 {np.median(lsc):.2f}; {time.time()-t0:.0f}s", flush=True)
