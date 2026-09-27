"""Backdrops for the row reel from the FULL H3DGS model (Paul, 2026-09-27: grey patches in the reel = regions beyond the
side-car's cut where it has no gaussians): every keyframe of the given blocks rendered at the block's own pose and
intrinsics through the H3DGS CHUNK whose cell contains the camera (the survey's merged hierarchy needs ~7 GB resident
plus large per-frame temporaries — it ran out of GPU memory beside the 04 chunk training), with the coarse scaffold's
gaussians filling everything outside that cell (the compact renderer's fill mode) and its skybox. Pixel-aligned with the
side-car overlay pass (same camera, same resolution scale). Writes <out>/<kf name> PNGs.
  python (h3dgs env) sidecar_row_backdrop.py --survey 05_13D_Jackal --blocks 018 019 ... --out <dir> [--scale 0.5] [--tau 0]"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians"); sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer")
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from read_write_model import read_images_binary
from hier_compact import CompactHierarchy
ap = argparse.ArgumentParser(); ap.add_argument("--survey", required=True); ap.add_argument("--blocks", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--cfg", default="lio_row100"); a = ap.parse_args()
S = Path("/home/paperspace/data/citrus_all") / a.survey; P = S / "experimental/h3dgs"; OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(P / "export_meta.json")); R_W = np.asarray(meta.get("world_rotation_to_zup") or meta["world_rotation_lio_to_h3dgs"], np.float64); GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
scaf = P / "output/scaffold/point_cloud/iteration_30000"; scaf_dir = str(scaf) if (scaf / "point_cloud.ply").exists() else ""
# frames (pose + intrinsics in the H3DGS frame) and the chunk whose cell holds each camera
cells = {}
for c in sorted(glob.glob(str(P / "camera_calibration/chunks/*_*"))):
    if (Path(c) / "sparse/0/images.bin").exists() and (P / "output/trained_chunks" / os.path.basename(c) / "hierarchy.hier_opt").exists():
        cells[os.path.basename(c)] = (np.loadtxt(Path(c) / "center.txt"), np.loadtxt(Path(c) / "extent.txt"))
frames = []
for b in a.blocks:
    tj = json.load(open(S / "prod/tassili/blocks_ns" / a.cfg / f"block_{b}" / "transforms.json"))
    if not str(tj.get("pose_convention", "")).startswith("opengl"): raise SystemExit(f"block {b}: untagged poses — convention must be checked first")
    W0, H0 = int(tj.get("w", 1280)), int(tj.get("h", 720)); K0 = (float(tj["fl_x"]), float(tj["fl_y"]), float(tj["cx"]), float(tj["cy"]))
    for f in tj["frames"]:
        c2w_h = R_W @ (np.asarray(f["transform_matrix"], np.float64) @ GL2CV); cc = c2w_h[:3, 3]
        own = [n for n, (ctr, ext) in cells.items() if abs(cc[0] - ctr[0]) <= ext[0] / 2 and abs(cc[1] - ctr[1]) <= ext[1] / 2]
        if not own: own = [min(cells, key=lambda n: np.hypot(cc[0] - cells[n][0][0], cc[1] - cells[n][0][1]))]
        frames.append((own[0], Path(f["file_path"]).name, c2w_h, W0, H0, K0))
by_chunk = {}
for fr in frames: by_chunk.setdefault(fr[0], []).append(fr)
print(f"[backdrop] {len(frames)} frames over chunks {{{', '.join(f'{k}: {len(v)}' for k, v in by_chunk.items())}}}", flush=True)
class Cam: pass
def make_cam(c2w_h, W, H, fx, fy, cx, cy):
    k = Cam(); k.image_width, k.image_height = W, H; k.FoVx = 2 * math.atan(W / (2 * fx)); k.FoVy = 2 * math.atan(H / (2 * fy)); k.image_name = "reel"
    w2c = np.linalg.inv(c2w_h); k.world_view_transform = torch.tensor(getWorld2View2(c2w_h[:3, :3], w2c[:3, 3])).float().transpose(0, 1).cuda()
    k.projection_matrix = torch.tensor(getProjectionMatrix(0.01, 100.0, k.FoVx, k.FoVy, cx / W, cy / H)).float().transpose(0, 1).cuda()
    k.full_proj_transform = (k.world_view_transform.unsqueeze(0).bmm(k.projection_matrix.unsqueeze(0))).squeeze(0); k.camera_center = k.world_view_transform.inverse()[3, :3]; return k
n = 0; t1 = time.time()
for cname, frs in by_chunk.items():
    t0 = time.time(); ch = CompactHierarchy(str(P / "output/trained_chunks" / cname / "hierarchy.hier_opt"), scaf_dir, [cells[cname]] if scaf_dir else None)
    print(f"[backdrop] chunk {cname}: {ch.n_hier} nodes + skybox {ch.skybox} + fill {ch.fill}, resident {ch.gpu_gib():.2f} GiB, loaded in {time.time()-t0:.0f}s; {len(frs)} frames", flush=True)
    for _, name, c2w_h, W0, H0, (fx0, fy0, cx0, cy0) in frs:
        W, H = int(round(W0 * a.scale)), int(round(H0 * a.scale)); s = W / W0
        with torch.no_grad(): im, _ = ch.render(make_cam(c2w_h, W, H, fx0 * s, fy0 * s, cx0 * s, cy0 * s), a.tau)
        cv2.imwrite(str(OUT / name), (im.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)); n += 1
    del ch; torch.cuda.empty_cache()
print(f"[backdrop] {n} frames -> {OUT} in {time.time()-t1:.0f}s", flush=True)
