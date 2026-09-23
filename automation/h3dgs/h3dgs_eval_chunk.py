"""Held-out evaluation of a merged H3DGS hierarchy with the VRAM-bounded compact renderer.
Each test camera is rendered at the pose of the CHUNK that owns it (the pose the hierarchy was trained on);
`--aligned` additionally scores the global/aligned pose (what render_hierarchy.py --eval uses).
  python h3dgs_eval_compact.py <proj_dir> [--taus 0 3 6] [--hier output/merged.hier] [--out output/eval_compact]
Writes <out>/scores.json and prints per-tau summaries (full-frame PSNR, FG-masked PSNR when fg masks exist)."""
import argparse, json, math, os, sys, time, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians"); sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians/preprocess")
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer")
from read_write_model import read_images_binary, read_cameras_binary, qvec2rotmat
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from hier_compact import CompactHierarchy
ap = argparse.ArgumentParser(); ap.add_argument("proj"); ap.add_argument("--taus", nargs="+", type=float, default=[0.0, 3.0, 6.0])
ap.add_argument("--hier", default="output/merged.hier"); ap.add_argument("--out", default="output/eval_compact"); ap.add_argument("--aligned", action="store_true")
ap.add_argument("--save", type=int, default=12, help="save this many renders per tau (first N test views)")
ap.add_argument("--fg", default="", help="foreground-mask dir override (export_meta fg_masks otherwise)")
ap.add_argument("--only_chunk", default="", help="score only the test views owned by this chunk (single-chunk hierarchies)")
ap.add_argument("--train_sample", type=int, default=0, help="score N evenly spaced TRAINING views instead of the held-out set (fit diagnostic)")
ap.add_argument("--exposure_json", default="", help="apply per-image trained exposure (chunk's exposure.json) before scoring")
ap.add_argument("--right_half", action="store_true", help="score only the right half of each image (H3DGS train_test_exp protocol)")
ap.add_argument("--sky", default="", help="sky-mask dir (255 = sky); adds a sky-masked PSNR over the non-sky pixels (default: the survey's prod/tassili/sky_masks)")
a = ap.parse_args(); PROJ = Path(a.proj); CC = PROJ / "camera_calibration"; OUT = PROJ / a.out; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(PROJ / "export_meta.json")); fg_dir = a.fg or meta.get("fg_masks"); FG = Path(fg_dir) if fg_dir and Path(fg_dir).exists() else None
sky_dir = a.sky or meta.get("sky_masks") or str(PROJ.parent.parent / "prod/tassili/sky_masks"); SKY = Path(sky_dir) if Path(sky_dir).exists() else None
cam = read_cameras_binary(str(CC / "aligned/sparse/0/cameras.bin"))[1]; fx, fy, cx, cy = cam.params[:4]; W, H = cam.width, cam.height
test = [l.strip() for l in open(CC / "aligned/sparse/0/test.txt") if l.strip()]
if a.train_sample:
    tset = set(test); allnames = sorted(im.name for im in read_images_binary(str(CC / "aligned/sparse/0/images.bin")).values() if im.name not in tset)
    test = allnames[::max(1, len(allnames) // a.train_sample)][:a.train_sample]
EXPO = json.load(open(a.exposure_json)) if a.exposure_json else {}; N_EXPO = 0
def apply_expo(im, name):
    """H3DGS/3DGS trained exposure: 3x4 affine on RGB (gaussian_renderer.render with use_trained_exp)."""
    global N_EXPO
    e = EXPO.get(name, EXPO.get(Path(name).stem))
    if e is None: return im
    e = torch.tensor(e, dtype=torch.float32, device=im.device); N_EXPO += 1
    return (torch.matmul(im.permute(1, 2, 0), e[:3, :3]).permute(2, 0, 1) + e[:3, 3, None, None]).clamp(0, 1)
def crop_right(t):
    return t[..., t.shape[-1] // 2:] if a.right_half else t
aligned = {im.name: im for im in read_images_binary(str(CC / "aligned/sparse/0/images.bin")).values()}
chunks = {}
for cdir in sorted((CC / "chunks").glob("*_*")):
    c = np.loadtxt(cdir / "center.txt"); e = np.loadtxt(cdir / "extent.txt")
    chunks[cdir.name] = (c, e, {im.name: im for im in read_images_binary(str(cdir / "sparse/0/images.bin")).values()})
def c2w_of(im):
    w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec; return np.linalg.inv(w2c)
def owner_pose(name):
    best = None
    for cname, (c, e, ims) in chunks.items():
        if name not in ims: continue
        m = c2w_of(ims[name]); inside = abs(m[0, 3] - c[0]) <= e[0] / 2 and abs(m[1, 3] - c[1]) <= e[1] / 2
        if inside: return cname, m
        best = best or (cname, m)
    return best
class Cam: pass
def make_cam(c2w):
    k = Cam(); k.image_width, k.image_height = W, H
    k.FoVx = 2 * math.atan(W / (2 * fx)); k.FoVy = 2 * math.atan(H / (2 * fy)); k.image_name = "eval"
    w2c = np.linalg.inv(c2w)
    k.world_view_transform = torch.tensor(getWorld2View2(c2w[:3, :3], w2c[:3, 3])).float().transpose(0, 1).cuda()
    k.projection_matrix = torch.tensor(getProjectionMatrix(0.01, 100.0, k.FoVx, k.FoVy, cx / W, cy / H)).float().transpose(0, 1).cuda()
    k.full_proj_transform = (k.world_view_transform.unsqueeze(0).bmm(k.projection_matrix.unsqueeze(0))).squeeze(0); k.camera_center = k.world_view_transform.inverse()[3, :3]
    return k
def psnr(a, b, m=None):
    d = (a - b) ** 2
    if m is not None: d = d[:, m]
    return 10 * math.log10(1.0 / max(d.mean().item(), 1e-12))
t0 = time.time(); ch = CompactHierarchy(str(PROJ / a.hier), str(PROJ / "output/scaffold/point_cloud/iteration_30000"))
print(f"[eval] {a.hier}: {ch.N} nodes loaded in {time.time()-t0:.0f}s, resident {ch.gpu_gib():.2f} GiB; {len(test)} test views; fg masks: {bool(FG)}", flush=True)
rows = []
for tau in a.taus:
    tdir = OUT / f"render_{tau:g}"; tdir.mkdir(exist_ok=True); saved = 0; t1 = time.time()
    for name in test:
        op = owner_pose(name)
        if op is None: continue
        cname, m = op
        if a.only_chunk and cname != a.only_chunk: continue
        gt = torch.from_numpy(np.asarray(Image.open(CC / "rectified/images" / name).convert("RGB"), np.float32) / 255).permute(2, 0, 1).cuda()
        mask = None
        if FG and (FG / name).exists(): mask = torch.from_numpy(np.asarray(Image.open(FG / name).convert("L")) > 0).cuda()
        if not rows: print(f"[eval] before first render: free {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB, torch reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
        try:
            im, n = ch.render(make_cam(m), tau)
        except Exception as ex:
            print(f"[eval] render failed on {name} ({cname}): {type(ex).__name__}: {str(ex)[:120]} | free {torch.cuda.mem_get_info()[0]/2**30:.2f} GiB reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True); raise
        if EXPO: im = apply_expo(im, name)
        if saved < a.save: Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(tdir / name); saved += 1
        ims, gts = crop_right(im), crop_right(gt); masks = crop_right(mask) if mask is not None else None   # train_test_exp protocol: score the right half only
        nosky = None
        if SKY and (SKY / name).exists(): nosky = crop_right(torch.from_numpy(np.asarray(Image.open(SKY / name).convert("L")) == 0).cuda())
        r = {"name": name, "tau": tau, "chunk": cname, "psnr": psnr(ims, gts), "psnr_fg": psnr(ims, gts, masks) if masks is not None else None,
             "psnr_nosky": psnr(ims, gts, nosky) if nosky is not None else None, "nodes": n}
        if a.aligned and name in aligned:
            im2, _ = ch.render(make_cam(c2w_of(aligned[name])), tau); r["psnr_aligned_pose"] = psnr(crop_right(apply_expo(im2, name) if EXPO else im2), gts)
        rows.append(r)
    rs = [r for r in rows if r["tau"] == tau]
    fg = [r["psnr_fg"] for r in rs if r["psnr_fg"] is not None]
    msg = f"[eval] tau {tau:g}: {len(rs)} views in {time.time()-t1:.0f}s — full-frame PSNR mean {np.mean([r['psnr'] for r in rs]):.2f} median {np.median([r['psnr'] for r in rs]):.2f} dB"
    if fg: msg += f"; FG-masked mean {np.mean(fg):.2f} median {np.median(fg):.2f} dB"
    ns = [r["psnr_nosky"] for r in rs if r["psnr_nosky"] is not None]
    if ns: msg += f"; SKY-MASKED mean {np.mean(ns):.2f} median {np.median(ns):.2f} dB"
    if a.aligned: msg += f"; at the aligned (pre-chunk-BA) pose: {np.mean([r['psnr_aligned_pose'] for r in rs]):.2f} dB"
    print(msg, flush=True)
json.dump(rows, open(OUT / "scores.json", "w"), indent=1); print("[eval] DONE", flush=True)
