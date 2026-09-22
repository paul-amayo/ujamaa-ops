"""Seam check on the merged H3DGS hierarchy of 05_13D: render the SAME keyframes the per-block seam profile
used (±20 kf around the 21->22, 35->36, 7->8 hand-offs) from the single merged model (tau = 0, all leaves)
and score full-frame PSNR against the photo, plus a GT | hierarchy strip for the 21->22 window.
Run from the repo dir in the h3dgs env:
  python /home/paperspace/logs/h3dgs_seam_eval.py -s <proj>/camera_calibration/aligned -i ../rectified/images \
     --model_path <proj>/output --hierarchy <proj>/output/merged.hier --scaffold_file <proj>/output/scaffold/point_cloud/iteration_30000
"""
import json, math, os, sys, urllib.request
import numpy as np, torch
from argparse import ArgumentParser
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/paperspace/code/hierarchical-3d-gaussians")
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render_post
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

parser = ArgumentParser()
lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
parser.add_argument("--tau", type=float, default=0.0)
parser.add_argument("--out", default="/home/paperspace/logs/h3dgs_seam")
args = parser.parse_args(sys.argv[1:])
os.makedirs(args.out, exist_ok=True)
dataset, pipe = lp.extract(args), pp.extract(args)
gaussians = GaussianModel(dataset.sh_degree); gaussians.active_sh_degree = dataset.sh_degree
scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)
cams = {c.image_name: c for c in list(scene.getTrainCameras()) + list(scene.getTestCameras())}
print(f"[hier] {gaussians._xyz.size(0)} nodes; {len(cams)} cameras loaded", flush=True)

# the same windows as seam_profile2.py (trajectory order + owner block from the demo server)
traj = json.load(urllib.request.urlopen("http://127.0.0.1:8001/scene/trajectory?stride=1"))["frames"]
own = [(f["image_name"], f["block"]) for f in traj if f.get("block") is not None]
bounds = {f"{own[i-1][1]}->{own[i][1]}": i for i in range(1, len(own)) if own[i][1] != own[i-1][1]}
SEAMS, SPAN = ["21->22", "35->36", "7->8"], 20
N = gaussians._xyz.size(0)
ri, pi, nri = (torch.zeros(N).int().cuda() for _ in range(3)); iw = torch.zeros(N).float().cuda(); ns = torch.zeros(N).int().cuda()

@torch.no_grad()
def render(vp):
    for a in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"):
        setattr(vp, a, getattr(vp, a).cuda())
    thr = (2 * (args.tau + 0.5)) * math.tan(vp.FoVx * 0.5) / (0.5 * vp.image_width)
    n = expand_to_size(gaussians.nodes, gaussians.boxes, thr, vp.camera_center, torch.zeros(3), ri, pi, nri)
    idx = ri[:n].int().contiguous(); nidx = nri[:n].contiguous()
    get_interpolation_weights(nidx, thr, gaussians.nodes, gaussians.boxes, vp.camera_center.cpu(), torch.zeros(3), iw, ns)
    im = render_post(vp, gaussians, pipe, torch.zeros(3, device="cuda"), render_indices=idx, parent_indices=pi,
                     interpolation_weights=iw, num_node_kids=ns, use_trained_exp=False)["render"]
    return torch.clamp(im, 0, 1), n

def psnr(a, b): return float(10 * torch.log10(1.0 / torch.mean((a - b) ** 2)))
rows, strip = [], {}
for sname in SEAMS:
    bi = bounds[sname]
    for off in range(-SPAN, SPAN):
        j = bi + off
        if not (0 <= j < len(own)): continue
        name, owner = own[j]
        stem = name.rsplit(".", 1)[0]
        vp = cams.get(name) or cams.get(stem)
        if vp is None: continue
        im, n = render(vp); gt = torch.clamp(vp.original_image.cuda(), 0, 1)
        p = psnr(im, gt)
        rows.append({"seam": sname, "off": off, "name": name, "owner": owner, "psnr": round(p, 2), "n_render": int(n), "test": name in set(x.image_name for x in scene.getTestCameras())})
        print(f"[hseam] {sname:>7} off {off:+3d} {name} owner {owner}: PSNR {p:.2f} ({n} gaussians)", flush=True)
        if sname == "21->22" and off in (-4, -1, 0, 2, 5, 10):
            strip[off] = (Image.fromarray((gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)),
                          Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)), rows[-1])
json.dump(rows, open(os.path.join(args.out, "h3dgs_seam.json"), "w"))
for sname in SEAMS:
    rs = [r for r in rows if r["seam"] == sname]
    print(f"[hseam] {sname}: " + " ".join(f"{r['off']:+d}:{r['psnr']:.1f}" for r in rs), flush=True)
allr = [r for r in rows]
near = [r["psnr"] for r in allr if abs(r["off"]) <= 2]; far = [r["psnr"] for r in allr if abs(r["off"]) >= 10]
print(f"[hseam] SUMMARY tau={args.tau}: at the former hand-off (|off|<=2) mean {np.mean(near):.2f} dB vs >=2 m away {np.mean(far):.2f} dB; hand-off frames (off 0): {[r['psnr'] for r in allr if r['off']==0]}", flush=True)
offs = sorted(strip); pad, th, sw, sh = 4, 22, 640, 360
canvas = Image.new("RGB", (2 * (sw + pad) + pad, len(offs) * (sh + th + pad) + pad + th), (20, 20, 20)); d = ImageDraw.Draw(canvas)
for c, lab in enumerate(["ground-truth photo", f"merged hierarchy (tau {args.tau})"]): d.text((pad + c * (sw + pad), 4), lab, fill=(255, 255, 255))
for r, off in enumerate(offs):
    gt, im, row = strip[off]; y = th + pad + r * (sh + th + pad)
    d.text((pad, y), f"keyframe {row['name']}  off {off:+d} ({abs(off)*0.2:.1f} m {'before' if off < 0 else 'after'} the old hand-off)  PSNR {row['psnr']} dB", fill=(255, 230, 120))
    for c, x in enumerate((gt, im)): canvas.paste(x.resize((sw, sh)), (pad + c * (sw + pad), y + th))
canvas.save(os.path.join(args.out, "h3dgs_seam_21_22_strip.png"))
print("[hseam] DONE", flush=True)
