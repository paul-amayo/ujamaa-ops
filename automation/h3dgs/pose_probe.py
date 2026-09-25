#
# Test-time pose refinement probe for a trained H3DGS chunk (UJAMAA, 2026-09-25; Paul's goal: Klapmuts H3DGS held-out
# sky-masked PSNR > 20 dB, "iterate on poses"). Geometry, opacity and colour of the hierarchy are FROZEN. Each view is
# rendered through gsplat at the tau expansion the compact evaluator uses (+ the scaffold skybox and, with --fill, the
# scaffold gaussians outside the chunk cell), scored sky-masked against the rectified photo at its SfM pose, then a
# 6-DoF camera-frame correction of that pose is optimised (L1 on non-sky pixels, coarse-to-fine) and the view is scored
# again at the corrected pose. Held-out views measure how much of the held-out gap is per-view pose error that the
# reconstruction cannot absorb; a sample of TRAINING views is the fit diagnostic (the trainer absorbed their pose error
# into the geometry, so they should gain much less). Nothing here changes the survey's poses or hierarchy.
#   python pose_probe.py -s <chunk dir> --model_path <trained chunk dir> --hierarchy <.hier_opt> -i ../../rectified/images
#          --eval --scaffold_file <scaffold point_cloud dir> [--fill] [--tau 3] [--steps 150] [--train_sample 16] [--sky <dir>]
# Writes <model_path>/pose_probe/scores.json (+ GT|before|after strips for the first --save owned held-out views) and
# prints [pose-probe] summaries (before/after PSNR, gain, correction sizes in scene units and degrees).
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from PIL import Image
from plyfile import PlyData
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gsplat import rasterization
from train_features import expand_view, gsplat_cam
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess")); from read_write_model import read_cameras_binary


def scaffold_fill(scaffold_dir, cells):
    """Scaffold gaussians OUTSIDE the chunk cell(s) — what the chunk saw during training and what neighbouring chunks
    provide after the merge (the compact evaluator's --fill) — as plain gsplat inputs. The skybox rows (first nsky) are
    skipped: create_from_hier already appended them to the model."""
    v = PlyData.read(os.path.join(scaffold_dir, "point_cloud.ply"))["vertex"]
    nsky = int(open(os.path.join(scaffold_dir, "pc_info.txt")).readline())
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float32)
    names = sorted([p.name for p in v.properties if p.name.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    rest = np.stack([v[n] for n in names], 1).astype(np.float32) if names else np.zeros((len(xyz), 0), np.float32)
    op = v["opacity"].astype(np.float32); sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float32)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float32)
    keep = np.ones(len(xyz), bool); keep[:nsky] = False
    for c, e in cells: keep &= ~((np.abs(xyz[:, 0] - c[0]) <= e[0] / 2) & (np.abs(xyz[:, 1] - c[1]) <= e[1] / 2))
    sel = np.nonzero(keep)[0]
    K = rest.shape[1] // 3 if rest.shape[1] else 0
    sh = np.zeros((len(sel), 16, 3), np.float32); sh[:, 0, :] = dc[sel]
    if K:   # f_rest is coefficient-major (all R coeffs, then G, then B)
        r = rest[sel].reshape(len(sel), 3, K).transpose(0, 2, 1); sh[:, 1:1 + min(K, 15), :] = r[:, :15, :]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()
    q = rot[sel] / np.maximum(np.linalg.norm(rot[sel], axis=1, keepdims=True), 1e-8)
    return t(xyz[sel]), t(np.exp(sc[sel])), t(q), t(1 / (1 + np.exp(-op[sel]))), t(sh), len(sel), nsky


def se3_apply(om, t, w2c0):
    """Camera-frame correction: w2c = [R(om) | t] @ w2c0 (rotation about / translation along the camera axes; the camera
    centre moves by |t| in world units)."""
    th = torch.sqrt((om * om).sum() + 1e-12); k = om / th; z = 0 * k[0]
    Kx = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]), torch.stack([-k[1], k[0], z])])
    R = torch.eye(3, device=om.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)
    T = torch.cat([torch.cat([R, t[:, None]], 1), torch.tensor([[0, 0, 0, 1.0]], device=om.device)], 0)
    return T @ w2c0


def render_K(g, w2c, Ks, w, h):
    """gsplat RGB (SH degree 3, classic mode, black background) for an explicit (already scaled) K; g = (means, scales, rots, opac, sh)."""
    means, scales, rots, opac, sh = g
    rgb, alpha, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=sh, viewmats=w2c[None], Ks=Ks[None],
                                  width=w, height=h, sh_degree=3, packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", rasterize_mode="classic")
    return rgb[0].permute(2, 0, 1).clamp(0, 1)


def render(g, w2c, K, W, H, scale=1.0):
    h, w = int(round(H * scale)), int(round(W * scale))
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda"); Ks[:2] *= (w / W)
    return render_K(g, w2c, Ks, w, h)


def rs_render(g, w2c0, K, W, H, scale, om0, t0, omv, tv, nb):
    """Rolling-shutter model: the pose varies linearly with the image row (constant velocity over the readout),
    pose(row) = exp(xi0 + f(row) * xiv) @ w2c0 with f from -0.5 (top row) to +0.5 (bottom row). Rendered as nb horizontal
    bands, each at its band-centre pose through a principal-point-shifted K (exact crop, no wasted pixels)."""
    h, w = int(round(H * scale)), int(round(W * scale))
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda"); Ks[:2] *= (w / W)
    edges = [int(round(h * b / nb)) for b in range(nb + 1)]; ims = []
    for b in range(nb):
        r0, r1 = edges[b], edges[b + 1]; fb = (b + 0.5) / nb - 0.5
        Kb = Ks.clone(); Kb[1, 2] -= r0
        ims.append(render_K(g, se3_apply(om0 + fb * omv, t0 + fb * tv, w2c0), Kb, w, r1 - r0))
    return torch.cat(ims, 1)


def psnr(a, b, m=None):
    d = (a - b) ** 2
    if m is not None: d = d[:, m]
    return 10 * math.log10(1.0 / max(d.mean().item(), 1e-12))


def main():
    parser = ArgumentParser(); lp = ModelParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--tau", type=float, default=3.0); parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr_rot", type=float, default=1e-3, help="Adam lr on the rotation vector (rad)")
    parser.add_argument("--lr_t", type=float, default=5e-3, help="Adam lr on the camera-frame translation (scene units)")
    parser.add_argument("--train_sample", type=int, default=16); parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--fill", action="store_true"); parser.add_argument("--sky", default=""); parser.add_argument("--save", type=int, default=6)
    parser.add_argument("--out", default="pose_probe"); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--all_test", action="store_true", help="score every held-out view the chunk holds, not only the ones inside its cell")
    parser.add_argument("--rs_bands", type=int, default=0, help="> 1: after the 6-DoF fit, also fit a rolling-shutter (constant-velocity, row-linear) pose with this many bands")
    parser.add_argument("--rs_views", type=int, default=0, help="run the rolling-shutter stage on at most this many held-out views (0 = all probed views)")
    args = parser.parse_args(sys.argv[1:]); safe_state(args.quiet)
    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree); gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)
    N = gaussians._xyz.size(0)
    cam0 = list(read_cameras_binary(os.path.join(dataset.source_path, "sparse", "0", "cameras.bin")).values())[0]
    fx, fy, cx, cy = [float(x) for x in cam0.params[:4]]; K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    bufs = [torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(), torch.zeros(N).float().cuda(), torch.zeros(N).int().cuda()]
    cell = (np.loadtxt(os.path.join(dataset.source_path, "center.txt")), np.loadtxt(os.path.join(dataset.source_path, "extent.txt")))
    FILL = None
    if args.fill and dataset.scaffold_file:
        FILL = scaffold_fill(dataset.scaffold_file, [cell])
        print(f"[pose-probe] scaffold fill: {FILL[5]} gaussians outside the cell (skybox {FILL[6]} rows skipped)", flush=True)
    proj = os.path.abspath(os.path.join(dataset.model_path, "..", ".."))
    meta = json.load(open(os.path.join(proj, "export_meta.json"))) if os.path.exists(os.path.join(proj, "export_meta.json")) else {}
    sky_dir = args.sky or meta.get("sky_masks") or os.path.join(meta.get("survey_root", os.path.join(proj, "..", "..")), "prod", "tassili", "sky_masks")
    SKY = sky_dir if os.path.isdir(sky_dir) else None
    out_dir = os.path.join(dataset.model_path, args.out); os.makedirs(out_dir, exist_ok=True)
    print(f"[pose-probe] hierarchy {dataset.hierarchy}: {N} nodes (skybox {gaussians.skybox_points}); tau {args.tau}; sky masks {SKY}; "
          f"cell centre {cell[0][:2].round(2).tolist()} extent {cell[1][:2].round(1).tolist()}", flush=True)
    stages = [(0.25, args.steps * 4 // 15, 1.0), (0.5, args.steps * 4 // 15, 0.5), (1.0, args.steps - 2 * (args.steps * 4 // 15), 0.25)]
    expo = gaussians.pretrained_exposures or {}

    def probe(cam, tag):
        for a in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"): setattr(cam, a, getattr(cam, a).cuda())
        W, H = int(cam.image_width), int(cam.image_height)
        thr = (2 * (args.tau + 0.5)) * math.tan(cam.FoVx * 0.5) / (0.5 * W)
        with torch.no_grad():
            means, scales, rots, opac, ri, pi, w = expand_view(gaussians, cam, thr, bufs)
            sh = gaussians.get_features; shs = (w.unsqueeze(2) * sh[ri] + (1 - w).unsqueeze(2) * sh[pi]).contiguous()
            if gaussians.skybox_points:
                sk = torch.arange(N - gaussians.skybox_points, N, device="cuda")
                means = torch.cat([means, gaussians.get_xyz[sk]]); scales = torch.cat([scales, gaussians.get_scaling[sk]])
                rots = torch.cat([rots, gaussians.get_rotation[sk]]); opac = torch.cat([opac, gaussians.get_opacity[sk].squeeze(-1)]); shs = torch.cat([shs, sh[sk]])
            if FILL is not None:
                means = torch.cat([means, FILL[0]]); scales = torch.cat([scales, FILL[1]]); rots = torch.cat([rots, FILL[2]]); opac = torch.cat([opac, FILL[3]]); shs = torch.cat([shs, FILL[4]])
        g = (means.contiguous(), scales.contiguous(), rots.contiguous(), opac.contiguous(), shs.contiguous())
        viewmat, _, _, _ = gsplat_cam(cam, K); w2c0 = viewmat[0].detach()
        name = cam.image_name if cam.image_name.endswith(".png") else cam.image_name + ".png"
        gt = cam.original_image.cuda().float()
        e = expo.get(cam.image_name, expo.get(name))
        def finish(im):   # trained per-image exposure (training views only; held-out views have none), as render_post applies it
            if e is None: return im
            return (torch.matmul(im.permute(1, 2, 0), e[:3, :3]).permute(2, 0, 1) + e[:3, 3, None, None]).clamp(0, 1)
        nosky = None
        if SKY and os.path.exists(os.path.join(SKY, name)):
            nosky = torch.from_numpy(np.asarray(Image.open(os.path.join(SKY, name)).convert("L")) == 0).cuda()
            if nosky.shape != (H, W): nosky = F.interpolate(nosky[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
        with torch.no_grad(): im0 = finish(render(g, w2c0, K, W, H)); p0 = psnr(im0, gt, nosky); p0f = psnr(im0, gt)
        om = torch.zeros(3, device="cuda", requires_grad=True); t = torch.zeros(3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([{"params": [om], "lr": args.lr_rot}, {"params": [t], "lr": args.lr_t}])
        loss0 = loss1 = None
        for scale, steps, lr_mult in stages:
            h, w_ = int(round(H * scale)), int(round(W * scale))
            gt_s = F.interpolate(gt[None], size=(h, w_), mode="area")[0] if scale != 1.0 else gt
            m_s = None if nosky is None else (F.interpolate(nosky[None, None].float(), size=(h, w_), mode="area")[0, 0] > 0.5 if scale != 1.0 else nosky)
            for grp, base in zip(opt.param_groups, (args.lr_rot, args.lr_t)): grp["lr"] = base * lr_mult
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                im = finish(render(g, se3_apply(om, t, w2c0), K, W, H, scale))
                d = (im - gt_s).abs(); loss = d[:, m_s].mean() if m_s is not None else d.mean()
                if loss0 is None: loss0 = float(loss)
                loss.backward(); opt.step(); loss1 = float(loss)
        with torch.no_grad():
            w2c1 = se3_apply(om, t, w2c0); im1 = finish(render(g, w2c1, K, W, H)); p1 = psnr(im1, gt, nosky); p1f = psnr(im1, gt)
        cc = cam.camera_center.detach().cpu().numpy(); inside = bool(abs(cc[0] - cell[0][0]) <= cell[1][0] / 2 and abs(cc[1] - cell[0][1]) <= cell[1][1] / 2)
        r = {"name": name, "set": tag, "inside_cell": inside, "psnr_nosky_before": p0, "psnr_nosky_after": p1, "psnr_before": p0f, "psnr_after": p1f,
             "gain_nosky": p1 - p0, "rot_deg": float(om.norm()) * 180 / math.pi, "shift_units": float(t.norm()), "t_cam": t.detach().cpu().tolist(),
             "om": om.detach().cpu().tolist(), "loss_first": loss0, "loss_last": loss1, "nodes": int(ri.numel()), "sky_masked": nosky is not None}
        print(f"[pose-probe] {tag} {name} inside={inside}: sky-masked {p0:.2f} -> {p1:.2f} dB (gain {p1 - p0:+.2f}); rot {r['rot_deg']:.3f} deg, shift {r['shift_units']:.4f} units; "
              f"L1 {loss0:.4f} -> {loss1:.4f}; {ri.numel()} nodes", flush=True)
        im2 = None
        if args.rs_bands > 1 and (tag != "test" or not args.rs_views or rs_budget[0] > 0):
            if tag == "test": rs_budget[0] -= 1
            om0 = om.detach().clone().requires_grad_(True); t0 = t.detach().clone().requires_grad_(True)
            omv = torch.zeros(3, device="cuda", requires_grad=True); tv = torch.zeros(3, device="cuda", requires_grad=True)
            opt2 = torch.optim.Adam([{"params": [om0], "lr": args.lr_rot * 0.5}, {"params": [t0], "lr": args.lr_t * 0.5}, {"params": [omv], "lr": args.lr_rot}, {"params": [tv], "lr": args.lr_t}])
            bases = (args.lr_rot * 0.5, args.lr_t * 0.5, args.lr_rot, args.lr_t)
            for scale, steps, lr_mult in stages:
                h, w_ = int(round(H * scale)), int(round(W * scale))
                gt_s = F.interpolate(gt[None], size=(h, w_), mode="area")[0] if scale != 1.0 else gt
                m_s = None if nosky is None else (F.interpolate(nosky[None, None].float(), size=(h, w_), mode="area")[0, 0] > 0.5 if scale != 1.0 else nosky)
                for grp, base in zip(opt2.param_groups, bases): grp["lr"] = base * lr_mult
                for _ in range(steps):
                    opt2.zero_grad(set_to_none=True)
                    im = finish(rs_render(g, w2c0, K, W, H, scale, om0, t0, omv, tv, args.rs_bands))
                    d = (im - gt_s).abs(); loss = d[:, m_s].mean() if m_s is not None else d.mean()
                    loss.backward(); opt2.step()
            with torch.no_grad():
                im2 = finish(rs_render(g, w2c0, K, W, H, 1.0, om0, t0, omv, tv, args.rs_bands)); p2 = psnr(im2, gt, nosky)
            r.update(psnr_nosky_rs=p2, gain_rs_vs_6dof=p2 - p1, rs_rot_deg=float(omv.norm()) * 180 / math.pi, rs_shift_units=float(tv.norm()),
                     rs_omv=omv.detach().cpu().tolist(), rs_tv=tv.detach().cpu().tolist(), rs_loss_last=float(loss))
            print(f"[pose-probe]   rolling-shutter ({args.rs_bands} bands) {name}: {p1:.2f} -> {p2:.2f} dB (gain vs 6-DoF {p2 - p1:+.2f}); top-to-bottom rot {r['rs_rot_deg']:.3f} deg, "
                  f"shift {r['rs_shift_units']:.4f} units (cam-frame {np.round(r['rs_tv'], 4).tolist()})", flush=True)
        return r, im0, im1 if im2 is None else torch.cat([im1, im2], 2), gt

    rows = []; saved = 0; t0 = time.time(); rs_budget = [args.rs_views]
    test_cams = scene.getTestCameras()
    for i in range(len(test_cams)):
        if args.max_views and sum(1 for r in rows if r["set"] == "test") >= args.max_views: break
        cam = test_cams[i]
        for a in ("camera_center",): setattr(cam, a, getattr(cam, a).cuda())
        cc = cam.camera_center.detach().cpu().numpy()
        if not args.all_test and not (abs(cc[0] - cell[0][0]) <= cell[1][0] / 2 and abs(cc[1] - cell[0][1]) <= cell[1][1] / 2): continue
        r, im0, im1, gt = probe(cam, "test"); rows.append(r)
        if saved < args.save:
            strip = torch.cat([gt, im0, im1], 2); strip = F.interpolate(strip[None], scale_factor=0.5, mode="area")[0]
            Image.fromarray((strip.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, r["name"].replace(".png", "") + "_gt_before_after.png")); saved += 1
    train_cams = scene.getTrainCameras(); nT = len(train_cams)
    if args.train_sample and nT:
        for i in range(0, nT, max(1, nT // args.train_sample))[: args.train_sample]:
            r, *_ = probe(train_cams[i], "train"); rows.append(r)
    for tag in ("test", "train"):
        rs = [r for r in rows if r["set"] == tag]
        if not rs: continue
        b = np.array([r["psnr_nosky_before"] for r in rs]); a = np.array([r["psnr_nosky_after"] for r in rs]); g = a - b
        print(f"[pose-probe] {tag}: {len(rs)} views — sky-masked PSNR median {np.median(b):.2f} -> {np.median(a):.2f} dB, mean {b.mean():.2f} -> {a.mean():.2f}; "
              f"gain median {np.median(g):+.2f} (p10 {np.percentile(g, 10):+.2f}, p90 {np.percentile(g, 90):+.2f}); "
              f"correction median rot {np.median([r['rot_deg'] for r in rs]):.3f} deg (p90 {np.percentile([r['rot_deg'] for r in rs], 90):.3f}), "
              f"shift {np.median([r['shift_units'] for r in rs]):.4f} units (p90 {np.percentile([r['shift_units'] for r in rs], 90):.4f})", flush=True)
        rr = [r for r in rs if "psnr_nosky_rs" in r]
        if rr:
            g2 = np.array([r["gain_rs_vs_6dof"] for r in rr]); tot = np.array([r["psnr_nosky_rs"] - r["psnr_nosky_before"] for r in rr])
            print(f"[pose-probe] {tag} rolling-shutter ({args.rs_bands} bands): {len(rr)} views — sky-masked median {np.median([r['psnr_nosky_before'] for r in rr]):.2f} -> {np.median([r['psnr_nosky_rs'] for r in rr]):.2f} dB; "
                  f"gain vs 6-DoF median {np.median(g2):+.2f} (p90 {np.percentile(g2, 90):+.2f}), total gain median {np.median(tot):+.2f}; top-to-bottom shift median {np.median([r['rs_shift_units'] for r in rr]):.4f} units "
                  f"(p90 {np.percentile([r['rs_shift_units'] for r in rr], 90):.4f}), rot median {np.median([r['rs_rot_deg'] for r in rr]):.3f} deg", flush=True)
    json.dump({"args": vars(args), "cell": [cell[0].tolist(), cell[1].tolist()], "rows": rows, "seconds": time.time() - t0}, open(os.path.join(out_dir, "scores.json"), "w"), indent=1)
    print(f"[pose-probe] DONE {len(rows)} views in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
