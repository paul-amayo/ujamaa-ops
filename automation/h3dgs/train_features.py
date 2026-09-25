#
# HiGH feature stage for a trained hierarchy (UJAMAA, 2026-09-25; Paul: "patch hierarchical 3dgs so that it also does
# high features in the second stage"). Runs AFTER train_post on a hierarchy (chunk hierarchy.hier_opt or merged.hier):
# geometry, opacity and colour are frozen; every node gets a 32-d HiGH feature, rendered through gsplat with the same
# per-view node/parent expansion train_post uses, and trained with HiGH's per-pixel Lorentz geodesic loss against the
# per-block supervision (semantic_v2_B PNGs + the embedder's colour->target lookup table). Output: <model_path>/
# features.bin (float32 [N,32], node order of the hierarchy) + features.json.
#   python train_features.py -s <chunk dir> --model_path <trained chunk dir> --hierarchy <.hier_opt> -i <images dir>
#          --sem_blocks <blocks_ns cfg dir with block_*/semantic_v2_B> [--lut <lookup.npy>] [--iters 2000] [--lr 5e-3]
#          [--check]   # --check: render RGB through gsplat for one view and compare with the H3DGS rasterizer (geometry contract)
import os, sys, json, math, glob, time
import numpy as np
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render_post
from utils.general_utils import safe_state
from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights
from gsplat import rasterization
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "preprocess")); from read_write_model import read_cameras_binary

# ---- Lorentz helpers (vendored from high/encoders/lorentz.py, pure torch) ----
def exp_map0(x, curv=1.0, eps=1e-8):
    rc = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    return torch.sinh(torch.clamp(rc, min=eps, max=math.asinh(2 ** 15))) * x / torch.clamp(rc, min=eps)
def rowwise_inner(x, y, curv=1.0):
    xt = torch.sqrt(1 / curv + (x ** 2).sum(-1, keepdim=True)); yt = torch.sqrt(1 / curv + (y ** 2).sum(-1, keepdim=True))
    return (x * y).sum(-1, keepdim=True) - xt * yt
def geodesic(pred, target, curv=1.0):
    """HighModel.safe_geodesic_loss: clamp to +-12, exp-map both, acosh(-<x,y>_L) with domain guard. pred/target [..., D]."""
    x = exp_map0(torch.clamp(pred, -12.0, 12.0), curv); y = exp_map0(torch.clamp(target, -12.0, 12.0), curv)
    return torch.acosh(torch.clamp(-rowwise_inner(x, y, curv), min=1.0 + 1e-7)).squeeze(-1)

def direct_collate(x): return x

def expand_view(gaussians, cam, limit, bufs):
    """train_post's per-view hierarchy expansion + render_post's python interpolation -> plain gaussians (no grad on
    geometry) and the index/weight tensors needed to interpolate the feature parameter the same way."""
    render_indices, parent_indices, nodes_for_render, interp_w, num_sib = bufs
    to_render = expand_to_size(gaussians.nodes, gaussians.boxes, limit, cam.camera_center, torch.zeros((3)), render_indices, parent_indices, nodes_for_render)
    idx = render_indices[:to_render].int(); node_idx = nodes_for_render[:to_render]
    get_interpolation_weights(node_idx, limit, gaussians.nodes, gaussians.boxes, cam.camera_center.cpu(), torch.zeros((3)), interp_w, num_sib)
    n = idx.size(0); ri = idx.long(); pi = parent_indices[:n].long(); w = interp_w[:n].unsqueeze(1); wi = 1 - w
    with torch.no_grad():
        xyz, sc, rot, op = gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation, gaussians.get_opacity
        means = w * xyz[ri] + wi * xyz[pi]; scales = w * sc[ri] + wi * sc[pi]
        par = rot[pi].clone(); r = rot[ri]; par[(r * par).sum(1) < 0] *= -1; rots = w * r + wi * par
        opac = w * op[ri] + wi * op[pi]
    return means.contiguous(), scales.contiguous(), rots.contiguous(), opac.contiguous().squeeze(-1), ri, pi, w

def gsplat_cam(cam, K):
    viewmat = cam.world_view_transform.transpose(0, 1).contiguous().float().cuda()[None]   # H3DGS stores W2C transposed
    Ks = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
    return viewmat, Ks, int(cam.image_width), int(cam.image_height)

def load_lut_and_index(sem_blocks, lut_path):
    """name -> (semantic PNG, that block's colour->target table). The tables are `semantic_v2_B/high_targets.npz`
    (colors [n,3] uint8, targets [n,32] float32) written by export_high_targets.py in the nerf_new env: palette colour
    -> vocab word -> CLIP text embedding -> HyperEmbedder.encode_features -> log_map0, exactly HighDataloader.create().
    (The clip_cache `_lookup.npy` tables in the citrus blocks are all zeros at the mask colours — RGB-only runs — so
    the first attempt skipped every view, 2026-09-25.)"""
    blocks = sorted(glob.glob(os.path.join(sem_blocks, "block_[0-9][0-9][0-9]")))
    index, tables = {}, {}
    for b in blocks:
        tp = lut_path or os.path.join(b, "semantic_v2_B", "high_targets.npz")
        if not os.path.exists(tp): continue
        for q in glob.glob(os.path.join(b, "semantic_v2_B", "*.png")): index.setdefault(os.path.basename(q), (q, tp))
    return tables, index, lut_path or "per-block semantic_v2_B/high_targets.npz", len(blocks)

def block_table(tables, path):
    if path not in tables:
        z = np.load(path); cols = z["colors"].astype(np.int64); keys = (cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]
        tables[path] = dict(zip(keys.tolist(), z["targets"].astype(np.float32)))
    return tables[path]

def target_for(name, index, tables, W, H, cache):
    """Per-view target as (palette-index map [H,W] int64, table [n_colours,32]); colours absent from the block's table
    (background black, unknown) get a zero row = unlabelled. Returns the [H,W,32] target on the GPU, or None when the
    view has no labelled pixel."""
    if name in cache:
        c = cache[name]
        return None if c is None else c[1].cuda()[c[0].cuda()]
    ent = index.get(name)
    if ent is None: cache[name] = None; return None
    p, tp = ent; tab = block_table(tables, tp)
    from PIL import Image
    im = np.asarray(Image.open(p).convert("RGB"))
    if im.shape[0] != H or im.shape[1] != W: im = np.asarray(Image.fromarray(im).resize((W, H), Image.NEAREST))
    key = (im[..., 0].astype(np.int64) << 16) | (im[..., 1].astype(np.int64) << 8) | im[..., 2].astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    table = np.stack([tab.get(int(k), np.zeros(32, np.float32)) for k in uniq])                  # [n_colours,32]
    if not (np.abs(table).sum(1) > 1e-6).any(): cache[name] = None; return None
    c = (torch.from_numpy(inv.reshape(H, W).astype(np.int64)), torch.from_numpy(table)); cache[name] = c
    return c[1].cuda()[c[0].cuda()]

def main():
    parser = ArgumentParser(); lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    parser.add_argument("--sem_blocks", required=True); parser.add_argument("--lut", default="")
    parser.add_argument("--iters", type=int, default=2000); parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--feat_dim", type=int, default=32); parser.add_argument("--check", action="store_true")
    parser.add_argument("--out", default="features.bin"); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--eval_views", type=int, default=0, help="load <model_path>/<out>, render this many HELD-OUT views' feature maps, score them against the supervision (geodesic on labelled px) and write feat_eval/<name>.npz + PCA PNGs")
    args = parser.parse_args(sys.argv[1:]); safe_state(args.quiet)
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree); gaussians.active_sh_degree = dataset.sh_degree
    scene = Scene(dataset, gaussians, resolution_scales=[1], create_from_hier=True)
    N = gaussians._xyz.size(0); print(f"[feat] hierarchy {dataset.hierarchy}: {N} nodes, skybox {gaussians.skybox_points}", flush=True)
    cam0 = list(read_cameras_binary(os.path.join(dataset.source_path, "sparse", "0", "cameras.bin")).values())[0]
    fx, fy, cx, cy = [float(x) for x in cam0.params[:4]]; K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    bufs = [torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(), torch.zeros(N).int().cuda(), torch.zeros(N).float().cuda(), torch.zeros(N).int().cuda()]
    train_cams = scene.getTrainCameras(); limmax, limmin = 0.1, 0.005
    if args.check:   # geometry contract: RGB through gsplat (SH evaluated by gsplat) vs the H3DGS rasterizer on one view
        cam = train_cams[len(train_cams) // 2]
        for a in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"): setattr(cam, a, getattr(cam, a).cuda())
        limit = 0.02; means, scales, rots, opac, ri, pi, w = expand_view(gaussians, cam, limit, bufs)
        with torch.no_grad():
            sh = gaussians.get_features; shs = (w.unsqueeze(2) * sh[ri] + (1 - w).unsqueeze(2) * sh[pi]).contiguous()
            viewmat, Ks, W, H = gsplat_cam(cam, K)
            rgb, alpha, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=shs, viewmats=viewmat, Ks=Ks, width=W, height=H, sh_degree=gaussians.active_sh_degree, packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", rasterize_mode="classic")
            ref = render_post(cam, gaussians, pipe, torch.zeros(3, device="cuda"), render_indices=bufs[0][:ri.numel()].int(), parent_indices=bufs[1], interpolation_weights=bufs[3], num_node_kids=bufs[4], use_trained_exp=False)["render"]
            g = rgb[0].permute(2, 0, 1).clamp(0, 1); mse = ((g - ref) ** 2).mean().item(); psnr_gt = -10 * math.log10(max(((g - cam.original_image.cuda()) ** 2).mean().item(), 1e-12))
            print(f"[check] {cam.image_name}: gsplat vs H3DGS rasterizer {(-10*math.log10(max(mse,1e-12))):.1f} dB agreement; gsplat vs photo {psnr_gt:.2f} dB; {ri.numel()} nodes expanded", flush=True)
        return
    lut, index, lut_path, nb = load_lut_and_index(args.sem_blocks, args.lut)
    if args.eval_views:
        F = torch.from_numpy(np.fromfile(os.path.join(dataset.model_path, args.out), dtype=np.float32).reshape(N, args.feat_dim)).cuda()
        ed = os.path.join(dataset.model_path, "feat_eval"); os.makedirs(ed, exist_ok=True); cache = {}; scores = []
        from PIL import Image
        test_cams = scene.getTestCameras(); nT = len(test_cams); step = max(1, nT // args.eval_views)
        for k in range(0, nT, step)[: args.eval_views]:
            cam = test_cams[k]
            for a in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"): setattr(cam, a, getattr(cam, a).cuda())
            name = cam.image_name if cam.image_name in index else cam.image_name + ".png"; viewmat, Ks, W, H = gsplat_cam(cam, K)
            with torch.no_grad():
                means, scales, rots, opac, ri, pi, w = expand_view(gaussians, cam, 0.02, bufs); f = w * F[ri] + (1 - w) * F[pi]
                out, alpha, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=f, viewmats=viewmat, Ks=Ks, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", rasterize_mode="classic")
                pred = out[0]; tgt = target_for(name, index, lut, W, H, cache)
                rec = {"name": name, "nodes": int(ri.numel())}
                if tgt is not None:
                    lab = tgt.abs().sum(-1) > 1e-6; d = geodesic(pred[lab], tgt[lab]); rec.update(geodesic_mean=float(d.mean()), labelled=int(lab.sum()), norm_pred_labelled=float(pred[lab].norm(dim=-1).mean()), norm_target=float(tgt[lab].norm(dim=-1).mean()))
                scores.append(rec); print(f"[feat-eval] {rec}", flush=True)
                np.savez_compressed(os.path.join(ed, name.replace(".png", "") + ".npz"), features=pred.half().cpu().numpy(), alpha=alpha[0, ..., 0].half().cpu().numpy())
                # PCA of the 32-d map -> RGB, fitted on pixels the model covers (alpha > 0.5), beside the semantic PNG
                X = pred.reshape(-1, args.feat_dim); m = (alpha[0, ..., 0].reshape(-1) > 0.5)
                mu = X[m].mean(0); _, _, V = torch.pca_lowrank(X[m] - mu, q=3); P = ((X - mu) @ V[:, :3]); lo, hi = P[m].quantile(0.02, 0), P[m].quantile(0.98, 0)
                rgb = ((P - lo) / (hi - lo + 1e-6)).clamp(0, 1).reshape(H, W, 3); rgb[~m.reshape(H, W)] = 0
                sem = np.asarray(Image.open(index[name][0]).convert("RGB").resize((W, H), Image.NEAREST)) if name in index else np.zeros((H, W, 3), np.uint8)
                Image.fromarray(np.concatenate([(rgb.cpu().numpy() * 255).astype(np.uint8), sem], 1)).save(os.path.join(ed, name.replace(".png", "") + "_pca_vs_sem.png"))
        g = [s["geodesic_mean"] for s in scores if "geodesic_mean" in s]
        print(f"[feat-eval] {len(scores)} held-out views: geodesic-to-target mean {np.mean(g):.3f} (min {np.min(g):.3f} max {np.max(g):.3f}) over {sum(s.get('labelled', 0) for s in scores)} labelled px; outputs in {ed}", flush=True)
        json.dump(scores, open(os.path.join(ed, "scores.json"), "w"), indent=1); return
    covered = sum(1 for c in train_cams if c.image_name + ".png" in index or c.image_name in index)
    print(f"[feat] supervision: {len(index)} semantic PNGs over {nb} blocks, LUT {lut_path}; {covered}/{len(train_cams)} train views covered", flush=True)
    feats = torch.nn.Parameter(torch.zeros(N, args.feat_dim, device="cuda")); optim = torch.optim.Adam([feats], lr=args.lr, eps=1e-15)
    loader = DataLoader(train_cams, num_workers=4, prefetch_factor=1, persistent_workers=True, collate_fn=direct_collate, shuffle=True)
    it, t0, ema, cache, skipped = 0, time.time(), None, {}, 0
    while it < args.iters:
        it_at_epoch = it
        for batch in loader:
            for cam in batch:
                if it >= args.iters: break
                name = cam.image_name if cam.image_name in index else cam.image_name + ".png"
                for a in ("world_view_transform", "projection_matrix", "full_proj_transform", "camera_center"): setattr(cam, a, getattr(cam, a).cuda())
                viewmat, Ks, W, H = gsplat_cam(cam, K); tgt = target_for(name, index, lut, W, H, cache)
                if tgt is None: skipped += 1; continue
                limit = math.pow(2, torch.rand(1).item() * (math.log2(limmax) - math.log2(limmin)) + math.log2(limmin))
                means, scales, rots, opac, ri, pi, w = expand_view(gaussians, cam, limit, bufs)
                f = w * feats[ri] + (1 - w) * feats[pi]
                if gaussians.skybox_points:   # skybox nodes carry no feature and no gradient
                    sk = torch.arange(N - gaussians.skybox_points, N, device="cuda")
                    with torch.no_grad():
                        means = torch.cat([means, gaussians.get_xyz[sk]]); scales = torch.cat([scales, gaussians.get_scaling[sk]]); rots = torch.cat([rots, gaussians.get_rotation[sk]]); opac = torch.cat([opac, gaussians.get_opacity[sk].squeeze(-1)])
                    f = torch.cat([f, torch.zeros(sk.numel(), args.feat_dim, device="cuda")])
                out, _, _ = rasterization(means=means, quats=rots, scales=scales, opacities=opac, colors=f, viewmats=viewmat, Ks=Ks, width=W, height=H, sh_degree=None, packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", rasterize_mode="classic")
                pred = out[0]; tgt_c = tgt; labelled = tgt_c.abs().sum(-1) > 1e-6
                d = geodesic(pred[labelled], tgt_c[labelled]); loss = d.mean()
                optim.zero_grad(set_to_none=True); loss.backward(); optim.step(); it += 1
                ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
                if it % 100 == 0 or it == 1: print(f"[feat] it {it} loss {loss.item():.3f} ema {ema:.3f} labelled {int(labelled.sum())} nodes {ri.numel()} ({time.time()-t0:.0f}s)", flush=True)
        if it == it_at_epoch: sys.exit(f"[feat] ABORT: a full pass over {len(train_cams)} views produced no labelled pixels ({skipped} skipped) — supervision/LUT mismatch")
    F = feats.detach().cpu().numpy().astype(np.float32); F.tofile(os.path.join(dataset.model_path, args.out))
    json.dump({"n_nodes": N, "dim": args.feat_dim, "iters": it, "lr": args.lr, "final_ema_loss": ema, "skipped_views": skipped, "lut": lut_path, "sem_blocks": args.sem_blocks, "hierarchy": dataset.hierarchy}, open(os.path.join(dataset.model_path, args.out + ".json"), "w"), indent=1)
    print(f"[feat] wrote {os.path.join(dataset.model_path, args.out)} [{N},{args.feat_dim}] after {it} iters, ema loss {ema:.3f}, {skipped} unsupervised views skipped, {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
