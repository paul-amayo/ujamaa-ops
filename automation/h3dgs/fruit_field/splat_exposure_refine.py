"""Post-geometry exposure for splatfacto (Paul 2026-10-08: "the per frame exposure, can that be a post geometry optimisation?"): load a
trained splatfacto, FREEZE means / scales / quats / opacities, and re-optimise the colours (features_dc, features_rest) JOINTLY with a
per-training-view 3x4 affine (H3DGS's exposure model, identity-initialised) for ITERS steps of L1 on the training views. Then score:
training views (the same 60 as trainview_psnr.py) with their affines, and the held-out views raw. nerf_new pixi env:
  python splat_exposure_refine.py <config.yml> <workspace> [iters=3000] [lr_colour=0.0025] [lr_affine=0.001]"""
import json, math, re, sys, time
from pathlib import Path
import numpy as np, torch
from nerfstudio.utils.eval_utils import eval_setup
import os; ALT = os.environ.get("ALT") == "1"; cfg, WS = Path(sys.argv[1]), Path(sys.argv[2]); ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000; LR_C = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0025; LR_A = float(sys.argv[5]) if len(sys.argv) > 5 else 0.001
_, pipeline, _, _ = eval_setup(cfg, test_mode="test"); model = pipeline.model; dm = pipeline.datamanager; dev = model.device
tr = dm.train_dataset; ev = dm.eval_dataset; names = [Path(p).name for p in tr.image_filenames]; n_tr = len(names)
for k in ("means", "scales", "quats", "opacities"): model.gauss_params[k].requires_grad_(False)
colour = [model.gauss_params["features_dc"], model.gauss_params["features_rest"]]
A = torch.nn.Parameter(torch.eye(3, 4, device=dev)[None].repeat(n_tr, 1, 1))   # per-view exposure, identity init
opt = torch.optim.Adam([{"params": [colour[0]], "lr": LR_C}, {"params": [colour[1]], "lr": LR_C / 20}, {"params": [A], "lr": LR_A}])   # splatfacto's own ratio: features_rest at dc/20 (v1 used the same lr for both and drifted: 22.53 -> 21.64 raw); model.eval(); torch.set_grad_enabled(True)
def gt(ds, i): return ds[i]["image"].to(dev)[..., :3]
def render(ds, i):
    cam = ds.cameras[i:i + 1].to(dev); out = model.get_outputs(cam); return out["rgb"]
def psnr(a, b): return 10 * math.log10(1 / max(float(((a.clamp(0, 1) - b) ** 2).mean()), 1e-10))
with torch.no_grad():
    pick = sorted(range(n_tr), key=lambda i: int(re.sub(r"\D", "", names[i]) or 0))[::max(1, n_tr // 60)][:60]
    base_tr = [psnr(render(tr, i), gt(tr, i)) for i in pick]; base_ev = [psnr(render(ev, i), gt(ev, i)) for i in range(len(ev))]
print(f"[refine] {cfg.parent.parent.name}: {n_tr} training views, {len(ev)} held-out; before: training-view PSNR median {np.median(base_tr):.2f} (60 views, raw), held-out median {np.median(base_ev):.2f} mean {np.mean(base_ev):.2f}", flush=True)
rng = np.random.default_rng(0); t0 = time.time(); run = []
for it in range(1, ITERS + 1):
    i = int(rng.integers(n_tr)); rgb = render(tr, i); g = gt(tr, i)
    if ALT:   # v3: the affine has a closed form given the render (least squares, like the post-hoc fit) - solve it exactly on every visit and take only the colour step with it
        with torch.no_grad():
            X = torch.cat([rgb.reshape(-1, 3), torch.ones(rgb.shape[0] * rgb.shape[1], 1, device=dev)], 1); sol = torch.linalg.lstsq(X, g.reshape(-1, 3)).solution; A.data[i] = torch.cat([sol[:3], sol[3][:, None]], 1)   # y = x @ W + b: W = sol[:3] (3x3) goes in M[:, :3] as applied below; sol.T put W transposed there (v3 bug)
    M = A[i].detach() if ALT else A[i]; rgb_e = rgb @ M[:, :3] + M[:, 3]   # pixel-row x E + offset (H3DGS convention)
    loss = (rgb_e - g).abs().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run.append(float(loss))
    if it % 500 == 0: print(f"[refine] it {it}: L1 {np.mean(run[-500:]):.4f} ({(time.time() - t0) / it:.3f} s/it)", flush=True)
with torch.no_grad():
    tr_raw = [psnr(render(tr, i), gt(tr, i)) for i in pick]; tr_exp = [psnr(render(tr, i) @ A[i][:, :3] + A[i][:, 3], gt(tr, i)) for i in pick]; ev_raw = [psnr(render(ev, i), gt(ev, i)) for i in range(len(ev))]
    Am = A.mean(0); ev_mean = [psnr(render(ev, i) @ Am[:, :3] + Am[:, 3], gt(ev, i)) for i in range(len(ev))]
    off = (A[:, :, :3] - torch.eye(3, device=dev)).abs(); print(f"[refine] affines: diagonal {A[:, [0, 1, 2], [0, 1, 2]].min():.3f}-{A[:, [0, 1, 2], [0, 1, 2]].max():.3f}, off-diagonal max {off[:, [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]].max():.3f}, offset max |b| {A[:, :, 3].abs().max():.3f}")
print(f"[refine] after {ITERS} its ({(time.time() - t0) / 60:.1f} min): training views raw {np.median(tr_raw):.2f} -> WITH own affine {np.median(tr_exp):.2f} (mean {np.mean(tr_exp):.2f}); held-out raw median {np.median(ev_raw):.2f} mean {np.mean(ev_raw):.2f}, with the mean affine {np.median(ev_mean):.2f}   [H3DGS served: training 24.29 with exposure / 17.05 raw; held-out 15.76 raw]", flush=True)
torch.save({"A": A.detach().cpu(), "names": names, "features_dc": colour[0].detach().cpu(), "features_rest": colour[1].detach().cpu()}, WS / f"exposure_refine_{cfg.parent.parent.name}.pt")
