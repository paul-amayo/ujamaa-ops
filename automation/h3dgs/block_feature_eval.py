"""Reference for the H3DGS feature stage: the per-block HiGH checkpoint's own feature render on the same keyframes,
scored with the same geodesic-to-target metric against the same high_targets.npz tables. Run in the nerf_new pixi env
with HIGH_EMBEDDER_CKPT set.
  python block_feature_eval.py <blocks_ns cfg dir> <kf name> [<kf name> ...]"""
import os, sys, glob, json, math, numpy as np, torch
from pathlib import Path
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
import high.encoders.lorentz as L
CFG = Path(sys.argv[1]); names = sys.argv[2:]
def owner(kf):
    for b in sorted(CFG.glob("block_[0-9][0-9][0-9]")):
        if (b / "semantic_v2_B" / kf).exists() and glob.glob(str(b / "splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml")): return b
def target(b, kf, W, H):
    z = np.load(b / "semantic_v2_B" / "high_targets.npz"); cols = z["colors"].astype(np.int64); keys = (cols[:, 0] << 16) | (cols[:, 1] << 8) | cols[:, 2]
    tab = dict(zip(keys.tolist(), z["targets"].astype(np.float32)))
    im = np.asarray(Image.open(b / "semantic_v2_B" / kf).convert("RGB").resize((W, H), Image.NEAREST))
    key = (im[..., 0].astype(np.int64) << 16) | (im[..., 1].astype(np.int64) << 8) | im[..., 2].astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True); table = np.stack([tab.get(int(k), np.zeros(32, np.float32)) for k in uniq])
    return torch.from_numpy(table)[torch.from_numpy(inv.reshape(H, W))]
def geodesic(pred, tgt, curv=1.0):
    x = L.exp_map0(torch.clamp(pred, -12, 12), curv=curv); y = L.exp_map0(torch.clamp(tgt, -12, 12), curv=curv)
    return torch.acosh(torch.clamp(-L.rowwise_inner(x, y, curv), min=1 + 1e-7)).squeeze(-1)
loaded = {}
for kf in names:
    b = owner(kf)
    if b is None: print(f"[block-eval] {kf}: no owner block with a glref stage-2 checkpoint"); continue
    if b not in loaded:
        cfg = sorted(glob.glob(str(b / "splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml")))[-1]
        config, pipeline, ckpt, step = eval_setup(Path(cfg), test_mode="test"); pipeline.eval(); loaded[b] = pipeline
        pipeline.model.image_encoder.set_positives(["tree"])          # forces the feature pass in eval
    pipeline = loaded[b]; dm = pipeline.datamanager
    cam, gt = None, None
    for ds in (dm.train_dataset, dm.eval_dataset):
        for i in range(len(ds)):
            if Path(ds.image_filenames[i]).name == kf: cam = ds.cameras[i:i + 1]; break
        if cam is not None: break
    if cam is None: print(f"[block-eval] {kf}: not in {b.name}'s datasets"); continue
    with torch.no_grad(): out = pipeline.model.get_outputs_for_camera(cam.to(pipeline.device))
    feat = out["high_features"].float().cpu(); H, W = feat.shape[:2]; tgt = target(b, kf, W, H); lab = tgt.abs().sum(-1) > 1e-6
    d = geodesic(feat[lab], tgt[lab]); print(f"[block-eval] {kf} via {b.name} stage-2 glref: geodesic-to-target mean {float(d.mean()):.3f} over {int(lab.sum())} labelled px; feature norm {float(feat[lab].norm(dim=-1).mean()):.2f} vs target {float(tgt[lab].norm(dim=-1).mean()):.2f}", flush=True)
