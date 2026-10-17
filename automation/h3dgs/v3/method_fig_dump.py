#!/usr/bin/env python3
"""method_fig_dump.py (2026-10-17) — the Gaussians of one v3 cell in the survey (LIO) frame for the HiGH paper's method figure: means, colours
(SH DC), opacities and the census label of each (features_B.bin.labels.npy), a fixed random subset. Chain: run frame -> (dataparser_transforms
transform, scale)^-1 -> H3DGS z-up frame -> (export_meta world_rotation_lio_to_h3dgs)^-1 -> LIO, the inverse of v3_render_service's camera chain.
With --hier the centroid of each tree's Gaussians is compared with its scene-graph marker (the check that the chain is right).
nerf_new pixi env:  method_fig_dump.py --config <run config.yml> --meta <export_meta.json> --identity <identity dir> --out <npz> [--n 200000] [--hier json]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--meta', required=True); ap.add_argument('--identity', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=200000); ap.add_argument('--hier', default=''); a = ap.parse_args(); CFG = Path(a.config)
dp = json.load(open(CFG.parent / 'dataparser_transforms.json')); T = np.eye(4); T[:3] = np.array(dp['transform'], dtype=np.float64); S = float(dp['scale'])
meta = json.load(open(a.meta)); R_W = np.array(meta.get('world_rotation_to_zup') or meta['world_rotation_lio_to_h3dgs'], dtype=np.float64)
ck = sorted((CFG.parent / 'nerfstudio_models').glob('step-*.ckpt'))[-1]; sd = torch.load(ck, map_location='cpu', weights_only=False)['pipeline']
g = {k.split('gauss_params.')[1]: v for k, v in sd.items() if 'gauss_params.' in k}; print('[dump]', ck.name, {k: tuple(v.shape) for k, v in g.items()})
means = g['means'].double().numpy(); N = means.shape[0]
p_h = (np.linalg.inv(T) @ np.c_[means / S, np.ones(N)].T).T[:, :3]; p_lio = (np.linalg.inv(R_W) @ np.c_[p_h, np.ones(N)].T).T[:, :3]
col = np.clip(0.28209479177 * g['features_dc'].float().numpy() + 0.5, 0, 1); opa = torch.sigmoid(g['opacities'].float()).numpy().reshape(-1)
lab = np.load(Path(a.identity) / 'features_B.bin.labels.npy'); assert lab.shape[0] == N, (lab.shape, N)
rng = np.random.default_rng(0); keep = np.where(opa > 0.1)[0]; sel = rng.choice(keep, min(a.n, len(keep)), replace=False)
np.savez_compressed(a.out, xyz=p_lio[sel].astype(np.float32), rgb=col[sel].astype(np.float32), opacity=opa[sel].astype(np.float32), label=lab[sel], n_total=N, n_opaque=len(keep))
print(f'[dump] {len(sel)} of {N} gaussians ({len(keep)} with opacity > 0.1); LIO bbox min {p_lio[sel].min(0).round(1)} max {p_lio[sel].max(0).round(1)}')
if a.hier:
    H = json.load(open(a.hier)); objs = {int(o['id']): np.array(o['xyz'], dtype=np.float64) for o in H['objects']}; d = []
    for t in sorted(set(int(v) for v in np.unique(lab[sel]))):
        if 0 <= t < 10000 and t in objs:
            m = lab[sel] == t; c = p_lio[sel][m].mean(0); d.append(np.linalg.norm((c - objs[t])[[0, 2]])); print(f'[check] tree {t}: {int(m.sum())} gaussians, centroid - marker {np.linalg.norm(c - objs[t]):.2f} m (ground plane {d[-1]:.2f} m)')
    print(f'[check] ground-plane centroid-marker distance over {len(d)} trees: median {np.median(d):.2f} m, max {np.max(d):.2f} m')
