#!/usr/bin/env python3
"""Lindendhof identity, step 2b (2026-10-12): trunk masks -> 3D trunk observations from the LiDAR instead of the splat's depth.
Step 2 (trunk_obs.py, v3 expected depth) smeared the trunks into ~2 m-wide bands across the rows: at thin trunks the splat's depth
mixes trunk and background. Here each lane's LiDAR init cloud (init_lidar.ply, LO world frame = the frame of the placed cameras in
transforms_ref_lo.json) is projected into the frame; the points falling inside a trunk mask whose camera depth is within BAND m of
the mask's near surface (10th percentile) are the trunk; their median is the observation.
system python3 (numpy):  trunk_obs_lidar.py <survey root> <trunk mask root> [lanes=1,2,3,4,5] [band=0.3] [max_range=12]
  -> <trunk mask root>/lane<n>/trunk_obs.json (same fields as trunk_obs.py; the splat-depth file is kept as trunk_obs_splat.json)"""
import json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
S, M = Path(sys.argv[1]), Path(sys.argv[2]); LANES = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4,5').split(',')]
BAND = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3; MAXR = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0; ZLO, ZHI = float(os.environ.get('ZLO', '0.2')), float(os.environ.get('ZHI', '0.7'))   # anchor band above the local ground (Lindendhof trunk base 0.2-0.7; Dec grow bags 0.05-0.45)
def read_ply(p):
    with open(p, 'rb') as f:
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith('element vertex'): n = int(line.split()[-1])
            if line == 'end_header': break
        a = np.frombuffer(f.read(n * 15), dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]), count=n)
    return np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
GL = np.diag([1.0, -1.0, -1.0, 1.0])
for n in LANES:
    t0 = time.time(); LD = S / f'experimental/lane{n}'; P = read_ply(LD / 'init_lidar.ply'); J = json.load(open(LD / 'transforms_ref_lo.json'))
    fx, fy, cx, cy, W, H = J['fl_x'], J['fl_y'], J['cx'], J['cy'], int(J['w']), int(J['h'])
    c2w = {f['file_path'].split('/')[-1]: np.array(f['transform_matrix'], np.float64) @ GL for f in J['frames']}
    od = M / f'lane{n}'
    if (od / 'trunk_obs.json').exists() and not (od / 'trunk_obs_splat.json').exists(): os.rename(od / 'trunk_obs.json', od / 'trunk_obs_splat.json')
    obs = []; nf = 0
    for jf in sorted(od.glob('f_*.json')):
        inst = json.load(open(jf)); fname = jf.name.replace('.json', '.png')
        if not inst or fname not in c2w: continue
        T = c2w[fname]; C = T[:3, 3]; near = np.abs(P[:, :2] - C[:2]).max(1) < MAXR; Q = P[near]
        pc = (Q - C) @ T[:3, :3]                      # world -> OpenCV camera (R^T (p - c))
        ok = pc[:, 2] > 0.3; Q, pc = Q[ok], pc[ok]
        u = np.round(fx * pc[:, 0] / pc[:, 2] + cx).astype(int); v = np.round(fy * pc[:, 1] / pc[:, 2] + cy).astype(int)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H); Q, pc, u, v = Q[inb], pc[inb], u[inb], v[inb]
        lab = np.array(Image.open(jf.with_suffix('.png'))); L = lab[v, u]
        # TRUNK BASE ONLY (2026-10-12): the trees branch into several upright leaders ~0.5 m apart, which SAM3 also calls "tree trunk";
        # the LiDAR trunk band (ground + 0.25..0.70 m) repeats every ~1.46 m = the real tree pitch. Keep only mask points in that band.
        gr = Q[np.abs(Q[:, :2] - C[:2]).max(1) < 2.5]; g = np.percentile(gr[:, 2], 5) if len(gr) > 50 else np.percentile(Q[:, 2], 3)
        base = (Q[:, 2] > g + ZLO) & (Q[:, 2] < g + ZHI)
        for it in inst:
            m = (L == it['idx']) & base
            if m.sum() < 8: continue
            z = pc[m, 2]; z0 = np.percentile(z, 10); sel = z <= z0 + BAND
            if sel.sum() < 5: continue
            q = Q[m][sel]
            obs.append({'frame': fname, 'idx': it['idx'], 'score': it['score'], 'area': it['area'], 'xyz': np.median(q, 0).round(3).tolist(),
                        'z_low': round(float(np.percentile(q[:, 2], 5)), 3), 'range_m': round(float(np.median(z[sel])), 2), 'n_px': int(sel.sum())})
        nf += 1
    json.dump(obs, open(od / 'trunk_obs.json', 'w'), indent=0)
    print(f'[trunk-obs-lidar] lane {n}: {len(obs)} observations from {nf} frames ({len(P):,} LiDAR points) in {time.time() - t0:.0f}s; '
          f'range median {np.median([o["range_m"] for o in obs]):.1f} m; LiDAR points per trunk median {np.median([o["n_px"] for o in obs]):.0f}', flush=True)
