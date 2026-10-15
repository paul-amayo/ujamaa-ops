#!/usr/bin/env python3
"""votes_from_mass.py <tree> [tag] [--min-w 0.005] — disambiguation of touching apples (2026-10-15 20:4x, Paul: "can we disambiguate them"):
rebuild the vote table of the mask-linked registry from the RENDERED MASS (maskgraph_<tag>.npz: alpha mass of every gaussian on every SAM3
mask, occlusion included) instead of the binary depth test: a gaussian votes for a mask only where it contributes >= min_w of that mask's
mass, so an apple hidden behind another contributes ~nothing to the front apple's mask. Fruit-gaussian set, cameras, lifted masks and the
record rules stay the same. Writes masklink_votes_<tag>_mass.npz (+ cameras) for fruit_registry_masklink.py --tag <tag>_mass."""
import argparse, shutil
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('tree'); ap.add_argument('tag', nargs='?', default='plain30k'); ap.add_argument('--min-w', type=float, default=0.005); ap.add_argument('--all', action='store_true', help='ALL gaussians that render a mask vote (no census fruit share — 10-15 21:5x, Paul: fruit/leaf is relevancy, not registry); needs means_<tag>.npy'); a = ap.parse_args()
R = Path(f'/home/paperspace/data/public/fruitnerf/high/{a.tree}/fruit_registry'); v = np.load(R / f'masklink_votes_{a.tag}.npz'); d = np.load(R / f'maskgraph_{a.tag}.npz'); cams = np.load(R / f'cameras_{a.tag}.npz')
assert list(v['names']) == list(d['names']) == list(cams['names']), 'view order mismatch'
if a.all:
    P_all = np.load(R / f'means_{a.tag}.npy'); fid = np.arange(int(d['N'])); P = P_all; Wf = np.zeros(len(fid), np.float32); Wn = Wf
else:
    fid = v['fid']; P = v['P']; Wf = v['Wf']; Wn = v['Wn']
pos = np.full(int(d['N']), -1, np.int64); pos[fid] = np.arange(len(fid))   # gaussian id -> row
mrow, gid, w, mview, mlab, mmass = d['mrow'], d['gid'], d['w'], d['mview'], d['mlab'], d['mmass']
keep = (w >= a.min_w * mmass[mrow]) & (pos[gid] >= 0); vi = mview[mrow[keep]].astype(np.int32); lab = mlab[mrow[keep]].astype(np.int16); row = pos[gid[keep]].astype(np.int32)
# pixel positions of the votes (for the figures): project the fruit gaussians with the cached cameras
VM, KK = cams['viewmat'], cams['K']; pc = np.einsum('nij,nj->ni', VM[vi, :3, :3], P[row]) + VM[vi, :3, 3]; z = np.maximum(pc[:, 2], 1e-6); uv = np.stack([KK[vi, 0, 0] * pc[:, 0] / z + KK[vi, 0, 2], KK[vi, 1, 1] * pc[:, 1] / z + KK[vi, 1, 2]], 1).round().astype(np.int16)
nvis = np.bincount(vi, minlength=len(v['names']))
suffix = '_massall' if a.all else '_mass'
np.savez_compressed(R / f'masklink_votes_{a.tag}{suffix}.npz', P=P, fid=fid, Wf=Wf, Wn=Wn, vi=vi, lab=lab, row=row, uv=uv, names=v['names'], nvis=nvis, masks=v['masks'], C=v['C'], Fd=v['Fd'], UPc=v['UPc'], N=v['N'], M=len(fid), share=(0.0 if a.all else v['share']), mass=v['mass'], depth_tol=-1.0, depth_tol_r=0.0)
shutil.copy(R / f'cameras_{a.tag}.npz', R / f'cameras_{a.tag}{suffix}.npz')
print(f'[votes-from-mass] {a.tree}: {len(row)} votes (was {len(v["row"])} with the depth test on fruit gaussians) from {len(np.unique(row))} gaussians ({"all" if a.all else "fruit"}), min_w {a.min_w} of the mask mass', flush=True)
