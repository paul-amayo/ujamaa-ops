#!/usr/bin/env python3
"""azalai_online_prep.py (11 Nov 2026) — RSS Azalai pilot AZ-2 "online along-row station and the counting baseline", step 1: what
the robot has seen so far, CAUSALLY, for one revisit pass and one pose source. No station, no match and no outcome is computed here.
Implements "What the robot has (causal)" of Azalai_RSS/numbers/pilots_20261111/ONLINE_design.md (5d76dfe):
  * lane passes of the revisit (the RR design's rule, copied from azalai_rr_prep.py);
  * pose source W_k (camera-to-world in the source's own world): 'graph' = the optimised graph pose c2w[k]; 'lo' = the survey's raw
    laser odometry (laser_dump/lo_poses.npz: positions linear, rotations slerp at the keyframe stamp, no clock offset) times the
    inverse of rig.json laser_to_camera_left; 'lo_viopatch_icp' = the same on lo_poses_viopatch_icp.npz (secondary);
  * every census return of the lane's keyframes brought back to the camera frame of its own keyframe (the inverse of the pose that
    placed it, exact) and placed in the source's world by W of that keyframe;
  * at every keyframe k of a lane pass the PRODUCTION instancing of bag_census_lidar.py, unchanged, on the returns of keyframes
    k0..k only (masks with score >= floor 0.2; ground-plane density on a 5 cm grid; Gaussian sigma 2 cells; local maxima over a
    0.6 m footprint above the 20th percentile of occupied cells; returns assigned within 0.35 m; bag = median of its returns).
Writes <out>/causal_<pass>_<source>.npz: per lane and keyframe the bags seen so far (3-D, source world) and the poses W.
nerf_new python3.10:  azalai_online_prep.py --rev apr1130 --source lo --out /mnt/shared/azalai_online/causal"""
import argparse, json, time, numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
ap = argparse.ArgumentParser(); ap.add_argument('--rev', required=True); ap.add_argument('--source', required=True, choices=['graph', 'lo', 'lo_viopatch_icp']); ap.add_argument('--out', required=True); a = ap.parse_args()
KL = Path('/home/paperspace/data/klapmuts'); X = KL / 'dec_2025_ten_rows/experimental'; SK = X / 'sankofa/jirs_20261023/lidar_fixedreader'; STPG = X / 'azalai/stpg_dec_apr_20261021/dec_canonical_fixedreader/stpg.json'
AXIS_Z, JUMP, MINLEN = 0.8, 0.6, 30                      # lane pass, RR design
V, FOOT, ASSIGN_R, PCT = 0.05, 0.6, 0.35, 20             # production instancing (bag_census_lidar.py, ledger v5)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); EXP = json.load(open(STPG))['experiences'][a.rev]; C = np.array(EXP['c2w'], float); kts = np.array(EXP['ts_ms'], np.int64); R = Path(EXP['survey'])
def passes(C):   # copied from azalai_rr_prep.py
    ok = np.isfinite(C).all((1, 2)); z = C[:, 2, 2]; x = C[:, 0, 3]; runs = []; cur = []
    for k in range(len(C)):
        s = 0 if (not ok[k] or abs(z[k]) <= AXIS_Z) else int(np.sign(z[k]))
        if s != 0 and cur and s == cur[0][1] and k == cur[-1][0] + 1 and abs(x[k] - x[cur[-1][0]]) <= JUMP: cur.append((k, s))
        else:
            if len(cur) >= MINLEN: runs.append(cur)
            cur = [(k, s)] if s != 0 else []
    if len(cur) >= MINLEN: runs.append(cur)
    return [(r[0][0], r[-1][0], r[0][1]) for r in runs]
L = passes(C); print(f'[online-prep] {a.rev} / {a.source}: {len(L)} lane passes, senses {[s for _, _, s in L]}', flush=True)
# ---------------------------------------------------------------- pose source
W = np.full_like(C, np.nan); lane_ok = [True] * len(L); src_file = 'stpg c2w'
if a.source == 'graph': W = C.copy(); GA = (0, 2)
else:
    fn = 'lo_poses.npz' if a.source == 'lo' else 'lo_poses_viopatch_icp.npz'
    f = (X / 'laser_dump' / fn) if a.rev == 'dec' else (R / 'experimental/laser_dump' / fn)
    if not f.exists() and a.rev == 'apr1054' and a.source == 'lo': f = R / 'experimental/lane2_apr_full/laser_dump/lo_poses.npz'   # the only laser odometry of April 10:54 (lane 2)
    src_file = str(f); lo = np.load(f); lts = lo['ts_ms'].astype(np.int64); LT = lo['T'].astype(float); sl = Slerp(lts.astype(float), Rotation.from_matrix(LT[:, :3, :3]))
    C2L = np.linalg.inv(np.array(json.load(open(R / 'prod/monos/rig.json'))['laser_to_camera_left'], float))
    def lo_at(t):   # as tenrows_lo_poses.py / bag_census_lidar.py --poses lo
        j = int(np.clip(np.searchsorted(lts, t), 1, len(lts) - 1)); w = (t - lts[j - 1]) / max(lts[j] - lts[j - 1], 1e-9)
        M = np.eye(4); M[:3, :3] = sl([float(t)]).as_matrix()[0]; M[:3, 3] = (1 - w) * LT[j - 1, :3, 3] + w * LT[j, :3, 3]; return M
    for i, (k0, k1, s) in enumerate(L):
        if kts[k0] < lts[0] or kts[k1] > lts[-1]: lane_ok[i] = False; continue
        for k in range(k0, k1 + 1): W[k] = lo_at(int(kts[k])) @ C2L
    GA = (0, 1)
    print(f'[online-prep] laser odometry {f}: {len(lts)} poses; lanes covered {[i + 1 for i, o in enumerate(lane_ok) if o]}; gaps over 300 ms at {[(int(i), int(d)) for i, d in enumerate(np.diff(lts)) if d > 300]}', flush=True)
# ---------------------------------------------------------------- returns in the camera frame, then in the source's world
z = np.load(SK / f'census_{a.rev}.npz'); xyz = z['xyz'].astype(np.float64); rk = z['k'].astype(np.int64); keep = z['score'][z['det']] >= float(z['floor'])
assert (np.diff(rk) >= 0).all(), 'census returns are not ordered by keyframe'
assert str(z['frame']).startswith(f'STPG experience {a.rev} c2w world'), str(z['frame'])
Rk = C[rk][:, :3, :3]; tk = C[rk][:, :3, 3]; pc = np.einsum('nj,nji->ni', xyz - tk, Rk)            # pc = R^T (pw - t)
chk = np.abs(np.einsum('nij,nj->ni', Rk[:1000], pc[:1000]) + tk[:1000] - xyz[:1000]).max(); assert chk < 1e-4, chk
t0 = time.time(); S_lane, S_k, S_start, S_n, B_xyz, B_n = [], [], [], [], [], []; nb = 0
for i, (k0, k1, s) in enumerate(L):
    if not lane_ok[i]: print(f'[online-prep] lane {i + 1}: no laser odometry, skipped', flush=True); continue
    m = (rk >= k0) & (rk <= k1) & keep; kk = rk[m]; Wk = W[kk]; pw = np.einsum('nij,nj->ni', Wk[:, :3, :3], pc[m]) + Wk[:, :3, 3]; P = pw[:, list(GA)]
    assert np.isfinite(pw).all()
    ends = np.searchsorted(kk, np.arange(k0, k1 + 1), side='right')
    for k, e in zip(range(k0, k1 + 1), ends):
        S_lane.append(i); S_k.append(k); S_start.append(nb)
        if e == 0: S_n.append(0); continue
        Pk = P[:e]; Xk = pw[:e]
        i0 = np.floor(Pk.min(0) / V).astype(int) - 2; I = np.floor(Pk[:, 0] / V).astype(int) - i0[0]; J = np.floor(Pk[:, 1] / V).astype(int) - i0[1]
        g = np.zeros((J.max() + 3, I.max() + 3), np.float32); np.add.at(g, (J, I), 1); sm = ndimage.gaussian_filter(g, sigma=2); r = int(FOOT / V)
        pk = (sm == ndimage.maximum_filter(sm, size=2 * r + 1)) & (sm > np.percentile(sm[sm > 0], PCT)); jj, ii = np.nonzero(pk); bags = np.c_[(ii + i0[0]) * V, (jj + i0[1]) * V]
        if not len(bags): S_n.append(0); continue
        dd, bi = cKDTree(bags).query(Pk); asg = np.where(dd <= ASSIGN_R, bi, -1); o = np.argsort(asg, kind='stable'); sa = asg[o]; cut = np.searchsorted(sa, np.arange(len(bags) + 1)); n_here = 0
        for b in range(len(bags)):
            if cut[b + 1] > cut[b]: B_xyz.append(np.median(Xk[o[cut[b]:cut[b + 1]]], 0)); B_n.append(cut[b + 1] - cut[b]); n_here += 1
        S_n.append(n_here); nb += n_here
    print(f'[online-prep] lane {i + 1} ({s:+d}): keyframes {k0}..{k1}, {int(m.sum()):,} returns, {S_n[-1]} bags seen by the last keyframe; {time.time() - t0:.0f} s', flush=True)
np.savez_compressed(OUT / f'causal_{a.rev}_{a.source}.npz', lanes=np.array(L), lane_ok=np.array(lane_ok), W=W, step_lane=np.array(S_lane), step_k=np.array(S_k), step_start=np.array(S_start), step_n=np.array(S_n),
                    bags_xyz=np.array(B_xyz, np.float64).reshape(-1, 3), bags_n=np.array(B_n), ground_axes=np.array(GA), source=a.source, source_file=src_file)
print(f'[online-prep] wrote causal_{a.rev}_{a.source}.npz: {len(S_k)} steps, {nb:,} bag records; {time.time() - t0:.0f} s', flush=True)
