#!/usr/bin/env python3
"""smac_dump_splits.py (30 Oct 2026) — write a survey's laser scans and LIO poses in the per-keyframe split format the SMaC app
(aru_segmap, test/smac-evaluation) reads: <out>/laser/laser_<i>.ply (binary, sensor frame, points within --rmax m) and
<out>/transforms/transform_<i>.txt (4x4, the absolute lidar pose at the scan's stamp, composed from transform_lio.monolithic
with the destination-stamp convention of survey_paths.poses_at_keyframes and slerp inside the bracketing increment), i = 1..N.
nerf_new python3.10:  smac_dump_splits.py <survey root> <out dir> [--every 1] [--rmax 40] [--max 0]"""
import sys, argparse, struct, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib'); import aru_py_logger
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts'); import survey_paths as sp
ap = argparse.ArgumentParser(); ap.add_argument('survey'); ap.add_argument('out'); ap.add_argument('--every', type=int, default=1); ap.add_argument('--rmax', type=float, default=40.0); ap.add_argument('--max', type=int, default=0); a = ap.parse_args()
M = sp.monos(a.survey); O = Path(a.out); (O / 'laser').mkdir(parents=True, exist_ok=True); (O / 'transforms').mkdir(exist_ok=True)
incs3 = sp._read_increments(M / 'transform_lio.monolithic', with_dest=True)
inc_ts = np.asarray([dst if dst is not None else src for src, dst, _ in incs3], np.int64); incs = [m for _, _, m in incs3]
absolute = np.empty((len(incs), 4, 4)); absolute[0] = np.eye(4)
for k in range(1, len(incs)): absolute[k] = absolute[k - 1] @ incs[k]
def pose_at(t):
    j = int(np.searchsorted(inc_ts, t, side='left'))
    if j <= 0 or j >= len(incs): return None
    ratio = float(inc_ts[j] - t) / float(inc_ts[j] - inc_ts[j - 1]); partial = sp._slerp_partial(incs[j], ratio)
    return absolute[j - 1] @ (incs[j] @ np.linalg.inv(partial))
def write_ply(path, P):
    with open(path, 'wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode()); f.write(P.astype('<f4').tobytes())
lg = aru_py_logger.LaserLogger(str(M / 'laser.monolithic'), False); n = i = 0; skipped = 0; stamps = []
while True:
    try: r = lg.read_from_file()
    except Exception: break
    if r is None or r[0] is None or getattr(r[0], 'size', 0) == 0: break
    n += 1
    if (n - 1) % a.every: continue
    t = int(r[1]); T = pose_at(t)
    if T is None: skipped += 1; continue
    P = np.asarray(r[0], np.float32)[:, :3]; P = P[np.isfinite(P).all(1)]; P = P[(np.abs(P).sum(1) > 0) & (np.linalg.norm(P, axis=1) < a.rmax)]
    i += 1; write_ply(O / 'laser' / f'laser_{i}.ply', P); np.savetxt(O / 'transforms' / f'transform_{i}.txt', T, fmt='%.9f'); stamps.append((i, t))
    if i % 500 == 0: print(f'[dump] {i} scans written ({n} read)', flush=True)
    if a.max and i >= a.max: break
np.savetxt(O / 'stamps.txt', np.array(stamps, np.int64), fmt='%d')
print(f'[dump] {i} scans written to {O} ({n} read, {skipped} outside the pose stream); poses span {np.linalg.norm(absolute[-1][:3, 3] - absolute[0][:3, 3]):.1f} m end to end', flush=True)
