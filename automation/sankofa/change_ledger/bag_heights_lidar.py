"""Per-bag plant height from each survey's lidar: scans read with aru_py_logger.LaserLogger (the lab reader), placed in the survey's laser world by the odometry
stream (prod/monos/transform_lio: Dec = KISS-ICP, Apr = LIO), then for every ledger bag the 95th
percentile of point heights within R m of the bag against the local ground (5th percentile within 1.5 m).
usage: bag_heights_lidar.py <survey root> <bags npy (N,2) laser-world xy> <out npz>"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); import survey_paths as sp
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib")
import aru_py_logger  # the lab's reader: LaserLogger -> ((N,3) float64 points, ts_ms); pbLaser stores float32, packed

def laser_records(path):
    lg = aru_py_logger.LaserLogger(str(path), False)
    while True:
        try:
            r = lg.read_from_file()
        except Exception:
            return
        pts = next((x for x in r if hasattr(x, "shape") and x.ndim == 2), None)
        ts = next((x for x in r if isinstance(x, int)), None)
        if pts is None or pts.size == 0 or ts is None:
            return
        yield ts, np.ascontiguousarray(pts[:, :3], dtype=np.float64)

survey, bags_npy, out = sys.argv[1], sys.argv[2], sys.argv[3]
mono = sp.monos(survey); laser = mono / "laser.monolithic"
if not laser.exists(): laser = mono / "monolithics" / "laser.monolithic"
incs = sp._read_increments(mono / "transform_lio.monolithic"); inc_ts = np.asarray([t for t, _ in incs], dtype=np.int64)
absolute = np.empty((len(incs), 4, 4)); absolute[0] = np.eye(4)
for k in range(1, len(incs)): absolute[k] = absolute[k - 1] @ incs[k][1]
def pose_at(t):
    j = int(np.searchsorted(inc_ts, t));
    if j <= 0 or j >= len(incs): return None
    r = float(inc_ts[j] - t) / float(inc_ts[j] - inc_ts[j - 1]); part = sp._slerp_partial(incs[j][1], r)
    return absolute[j - 1] @ (incs[j][1] @ np.linalg.inv(part))
bags = np.load(bags_npy); tree = cKDTree(bags)
R_BAG, R_GND, Z_MAX = 0.35, 1.5, 2.0
heights = [[] for _ in bags]; grounds = [[] for _ in bags]
t0 = time.time(); n = 0; traj = []
for ts, P in laser_records(laser):
    T = pose_at(ts)
    if T is None: continue
    n += 1; traj.append(T[:3, 3])
    W = (T[:3, :3] @ P.T).T + T[:3, 3]
    near = np.linalg.norm(W[:, :2] - T[:2, 3], axis=1) < 12.0; W = W[near]
    if len(W) == 0: continue
    # points within R_GND of any bag: ground candidates; within R_BAG: plant candidates
    d, i = tree.query(W[:, :2], k=1, distance_upper_bound=R_GND)
    ok = np.isfinite(d)
    for bi in np.unique(i[ok]):
        m = ok & (i == bi)
        grounds[bi].append(W[m, 2].astype(np.float32))            # pooled over all scans; percentile taken once at the end
        mm = m & (d <= R_BAG)
        if mm.any(): heights[bi].append(W[mm, 2].astype(np.float32))
    if n % 500 == 0: print(f"  {n} scans {time.time() - t0:.0f} s", flush=True)
traj = np.array(traj); dtraj, _ = cKDTree(traj[:, :2]).query(bags)
print(f"{Path(survey).name}: {n} scans in {time.time() - t0:.0f} s; bags within 2 m of the laser-world trajectory: {int((dtraj <= 2).sum())} of {len(bags)} (median {np.median(dtraj):.2f} m)")
H = np.full(len(bags), np.nan); G = np.full(len(bags), np.nan); N = np.zeros(len(bags), int)
for bi in range(len(bags)):
    if heights[bi] and grounds[bi]:
        z = np.concatenate(heights[bi]); g = float(np.percentile(np.concatenate(grounds[bi]), 5)); z = z[(z - g) < Z_MAX]
        if len(z) >= 30: H[bi] = np.percentile(z - g, 95); G[bi] = g; N[bi] = len(z)
print(f"  bags with a height: {int(np.isfinite(H).sum())}; height median {np.nanmedian(H):.2f} m, IQR {np.nanpercentile(H, 25):.2f}-{np.nanpercentile(H, 75):.2f}")
sens = np.array([traj[np.argmin(np.linalg.norm(traj[:, :2] - b, axis=1)), 2] for b in bags]) - G
print(f"  sensor above local ground per bag: median {np.nanmedian(sens):.2f} m")
np.savez(out, height=H, ground=G, n_points=N, bags=bags, traj=traj, sensor_above_ground=sens)
