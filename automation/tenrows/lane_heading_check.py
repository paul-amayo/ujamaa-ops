"""Check a lane's placed cameras against its LiDAR odometry (2026-10-06): heading of the warped optical axis minus the LO camera's,
as deviation from the median offset (the extrinsic), plus orientation and position residuals. December lane 2 (the recipe of
record, 29.7 dB) measures p50 0.24 / p90 0.98 / p99 2.93 / max 3.43 deg, 21 frames > 2 deg.
  h3dgs env: lane_heading_check.py <lane dir> [warped json, default transforms_ref_lo.json]"""
import json, re, sys
from pathlib import Path
import numpy as np
LD = Path(sys.argv[1]); WJ = sys.argv[2] if len(sys.argv) > 2 else "transforms_ref_lo.json"; GL = np.diag([1.0, -1, -1, 1])
def load(p):
    J = json.load(open(p)); return {f["file_path"].split("/")[-1]: np.array(f["transform_matrix"], float) @ GL for f in J["frames"]}
lo, wp = load(LD / "transforms_lo.json"), load(LD / WJ); names = sorted(wp, key=lambda n: int(re.sub(r"\D", "", n)))
yaw = np.array([np.degrees(np.arctan2(wp[n][1, 2], wp[n][0, 2]) - np.arctan2(lo[n][1, 2], lo[n][0, 2])) for n in names]); yaw = (yaw + 180) % 360 - 180; dev = np.abs(yaw - np.median(yaw))
ang = np.array([np.degrees(np.arccos(np.clip((np.trace(wp[n][:3, :3].T @ lo[n][:3, :3]) - 1) / 2, -1, 1))) for n in names]); pos = np.array([np.linalg.norm(wp[n][:3, 3] - lo[n][:3, 3]) for n in names])
bad = [names[i] for i in np.where(dev > 2)[0]]
print(f"[heading-check] {LD.name}: {len(names)} frames {names[0]}..{names[-1]}; heading dev from median p50 {np.median(dev):.2f} p90 {np.percentile(dev, 90):.2f} p99 {np.percentile(dev, 99):.2f} max {dev.max():.2f} deg "
      f"(> 2 deg: {len(bad)}{', ' + bad[0] + '..' + bad[-1] if bad else ''}); orientation vs LO median {np.median(ang):.2f} p90 {np.percentile(ang, 90):.2f} max {ang.max():.2f} deg; position p50 {np.median(pos) * 100:.1f} p90 {np.percentile(pos, 90) * 100:.1f} max {pos.max() * 100:.1f} cm", flush=True)
