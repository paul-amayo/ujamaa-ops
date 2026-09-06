"""Re-label the ZED camera body axes to the IMAGE frame: inc' = C·inc·C, C = diag(-1,-1,1,1)
(measured against COLMAP on block_013: stored camera x,y are negated vs the images)."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
import ensure_laser_frame_poses as elf
M = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics"); C = np.diag([-1.0, -1.0, 1.0, 1.0])
src = elf.read_transforms(M/"zed_transform.monolithic"); msgs = [m for m, _ in src]; mats = [C @ np.asarray(mm, float) @ C for _, mm in src]
out = M/"zed_transform_rollfix.monolithic"; elf.write_transforms(msgs, mats, out); (M/(out.name + ".index")).unlink(missing_ok=True)
T = np.eye(4); P = []
for m in mats: T = T @ m; P.append(T[:3, 3].copy())
P = np.array(P); print(f"[rollfix] wrote {out.name} ({out.stat().st_size} B), {len(mats)} increments; integrated path ptp {np.ptp(P, 0).round(1)} (x,y,z) — expect same extents as before, x and y mirrored")
