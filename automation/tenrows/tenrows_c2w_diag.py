from PIL import Image  # noqa
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from survey_paths import _read_increments, _slerp_partial, load_l2c
from rig_calib import l2c_flat
import aru_nerf_interface as a
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R/"prod/monos/monolithics"
incs = _read_increments(MD/"transform_lio.monolithic"); m0 = incs[1][1]
print(f"[diag] _read_increments -> {len(incs)} entries; matrix type {type(m0).__name__} shape {np.shape(m0)}; entry[1] last row {np.asarray(m0).reshape(-1)[12:16].round(4) if np.size(m0)==16 else 'n/a'}")
inc_ts = np.asarray([t for t, _ in incs], np.int64); absolute = np.empty((len(incs), 4, 4)); absolute[0] = np.eye(4)
for k in range(1, len(incs)): absolute[k] = absolute[k-1] @ incs[k][1]
l2c = load_l2c(MD); l2ci = np.linalg.inv(l2c)
def T_at(t):
    j = int(np.searchsorted(inc_ts, t, side="left")); ratio = float(inc_ts[j]-t)/float(inc_ts[j]-inc_ts[j-1]); partial = _slerp_partial(incs[j][1], ratio)
    return absolute[j-1] @ (incs[j][1] @ np.linalg.inv(partial))
kidx = json.loads((MD/"kf_index.json").read_text())
rig = json.loads((R/"prod/monos/rig.json").read_text())["intrinsics"]
proj = a.LaserProjector(image_mono=str(MD/"image_left.monolithic"), transform_mono=str(MD/"transform_lio.monolithic"), laser_mono=str(MD/"laser.monolithic"),
    fx=rig["fx"], fy=rig["fy"], cx=rig["cx"], cy=rig["cy"], img_w=1280, img_h=720, laser_match_tol_ns=80_000_000, l2c=l2c_flat(MD))
def ang(Ra, Rb): return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
for K in (50, 300, 600, 840, 1200, 1700):
    e = kidx[K]; d = proj.project_debug(int(e["image_idx"]))
    if "c2w" not in d: print(f"K{K}: no c2w ({d.get('error')})"); continue
    C = np.asarray(d["c2w"], float).reshape(4, 4); T = T_at(int(e["ts_ms"]))
    cands = {"L2C·T·L2C⁻¹ (blocks)": l2c @ T @ l2ci, "T": T, "T·L2C⁻¹": T @ l2ci, "T·L2C": T @ l2c, "L2C⁻¹·T·L2C": l2ci @ T @ l2c}
    best = min(cands, key=lambda k: np.linalg.norm(cands[k][:3, 3] - C[:3, 3]) + ang(cands[k][:3, :3], C[:3, :3]))
    blk = cands["L2C·T·L2C⁻¹ (blocks)"]
    print(f"K{K:5d} img {e['image_idx']:5d}: projector c2w pos {C[:3,3].round(2)} | blocks-formula pos {blk[:3,3].round(2)} | Δpos {np.linalg.norm(blk[:3,3]-C[:3,3]):6.2f} m Δrot {ang(blk[:3,:3], C[:3,:3]):5.2f}° | best match: {best} (Δpos {np.linalg.norm(cands[best][:3,3]-C[:3,3]):.3f} m, Δrot {ang(cands[best][:3,:3], C[:3,:3]):.3f}°)")
