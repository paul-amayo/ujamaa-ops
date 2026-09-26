"""Keyframe cut on the LiDAR-odometry poses, process B (writer side; aru_py_logger only — reader and writer bindings
clash on protobuf descriptors, hence two processes). Writes
  monolithics/image_left_kf20cm_lo.monolithic  (verbatim keyframe pixels, original image stamps; K = entry index)
  monolithics/transform_lo.monolithic          (laser-frame LO pose increments at EVERY image stamp, the pilot
                                                convention: first record identity, then inv(prev) @ curr with
                                                source = prev stamp, dest = stamp — the same shape as transform_lio)
and checks the stream by re-reading it with survey_paths._read_increments.
  env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=<cp310 aru_py_logger> <nerf_new python3.10> tenrows_lo_kfcut_B.py"""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts"); from survey_paths import _read_increments
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R / "prod/monos/monolithics"; PNG = R / "prod/scratch_sam3_lo"
index = json.loads((MD / "kf_index_lo.json").read_text())
out = MD / "image_left_kf20cm_lo.monolithic"
for stale in (out, Path(str(out) + ".index")): stale.unlink(missing_ok=True)
lg = aru_py_logger.MonoImageLogger(str(out), True)
for e in index:
    img = np.ascontiguousarray(np.asarray(Image.open(PNG / f"kf_{e['K']:06d}.png").convert("RGB"))); lg.write_to_file(img, int(e["ts_ms"]))
print(f"[kfcut-lo-B] wrote {len(index)} keyframes -> {out.name} ({out.stat().st_size / 1e6:.0f} MB)", flush=True)
z = np.load(MD / "lo_image_poses_laser.npz"); ts = z["ts_ms"]; T = z["T"]
tout = MD / "transform_lo.monolithic"
for stale in (tout, Path(str(tout) + ".index")): stale.unlink(missing_ok=True)
tl = aru_py_logger.TransformLogger(str(tout), True); prev = None
for i in range(len(ts)):
    if prev is None: tl.write_to_file(np.eye(4), int(ts[i]), int(ts[i]))
    else: tl.write_to_file(np.ascontiguousarray(np.linalg.inv(T[prev]) @ T[i]), int(ts[prev]), int(ts[i]))
    prev = i
del tl
incs = _read_increments(tout); Tc = np.eye(4); P = []
for _, m in incs:
    m = np.asarray(m, float); Tc = Tc @ (m.reshape(4, 4, order="F") if m.ndim == 1 else m); P.append(Tc[:3, 3].copy())
P = np.array(P); err = np.linalg.norm(P - T[:len(P), :3, 3], axis=1)
print(f"[kfcut-lo-B] transform_lo.monolithic: {len(incs)} increments; re-read prefix product vs the LO poses: max |dp| {err.max():.4f} m (should be ~0)", flush=True)
