"""Keyframe cut, process A (reader side; no aru_py_logger in this process).
Rule = the C++ dump_keyframe_image_monolithic: keep image 0; keep image i when the
camera has translated > MIN_DIST from the LAST KEPT image (pose interpolated at
the image timestamp). Poses: drift-corrected CAMERA-frame stream (zed_transform_inscorr),
prefix-integrated, linearly interpolated at image timestamps.
Writes: prod/scratch_sam3/kf_%06d.png (RGB, verbatim from image_left.monolithic) and
prod/monos/monolithics/kf_index.json ({K: image_idx, ts_ms, position})."""
from PIL import Image  # noqa (PIL before the binding)
import sys, json, numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
from survey_paths import _read_increments
import aru_nerf_interface as a
from pathlib import Path
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R/"prod/monos/monolithics"; OUT = R/"prod/scratch_sam3"; OUT.mkdir(parents=True, exist_ok=True)
MIN_DIST = 0.20
incs = _read_increments(MD/"zed_transform_inscorr.monolithic")
ts = np.array([s for s, _ in incs], float); dest = np.r_[ts[1:], ts[-1] + (ts[-1] - ts[-2])]
T = np.eye(4); pos = []
for _, m in incs: T = T @ m.reshape(4, 4, order="F"); pos.append(T[:3, 3].copy())
pos = np.array(pos)
rdr = a.ImageMonoReader(image_mono=str(MD/"image_left.monolithic")); n = rdr.num_images()
its = np.array([rdr.timestamp(i) for i in range(n)], float)
P = np.stack([np.interp(its, dest, pos[:, k]) for k in range(3)], 1)
kept = [0]; last = P[0]
for i in range(1, n):
    if np.linalg.norm(P[i] - last) > MIN_DIST: kept.append(i); last = P[i]
print(f"[kfcut] {n} images -> {len(kept)} keyframes (MIN_DIST {MIN_DIST} m)", flush=True)
index = [{"K": K, "image_idx": int(i), "ts_ms": int(its[i]), "pos": P[i].round(4).tolist()} for K, i in enumerate(kept)]
(MD/"kf_index.json").write_text(json.dumps(index)); print("[kfcut] kf_index.json written", flush=True)
for K, i in enumerate(kept):
    f = OUT/f"kf_{K:06d}.png"
    if f.exists() and f.stat().st_size > 0: continue
    img = np.asarray(rdr.read_index(i))[:, :, :3]
    Image.fromarray(img).save(f, compress_level=1)
    if K % 300 == 0: print(f"  kf {K}/{len(kept)}", flush=True)
print(f"[kfcut] wrote {len(kept)} PNGs to {OUT} + kf_index.json", flush=True)
