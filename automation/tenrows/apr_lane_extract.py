"""April Klapmuts lane 2 (Paul 2026-10-05: "run the lane recipe on april lane 2 after the demo"): the full 15 Hz left stream of
the stretch where April drove December's lane 2. The window comes from April's keyframes: the December->April plant-ledger
registration (klapmuts_ledger_v5: p_apr = R(1.24 deg) p_dec + t) puts December's lane-2 path on April keyframes
kf_000224..kf_000346 (123 keyframes, 0.6 m, heading along the lane; the ledger's April frame = (y, -x) of the April H3DGS export).
Timestamps of those keyframes are read from image_left_kf20cm.monolithic (keyframe index = record index); every frame of
image_left.monolithic between them is written as <out>/images/f_<stream index>.png with <out>/stamps.json, exactly as
tenrows_lane_extract.py does for December.
  PYTHONPATH=<cp310 aru_py_logger> env -u LD_LIBRARY_PATH -u LD_PRELOAD <nerf_new python3.10> apr_lane_extract.py <kf_first> <kf_last> <out>"""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
M = __import__("os").environ.get("LANE_SURVEY", "/home/paperspace/data/klapmuts/apr_2026_zed") + "/prod/monos"; k0, k1, OUT = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]); (OUT / "images").mkdir(parents=True, exist_ok=True)
lg = aru_py_logger.MonoImageLogger(f"{M}/image_left_kf20cm.monolithic", False); i = 0; t0 = t1 = None
while not lg.end_of_file():
    img, ts = lg.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    if i == k0: t0 = float(ts)
    if i == k1: t1 = float(ts); break
    i += 1
assert t0 is not None and t1 is not None, (k0, k1, i)
print(f"[apr-lane] keyframes {k0}..{k1} -> stamps {t0:.0f}..{t1:.0f} ({(t1 - t0) / 1000:.1f} s)", flush=True)
lg = aru_py_logger.MonoImageLogger(f"{M}/image_left.monolithic", False); i = 0; kept = {}; tt = time.time()
while not lg.end_of_file():
    img, ts = lg.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    if t0 <= ts <= t1:
        name = f"f_{i:05d}.png"; cv2.imwrite(str(OUT / "images" / name), np.asarray(img)); kept[name] = int(ts)
    if ts > t1 + 1000: break
    i += 1
json.dump(kept, open(OUT / "stamps.json", "w")); json.dump({"kf_first": k0, "kf_last": k1, "t0_ms": t0, "t1_ms": t1}, open(OUT / "window.json", "w"))
print(f"[apr-lane] {len(kept)} frames (stream {min(int(k[2:7]) for k in kept)}..{max(int(k[2:7]) for k in kept)}) -> {OUT / 'images'} in {time.time() - tt:.0f}s", flush=True)
