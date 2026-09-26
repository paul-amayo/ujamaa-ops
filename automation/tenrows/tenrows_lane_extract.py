"""All left frames of one ten_rows lane from the full 15 Hz stream (Paul, 2026-09-26: "prove in one row"): every image in
image_left.monolithic whose stamp lies in [t0, t1] ms is written with the canonical aru_py_logger reader + cv2.imwrite
(BGR-native -> correct colours on disk) as <out>/images/f_<stream index>.png, plus <out>/stamps.json {name: ts_ms}.
  PYTHONPATH=<cp310 aru_py_logger> env -u LD_LIBRARY_PATH -u LD_PRELOAD <nerf_new python3.10> tenrows_lane_extract.py <t0_ms> <t1_ms> <out dir>"""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
t0, t1, OUT = float(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]); (OUT / "images").mkdir(parents=True, exist_ok=True)
lg = aru_py_logger.MonoImageLogger("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics/image_left.monolithic", False); i = 0; kept = {}; tt = time.time()
while not lg.end_of_file():
    img, ts = lg.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    if t0 <= ts <= t1:
        name = f"f_{i:05d}.png"; cv2.imwrite(str(OUT / "images" / name), np.asarray(img)); kept[name] = int(ts)
    if ts > t1 + 1000: break
    i += 1
json.dump(kept, open(OUT / "stamps.json", "w"))
print(f"[lane] {len(kept)} frames in [{t0:.0f}, {t1:.0f}] ms (stream indices {min(int(k[2:7]) for k in kept)}..{max(int(k[2:7]) for k in kept)}) -> {OUT / 'images'} in {time.time() - tt:.0f}s", flush=True)
