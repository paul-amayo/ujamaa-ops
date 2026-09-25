"""Dump the ten_rows laser scans (sensor frame, as recorded) to a float32 memmap for LiDAR odometry:
<out>/scans_f32.npy [N, 64000, 3], <out>/scan_ts_ms.npy [N] (int64 ms, camera clock). Zero rows = no return.
Run with nerf_new python3.10 + PYTHONPATH of the cp310 aru_py_logger build."""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib"); import aru_py_logger
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics/laser.monolithic")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental/laser_dump"); OUT.mkdir(parents=True, exist_ok=True)
lg = aru_py_logger.LaserLogger(str(SRC), False); scans, ts = [], []; t0 = time.time(); n = 0
mm = None
while n < 6000:   # LaserLogger has no end_of_file(): stop on an empty/None record or a read exception
    try: r = lg.read_from_file()
    except Exception as e: print(f"[dump] read stopped at {n}: {type(e).__name__}", flush=True); break
    if r is None or r[0] is None or getattr(r[0], "size", 0) == 0: break
    pts, t = np.asarray(r[0], np.float32), int(r[1])
    if mm is None: mm = np.lib.format.open_memmap(OUT / "scans_f32.npy", mode="w+", dtype=np.float32, shape=(6000, pts.shape[0], 3))
    mm[n] = pts if pts.shape[0] == mm.shape[1] else np.pad(pts, ((0, mm.shape[1] - pts.shape[0]), (0, 0)))[:mm.shape[1]]; ts.append(t); n += 1
    if n % 500 == 0: print(f"[dump] {n} scans ({time.time() - t0:.0f}s)", flush=True)
mm.flush(); np.save(OUT / "scan_ts_ms.npy", np.array(ts, np.int64))
valid = np.count_nonzero(np.abs(mm[:n]).sum(2) > 0, axis=1)
print(f"[dump] {n} scans, {(ts[-1] - ts[0]) / 1e3:.0f} s at {n / ((ts[-1] - ts[0]) / 1e3):.1f} Hz; valid returns per scan median {int(np.median(valid))}; wrote {OUT} (memmap sized 6000; use the first {n} rows) in {(time.time() - t0) / 60:.1f} min", flush=True)
np.save(OUT / "n_scans.npy", np.array([n]))
