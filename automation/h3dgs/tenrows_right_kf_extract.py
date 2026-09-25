"""Right-camera PNGs for a set of ten_rows keyframes (stereo experiment, 2026-09-25). Walks image_right.monolithic once
with the canonical aru_py_logger reader (yields BGR; cv2.imwrite expects BGR — same convention as extract_kf_pngs.py),
matches each record to the keyframes' LEFT timestamps (prod/monos/monolithics/kf_index.json ts_ms) within --tol ms and
writes <out>/kf_<K:06d>_R.png. Reports the left/right stamp offsets (the ZED node stamps both images identically).
  env -u LD_LIBRARY_PATH -u LD_PRELOAD <nerf_new python3.10> tenrows_right_kf_extract.py --names <file: kf_XXXXXX.png per line>
      --out <dir> [--mono <image_right.monolithic>] [--kf-index <kf_index.json>] [--tol 4]"""
import argparse, bisect, json, re, sys, time
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/lib"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
import aru_py_logger  # noqa: E402
M = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics")
p = argparse.ArgumentParser(); p.add_argument("--names", required=True); p.add_argument("--out", required=True)
p.add_argument("--mono", default=str(M / "image_right.monolithic")); p.add_argument("--kf-index", default=str(M / "kf_index.json")); p.add_argument("--tol", type=float, default=4.0)
a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
kf = {int(e["K"]): float(e["ts_ms"]) for e in json.load(open(a.kf_index))}
want = sorted({int(re.search(r"kf_(\d+)", l).group(1)) for l in open(a.names) if "kf_" in l})
targets = sorted((kf[k], k) for k in want if k in kf); tts = [t for t, _ in targets]
print(f"[right-kf] {len(want)} keyframes wanted, {len(targets)} with a left timestamp; span {tts[0]:.0f}..{tts[-1]:.0f} ms", flush=True)
def to_ms(ts):
    ts = float(ts)
    return ts / 1e6 if ts > 1e15 else (ts if ts > 1e11 else ts * 1e3)
logger = aru_py_logger.MonoImageLogger(a.mono, False); n = 0; hits = {}; dts = []; t0 = time.time(); first = None
while not logger.end_of_file():
    img, ts = logger.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    ms = to_ms(ts); n += 1
    if first is None: first = (ts, ms); print(f"[right-kf] first record raw ts {ts} -> {ms:.0f} ms; first target {tts[0]:.0f} ms (dt {ms - tts[0]:.0f} ms); image {img.shape}", flush=True)
    if ms > tts[-1] + 2000: break
    i = bisect.bisect_left(tts, ms); cand = [j for j in (i - 1, i) if 0 <= j < len(tts)]
    if not cand: continue
    j = min(cand, key=lambda j: abs(tts[j] - ms)); dt = ms - tts[j]
    if abs(dt) <= a.tol:
        k = targets[j][1]
        if k not in hits or abs(dt) < abs(hits[k]):
            cv2.imwrite(str(out / f"kf_{k:06d}_R.png"), img); hits[k] = dt; dts.append(dt)
    if n % 5000 == 0: print(f"[right-kf] {n} records, {len(hits)} matched ({time.time() - t0:.0f}s)", flush=True)
missing = [k for _, k in targets if k not in hits]
print(f"[right-kf] done: {n} records read, {len(hits)}/{len(targets)} keyframes matched within {a.tol} ms, |dt| median {np.median(np.abs(dts)) if dts else float('nan'):.2f} ms max {np.max(np.abs(dts)) if dts else float('nan'):.2f} ms; "
      f"missing {len(missing)} {missing[:8]}; {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
