"""Rebuild ten_rows' keyframe image stream and PNGs from a (corrected) full image stream using the existing kf_index.json
(the 20-cm keyframe selection is geometric and unchanged). One sequential pass over image_left.monolithic; entry K of the
new image_left_kf20cm.monolithic is keyframe K with its original timestamp, and prod/scratch_sam3/kf_%06d.png is rewritten.
Run with nerf_new's python3.10 + PYTHONPATH of the aru_py_logger build (same env as the ingest).
  python tenrows_kf_regen.py <monolithics_dir> <scratch_sam3_dir>"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/lib"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
import aru_py_logger
MD, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
kf = json.load(open(MD / "kf_index.json")); want = {e["image_idx"]: e for e in kf}; N = len(kf)
src = aru_py_logger.MonoImageLogger(str(MD / "image_left.monolithic"), False)
kfp = MD / "image_left_kf20cm.monolithic"
for p in (kfp, kfp.with_suffix(".monolithic.index")):
    if p.exists(): p.unlink()
dst = aru_py_logger.MonoImageLogger(str(kfp), True)
i = 0; written = 0; t0 = time.time(); first = None
while not src.end_of_file():
    img, ts = src.read_from_file()
    if img is None or getattr(img, "size", 0) == 0: break
    if i in want:
        e = want[i]; K = e["K"]
        dst.write_to_file(img, int(e["ts_ms"]))
        cv2.imwrite(str(OUT / f"kf_{K:06d}.png"), img)
        if first is None:
            top = img[: img.shape[0] // 6].reshape(-1, 3).mean(0); first = f"kf_{K:06d} sky rows B {top[0]:.0f} G {top[1]:.0f} R {top[2]:.0f} (BGR array as stored; expect B > R for a blue-ish sky)"
        written += 1
        if written % 250 == 0: print(f"[kf-regen] {written}/{N} keyframes ({time.time()-t0:.0f}s)", flush=True)
    i += 1
del dst
print(f"[kf-regen] done: {written}/{N} keyframes from {i} full-stream frames in {(time.time()-t0)/60:.1f} min; {first}")
if written != N: sys.exit(f"[kf-regen] MISMATCH: wrote {written}, index has {N}")
