"""Keyframe cut, process B (writer side; aru_py_logger only): kf PNGs + kf_index.json
-> image_left_kf20cm.monolithic (verbatim pixels, original image timestamps)."""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib")
import aru_py_logger
R = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows"); MD = R/"prod/monos/monolithics"; PNG = R/"prod/scratch_sam3"
index = json.loads((MD/"kf_index.json").read_text())
out = MD/"image_left_kf20cm.monolithic"
for stale in (out, Path(str(out) + ".index")): stale.unlink(missing_ok=True)
lg = aru_py_logger.MonoImageLogger(str(out), True)
for e in index:
    img = np.ascontiguousarray(np.asarray(Image.open(PNG/f"kf_{e['K']:06d}.png").convert("RGB")))
    lg.write_to_file(img, int(e["ts_ms"]))
print(f"[kfcut-B] wrote {len(index)} keyframes -> {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
