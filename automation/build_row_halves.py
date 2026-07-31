#!/usr/bin/env python3
"""Half-row partition (MAX BLOCK LENGTH rule): detect row-passes like
build_row_blocks.py, then split each pass at its midpoint into A/B halves that
OVERLAP by one frame (for compositing). Parameterized by survey root.
Usage: build_row_halves.py <survey_root> [outname=lio_row_halves]"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(sys.argv[1])
OUTCFG = ROOT / "blocks_ns" / (sys.argv[2] if len(sys.argv) > 2 else "lio_row_halves")
lio = sorted(json.loads((ROOT / "lio_image_poses_kf20cm.json").read_text()), key=lambda p: p["image_idx"])
names = [p["image_name"] for p in lio]
name2tf = {p["image_name"]: p["transform"] for p in lio}
name2ts = {p["image_name"]: p.get("timestamp_ns") for p in lio}
FGMASKS = ROOT / "fg_masks"
# Image dir: prefer the dir whose contents actually match the json's names
# (03 keeps json-named frames in images/, while kf_images/ uses kf_%06d —
# checking existence of the dir alone is NOT enough, check a real name).
_probe = json.loads((ROOT / "lio_image_poses_kf20cm.json").read_text())[0]["image_name"]
P = np.array([np.array(p["transform"])[:3, 3] for p in lio])
sp = P.max(0) - P.min(0); a, b = sorted(np.argsort(-sp)[:2])
Z = P[:, b]

dz = np.diff(Z); sign = np.sign(dz); sign[sign == 0] = 1
dirn = np.sign(np.convolve(sign, np.ones(15) / 15, mode="same"))
rev = list(np.where(np.diff(dirn) != 0)[0] + 1)
bounds = [0] + rev + [len(P)]
passes = [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]
passes = [p for p in passes if len(p) > 20]

# split each pass at midpoint; halves share the split frame
halves = []
for p in passes:
    m = len(p) // 2
    halves.append(p[: m + 1])
    halves.append(p[m:])

src_candidates = sorted((ROOT / "blocks_ns").glob("lio_arc*/block_000/transforms.json"))
src = json.loads(src_candidates[0].read_text())
intr = {k: src[k] for k in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2", "w", "h", "camera_model") if k in src}
KFIMG = next((d for d in (ROOT / "images", ROOT / "kf_images")
              if (d / _probe).exists()), None)
if KFIMG is None:
    raise SystemExit(f"no image dir contains {_probe} — checked images/ and kf_images/")
print(f"[build_row_halves] image dir: {KFIMG}")

print(f"{len(passes)} row-passes -> {len(halves)} half-row blocks")
for i, pf in enumerate(halves):
    bd = OUTCFG / f"block_{i:03d}"; bd.mkdir(parents=True, exist_ok=True)
    frames = []
    for j in pf:
        nm = names[j]
        fr = {"file_path": str(KFIMG / nm), "transform_matrix": name2tf[nm]}
        if name2ts.get(nm) is not None:
            fr["timestamp_ns"] = name2ts[nm]
        mp = FGMASKS / nm
        if mp.exists():
            fr["mask_path"] = str(mp)
        frames.append(fr)
    tj = dict(intr); tj["ply_file_path"] = "init_da3.ply"; tj["frames"] = frames
    (bd / "transforms.json").write_text(json.dumps(tj, indent=2))
    print(f"  block_{i:03d}: {len(frames)}f  kf {names[pf[0]]}..{names[pf[-1]]}  X~{P[pf,a].mean():.1f}m")
print(f"\nwrote {len(halves)} blocks to {OUTCFG}")
