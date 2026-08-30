#!/usr/bin/env python3
"""Render splat depth for a fruit3d workdir — the ZED-hole recovery lever.

76% of fruit detections lost their 3D point to ZED depth holes (median
fruit is 91 px). The trained splats render DENSE metric depth at the same
native intrinsics, so: for every prepped native frame, pick the nearest
block by camera-centre, render depth at the frame's interpolated pose, and
write <workdir>/depth_splat/<name>.png (uint16 mm, same contract as the
ZED pngs). fruit3d_detect.py consumes it with FRUIT3D_DEPTH_DIR=depth_splat.

Runs in the nerf_new pixi env (BlockRenderer). GPU-light (~0.1 s/frame).
  usage: fruit3d_splatdepth.py --workdir W [W2 ...] --cfg-dir <blocks_ns/cfg>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/interfaces/splat_viewer")


def block_centres(cfg_dir):
    out = []
    for bd in sorted(cfg_dir.glob("block_*")):
        tj = bd / "transforms.json"
        if not tj.exists():
            continue
        fr = json.loads(tj.read_text())["frames"]
        C = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in fr])
        out.append((bd, C.mean(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", nargs="+", type=Path, required=True)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--run-glob",
                    default="splat_runs_FEATFIX/stage2_censusinit_*/high/*/config.yml")
    a = ap.parse_args()

    from block_renderer import BlockRenderer
    import cv2

    centres = block_centres(a.cfg_dir)
    renderers = {}

    def renderer_for(pos):
        bd = min(centres, key=lambda c: np.linalg.norm(c[1] - pos))[0]
        if bd not in renderers:
            cfgs = sorted(bd.glob(a.run_glob))
            if not cfgs:
                return None, bd.name
            renderers[bd] = BlockRenderer(str(cfgs[-1]), block_id=bd.name)
        return renderers[bd], bd.name

    for work in a.workdir:
        meta = json.loads((work / "meta.json").read_text())
        dd = work / "depth_splat"
        dd.mkdir(exist_ok=True)
        n_ok, miss = 0, {}
        for rec in meta["frames"]:
            P = np.asarray(rec["pose"])
            img = cv2.imread(str(work / "frames" / f"{rec['name']}.png"))
            H, W = img.shape[:2]
            br, bname = renderer_for(P[:3, 3])
            if br is None:
                miss[bname] = miss.get(bname, 0) + 1
                continue
            out = br.render(P, W, H, want_depth=True)
            d = out["depth"]
            if d is None:
                miss[bname] = miss.get(bname, 0) + 1
                continue
            mm = np.clip(np.nan_to_num(d, nan=0.0) * 1000.0, 0, 65535)
            cv2.imwrite(str(dd / f"{rec['name']}.png"), mm.astype(np.uint16))
            n_ok += 1
        print(f"[splatdepth] {work.name}: {n_ok}/{len(meta['frames'])} frames "
              f"rendered" + (f"; no run for {miss}" if miss else ""), flush=True)


if __name__ == "__main__":
    main()
