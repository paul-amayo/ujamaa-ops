#!/usr/bin/env python3
"""dec_2025_ten_rows: whole-survey LiDAR cloud — every scan, full 360 deg.

Design (verified on this box 2026-09-05):
  * LaserProjector.project(i) returns only the ~15.6k points inside the camera
    frustum; a 32-ring scan holds 64k points over 360 deg (37k 'behind', 12k
    'outside' the image). project_debug(i)['p_world'] carries the FULL scan
    lifted to world — checked against c2w @ L2C @ T_motion @ raw to 1.8e-5.
  * Images run at 15 Hz, scans at 10 Hz, so ~1.5 images match each scan;
    lifting per image would place each scan twice at slightly different
    interpolated poses. We pick ONE image per scan (smallest |dt|) -> at most
    3,943 lifts, one per scan.
  * Edge frames (before the first / after the last odometry record) return an
    'error' dict ("no LIO pose at ts_img") and are skipped, not fatal.

Frame contract (Paul's flag): the ZED odometry is CAMERA-frame; the projector
assumes LIDAR-frame Tf in c2w = L2C * Tf * L2C^-1. We feed the conjugated
transform_lio_laserframe.monolithic (ensure_laser_frame_poses.py, detected
flat-y camera -> converted flat-z laser), and pass the rig's A300 L2C — without
it the binding silently falls back to the Jackal extrinsic.

Run in nerf_new python3.10 with aru_nerf_interface on PYTHONPATH. Never import
pbTransform_pb2 in this process (descriptor clash).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
import aru_nerf_interface as a  # noqa: E402
from rig_calib import load_rig, l2c_flat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monos", required=True, type=Path)
    ap.add_argument("--transform", default="transform_lio_laserframe.monolithic",
                    help="pose monolithic (MUST be lidar-frame for the projector)")
    ap.add_argument("--rmax", type=float, default=40.0, help="keep returns within this of the sensor (m)")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True, help="output stem (.npz/.ply)")
    args = ap.parse_args()

    M = args.monos
    rig = load_rig(M)
    K = rig["intrinsics"]
    proj = a.LaserProjector(
        image_mono=str(M / "image_left.monolithic"),
        transform_mono=str(M / args.transform),
        laser_mono=str(M / "laser.monolithic"),
        fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"],
        img_w=int(K.get("img_w", 1280)), img_h=int(K.get("img_h", 720)),
        laser_match_tol_ns=80_000_000, l2c=l2c_flat(M))
    n = proj.num_images()
    print(f"[cloud] {n} images; rmax {args.rmax} m, voxel {args.voxel} m; "
          f"L2C from rig '{rig.get('rig')}'", flush=True)

    # pass 1: one image per scan (smallest |dt| within tolerance) — cheap, no projection
    best = {}
    for i in range(n):
        mi = proj.match_info(i)
        if not mi.get("within_tol"):
            continue
        li, dt = int(mi["laser_idx"]), abs(int(mi["dt_ns"]))
        if li not in best or dt < best[li][1]:
            best[li] = (i, dt)
    print(f"[cloud] scans reachable: {len(best)} (one image each; median |dt| "
          f"{np.median([v[1] for v in best.values()]):.0f} ms)", flush=True)

    # pass 2: lift the FULL scan for each chosen image
    keys, cams, raw_total, lifted, edge = [], [], 0, 0, 0
    t0 = time.time()
    for k, (li, (i, _)) in enumerate(sorted(best.items())):
        d = proj.project_debug(int(i))
        if "error" in d or "p_world" not in d:
            edge += 1
            continue
        pw = np.asarray(d["p_world"], dtype=np.float64)[:, :3]
        cam = np.asarray(d["c2w"], dtype=np.float64).reshape(4, 4)[:3, 3]
        cams.append(cam)
        dist = np.linalg.norm(pw - cam, axis=1)
        pw = pw[dist <= args.rmax]
        if len(pw) == 0:
            continue
        raw_total += len(pw)
        lifted += 1
        keys.append(np.unique(np.round(pw / args.voxel).astype(np.int32), axis=0))
        if k % 500 == 0:
            print(f"  scan {k}/{len(best)}: {raw_total:,} pts kept, "
                  f"{sum(len(q) for q in keys):,} pre-merge voxels ({time.time()-t0:.0f}s)", flush=True)

    if not keys:
        sys.exit("[cloud] NO points lifted — check pose frame / index files")
    uq, cnt = np.unique(np.concatenate(keys), axis=0, return_counts=True)
    xyz = (uq.astype(np.float64) * args.voxel).astype(np.float32)
    cams = np.array(cams)
    ext = xyz.max(0) - xyz.min(0)
    tp = cams.max(0) - cams.min(0)
    flat = int(np.argmin(tp))
    print(f"[cloud] scans lifted {lifted}/{len(best)} (edge/no-pose skipped {edge}); "
          f"raw pts kept {raw_total:,}; voxels {len(xyz):,}; returns/voxel median {np.median(cnt):.0f} "
          f"p90 {np.percentile(cnt, 90):.0f} max {cnt.max()}")
    print(f"[cloud] cloud extent (x,y,z) = {np.round(ext, 1)} m")
    print(f"[cloud] trajectory ptp (x,y,z) = {np.round(tp, 2)} m -> flat axis {'xyz'[flat]} "
          f"({'camera-world, y down: EXPECTED' if flat == 1 else 'UNEXPECTED — check frame'})")

    np.savez_compressed(args.out.with_suffix(".npz"), xyz=xyz, count=cnt.astype(np.int32),
                        cams=cams.astype(np.float32), voxel=args.voxel, rmax=args.rmax)
    h = -xyz[:, 1]                                  # camera-world: y down -> height = -y
    lo, hi = np.percentile(h, 2), np.percentile(h, 98)
    grey = (40 + 200 * np.clip((h - lo) / max(1e-6, hi - lo), 0, 1)).astype(np.uint8)
    with open(args.out.with_suffix(".ply"), "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(xyz)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n"
                b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                b"property int count\nend_header\n")
        rec = np.zeros(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3), ("count", "<i4")])
        rec["xyz"] = xyz; rec["rgb"] = np.stack([grey] * 3, 1); rec["count"] = cnt
        rec.tofile(f)
    print(f"[cloud] wrote {args.out.with_suffix('.npz').name} + .ply", flush=True)


if __name__ == "__main__":
    main()
