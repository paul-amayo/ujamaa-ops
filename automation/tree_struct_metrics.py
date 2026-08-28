#!/usr/bin/env python3
"""Per-tree structural metrics from the gen2 census-owned gaussians (P2 v1).

For every block: stage1 gaussian means (ns space) -> survey LIO frame via the
run's dataparser_transforms.json; ownership from the census interaction W
(argmax over entities, tot > 1.0 floor, tree rows only — fruit rows are
labels >= FRUIT_ID_BASE and excluded). Metrics per tree across its blocks:
gaussian count, centroid, height (p97-p3 of Y, the up axis in the LIO camera
world frame), canopy radius (p90 horizontal distance to centroid), robust
bbox volume. Overlap-zone gaussians are counted once per hosting block —
extents are percentile-robust to that; n_gauss is seam-inflated (noted).

Runs under the nerf_new pixi env (torch for ckpt reads).
Output: <root>/prod/bateleur/scene_graph/tree_struct_metrics_v1.json
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import torch

FRUIT_ID_BASE = 10000


def block_points(bd):
    w = bd / "splat_runs_FEATFIX/interaction_W.npz"
    if not w.exists():
        return None, f"no interaction_W: {bd.name}"
    cks = sorted(bd.glob(
        "splat_runs_STAGE1/*/high/*/nerfstudio_models/*.ckpt"),
        key=lambda p: p.stat().st_mtime)
    if not cks:
        return None, f"no stage1 ckpt: {bd.name}"
    ck = cks[-1]
    dpt = ck.parent.parent / "dataparser_transforms.json"
    if not dpt.exists():
        return None, f"no dataparser_transforms: {bd.name}"
    d = json.loads(dpt.read_text())
    T = np.eye(4)
    T[:3] = np.asarray(d["transform"], np.float64)
    scale = float(d["scale"])
    z = np.load(w)
    W, labels = z["W"], z["labels"]
    sd = torch.load(ck, map_location="cpu")["pipeline"]
    means = next(v for k, v in sd.items()
                 if k.endswith("gauss_params.means")).numpy().astype(np.float64)
    if means.shape[0] != W.shape[1]:
        return None, (f"W/ckpt mismatch {bd.name}: "
                      f"W {W.shape[1]} vs means {means.shape[0]}")
    pts = (np.linalg.inv(T) @ np.c_[means / scale,
                                    np.ones(len(means))].T).T[:, :3]
    tot = W.sum(0)
    assign = W.argmax(0)
    owned = tot > 1.0
    out = {}
    for row, lab in enumerate(labels):
        if lab >= FRUIT_ID_BASE:
            continue
        m = owned & (assign == row)
        if m.sum() >= 30:
            out[int(lab)] = pts[m]
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    root = args.root

    per_tree = {}
    skipped = []
    blocks = sorted(root.glob("prod/tassili/blocks_ns/*/block_*"))
    for bd in blocks:
        got, err = block_points(bd)
        if err:
            skipped.append(err)
            continue
        for tid, p in got.items():
            per_tree.setdefault(tid, []).append(p)

    # census blend attribution smears ownership onto background gaussians
    # along viewing rays; anchor-trim to the hierarchy centroid before
    # measuring (untrimmed: 05 read 0.86 m heights / 6.6 m canopy radii).
    H = json.loads((root / "prod/bateleur/scene_graph/marker_hierarchy.json")
                   .read_text())
    anchor = {o["id"]: np.asarray(o["xyz"], np.float64) for o in H["objects"]}

    metrics = {}
    for tid, chunks in sorted(per_tree.items()):
        P = np.concatenate(chunks)
        n_raw = len(P)
        a = anchor.get(tid)
        if a is None:
            continue
        horiz_a = np.linalg.norm(P[:, [0, 2]] - a[[0, 2]], axis=1)
        P = P[(horiz_a < 3.0) & (np.abs(P[:, 1] - a[1]) < 6.0)]
        if len(P) < 30:
            continue
        c = P.mean(0)
        dy = np.percentile(P[:, 1], 97) - np.percentile(P[:, 1], 3)
        horiz = np.linalg.norm(P[:, [0, 2]] - c[[0, 2]], axis=1)
        vol = float(np.prod([np.percentile(P[:, a2], 97)
                             - np.percentile(P[:, a2], 3) for a2 in range(3)]))
        metrics[str(tid)] = {
            "n_gauss": int(len(P)), "n_raw": n_raw,
            "trim_kept": round(float(len(P)) / n_raw, 3),
            "n_blocks": len(chunks),
            "centroid_xyz": [round(float(x), 3) for x in c],
            "height_m": round(float(dy), 3),
            "canopy_r90_m": round(float(np.percentile(horiz, 90)), 3),
            "bbox_vol_m3": round(vol, 3)}

    h = [m["height_m"] for m in metrics.values()]
    r = [m["canopy_r90_m"] for m in metrics.values()]
    out = {"schema": "tree_struct_metrics/v1", "generated": str(date.today()),
           "source": "gen2 census argmax over stage1 gaussians (seam-inflated n_gauss)",
           "n_trees": len(metrics), "n_blocks": len(blocks),
           "skipped": skipped, "trees": metrics}
    dst = root / "prod/bateleur/scene_graph/tree_struct_metrics_v1.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"[structmet] {root.name}: {len(metrics)} trees from "
          f"{len(blocks) - len(skipped)}/{len(blocks)} blocks; "
          f"median height {np.median(h):.2f} m, canopy r90 {np.median(r):.2f} m"
          if metrics else f"[structmet] {root.name}: NO TREES")
    for s in skipped:
        print(f"[structmet]   skip: {s}")
    print(f"[structmet] -> {dst}")


if __name__ == "__main__":
    main()
