"""Stage C — unproject fruit detections to 3D and dedup by clustering.

Convention check is empirical: unprojects with both CV (+z forward) and GL
(-z forward) camera axes and keeps whichever lands the point cloud nearest the
tree's census centroid. Cluster = single-linkage union-find at eps metres.
Reports: physical fruit count at 10 fps vs keyframes-only, sightings/fruit.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

TAG = os.environ.get("FRUIT3D_TAG", "")
work = Path(sys.argv[1])
eps_list = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                               else ["0.05", "0.08", "0.12"])]
meta = json.loads((work / "meta.json").read_text())
dets = json.loads((work / f"detections{TAG}.json").read_text())
root, tree = Path(meta["root"]), meta["tree"]
pose_of = {r["name"]: np.array(r["pose"]) for r in meta["frames"]}

H = json.loads((root / "prod/bateleur/scene_graph/marker_hierarchy.json").read_text())
cent = np.array(next(o["xyz"] for o in H["objects"] if o["id"] == tree))

# intrinsics from any block transforms.json (one rig, one resolution)
tj = next(root.glob("prod/tassili/blocks_ns/*/block_*/transforms.json"))
T = json.loads(tj.read_text())
K = np.array([[T["fl_x"], 0, T["cx"]], [0, T["fl_y"], T["cy"]], [0, 0, 1]])
Kinv = np.linalg.inv(K)

# depth units: ZED PNGs are uint16 mm if median >> 100
all_d = [d["depth_med"] for r in dets for d in r["detections"] if d["depth_med"]]
if len(all_d) < 3:
    (work / f"dedup3d{TAG}.json").write_text(json.dumps(
        {"tree": tree, "n_events_10fps": 0, "n_events_kf": 0,
         "note": "too few depth-valid detections"}))
    print(f"[cluster] tree {tree}: only {len(all_d)} depth-valid "
          "detections — wrote empty dedup3d.json (expected, not a failure)")
    sys.exit(0)
scale = 0.001 if np.median(all_d) > 100 else 1.0
print(f"[cluster] {len(all_d)} depth-valid detections, median raw depth "
      f"{np.median(all_d):.0f} -> unit scale {scale}")

pts, tags = [], []
for r in dets:
    P = pose_of[r["name"]]
    for d in r["detections"]:
        src = d.get("depth_src", "fruit")
        if not d["depth_med"] or (src == "fruit" and d["depth_n"] < 4):
            continue
        z = d["depth_med"] * scale
        if not (0.5 < z < 12.0):
            continue
        ray = Kinv @ np.array([d["u"], d["v"], 1.0])
        pts.append((P, ray, z))
        tags.append({"name": r["name"], "is_kf": r["is_kf"], "ts": r["ts_ms"],
                     "score": d["score"], "area": d["area"], "depth_src": src})

def world(conv):
    out = np.empty((len(pts), 3))
    for i, (P, ray, z) in enumerate(pts):
        Xc = ray * z
        if conv == "gl":
            Xc = np.array([Xc[0], -Xc[1], -Xc[2]])
        out[i] = (P @ np.array([*Xc, 1.0]))[:3]
    return out

best = min(("cv", "gl"), key=lambda c: np.median(
    np.linalg.norm(world(c) - cent, axis=1)))
W = world(best)
d_cent = np.linalg.norm(W - cent, axis=1)
print(f"[cluster] convention {best.upper()}: median dist to census centroid "
      f"{np.median(d_cent):.2f} m (other: "
      f"{np.median(np.linalg.norm(world('gl' if best=='cv' else 'cv')-cent,axis=1)):.2f} m)")
keep = d_cent < 6.0  # sanity: fruit lives on the tree, not across the row
W, tags = W[keep], [t for t, k in zip(tags, keep) if k]
print(f"[cluster] {keep.sum()}/{len(keep)} points within 6 m of tree centroid")
if len(W) < 2:
    (work / f"dedup3d{TAG}.json").write_text(json.dumps(
        {"tree": tree, "n_events_10fps": int(len(W)), "n_events_kf": 0,
         "note": "no unprojectable points survived the gates"}))
    print(f"[cluster] tree {tree}: {len(W)} usable points — wrote empty "
          "dedup3d.json (expected, not a failure)")
    sys.exit(0)

def cluster(P, eps):
    n = len(P)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        d = np.linalg.norm(P[i + 1:] - P[i], axis=1)
        for j in np.where(d < eps)[0] + i + 1:
            ra, rb = find(i), find(int(j))
            if ra != rb:
                parent[ra] = rb
    lab = np.array([find(i) for i in range(n)])
    return lab

def report(mask, label):
    P = W[mask]
    print(f"\n--- {label}: {len(P)} detection-events ---")
    rows = {}
    if len(P) == 0:
        for eps in eps_list:
            rows[eps] = {"clusters": 0, "multi": 0, "singletons": 0,
                         "max_sightings": 0}
        return rows
    for eps in eps_list:
        lab = cluster(P, eps)
        uniq, counts = np.unique(lab, return_counts=True)
        multi = int((counts >= 2).sum())
        print(f"  eps {eps*100:>4.0f} cm: {len(uniq):>3} clusters "
              f"({multi} seen >=2x, singletons {int((counts==1).sum())}); "
              f"sightings/fruit med {np.median(counts):.0f} max {counts.max()}")
        rows[eps] = {"clusters": int(len(uniq)), "multi": multi,
                     "singletons": int((counts == 1).sum()),
                     "max_sightings": int(counts.max())}
    return rows

kf_mask = np.array([t["is_kf"] for t in tags])
fr_mask = np.array([t["depth_src"] == "fruit" for t in tags])
res_all = report(np.ones(len(tags), bool), "10 fps (all native frames)")
res_kf = report(kf_mask, "keyframes only (same machinery)")
res_clean = report(fr_mask, "10 fps, fruit-pixel depth only (clean tier)")
res_clean_kf = report(fr_mask & kf_mask, "keyframes only, fruit-pixel depth")

out = {"tree": tree, "convention": best, "depth_scale": scale,
       "n_events_10fps": int(len(tags)), "n_events_kf": int(kf_mask.sum()),
       "per_eps_10fps": {str(k): v for k, v in res_all.items()},
       "per_eps_kf": {str(k): v for k, v in res_kf.items()},
       "per_eps_10fps_fruitpx": {str(k): v for k, v in res_clean.items()},
       "per_eps_kf_fruitpx": {str(k): v for k, v in res_clean_kf.items()},
       "points": [{"xyz": W[i].tolist(), **tags[i]} for i in range(len(tags))]}
(work / f"dedup3d{TAG}.json").write_text(json.dumps(out, indent=1))
print(f"\n[cluster] wrote {work}/dedup3d{TAG}.json")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
eps0 = eps_list[1] if len(eps_list) > 1 else eps_list[0]
lab = cluster(W, eps0)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (title, mask) in zip(axes, [("10 fps (fruit-px depth)", fr_mask),
                                    ("keyframes only (fruit-px depth)",
                                     fr_mask & kf_mask)]):
    P = W[mask]
    lb = cluster(P, eps0)
    for u in np.unique(lb):
        m = lb == u
        ax.scatter(P[m, 0], P[m, 2], s=26, alpha=.8)
    ax.scatter([cent[0]], [cent[2]], marker="*", s=240, color="black",
               label=f"tree {tree} centroid")
    ax.set_title(f"{title}: {len(np.unique(lb))} fruit @ eps {eps0*100:.0f} cm "
                 f"({mask.sum()} events)")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
fig.suptitle(f"3D fruit dedup, tree {tree} — top-down of unprojected detections")
fig.tight_layout()
fig.savefig(work / f"dedup3d{TAG}.png", dpi=130)
print(f"[cluster] fig -> {work}/dedup3d{TAG}.png")
