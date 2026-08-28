"""Stage A — prep native-rate (10 fps) frames for 3D fruit dedup on one tree.

For every native frame whose timestamp falls inside the tree's keyframe
sighting windows: decode the left image + nearest depth PNG, interpolate the
camera pose from transform_lio (same math as survey_paths.poses_at_keyframes,
at arbitrary ts), and warp-proxy the tree mask from the nearest keyframe's
trees_only supervision (dilated for the <=10 cm baseline slack).

Runs under SYSTEM python3 (numpy+scipy+cv2). Emits a self-contained workdir
that stage B (sam3 env) and stage C (cluster) consume.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/paperspace/code/aru_sil_core/src/scripts")
import survey_paths  # noqa: E402


def read_varint_f(f):
    result = shift = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        v = b[0]
        result |= (v & 0x7F) << shift
        if not v & 0x80:
            return result
        shift += 7


def index_mono(path, ts_field, data_field):
    """[(ts_ms, payload_offset, payload_len)] — header walk, payload seek."""
    out = []
    with open(path, "rb") as f:
        while True:
            size = read_varint_f(f)
            if size is None:
                break
            start = f.tell()
            ts, off, ln = None, None, None
            while f.tell() < start + size:
                tag = read_varint_f(f)
                if tag is None:
                    break
                fld, wire = tag >> 3, tag & 7
                if wire == 0:
                    v = read_varint_f(f)
                    if fld == ts_field:
                        ts = v
                elif wire == 2:
                    n = read_varint_f(f)
                    if fld == data_field:
                        off, ln = f.tell(), n
                    f.seek(n, 1)
                elif wire == 5:
                    f.seek(4, 1)
                elif wire == 1:
                    f.seek(8, 1)
            f.seek(start + size)
            if ts is not None and off is not None:
                out.append((normalize_ms(ts), off, ln))
    return out


def normalize_ms(t):
    t = int(t)
    while t > 10 ** 14:
        t //= 1000
    return t


def decode_at(path, off, ln, flags=cv2.IMREAD_UNCHANGED):
    with open(path, "rb") as f:
        f.seek(off)
        buf = np.frombuffer(f.read(ln), np.uint8)
    return cv2.imdecode(buf, flags)


def poses_at(root, query_ts):
    """camera world poses at arbitrary ms timestamps (survey_paths math)."""
    incs = survey_paths._read_increments(
        survey_paths.monos(root) / "transform_lio.monolithic")
    inc_ts = np.asarray([t for t, _ in incs], dtype=np.int64)
    absolute = np.empty((len(incs), 4, 4))
    absolute[0] = np.eye(4)
    for k in range(1, len(incs)):
        absolute[k] = absolute[k - 1] @ incs[k][1]
    l2c = survey_paths.load_l2c(root)
    l2c_inv = np.linalg.inv(l2c)
    poses = np.empty((len(query_ts), 4, 4))
    for i, t in enumerate(query_ts):
        j = int(np.searchsorted(inc_ts, t, side="left"))
        if j <= 0 or j >= len(incs):
            poses[i] = np.nan
            continue
        ratio = float(inc_ts[j] - t) / float(inc_ts[j] - inc_ts[j - 1])
        partial = survey_paths._slerp_partial(incs[j][1], ratio)
        T = absolute[j - 1] @ (incs[j][1] @ np.linalg.inv(partial))
        poses[i] = l2c @ T @ l2c_inv
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--tree", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pad-kf", type=int, default=1,
                    help="expand each sighting window by this many keyframes")
    ap.add_argument("--dilate", type=int, default=30,
                    help="tree-mask dilation px (nearest-kf warp slack)")
    args = ap.parse_args()
    root, tree = args.root, args.tree

    # --- tree's keyframe sightings from the prod fruit ledger ---
    kf_hits = {}
    for fe in sorted((survey_paths.bateleur(root) / "sam3_fruit").glob(
            "clip_*/frame_entries.json")):
        d = json.loads(fe.read_text())
        loc2kf = {f["local_idx"]: f["kf_idx"] for f in d["frames"]}
        for loc, ents in d["frame_entries"].items():
            n = sum(1 for e in ents if e.get("parent_census_id") == tree)
            if n:
                kf_hits[loc2kf[int(loc)]] = kf_hits.get(loc2kf[int(loc)], 0) + n
    kfs = sorted(kf_hits)
    if not kfs:
        sys.exit(f"PREP-FAIL: tree {tree} has no ledger sightings")
    print(f"[prep] tree {tree}: {sum(kf_hits.values())} detection-events "
          f"across {len(kfs)} keyframes: {kfs}")

    # --- supervision masks: which block dir holds each kf ---
    sup_of = {}
    for sup in root.glob(
            "prod/tassili/blocks_ns/*/block_*/supervision/trees_only"):
        for p in sup.glob("kf_*.png"):
            sup_of.setdefault(int(p.stem[3:]), []).append(p)

    # --- timestamp domains ---
    nat = index_mono(survey_paths.monos(root) / "image_left.monolithic", 3, 4)
    dep = index_mono(survey_paths.monos(root) / "image_depth.monolithic", 2, 1)
    kf_ts = np.array([normalize_ms(t) for t in __import__("mono_ts").mono_timestamps(
        survey_paths.monos(root) / "image_left_kf20cm.monolithic")], np.int64)
    nat_ts = np.array([t for t, _, _ in nat], np.int64)
    dep_ts = np.array([t for t, _, _ in dep], np.int64)
    in_nat = sum(1 for t in kf_ts if t in set(nat_ts.tolist()))
    print(f"[prep] native {len(nat_ts)} @10fps, depth {len(dep_ts)}, "
          f"kf {len(kf_ts)}; kf ts subset of native: {in_nat}/{len(kf_ts)}")

    # --- native frames inside the sighting windows ---
    groups, cur = [], [kfs[0]]
    for k in kfs[1:]:
        if k - cur[-1] <= 3:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    sel = set()
    for g in groups:
        lo = kf_ts[max(0, g[0] - args.pad_kf)]
        hi = kf_ts[min(len(kf_ts) - 1, g[-1] + args.pad_kf)]
        for i, t in enumerate(nat_ts):
            if lo <= t <= hi:
                sel.add(i)
    sel = sorted(sel)
    print(f"[prep] {len(groups)} windows -> {len(sel)} native frames "
          f"(vs {len(kfs)} keyframes = x{len(sel)/len(kfs):.1f})")

    poses = poses_at(root, nat_ts[sel])

    # --- emit workdir ---
    out = args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    meta, n_ok = [], 0
    for row, i in enumerate(sel):
        t = int(nat_ts[i])
        if np.isnan(poses[row]).any():
            continue
        # nearest keyframe that actually holds the tree, within 300 ms
        cand = [(abs(int(kf_ts[k]) - t), k) for k in kfs]
        dt_kf, k_near = min(cand)
        if dt_kf > 300:
            continue
        sup_paths = sup_of.get(k_near, [])
        m = None
        for sp in sup_paths:
            a = cv2.imread(str(sp), cv2.IMREAD_UNCHANGED)
            if a is not None and (a == tree).any():
                m = (a == tree).astype(np.uint8)
                break
        if m is None:
            continue
        img = decode_at(survey_paths.monos(root) / "image_left.monolithic",
                        nat[i][1], nat[i][2], cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, np.ones((args.dilate, args.dilate), np.uint8))
        j = int(np.argmin(np.abs(dep_ts - t)))
        dmat = decode_at(survey_paths.monos(root) / "image_depth.monolithic",
                         dep[j][1], dep[j][2])
        name = f"nat_{i:06d}"
        cv2.imwrite(str(out / "frames" / f"{name}.png"), img)
        cv2.imwrite(str(out / "masks" / f"{name}.png"), m * 255)
        cv2.imwrite(str(out / "depth" / f"{name}.png"), dmat)
        meta.append({"name": name, "nat_idx": i, "ts_ms": t,
                     "donor_kf": int(k_near), "kf_dt_ms": int(dt_kf),
                     "is_kf": bool(t in set(kf_ts.tolist())),
                     "depth_dt_ms": int(abs(int(dep_ts[j]) - t)),
                     "depth_dtype": str(dmat.dtype), "pose": poses[row].tolist()})
        n_ok += 1
    (out / "meta.json").write_text(json.dumps(
        {"root": str(root), "tree": tree, "kf_sightings": kf_hits,
         "frames": meta}, indent=1))
    d_kf = sum(1 for r in meta if r["is_kf"])
    print(f"[prep] wrote {n_ok} frames ({d_kf} are keyframes) -> {out}")
    if meta:
        dm = [r["depth_dt_ms"] for r in meta]
        print(f"[prep] depth ts offset ms: med {np.median(dm):.0f} "
              f"max {max(dm)}   depth dtype {meta[0]['depth_dtype']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
