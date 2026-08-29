#!/usr/bin/env python3
"""Cross-epoch fruit v4 — VIEW-NORMALIZED comparability.

v3 (3D-dedup counts) fixed sighting inflation but left one-sided pairs:
t72(05)=18 vs t73(04)=0 because 04's pass barely saw that tree (2 kf
sightings -> 10 native frames). v4 asks: on pairs where BOTH epochs had a
fair look (>= F native frames in the sighting windows and >= E depth-valid
events), do the epochs finally agree? Also reports fruit-per-100-frames as
an exposure-normalized rate. Writes fruit_cross_epoch_04_05_v4norm.json.
"""
import json
from pathlib import Path

import numpy as np

SUB = Path("/home/paperspace/data/citrus_all/sankofa_substrate")
ROOTS = {"04": Path("/home/paperspace/data/citrus_all/04_13D_Jackal"),
         "05": Path("/home/paperspace/data/citrus_all/05_13D_Jackal")}
EPS = "0.08"


def per_tree(root):
    out = {}
    for w in sorted((root / "prod/scratch_sam3").glob("fruit3d_t*")):
        dd = w / "dedup3d.json"
        meta = w / "meta.json"
        if not dd.exists() or not meta.exists():
            continue
        d = json.loads(dd.read_text())
        m = json.loads(meta.read_text())
        tier = d.get("per_eps_10fps_fruitpx", {}).get(EPS)
        out[d["tree"]] = {
            "frames": len(m.get("frames", [])),
            "events": d.get("n_events_10fps", 0),
            "corroborated": tier["multi"] if tier else 0}
    return out


c04, c05 = per_tree(ROOTS["04"]), per_tree(ROOTS["05"])
z = np.load(SUB / "assoc_04_05_v4.npz", allow_pickle=True)
pairs = [(int(p[0]), int(p[1])) for p in z["pairs"]]

rows = []
for a, b in pairs:
    r4 = c04.get(a, {"frames": 0, "events": 0, "corroborated": 0})
    r5 = c05.get(b, {"frames": 0, "events": 0, "corroborated": 0})
    rows.append({"tree_04": a, "tree_05": b,
                 "frames_04": r4["frames"], "frames_05": r5["frames"],
                 "events_04": r4["events"], "events_05": r5["events"],
                 "fruit_04": r4["corroborated"], "fruit_05": r5["corroborated"]})


def agree_stats(sel):
    a = np.array([r["fruit_04"] for r in sel], float)
    b = np.array([r["fruit_05"] for r in sel], float)
    n = len(sel)
    both = int(((a > 0) & (b > 0)).sum())
    neither = int(((a == 0) & (b == 0)).sum())
    agree = (both + neither) / n if n else float("nan")
    rho = (np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]
           if n > 3 else float("nan"))
    m = (a > 0) & (b > 0)
    rho_b = (np.corrcoef(a[m].argsort().argsort(), b[m].argsort().argsort())[0, 1]
             if m.sum() > 3 else float("nan"))
    return {"n": n, "both": both, "neither": neither,
            "one_sided": n - both - neither,
            "agreement": round(float(agree), 3),
            "spearman": round(float(rho), 3),
            "spearman_both": round(float(rho_b), 3) if rho_b == rho_b else None}


print(f"[v4] {len(rows)} pairs")
tiers = {}
for name, sel in [
        ("all", rows),
        ("viewed_F20", [r for r in rows if r["frames_04"] >= 20 and r["frames_05"] >= 20]),
        ("viewed_F40", [r for r in rows if r["frames_04"] >= 40 and r["frames_05"] >= 40]),
        ("evented_E20", [r for r in rows if r["events_04"] >= 20 and r["events_05"] >= 20])]:
    s = agree_stats(sel) if sel else None
    tiers[name] = s
    print(f"  {name:<12} {s}")

# exposure-normalized rate on the well-viewed subset
sel = [r for r in rows if r["frames_04"] >= 20 and r["frames_05"] >= 20]
for r in sel:
    r["rate_04"] = round(100.0 * r["fruit_04"] / r["frames_04"], 2)
    r["rate_05"] = round(100.0 * r["fruit_05"] / r["frames_05"], 2)
if len(sel) > 3:
    ra = np.array([r["rate_04"] for r in sel])
    rb = np.array([r["rate_05"] for r in sel])
    rr = np.corrcoef(ra.argsort().argsort(), rb.argsort().argsort())[0, 1]
    print(f"  fruit/100frames Spearman on viewed_F20 ({len(sel)}): {rr:+.3f}")
    tiers["rate_spearman_F20"] = round(float(rr), 3)

out = {"schema": "fruit_cross_epoch/v4-viewnorm", "eps_m": float(EPS),
       "tiers": tiers, "pairs": rows}
(SUB / "fruit_cross_epoch_04_05_v4norm.json").write_text(json.dumps(out, indent=1))
print(f"[v4] wrote {SUB/'fruit_cross_epoch_04_05_v4norm.json'}")
