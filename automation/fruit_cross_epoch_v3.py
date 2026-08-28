#!/usr/bin/env python3
"""Cross-epoch fruit v3 — join 3D-DEDUPLICATED counts across 04/05.

v1 (raw detection-events) and v2 (view-qualified) both failed to reconcile
epochs (57% presence agreement) because events count sightings, not fruit.
v3 joins the corroborated physical count from the 10 fps 3D dedup
(fruit-pixel-depth tier, eps 8 cm, clusters seen >=2x) over the same
assoc_04_05_v4 matched pairs. Writes fruit_cross_epoch_04_05_v3.json.
"""
import json
from pathlib import Path

import numpy as np

SUB = Path("/home/paperspace/data/citrus_all/sankofa_substrate")
ROOTS = {"04": Path("/home/paperspace/data/citrus_all/04_13D_Jackal"),
         "05": Path("/home/paperspace/data/citrus_all/05_13D_Jackal")}
EPS = "0.08"


def counts(root):
    out = {}
    for w in sorted((root / "prod/scratch_sam3").glob("fruit3d_t*/dedup3d.json")):
        d = json.loads(w.read_text())
        tier = d.get("per_eps_10fps_fruitpx", {}).get(EPS)
        out[d["tree"]] = {
            "corroborated": tier["multi"] if tier else 0,
            "clusters": tier["clusters"] if tier else 0,
            "events": d.get("n_events_10fps", 0)}
    return out


c04, c05 = counts(ROOTS["04"]), counts(ROOTS["05"])
print(f"[v3] dedup workdirs: 04 has {len(c04)} trees, 05 has {len(c05)}")
z = np.load(SUB / "assoc_04_05_v4.npz", allow_pickle=True)
rows = []
for p in z["pairs"]:
    t04, t05 = int(p[0]), int(p[1])
    a = c04.get(t04, {"corroborated": 0, "clusters": 0, "events": 0})
    b = c05.get(t05, {"corroborated": 0, "clusters": 0, "events": 0})
    rows.append({"tree_04": t04, "tree_05": t05, "match_m": round(float(p[2]), 3),
                 "fruit3d_04": a["corroborated"], "fruit3d_05": b["corroborated"],
                 "clusters_04": a["clusters"], "clusters_05": b["clusters"]})

n = len(rows)
a = np.array([r["fruit3d_04"] for r in rows], float)
b = np.array([r["fruit3d_05"] for r in rows], float)
both = int(((a > 0) & (b > 0)).sum())
only4 = int(((a > 0) & (b == 0)).sum())
only5 = int(((a == 0) & (b > 0)).sum())
neither = n - both - only4 - only5
agree = (both + neither) / n
rho = np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1]
m = (a > 0) & (b > 0)
rho_pos = (np.corrcoef(a[m].argsort().argsort(),
                       b[m].argsort().argsort())[0, 1]
           if m.sum() > 3 else float("nan"))
print(f"[v3] pairs {n}: both {both}, only04 {only4}, only05 {only5}, "
      f"neither {neither}")
print(f"[v3] presence agreement {agree*100:.1f}%  (v1 raw: 57%)")
print(f"[v3] Spearman all {rho:+.3f}, both-bearing({int(m.sum())}) {rho_pos:+.3f}")

out = {"schema": "fruit_cross_epoch/v3-3d-dedup",
       "eps_m": float(EPS), "tier": "fruit-pixel depth, sightings>=2",
       "n_pairs": n, "both": both, "only_04": only4, "only_05": only5,
       "presence_agreement": round(agree, 3),
       "spearman_all": round(float(rho), 3),
       "spearman_both_bearing": round(float(rho_pos), 3),
       "pairs": rows}
(SUB / "fruit_cross_epoch_04_05_v3.json").write_text(json.dumps(out, indent=1))
print(f"[v3] wrote {SUB/'fruit_cross_epoch_04_05_v3.json'}")
