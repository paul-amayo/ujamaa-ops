#!/usr/bin/env python3
"""Attach 3D-deduplicated fruit counts to a survey's marker hierarchy.

For each fruit node {id, tree_id, n_detections} adds:
  n_fruit3d    corroborated physical fruit (fruit-px depth, eps 8 cm, >=2 sightings)
  n_clusters3d all 3D clusters at that eps (corroborated + singletons)
n_detections (sighting-events) is kept for provenance. Idempotent; one-time
backup to marker_hierarchy.json.pre_fruit3d. Usage: attach_fruit3d_counts.py <root>
"""
import json
import sys
from pathlib import Path

EPS = "0.08"
root = Path(sys.argv[1])
hj = root / "prod/bateleur/scene_graph/marker_hierarchy.json"
h = json.loads(hj.read_text())

dedup = {}
for p in sorted((root / "prod/scratch_sam3").glob("fruit3d_t*/dedup3d.json")):
    d = json.loads(p.read_text())
    tier = d.get("per_eps_10fps_fruitpx", {}).get(EPS)
    dedup[d["tree"]] = {"n_fruit3d": tier["multi"] if tier else 0,
                        "n_clusters3d": tier["clusters"] if tier else 0}

if not dedup:
    sys.exit(f"[attach] {root.name}: no dedup3d.json found — nothing to attach")

bak = hj.with_suffix(".json.pre_fruit3d")
if not bak.exists():
    bak.write_text(hj.read_text())

n = 0
for f in h.get("fruits", []):
    d = dedup.get(f["tree_id"])
    if d is None:
        continue
    f.update(d)
    n += 1
hj.write_text(json.dumps(h, indent=1))
tot = sum(f.get("n_fruit3d", 0) for f in h.get("fruits", []))
bearing = sum(1 for f in h.get("fruits", []) if f.get("n_fruit3d", 0) > 0)
print(f"[attach] {root.name}: {n}/{len(h.get('fruits', []))} fruit nodes got 3D "
      f"counts -> {tot} corroborated fruit on {bearing} trees")
