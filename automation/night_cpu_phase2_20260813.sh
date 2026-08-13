#!/bin/bash
# Night CPU phase 2 (Paul: "what can cpu do for the remaining 9.5 hours") —
# the Q4 recluster is CPU-bound (SAM3 cached; lift+voxel-CC on CPU), made
# NON-DESTRUCTIVE via a symlinked workspace: sam3_v2_q4/ links clips.json +
# clip_* from sam3_v2, cluster writes ITS OWN global_ids.json there; the
# live registry is never touched. Then read-only comparison reports, a
# klapmuts 100-kf preview partition, and an 02 hierarchy re-fit probe.
# No rm, no git push, no live-artifact writes.
set -u
LOGS=/home/paperspace/logs
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
mark() { echo "[$(date +%H:%M:%S)] $1"; }
exec > "$LOGS/night_cpu2_20260813.log" 2>&1
cd /home/paperspace/code/nerf_new
mark "CPU phase 2 start"

# ---- P1. Q4 non-destructive recluster + comparison (03 first: the 40%
# selection-disagreement survey; then 04, 05) --------------------------------
for S in 03_13B_Jackal 04_13D_Jackal 05_13D_Jackal; do
  ROOT=/home/paperspace/data/citrus_all/$S
  Q4=$ROOT/sam3_v2_q4
  if [ ! -f "$Q4/global_ids.json" ]; then
    mkdir -p "$Q4"
    [ -e "$Q4/clips.json" ] || ln -s "$ROOT/sam3_v2/clips.json" "$Q4/clips.json"
    for C in "$ROOT"/sam3_v2/clip_*; do
      [ -e "$Q4/$(basename "$C")" ] || ln -s "$C" "$Q4/$(basename "$C")"
    done
    pixi run python "$SRC/cluster_tree_instances.py" \
      --data-dir "$ROOT" --out-name sam3_v2_q4 \
      --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 \
      --depth-back 1.5 --max-det-dist 10.0 \
      > "$LOGS/q4_recluster_${S}.log" 2>&1 \
      && mark "P1-DONE recluster $S" || { mark "P1-FAIL recluster $S"; continue; }
  else
    mark "P1-SKIP recluster $S (exists)"
  fi
done

# comparison report: new (mono-K) vs live registry, assoc-grade trees only
pixi run python - << 'EOF' > "$LOGS/q4_recluster_report.txt" 2>&1
import json
import numpy as np
def grade(g):
    out = {}
    for gid, s in g["stats"].items():
        if s["n_lidar_points"] >= 500 and s["n_frame_observations"] >= 5:
            out[int(gid)] = np.array([s["world_centroid"][0], s["world_centroid"][2]])
    return out
for S in ("03_13B_Jackal", "04_13D_Jackal", "05_13D_Jackal"):
    R = f"/home/paperspace/data/citrus_all/{S}"
    try:
        old = grade(json.load(open(f"{R}/sam3_v2/global_ids.json")))
        new = grade(json.load(open(f"{R}/sam3_v2_q4/global_ids.json")))
    except Exception as e:
        print(f"{S}: SKIP ({e})")
        continue
    if not old or not new:
        print(f"{S}: empty side old={len(old)} new={len(new)}")
        continue
    on = np.array(list(old.values())); nn_ = np.array(list(new.values()))
    d = np.linalg.norm(on[:, None] - nn_[None], axis=2)
    j = d.argmin(1); i = d.argmin(0)
    mutual = [(a, j[a], d[a, j[a]]) for a in range(len(on)) if i[j[a]] == a and d[a, j[a]] < 2.0]
    dd = np.array([m[2] for m in mutual]) if mutual else np.array([9.9])
    moved = int((dd > 0.5).sum())
    print(f"{S}: old {len(old)} vs new {len(new)} assoc-grade | mutual<2m {len(mutual)} "
          f"({100*len(mutual)/max(len(old),1):.0f}% of old) | centroid delta median "
          f"{np.median(dd):.3f} m p95 {np.percentile(dd,95):.3f} m | moved>0.5m {moved} | "
          f"old-only {len(old)-len(mutual)}  new-only {len(new)-len(mutual)}")
EOF
grep -q "assoc-grade" "$LOGS/q4_recluster_report.txt" \
    && mark "P1-DONE q4 comparison report" || mark "P1-FAIL q4 report"

# ---- P2. 03 association re-check on the q4 registry (CPU, non-destructive:
# assoc reads global_ids from sam3_v2 — run only if q4-03 differs; report-only
# via a copy of the abs tool pointed at q4 would need a flag, so instead we
# report the registry deltas above and leave assoc to daylight if they move.
mark "P2-NOTE assoc re-check deferred to daylight if P1 deltas are material"

# ---- P3. klapmuts 100-kf preview partition + mask-coverage report ---------
KL=/home/paperspace/data/klapmuts
if [ ! -d "$KL/blocks_ns/lio_row100_preview/block_000" ]; then
  pixi run python "$SRC/build_row_blocks.py" \
    --data-dir "$KL" --outcfg-name lio_row100_preview --max-kf 100 \
    > "$LOGS/klap_row100_preview.log" 2>&1 \
    && mark "P3a-DONE klap preview blocks ($(ls -d "$KL"/blocks_ns/lio_row100_preview/block_* 2>/dev/null | wc -l))" \
    || mark "P3a-FAIL klap preview"
fi
python3 - << 'EOF' >> "$LOGS/klap_row100_preview.log" 2>&1
import glob, json, os
blocks = sorted(glob.glob('/home/paperspace/data/klapmuts/blocks_ns/lio_row100_preview/block_*/transforms.json'))
tot = with_mask = 0
for b in blocks:
    for fr in json.load(open(b))['frames']:
        tot += 1
        mp = fr.get('mask_path')
        with_mask += bool(mp and os.path.exists(mp))
print(f"[preview] {len(blocks)} blocks, {tot} frames, mask coverage {with_mask}/{tot}"
      f" ({100*with_mask/max(tot,1):.1f}%) — re-block is GO only at ~100%")
EOF
mark "P3b-DONE klap mask-coverage report"

# ---- P4. 02 hierarchy re-fit probe (new out path; live hierarchy untouched)
pixi run python "$SRC/build_marker_hierarchy.py" \
  --semantic-monolithic /home/paperspace/data/citrus_all/02_13B_Jackal/filtered_semantic_v2.monolithic \
  --marker-monolithic /home/paperspace/data/citrus_all/02_13B_Jackal/scene_graph/markers_v2.monolithic \
  --out /home/paperspace/data/citrus_all/02_13B_Jackal/scene_graph/marker_hierarchy_refit_probe.json \
  > "$LOGS/02_hier_refit_probe.log" 2>&1 \
  && mark "P4-DONE 02 hierarchy re-fit probe (PCA free): $(python3 -c "
import json
p=json.load(open('/home/paperspace/data/citrus_all/02_13B_Jackal/scene_graph/marker_hierarchy_refit_probe.json')).get('_provenance',{})
print(p.get('n_objects'),'obj /',p.get('n_rows'),'rows')" 2>/dev/null)" \
  || mark "P4-FAIL 02 re-fit probe"

# ---- P5. Dec odom coverage report (read-only wire parse) ------------------
pixi run python - << 'EOF' > "$LOGS/dec_odom_report.txt" 2>&1
import sys
import numpy as np
sys.path.insert(0, '/home/paperspace/code/aru_sil_core/src/scripts')
from kf_domain import _read_increments, _kf_timestamps
from pathlib import Path
M = Path('/home/paperspace/data/klapmuts/dec_2025_a300/monolithics')
inc = _read_increments(M / 'zed_transform.monolithic')
ts = np.array([t for t, _ in inc])
P = np.eye(4); pts = []
for _, m in inc:
    P = P @ m
    pts.append(P[:3, 3].copy())
pts = np.array(pts)
kf_ts = _kf_timestamps(M / 'image_left_kf20cm.monolithic')
print(f"zed odom: {len(inc)} increments, ts span {(ts[-1]-ts[0])/1000:.1f} s")
print(f"trajectory span: x {pts[:,0].ptp():.1f} z {pts[:,2].ptp():.1f} y {pts[:,1].ptp():.1f} (composed raw, frame TBD)")
print(f"path length: {np.linalg.norm(np.diff(pts,axis=0),axis=1).sum():.1f} m (expect ~44 m)")
inside = ((kf_ts >= ts[0]) & (kf_ts <= ts[-1])).sum()
print(f"dec kf ts inside odom span: {inside}/{len(kf_ts)}")
print("NOTE: frame/convention (camera-frame odom? L2C applicability) = daylight work")
EOF
mark "P5-DONE dec odom coverage report"

mark "NIGHT-CPU2-DONE"
