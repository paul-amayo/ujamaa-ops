#!/usr/bin/env bash
# 02_13B DAY BATCH (CPU): verify the redownloaded bag set, resolve the stray
# .tmp, combine into combined.bag. Stops before monolithic ingest (that step
# runs interactively with the rosbag skill once this reports clean).
set -e
D=/home/paperspace/data/citrus_all/02_13B_Jackal
cd "$D"
echo "== bag inventory =="
ls -la *.bag | awk '{print $5, $9}'
echo "== stray tmp files =="
ls -la *.tmp 2>/dev/null || echo "none"
echo "== per-family counts =="
for fam in base mapir odom zed lidar velodyne adk; do
  c=$(ls ${fam}_*.bag 2>/dev/null | wc -l); [ "$c" -gt 0 ] && echo "$fam: $c"
done
echo "== integrity: rosbag reindex check on each bag (python rosbags) =="
python3 - <<'PY'
from pathlib import Path
from rosbags.rosbag1 import Reader
bad = []
for b in sorted(Path(".").glob("*.bag")):
    try:
        with Reader(b) as r:
            n = sum(1 for _ in zip(range(3), r.messages()))
        print(f"OK   {b.name}")
    except Exception as e:
        bad.append(b.name); print(f"BAD  {b.name}: {e}")
print(f"\n{len(bad)} bad bags" if bad else "\nall bags readable")
PY
echo "== combine =="
if [ ! -f combined.bag ]; then
  [ -f combine_bags.py ] || cp /home/paperspace/data/citrus_all/05_13D_Jackal/combine_bags.py .
  python3 combine_bags.py
else
  echo "combined.bag already exists: $(du -h combined.bag | cut -f1)"
fi
echo "== system-python ingest prereqs =="
python3 -c "import numpy, scipy; print('numpy+scipy OK (system python)')"
echo "02 DAY BATCH COMPLETE $(date -u)"
