#!/bin/bash
# dec_2025_ten_rows post-ingest chain: frame-fix -> (no symlinks) -> accumulate -> top-down.
# Runs the frame conversion in its OWN process (pbTransform_pb2 clashes with the
# C++ binding), then the accumulator against the laserframe file EXPLICITLY.
set -uo pipefail
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows
M=$R/prod/monos
MD=$M/monolithics
OUTDIR=$R/prod/tassili
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10
SRC=/home/paperspace/code/aru_sil_core/src/scripts
LOGS=$R/_logs
say(){ echo "[$(date '+%H:%M:%S')] POST $*"; }
mkdir -p $OUTDIR

# 0. inputs present (a missing input is indistinguishable from an empty one downstream)
for f in laser.monolithic image_left.monolithic zed_transform.monolithic; do
  [ -s "$MD/$f" ] || { say "MISSING $MD/$f"; exit 1; }
done
say "streams: $(ls -la $MD | grep -cE '\.monolithic$') monolithics present"

# 1. frame fix — dry-run first so the detection is on record, then the real write
say "frame-fix DRY RUN (expect: camera-frame, flat axis y)"
env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY $SRC/ensure_laser_frame_poses.py --data-dir $M --dry-run \
  2>&1 | tee $LOGS/ten_rows_laserframe_dryrun.log | grep -E "LASERFRAME"
say "frame-fix REAL"
env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY $SRC/ensure_laser_frame_poses.py --data-dir $M \
  2>&1 | tee $LOGS/ten_rows_laserframe.log | grep -E "LASERFRAME"
[ -s "$MD/transform_lio_laserframe.monolithic" ] || { say "no transform_lio_laserframe.monolithic written"; exit 1; }

# 2. no shims: drop the two transform_lio symlinks the tool creates; we reference
#    the laserframe file explicitly (Paul's no-symlink rule, chosen 2026-09-05)
for l in "$MD/transform_lio.monolithic" "$M/transform_lio.monolithic"; do
  if [ -L "$l" ]; then say "removing symlink $l -> $(readlink "$l")"; rm -f "$l"; fi
done
# a stale sub-64B index left beside the new mono would read as empty — clear it
find $MD -name "transform_lio_laserframe.monolithic.index" -size -64c -delete 2>/dev/null || true

# 3. accumulate over the FULL image stream (all 3,943 scans reachable)
say "accumulate (rmax 40 m, voxel 0.05, stride 1)"
export PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib:/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib
env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY /home/paperspace/logs/tenrows_laser_cloud.py \
  --monos $MD --transform transform_lio_laserframe.monolithic \
  --rmax 40 --voxel 0.05 --stride 1 --out $OUTDIR/ten_rows_laser_cloud \
  2>&1 | tee $LOGS/ten_rows_laser_cloud.log | grep -E "\[cloud\]"
[ -s "$OUTDIR/ten_rows_laser_cloud.ply" ] || { say "accumulate produced no PLY"; exit 1; }

# 4. top-down render
say "top-down render"
env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY /home/paperspace/logs/tenrows_topdown.py \
  --npz $OUTDIR/ten_rows_laser_cloud.npz --out $OUTDIR/ten_rows_laser_topdown.png 2>&1 | tail -3
say "POST DONE"
