#!/bin/bash
# Full rebuild on the roll-corrected pose stream: promote stream -> blocks (GL c2w) ->
# LiDAR inits re-lifted -> A/B arm -> split paths -> relaunch the stage-1 driver.
LOG=/home/paperspace/logs/tenrows_rebuild_rollfix.log; : > $LOG
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10
PP=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib
MD=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/monos/monolithics
B=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
cp $MD/transform_lio_laserframe_rollfix_inscorr.monolithic $MD/transform_lio.monolithic && rm -f $MD/transform_lio.monolithic.index && log "transform_lio.monolithic := roll-corrected + INS-corrected laser-frame stream ($(stat -c %s $MD/transform_lio.monolithic) B)"
for d in $B/block_0??; do rm -rf $d/splat_runs_STAGE1 $d/init_lidar.ply $d/stage1_eval.json; done; log "old block runs/inits cleared (diagnostic arms block_013_{gl,fix,colmap} kept)"
cd /home/paperspace/code/aru_sil_core/src/scripts && PYTHONPATH=$PP env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY -c "
from PIL import Image
import sys, runpy; sys.argv = ['build_row_blocks.py', '--data-dir', '/home/paperspace/data/klapmuts/dec_2025_ten_rows', '--outcfg-name', 'lio_row100', '--max-kf', '100']
runpy.run_path('/home/paperspace/code/aru_sil_core/src/scripts/build_row_blocks.py', run_name='__main__')" >> $LOG 2>&1 || { log "BLOCKS FAILED"; exit 1; }
log "blocks: $(ls -d $B/block_0?? | wc -l); $(grep -o '"pose_convention": "[a-z_]*"' $B/block_000/transforms.json)"
/home/paperspace/logs/tenrows_init_all.sh 2>&1 | grep -E "INIT" | tee -a $LOG
grep -E "LIDAR-init" /home/paperspace/logs/tenrows_init_all.log | awk '{gsub(",","",$9); print $9}' | sort -n | awk '{a[NR]=$1} END {print "[init] voxel points/block: min", a[1], "median", a[int((NR+1)/2)], "max", a[NR]}' | tee -a $LOG
cd /home/paperspace/logs && PYTHONPATH=$PP env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY tenrows_ab_build.py block_013 2>&1 | grep -E "\[ab\]|Error|assert" | tee -a $LOG
$PY - << 'PYX' | tee -a $LOG
import json
from pathlib import Path
B = Path("/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili/blocks_ns/lio_row100")
for arm in ("block_013", "block_013_full"):
    p = B/arm/"transforms.json"; t = json.loads(p.read_text()); by = {Path(f["file_path"]).name: f["file_path"] for f in t["frames"]}
    for k in ("train_filenames", "val_filenames", "test_filenames"): t[k] = [by[Path(x).name] for x in t[k]]
    p.write_text(json.dumps(t, indent=2)); print(f"[split] {arm}: train {len(t['train_filenames'])} test {len(t['test_filenames'])} (absolute)")
PYX
cp $B/block_013/init_lidar.ply $B/block_013_full/init_lidar.ply; rm -rf $B/block_013_full/splat_runs_STAGE1 $B/block_013_full/stage1_eval.json
log "relaunching stage-1 driver (A/B pair first, then the other 22 blocks)"
setsid nohup /home/paperspace/logs/tenrows_stage1_driver.sh > /home/paperspace/logs/tenrows_stage1_driver.out 2>&1 < /dev/null &
log "REBUILD-ROLLFIX DONE"
