#!/bin/bash
# ten_rows colour rebuild (Paul 2026-09-25: "just clear the entire ten rows outputs and regen").
# CORRECTED 16:5x: the monolithics are RIGHT (cv2-native BGR arrays, verified against the bag); the swap entered in the
# 09-05 keyframe PNG dump (aru_nerf_interface reader returns RGB, cv2.imwrite expects BGR). So: no re-ingest; regenerate
# the keyframe PNGs from the existing kf20cm stream with the canonical extract_kf_pngs.py (aru_py_logger reader ->
# imwrite = correct), then everything downstream of the PNGs. Stages (idempotent, logged):
#   1. discard the RGB-array re-ingest (monolithics_rgbfix) and verify the existing stream through the PNG path
#   2. keyframe PNGs (prod/scratch_sam3 cleared and rewritten; old ones quarantined)
#   3. sky masks (SAM3) on the corrected keyframes
#   4. LiDAR inits per block (colours come from the images), canonical + _ref dirs
#   5. when the image_farm fleet has finished: the H3DGS survey on the ZED-conf poses/camera with the LiDAR seed
set -uo pipefail
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows; M=$R/prod/monos; MD=$M/monolithics; SRC=/home/paperspace/code/aru_sil_core/src/scripts
PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10
SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
L=/home/paperspace/logs/tenrows_rgbfix_chain.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] RGBFIX $*" | tee -a $L; }
export PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib:/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib
# ---- 1. discard the wrong-convention re-ingest; verify the existing stream through the PNG path
if [ -d $M/monolithics_rgbfix ]; then say "1. discarding monolithics_rgbfix (RGB arrays = wrong convention)"; rm -rf $M/monolithics_rgbfix; fi
env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY - $MD << 'PY' 2>&1 | grep -E "\[check\]" | tee -a $L
import sys, cv2, numpy as np; sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/lib"); sys.path.insert(0, "/home/paperspace/code/aru_sil_core/build/datatypes")
import aru_py_logger
lg = aru_py_logger.MonoImageLogger(sys.argv[1] + "/image_left_kf20cm.monolithic", False); n = 0
while not lg.end_of_file() and n < 301: img, ts = lg.read_from_file(); n += 1
cv2.imwrite("/tmp/rgbfix_check.png", img); png = cv2.cvtColor(cv2.imread("/tmp/rgbfix_check.png"), cv2.COLOR_BGR2RGB); t = png[: png.shape[0] // 6].reshape(-1, 3).mean(0)
print(f"[check] kf20cm entry 300 through extract's PNG path: sky RGB {tuple(int(x) for x in t)} -> {'OK (R<B)' if t[0] < t[2] else 'SWAPPED'}"); assert t[0] < t[2]
PY
[ ${PIPESTATUS[0]} -eq 0 ] || { say "stream check FAILED"; exit 1; }
# ---- 2. keyframe PNGs from the kf20cm stream (canonical tool)
if [ ! -e $R/prod/scratch_sam3/KF_REGEN_OK ]; then
  say "2. keyframe PNGs (old scratch_sam3 -> prod/scratch_sam3_bgrswapped)"
  [ -d $R/prod/scratch_sam3 ] && [ ! -d $R/prod/scratch_sam3_bgrswapped ] && mv $R/prod/scratch_sam3 $R/prod/scratch_sam3_bgrswapped
  mkdir -p $R/prod/scratch_sam3
  env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY $SRC/extract_kf_pngs.py --data-dir $R --mono $MD/image_left_kf20cm.monolithic --out-dir $R/prod/scratch_sam3 > $R/_logs/ten_rows_kf_pngs_rgbfix.log 2>&1 || { say "PNG extraction FAILED"; exit 1; }
  N=$(ls $R/prod/scratch_sam3 | grep -c png); [ "$N" -ge 1990 ] && touch $R/prod/scratch_sam3/KF_REGEN_OK || { say "only $N PNGs"; exit 1; }
  env -u LD_LIBRARY_PATH -u LD_PRELOAD $PY -c "
import cv2; png = cv2.cvtColor(cv2.imread('$R/prod/scratch_sam3/kf_000300.png'), cv2.COLOR_BGR2RGB); t = png[: png.shape[0] // 6].reshape(-1, 3).mean(0); print('[check] new kf_000300.png sky RGB', tuple(int(x) for x in t), 'OK' if t[0] < t[2] else 'SWAPPED')" | tee -a $L
  say "   $N keyframe PNGs written"
fi
# ---- 3. sky masks
if [ ! -e $R/prod/tassili/sky_masks/SKY_REGEN_OK ]; then
  say "3. sky masks (old -> prod/tassili/sky_masks_bgrswapped)"; [ -d $R/prod/tassili/sky_masks ] && [ ! -d $R/prod/tassili/sky_masks_bgrswapped ] && mv $R/prod/tassili/sky_masks $R/prod/tassili/sky_masks_bgrswapped
  $SAM3_PY $SRC/build_sky_masks.py --data-dir $R > $R/_logs/ten_rows_sky_masks_rgbfix.log 2>&1 && touch $R/prod/tassili/sky_masks/SKY_REGEN_OK || { say "SKY MASKS FAILED"; exit 1; }
  say "   sky masks: $(ls $R/prod/tassili/sky_masks | grep -c png) PNGs"
fi
# ---- 4. LiDAR inits (colours from the images)
if [ ! -e $R/prod/tassili/LIDAR_INIT_RGBFIX_OK ]; then
  say "4. LiDAR inits per block (canonical + _ref)"; cd $SRC
  for BD in $R/prod/tassili/blocks_ns/lio_row100/block_0?? $R/prod/tassili/blocks_ns/lio_row100/block_0??_ref; do
    pixi run --manifest-path /home/paperspace/code/InstantSplat/pixi.toml python lidar_init_per_block.py --block-dir "$BD" --root "$R" --force >> $R/_logs/ten_rows_lidar_init_rgbfix.log 2>&1 && echo "   init ok $(basename $BD)" >> $L || say "   INIT FAIL $(basename $BD)"
  done
  touch $R/prod/tassili/LIDAR_INIT_RGBFIX_OK; say "   inits done: $(grep -c 'init ok' $L)"
fi
# ---- 5. H3DGS survey on the ZED-conf poses (waits for the image_farm fleet)
while pgrep -f "^bash /home/paperspace/logs/image_farm_fleet.sh" > /dev/null || pgrep -f "^bash /home/paperspace/logs/image_farm_fleet_rerun.sh" > /dev/null; do sleep 120; done
N_ZED=$(ls $R/prod/tassili/blocks_ns/lio_row100/block_0??/transforms_ref_zedconf.json 2>/dev/null | wc -l); say "5. fleet finished; $N_ZED blocks have transforms_ref_zedconf.json"
[ "$N_ZED" -ge 22 ] || { say "too few ZED-conf refined blocks — survey not launched"; exit 1; }
PROJ=$R/experimental/h3dgs_rgb_zed
say "   H3DGS survey -> $PROJ (transforms_ref_zedconf.json, camera 527.985,527.88,638.975,333.1835, LiDAR seed)"
H3DGS_TRANSFORMS_NAME=transforms_ref_zedconf.json H3DGS_INTRINSICS="527.985,527.88,638.975,333.1835" H3DGS_LIDAR_SEED=1 \
  bash /home/paperspace/logs/h3dgs_survey.sh $R $PROJ > /home/paperspace/logs/h3dgs_survey_h3dgs_rgb_zed.out 2>&1
say "   survey rc=$? : $(grep -E '\[eval\] tau 3|SURVEY|DONE' /home/paperspace/logs/h3dgs_survey_h3dgs_rgb_zed.out | tail -2 | tr '\n' ' ')"
say "CHAIN DONE"
