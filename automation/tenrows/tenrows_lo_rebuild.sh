#!/bin/bash
# tenrows_lo_rebuild.sh — rebuild ten_rows on the LiDAR-odometry pose stream with the 20 cm-OR-3 deg keyframe rule
# (Paul, 2026-09-26: purge the ZED-odometry products; oversample the turns; take 3 deg). Stages (each idempotent, marker
# files under prod/monos/monolithics/LO_REBUILD/):
#   1 swap   the survey's "current" pose stream, keyframe index, keyframe monolithic, keyframe PNGs, sky masks and block
#            config are moved to *_zedkf names and the LO versions take their canonical names (transform_lio.monolithic,
#            kf_index.json, image_left_kf20cm.monolithic, prod/scratch_sam3, prod/tassili/sky_masks, blocks_ns/lio_row100);
#            the running h3dgs_lo project keeps its old masks through its export_meta.json path
#   2 sky    build_sky_masks.py on the new keyframes (SAM3 env)
#   3 blocks build_row_blocks.py --outcfg-name lio_row100 --max-kf 100 (poses from the LO stream via kf_pose_records)
#   4 refine per-block GPU SIFT + exhaustive + GLOMAP (fixed ZED-conf camera), ORIENTATION-aware Sim(3) onto the LO odometry
#            -> block_NNN/transforms_ref_lo.json (klapmuts_refine_cam.sh + klapmuts_apply_refine_orient.py)
#   5 inits  lidar_init_per_block.py --force per block (LaserProjector on the LO stream)
# Then the H3DGS survey: H3DGS_TAG=ten_rows_lo3 H3DGS_TRANSFORMS_NAME=transforms_ref_lo.json H3DGS_INTRINSICS=<ZED conf>
# H3DGS_LIDAR_SEED=1 bash h3dgs_survey.sh <survey> <survey>/experimental/h3dgs_lo3 (chunk of block_010 first).
set -u
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows; M=$R/prod/monos; MD=$M/monolithics; T=$R/prod/tassili; MK=$MD/LO_REBUILD; mkdir -p $MK $R/_logs
SRC=/home/paperspace/code/aru_sil_core/src/scripts; PY=/home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10; SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
export PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib:/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-3.10/lib
unset LD_LIBRARY_PATH LD_PRELOAD CUDA_HOME
L=/home/paperspace/logs/tenrows_lo_rebuild.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== LO rebuild start"
# 1. swap
if [ ! -e $MK/swap.done ]; then
  for f in image_left_kf20cm_lo.monolithic kf_index_lo.json transform_lo.monolithic; do [ -e $MD/$f ] || { say "missing $MD/$f — run tenrows_lo_kfcut_A/B first"; exit 1; }; done
  [ -d $R/prod/scratch_sam3_lo ] || { say "missing prod/scratch_sam3_lo"; exit 1; }
  mv $MD/transform_lio.monolithic $MD/transform_lio_zedkf.monolithic; rm -f $MD/transform_lio.monolithic.index; cp $MD/transform_lo.monolithic $MD/transform_lio.monolithic
  mv $MD/kf_index.json $MD/kf_index_zedkf.json; cp $MD/kf_index_lo.json $MD/kf_index.json
  mv $MD/image_left_kf20cm.monolithic $MD/image_left_kf20cm_zedkf.monolithic; [ -e $MD/image_left_kf20cm.monolithic.index ] && mv $MD/image_left_kf20cm.monolithic.index $MD/image_left_kf20cm_zedkf.monolithic.index
  cp $MD/image_left_kf20cm_lo.monolithic $MD/image_left_kf20cm.monolithic; rm -f $MD/image_left_kf20cm.monolithic.index
  mv $R/prod/scratch_sam3 $R/prod/scratch_sam3_zedkf; mv $R/prod/scratch_sam3_lo $R/prod/scratch_sam3
  mv $T/sky_masks $T/sky_masks_zedkf; mkdir -p $T/sky_masks
  for meta in $R/experimental/h3dgs_lo/export_meta.json; do [ -e $meta ] && $PY - "$meta" <<'EOF'
import json, sys; p = sys.argv[1]; m = json.load(open(p)); m["sky_masks"] = m["sky_masks"].replace("/sky_masks", "/sky_masks_zedkf"); json.dump(m, open(p, "w"), indent=1); print("[swap] patched", p, "->", m["sky_masks"])
EOF
  done
  mv $T/blocks_ns/lio_row100 $T/blocks_ns/lio_row100_zedkf
  touch $MK/swap.done; say "swap done: transform_lio=LO stream, kf_index/kf monolithic/scratch_sam3 = 20cm-or-3deg keyframes ($(ls $R/prod/scratch_sam3 | wc -l) PNGs); old set kept under *_zedkf"
fi
# 2. sky masks
if [ ! -e $MK/sky.done ]; then
  t0=$(date +%s); $SAM3_PY $SRC/build_sky_masks.py --data-dir $R > $R/_logs/ten_rows_sky_masks_lo.log 2>&1 || { say "SKY MASKS FAILED ($(tail -1 $R/_logs/ten_rows_sky_masks_lo.log | cut -c1-120))"; exit 1; }
  touch $MK/sky.done; say "sky masks in $(( $(date +%s)-t0 ))s: $(ls $T/sky_masks | wc -l) files"
fi
# 3. blocks
if [ ! -e $MK/blocks.done ]; then
  t0=$(date +%s); (cd $SRC && $PY build_row_blocks.py --data-dir $R --outcfg-name lio_row100 --max-kf 100) > $R/_logs/ten_rows_blocks_lo.log 2>&1 || { say "BLOCKS FAILED ($(tail -1 $R/_logs/ten_rows_blocks_lo.log | cut -c1-120))"; exit 1; }
  touch $MK/blocks.done; say "blocks in $(( $(date +%s)-t0 ))s: $(ls -d $T/blocks_ns/lio_row100/block_[0-9][0-9][0-9] | wc -l) blocks"
fi
# 4. refine (GPU) — orientation-aware Sim(3) onto the LO odometry
if [ ! -e $MK/refine.done ]; then
  t0=$(date +%s); CAM_PARAMS="527.985,527.88,638.975,333.1835" APPLY_SCRIPT=/home/paperspace/logs/klapmuts_apply_refine_orient.py bash /home/paperspace/logs/klapmuts_refine_cam.sh $R lo > $R/_logs/ten_rows_refine_lo.log 2>&1
  n=$(ls $T/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/transforms_ref_lo.json 2>/dev/null | wc -l); nb=$(ls -d $T/blocks_ns/lio_row100/block_[0-9][0-9][0-9] | wc -l)
  [ "$n" -ge $(( nb * 9 / 10 )) ] || { say "REFINE: only $n/$nb blocks written"; exit 1; }
  touch $MK/refine.done; say "refine in $(( $(date +%s)-t0 ))s: $n/$nb blocks -> transforms_ref_lo.json; $(grep -h "apply-orient" $R/_logs/ten_rows_refine_lo.log /home/paperspace/logs/klapmuts_refine_lo_dec_2025_ten_rows.log 2>/dev/null | grep -oE 'orientation residual median [0-9.]+' | awk '{s+=$4;n++} END{if(n) printf "mean orientation residual %.2f deg over %d blocks", s/n, n}')"
fi
# 5. LiDAR inits (LaserProjector on the LO stream)
if [ ! -e $MK/inits.done ]; then
  t0=$(date +%s); ok=0
  for BD in $T/blocks_ns/lio_row100/block_[0-9][0-9][0-9]; do
    (cd $SRC && pixi run --manifest-path /home/paperspace/code/InstantSplat/pixi.toml python lidar_init_per_block.py --block-dir "$BD" --root "$R" --force) >> $R/_logs/ten_rows_lidar_init_lo.log 2>&1 && ok=$((ok+1)) || say "   INIT FAIL $(basename $BD)"
  done
  touch $MK/inits.done; say "inits in $(( $(date +%s)-t0 ))s: $ok blocks"
fi
say "=== LO rebuild DONE — next: H3DGS_TAG=ten_rows_lo3 H3DGS_TRANSFORMS_NAME=transforms_ref_lo.json H3DGS_INTRINSICS=527.985,527.88,638.975,333.1835 H3DGS_LIDAR_SEED=1 H3DGS_STOP_AFTER_CHUNKS=1 bash h3dgs_survey.sh $R $R/experimental/h3dgs_lo3"
