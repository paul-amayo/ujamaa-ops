#!/bin/bash
# LiDAR-seeded chunk variant: clone a source project's export/prior/aligned/scaffold (hardlinks), cut the chunk under
# test, replace its COLMAP seed points with the survey's LiDAR points (h3dgs_lidar_init.py), train + post-opt the
# chunk and score it on its own held-out views.
#   usage: h3dgs_lidar_variant.sh <survey_root> <src_proj> <dst_proj> <chunk> [lidar-init args...]   (env: H3DGS_BLOCK_VARIANT, H3DGS_TRAIN_EXTRA ...)
S=${1:?survey}; SRC=${2:?src}; DST=${3:?dst}; C=${4:?chunk}; shift 4; LARGS="$*"
L=/home/paperspace/logs/h3dgs_variants.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
say "=== LiDAR-init variant $(basename $DST) from $(basename $SRC), chunk $C ($LARGS)"
[ -e $DST/export_meta.json ] || $PY /home/paperspace/logs/h3dgs_export.py $S $DST 2>&1 | tail -1 | tee -a $L
CC=$DST/camera_calibration; SC=$SRC/camera_calibration; mkdir -p $CC/rectified $DST/output/scaffold
[ -e $CC/rectified/depths ] || cp -al $SC/rectified/depths $CC/rectified/depths
[ -e $CC/prior ] || cp -a $SC/prior $CC/prior
[ -e $CC/aligned ] || cp -a $SC/aligned $CC/aligned
[ -e $DST/output/scaffold/point_cloud ] || cp -al $SRC/output/scaffold/point_cloud $DST/output/scaffold/point_cloud
H3DGS_STOP_AFTER_CHUNKS=1 H3DGS_ONLY_CHUNKS="$C" /home/paperspace/logs/h3dgs_survey.sh $S $DST > /home/paperspace/logs/h3dgs_survey_$(basename $DST).out 2>&1
[ -e $CC/chunks/$C/sparse/0/points3D.ply ] || { say "chunk stage failed"; exit 1; }
$PY /home/paperspace/logs/h3dgs_lidar_init.py $S $DST $C $LARGS 2>&1 | tee -a $L
H3DGS_ONLY_CHUNKS="$C" /home/paperspace/logs/h3dgs_survey.sh $S $DST >> /home/paperspace/logs/h3dgs_survey_$(basename $DST).out 2>&1
say "survey rc=$? ($(grep -aE 'train chunk|post-opt chunk|FAILED' /home/paperspace/logs/h3dgs_survey_$(basename $DST).out | tail -2 | cut -c18-70 | tr '\n' ';'))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $DST --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --out output/eval_chunk_$C --save 12 2>&1 | grep -aE "^\[eval\] tau|Error" | cut -c1-260 | tee -a $L
say "VARIANT $(basename $DST) DONE"
