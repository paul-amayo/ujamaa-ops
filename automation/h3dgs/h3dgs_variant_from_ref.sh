#!/bin/bash
# Build a sibling H3DGS project that REUSES a source project's export, depth maps, fixed-pose SfM (prior),
# global-BA model (aligned) and scaffold via hardlinks/copies, so a recipe variant only re-runs the chunk stage
# and the chunk training. Then runs the survey script on it with whatever H3DGS_* env the caller sets, and
# scores the chunk(s) under test on their own held-out views.
#   usage: h3dgs_variant_from_ref.sh <survey_root> <src_proj> <dst_proj> <chunk>     (env: H3DGS_CHUNK_BA, H3DGS_ONLY_CHUNKS ...)
S=${1:?survey}; SRC=${2:?src proj}; DST=${3:?dst proj}; C=${4:?chunk}
L=/home/paperspace/logs/h3dgs_variants.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/logs/h3dgs_bin:$PATH; PY=/home/paperspace/miniconda3/envs/h3dgs/bin/python
say "=== variant $(basename $DST) from $(basename $SRC) (chunk $C; CHUNK_BA=${H3DGS_CHUNK_BA:-0} SKIP_GLOBAL_BA=${H3DGS_SKIP_GLOBAL_BA:-0} VARIANT=${H3DGS_BLOCK_VARIANT:-})"
[ -e $DST/export_meta.json ] || $PY /home/paperspace/logs/h3dgs_export.py $S $DST 2>&1 | tail -1 | tee -a $L
CC=$DST/camera_calibration; SC=$SRC/camera_calibration; mkdir -p $CC/rectified $DST/output/scaffold
[ -e $CC/rectified/depths ] || cp -al $SC/rectified/depths $CC/rectified/depths
[ -e $CC/prior ] || cp -a $SC/prior $CC/prior
if [ "${H3DGS_SKIP_GLOBAL_BA:-0}" != 1 ]; then [ -e $CC/aligned ] || cp -a $SC/aligned $CC/aligned; fi
[ -e $DST/output/scaffold/point_cloud ] || { [ "${H3DGS_SKIP_GLOBAL_BA:-0}" != 1 ] && cp -al $SRC/output/scaffold/point_cloud $DST/output/scaffold/point_cloud; }
H3DGS_ONLY_CHUNKS="$C" /home/paperspace/logs/h3dgs_survey.sh $S $DST > /home/paperspace/logs/h3dgs_survey_$(basename $DST).out 2>&1
say "survey rc=$? ($(grep -aE 'train chunk|post-opt chunk|FAILED' /home/paperspace/logs/h3dgs_survey_$(basename $DST).out | tail -2 | cut -c18-70 | tr '\n' ';'))"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 900 $PY /home/paperspace/logs/h3dgs_eval_chunk.py $DST --hier output/trained_chunks/$C/hierarchy.hier_opt --taus 0 --only_chunk $C --out output/eval_chunk_$C --save 12 2>&1 | grep -aE "^\[eval\] tau|Error" | cut -c1-260 | tee -a $L
say "VARIANT $(basename $DST) DONE"
