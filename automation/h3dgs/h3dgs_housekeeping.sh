#!/bin/bash
# Reclaim transient H3DGS artefacts in a project dir (safe on a running job: touches only finished chunks).
#   usage: h3dgs_housekeeping.sh <proj_dir>
PROJ=${1:?proj}; CC=$PROJ/camera_calibration; OUT=$PROJ/output; before=$(du -sm $PROJ | cut -f1)
for RC in $(ls $CC/raw_chunks 2>/dev/null); do [ -e $CC/chunks/$RC/sparse/0/images.bin ] && rm -rf $CC/raw_chunks/$RC/bundle_adjustment/images $CC/raw_chunks/$RC/bundle_adjustment/stereo $CC/raw_chunks/$RC/bundle_adjustment/database.db*; done
[ -e $CC/aligned/sparse/0/points3D.bin ] && rm -f $CC/rectified/database.db*
for T in $OUT/trained_chunks/*/; do [ -e $T/hierarchy.hier_opt ] && rm -rf $T/point_cloud $T/hierarchy.hier; done
rm -rf $OUT/trained_chunks_baddepth_* $OUT/trained_chunks_partial_* 2>/dev/null
after=$(du -sm $PROJ | cut -f1); echo "$PROJ: $((before/1024)) GB -> $((after/1024)) GB (reclaimed $(((before-after)/1024)) GB)"
