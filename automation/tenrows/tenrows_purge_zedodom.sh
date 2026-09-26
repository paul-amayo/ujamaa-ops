#!/bin/bash
# tenrows_purge_zedodom.sh — Paul, 2026-09-26 04:3x box: "stop all checkpoints in ten rows that have used zed odom, clear
# them from disk to save space". Every trained splat/hierarchy output of dec_2025_ten_rows so far was placed by the ZED
# visual odometry (directly or via the per-block Sim(3) into that world). Deleted below; NOT deleted: raw data
# (prod/monos), keyframe PNGs, masks, the per-block from-scratch SfM workspaces (colmap_glref_*, reusable), the
# transforms*.json pose files, init_lidar.ply (tiny), the right-camera PNGs, the stereo rig transform, the laser dump +
# KISS-ICP poses (the new pose source), prod/tassili/colmap_block_* SfM workspaces and the laser clouds (not checkpoints).
set -u
R=/home/paperspace/data/klapmuts/dec_2025_ten_rows; B=$R/prod/tassili/blocks_ns/lio_row100
L=/home/paperspace/logs/tenrows_purge_zedodom.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
say "=== purge start: $(df -h /home/paperspace/data | tail -1 | awk '{print $4" free"}')"
for d in h3dgs h3dgs_ref h3dgs_reffix_tr h3dgs_tr_lidar h3dgs_rgb_zed h3dgs_rgb_zed_stereo11 h3dgs_rgb_zed_stereo h3dgs_ins11 h3dgs_masks fleet_renders stereo_reg_11; do
  [ -e $R/experimental/$d ] && { s=$(du -sh $R/experimental/$d | cut -f1); rm -rf $R/experimental/$d; say "removed experimental/$d ($s)"; }
done
for d in $B/block_[0-9][0-9][0-9]_ref $B/block_[0-9][0-9][0-9]_ins $B/block_013_colmap $B/block_013_fix $B/block_013_fixp $B/block_013_full $B/block_013_full_ref $B/block_013_gl; do
  [ -e $d ] && { rm -rf $d; say "removed $(basename $d)"; }
done
for d in $B/block_[0-9][0-9][0-9]/splat_runs*; do [ -e $d ] && { rm -rf $d; say "removed $(basename $(dirname $d))/$(basename $d)"; }; done
say "=== purge done: $(df -h /home/paperspace/data | tail -1 | awk '{print $4" free"}'); remaining under blocks_ns: $(du -sh $B | cut -f1), experimental: $(du -sh $R/experimental | cut -f1)"
