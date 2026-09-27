#!/bin/bash
# sidecar_cleanup_20260927.sh — deletions authorised by Paul (2026-09-27 ~09:55 box, AskUserQuestion "Disk cleanup": all
# three options): (1) superseded side-car seed variants under 05/experimental/h3dgs_sidecar (keeps per block: the converted
# stage-1 checkpoint, both census npz, the bootstrap run, the FINAL ratio-2 seed; the block-style control keeps its
# void-row seed); (2) the stale stopped ten_rows projects; (3) the lane-2 ablation leftovers (keeps lane2/h3dgs = the
# served arm A, and lane2/colmap). Explicit paths only; prints what goes and the space freed.
set -u
SC=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar; TR=/home/paperspace/data/klapmuts/dec_2025_ten_rows/experimental
L=/home/paperspace/logs/sidecar_cleanup_20260927.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
before=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); say "=== cleanup start: ${before}G free"
gone=(); rmdir_(){ for p in "$@"; do if [ -e "$p" ]; then s=$(du -sh "$p" 2>/dev/null | cut -f1); rm -rf "$p" && gone+=("$s $p"); fi; done; }
# (1) side-car variants
for b in block_000 block_013 block_020; do
  d=$SC/$b; [ -d $d ] || continue
  rmdir_ $d/stage2_init_glref $d/stage2_init_census_glref $d/splat_runs_FEATFIX/stage2_censusinit_glref \
         $d/stage2_init_census_glref_f0.0 $d/splat_runs_FEATFIX/stage2_censusinit_glref_f0.0 \
         $d/stage2_init_census_glref_f0.1 $d/splat_runs_FEATFIX/stage2_censusinit_glref_f0.1 \
         $d/stage2_init_census_glref_bg_f1.0 $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0 \
         $d/stage2_init_census_glref_bg_f1.0_r4 $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r4 \
         $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_op0.95 $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_op0.8 \
         $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2_op0.95 $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r4_op0.95
done
d=$SC/blockstyle_ref/block_000
rmdir_ $d/stage2_init_glref $d/stage2_init_census_glref $d/splat_runs_FEATFIX/stage2_censusinit_glref $d/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_op0.95
# (2) stale stopped ten_rows projects
rmdir_ $TR/h3dgs_lo3s $TR/h3dgs_lo $TR/h3dgs_lo3s_rgb $TR/h3dgs_lo3sL $TR/h3dgs_lo3 $TR/h3dgs_lo3sL01
# (3) lane-2 ablation leftovers
rmdir_ $TR/lane2/h3dgs_b4m $TR/lane2/h3dgs_full30 $TR/lane2/h3dgs_kf30
for g in "${gone[@]}"; do say "removed $g"; done
after=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); say "=== cleanup done: ${before}G -> ${after}G free ($(( after - before ))G freed); kept per side-car block: $(ls $SC/block_000 | tr '\n' ' ')"
