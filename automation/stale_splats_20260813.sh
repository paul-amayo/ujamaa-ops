#!/bin/bash
# STALE SPLAT / EXPERIMENT DELETION LIST — 2026-08-13 census (disk 83% full,
# 321G free). REVIEW THEN RUN YOURSELF (rm is Paul's). Dry-run by default:
#   bash automation/stale_splats_20260813.sh          # prints what would go
#   bash automation/stale_splats_20260813.sh --delete # actually deletes
# Winning-config evidence lives in lab_notebook/ + PILLARS; nothing below is
# the last copy of a RESULT — only of run artifacts.
set -u
DO=${1:-}
del() {  # del <path> <reason>
    [ -e "$1" ] || return 0
    if [ "$DO" = "--delete" ]; then
        echo "DELETING  $(du -sBG "$1" 2>/dev/null | cut -f1)  $1  ($2)"
        rm -rf "$1"
    else
        echo "would rm  $(du -sBG "$1" 2>/dev/null | cut -f1)  $1  ($2)"
    fi
}
C=/home/paperspace/data/citrus_all
N=/home/paperspace/data/high/nerf

# ---- 1. 02 combined.bag — 81G ---------------------------------------------
# Ingest COMPLETE + verified (monolithics, LIO, kf mono, registry 177 trees,
# ledger epoch landed). Same post-ingest deletion every other survey got.
del "$C/02_13B_Jackal/combined.bag" "ingested 2026-07-22, registry+ledger done"

# ---- 2. 03 arc dedup fleet — ~250G ----------------------------------------
# 66-block June fleet: single-pass v16 (recipe DEAD — absgrad autopsy
# 2026-08-13), DA3-init, arc partition (superseded by row for semantics).
# splats.json only ever registered 2-3 entries — barely served. The row-
# halves fleet (80 halves, night queue HELD) is the replacement direction.
# CAVEAT: until the halves fleet trains, 03 has no complete trained fleet.
# Deletes RUN artifacts (ckpts, sweeps, logs); keeps transforms.json +
# markers/registry (survey-level) so halves work is unaffected.
for B in "$C"/03_13B_Jackal/blocks_ns/lio_arc_size15.0_ov0.10_kf20cm_dedup/block_*/; do
    del "${B}splat_runs_perpass"      "v16-era run"
    del "${B}splat_runs_lio_only_v2"  "v16-era run"
    for R in "${B}"splat_runs_*; do
        case "$R" in *perpass|*lio_only_v2) ;; *) del "$R" "v16-era run" ;; esac
    done
    del "${B}sweep_semantic.log"      "fat sweep log"
    del "${B}sweep_perpass_train.log" "fat sweep log"
    del "${B}da3"                     "DA3 outputs (init retired)"
done
del "$C/03_13B_Jackal/blocks_ns/lidar_pass1_100cm" "1G early probe config"

# ---- 3. 01 arc config — ~29G ----------------------------------------------
# Superseded by the WINNING lio_row C_20k fleet (row pointing 88.5 vs 72.2).
# KEEP block_000/transforms.json: build_row_blocks' default intrinsics donor.
for B in "$C"/01_13B_Jackal/blocks_ns/lio_arc_size15.0_ov0.10_kf20cm/block_*/; do
    case "$B" in *block_000/)
        for R in "${B}"splat_runs_* "${B}da3" "${B}colmap"; do del "$R" "arc run (keep transforms)"; done ;;
    *) del "$B" "arc block, superseded by lio_row" ;; esac
done

# ---- 4. 05 legacy configs — ~15G ------------------------------------------
# lio_arc: tonight's v16 mechanics test (recipe dead). lio_row: 57-64 m
# length-baseline blocks (violate the 100-kf rule; sweep results are in the
# notebook). lio_row100 is the live config — NOT touched.
del "$C/05_13D_Jackal/blocks_ns/lio_arc_size15.0_ov0.10_kf20cm" "v16 mechanics test"
del "$C/05_13D_Jackal/blocks_ns/lio_row" "length-baseline era (57-64 m blocks)"

# ---- 5. embedder sweeps — ~15G --------------------------------------------
# N/O/P/Q hyperparameter sweeps, superseded by the c20cos20 canonical recipe
# (insights §14). Results logged; ckpts stale.
for D in "$N"/sweep_*/; do del "$D" "embedder sweep, c20cos20 canon"; done
# old training outputs (NOT the clip_openclip_* target caches — those are
# ACTIVE, content-keyed inputs to training):
del "$N/outputs/baseline"      "old embedder baseline run"
del "$N/outputs/baseline_orig" "old embedder baseline run"
# CORAL 05 embedder — superseded by 05_13D_v1g (c20cos20) the same day:
del "$N/05_13D_v1" "CORAL config, replaced by 05_13D_v1g"

# ---- 6. VERIFY-BEFORE-DELETE (not touched by this script) ------------------
# - 04 block_001_L095_sky: splat_runs_BASE (4G) / BASELINE (2G) / TREELOD
#   (3G): experiment runs, but one of them holds the stage1 ckpt the
#   CANONICAL stage2_init_census derived from — confirm provenance first.
# - splat_runs_REGRESSION probe_* accumulate per harness run (~2G): keep the
#   newest calibrated pair; consider auto-prune in the harness.
# - /home/paperspace/data/high/nerf/outputs remainder (~34G): clip_openclip_*
#   CLIP-target caches are ACTIVE — do not delete.
echo
echo "Totals above are per-item; expected reclaim ~370-390G on the safe set."
