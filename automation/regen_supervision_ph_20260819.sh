#!/bin/bash
# Fleet supervision regeneration v2 — PARTITION HYGIENE end-to-end
# (Paul 2026-08-19 evening: "drop the ground filter to 0.3m and then
#  implement this across the fleet").
#
# What changed since the afternoon chain:
#   - cluster_tree_instances: --mask-partition-hygiene is DEFAULT ON
#     (smallest-on-top pixel ownership; drapes with discontinuous owned
#     regions dropped; lifts from owned pixels only) and --ground-margin
#     defaults to 0.3 m (0.6 ate low fringe canopies).
#   - markers_v2_from_sam3: supervision + marker lifts consume OWNED
#     regions — written supervision is disjoint by construction; the
#     entry-order overlap lottery is gone; unassigned masks carve void.
# Validated on 05 (sam3_v2_ph): 107 trees = 106 prod census +3 real
# sub-pitch splits -4 drape phantoms; -26% double-counted points.
#
# Per survey: recluster -> markers + semantic monolithics -> hierarchy.
# apr's 11 SAM3 clips from tonight's slot are BANKED — reused as-is.
# Then: quarantine per-block supervision-derived artifacts (stage1 splats
# SURVIVE), reset block pointers, relaunch the week queue.
set -uo pipefail
SRC=/home/paperspace/code/aru_sil_core/src/scripts
L=/home/paperspace/logs
STATE=$L/week_prod_20260814_state
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
cd /home/paperspace/code/nerf_new

survey_chain() {  # <root> <name> <minpts> <pitch> [extra cluster args...]
    local R=$1 N=$2 MP=$3 SP=$4; shift 4
    say "REGEN-PH $N: recluster (hygiene ON, margin 0.3, N=$MP pitch=$SP $*)"
    pixi run python "$SRC/cluster_tree_instances.py" \
        --data-dir "$R" \
        --min-pts-per-voxel "$MP" --split-pitch-m "$SP" \
        --depth-back 1.5 --cache-points "$@" \
        > "$L/regen_ph_${N}_cluster.log" 2>&1 \
        || { say "REGEN-PH $N cluster FAILED"; return 1; }
    say "REGEN-PH $N: markers + owned-region semantic monolithics"
    pixi run python "$SRC/markers_v2_from_sam3.py" \
        --data-dir "$R" --use-sam3-frames --depth-back 1.5 \
        --min-observations 2 --drift-m 0 \
        > "$L/regen_ph_${N}_markers.log" 2>&1 \
        || { say "REGEN-PH $N markers FAILED"; return 1; }
    say "REGEN-PH $N: hierarchy"
    pixi run python "$SRC/build_marker_hierarchy.py" \
        --semantic-monolithic "$R/prod/monos/filtered_semantic_v2.monolithic" \
        --marker-monolithic "$R/prod/bateleur/scene_graph/markers_v2.monolithic" \
        --dominant-direction-xz "1,0" \
        --out "$R/prod/bateleur/scene_graph/marker_hierarchy.json" \
        > "$L/regen_ph_${N}_hierarchy.log" 2>&1 \
        || { say "REGEN-PH $N hierarchy FAILED"; return 1; }
    say "REGEN-PH $N: done"
}

# SCOPE (Paul): hygiene code is fleet-wide DEFAULT, but only 04+05 are
# reclustered/regenerated tonight — 01/03/02/apr keep their current
# censuses and supervision until told otherwise.
C=/home/paperspace/data/citrus_all
survey_chain "$C/05_13D_Jackal" 05_13D_Jackal 5 4.2 \
    --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 --max-det-dist 10.0
survey_chain "$C/04_13D_Jackal" 04_13D_Jackal 5 4.2 \
    --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 --max-det-dist 10.0

# ---- quarantine per-block supervision-derived artifacts (stage1 survives)
for R in "$C"/05_13D_Jackal "$C"/04_13D_Jackal; do
    S=$(basename "$R")
    Q=$R/experimental/quarantine_20260819_ph
    mkdir -p "$Q"
    for BN in "$R"/prod/tassili/blocks_ns/*/; do
        [ -d "$BN" ] || continue
        for BD in "$BN"block_*/; do
            [ -d "$BD" ] || continue
            BID=$(basename "$BD")
            for A in semantic_v2_B supervision splat_runs_FEATFIX \
                     stage2_init_census stage2_init; do
                if [ -e "$BD$A" ]; then
                    mkdir -p "$Q/$BID"
                    mv "$BD$A" "$Q/$BID/" \
                        && say "QUARANTINE $S/$BID/$A"
                fi
            done
        done
        for VJ in "$BN"verdicts_*.json; do
            [ -e "$VJ" ] && mkdir -p "$Q" && mv "$VJ" "$Q/" \
                && say "QUARANTINE $S/$(basename "$VJ")"
        done
    done
done

# ---- reset rotation state for the regenerated surveys only, relaunch
for S in 05_13D_Jackal 04_13D_Jackal; do
    rm -f "$STATE/$S".block_*.FAILED
    [ -e "$STATE/$S.next" ] && echo 0 > "$STATE/$S.next"
done
say "state reset (05+04 only); relaunching queue"
setsid bash /home/paperspace/code/automation/week_prod_queue_20260814.sh \
    >> "$L/week_prod_20260814.log" 2>&1 < /dev/null &
disown
say "REGEN-PH-ALL-DONE"
