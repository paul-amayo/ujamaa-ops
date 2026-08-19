#!/bin/bash
# Fleet supervision regeneration, 2026-08-19 (Paul: "fix the entire
# supervision monolithic / bridge and everything"). Applies, per survey:
#   1. recluster with the density recipe + coverage veto (13B: N=1;
#      13D: N=5 + NMS 4.2; apr: calibrated below) -> clean member_to_global
#   2. markers_v2_from_sam3 -> markers + PAD-FREE semantic monolithics
#      (entry K = keyframe K) painted from the vetoed assignments
#   3. build_marker_hierarchy (derived-postdates-inputs)
# Then quarantines today's lag-1 per-block artifacts, zeros counters and
# relaunches the queue.
set -uo pipefail
SRC=/home/paperspace/code/aru_sil_core/src/scripts
L=/home/paperspace/logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
cd /home/paperspace/code/nerf_new

survey_chain() {  # <root> <name> <minpts> <pitch>
    local R=$1 N=$2 MP=$3 SP=$4
    say "REGEN $N: recluster (N=$MP pitch=$SP, coverage veto)"
    pixi run python "$SRC/cluster_tree_instances.py" \
        --data-dir "$R" --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 \
        --min-pts-per-voxel "$MP" --split-pitch-m "$SP" \
        --depth-back 1.5 --max-det-dist 10.0 --cache-points \
        > "$L/regen_${N}_cluster.log" 2>&1 || { say "REGEN $N cluster FAILED"; return 1; }
    say "REGEN $N: markers + pad-free semantic monos"
    pixi run python "$SRC/markers_v2_from_sam3.py" \
        --data-dir "$R" --use-sam3-frames --depth-back 1.5 --min-observations 2 --drift-m 0 \
        > "$L/regen_${N}_markers.log" 2>&1 || { say "REGEN $N markers FAILED"; return 1; }
    say "REGEN $N: hierarchy"
    pixi run python "$SRC/build_marker_hierarchy.py" \
        --semantic-monolithic "$R/prod/monos/filtered_semantic_v2.monolithic" \
        --marker-monolithic "$R/prod/bateleur/scene_graph/markers_v2.monolithic" \
        --dominant-direction-xz "1,0" \
        --out "$R/prod/bateleur/scene_graph/marker_hierarchy.json" \
        > "$L/regen_${N}_hierarchy.log" 2>&1 || { say "REGEN $N hierarchy FAILED"; return 1; }
    say "REGEN $N: done"
}

survey_chain /home/paperspace/data/citrus_all/05_13D_Jackal 05_13D_Jackal 5 4.2
survey_chain /home/paperspace/data/citrus_all/04_13D_Jackal 04_13D_Jackal 5 4.2
survey_chain /home/paperspace/data/citrus_all/01_13B_Jackal 01_13B_Jackal 1 0
survey_chain /home/paperspace/data/citrus_all/03_13B_Jackal 03_13B_Jackal 1 0
survey_chain /home/paperspace/data/citrus_all/02_13B_Jackal 02_13B_Jackal 1 0

# ---- apr: calibrate N on its own density first ------------------------------
A=/home/paperspace/data/klapmuts/apr_2026_zed
say "REGEN apr: baseline collect (caches points for calibration)"
pixi run python "$SRC/cluster_tree_instances.py" \
    --data-dir "$A" --voxel-size 0.1 --dilate-iters 2 --min-voxels 50 \
    --min-pts-per-voxel 1 --split-pitch-m 0 \
    --depth-back 1.5 --max-det-dist 10.0 --cache-points \
    > "$L/regen_apr_cluster_n1.log" 2>&1 || say "apr baseline collect FAILED"
say "REGEN apr: calibrating N vs the old 749 census"
pixi run python /home/paperspace/code/automation/apr_calibrate_n.py \
    > "$L/regen_apr_calibration.log" 2>&1 || say "apr calibration FAILED (falls back N=1)"
BEST=$(grep -a "^BEST" "$L/regen_apr_calibration.log" | tail -1)
N=$(echo "$BEST" | awk '{print $2}'); PITCH=$(echo "$BEST" | awk '{print $3}')
say "REGEN apr: chosen N=${N:-1} pitch=${PITCH:-1.0}"
survey_chain "$A" apr_2026_zed "${N:-1}" "${PITCH:-1.0}"

# ---- clear today's lag-1 / pre-veto per-block artifacts ---------------------
for SB in /home/paperspace/data/citrus_all/05_13D_Jackal /home/paperspace/data/citrus_all/04_13D_Jackal \
          /home/paperspace/data/citrus_all/01_13B_Jackal /home/paperspace/data/citrus_all/03_13B_Jackal; do
    NB=$(basename "$SB"); Q=$SB/experimental/quarantine_20260819_lag1
    for BD in "$SB"/prod/tassili/blocks_ns/lio_row100/block_*; do
        [ -d "$BD" ] || continue
        MOVED=0
        for AASSET in semantic_v2_B supervision splat_runs_FEATFIX stage2_init_census stage2_init; do
            [ -e "$BD/$AASSET" ] || continue
            mkdir -p "$Q/$(basename "$BD")"
            mv "$BD/$AASSET" "$Q/$(basename "$BD")/$AASSET" && MOVED=1
        done
        [ "$MOVED" = "1" ] && say "QUARANTINE $NB/$(basename "$BD"): lag-1 supervision artifacts"
    done
    # config-level verdicts recorded against lag-1 GT
    for VJ in "$SB"/prod/tassili/blocks_ns/lio_row100/verdicts_censusinit_fw2*.json; do
        [ -f "$VJ" ] || continue
        mkdir -p "$Q"; mv "$VJ" "$Q/$(basename "$VJ")"
    done
done

# ---- reset rotation + relaunch ----------------------------------------------
S=/home/paperspace/logs/week_prod_20260814_state
mkdir -p "$S/archive_20260819_regen"
mv "$S"/*.next "$S"/*.FAILED "$S"/round "$S/archive_20260819_regen/" 2>/dev/null || true
say "counters zeroed; relaunching queue"
setsid nohup bash /home/paperspace/code/automation/week_prod_queue_20260814.sh < /dev/null > /dev/null 2>&1 &
say "REGEN-ALL-DONE"
