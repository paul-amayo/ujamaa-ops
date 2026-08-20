#!/bin/bash
# PRE-PROD READINESS RECIPE (Paul 2026-08-15: "a pre-prod queue or recipe
# that launches, checks assets, fixes things" — all 7 surveys must be train
# ready BEFORE any training slot; no parking, problems surface HERE).
#
#   usage: preprod_readiness.sh            # check + fix + report all surveys
#          preprod_readiness.sh <survey>   # one survey
#
# Gates per survey (each = CHECK, then FIX if mechanical, else RED report):
#   R1 pose      mono-native pose source resolves (kf_domain contract);
#                FIX: root-level aliases for nested dec layouts
#                (transform_lio -> zed odom, kf mono link)
#   R2 kdomain   kf mono record count == kf_*.png count; FIX: re-extract
#   R3 masks     sky/fg dirs KF-NAMED + count >= K + .done;
#                FIX: quarantine stream-named relics -> experimental/ and
#                rebuild kf-named (GPU ~40 min, build_sky/fg_masks)
#   R4 registry  global_ids + hierarchy present; absent = BUILDS-IN-SLOT
#                (pipeline [3][4]) — reported, not fixed here
#   R5 supervision  probe-compile FIRST + MIDDLE canonical block (CPU)
#                and audit density AGAINST THE SURVEY OWN SAM3 REGISTRY:
#                pass = supervised ids >= RATIO_FLOOR% of registry trees
#                near the block path (derived bar — farms look different).
#                Encodes the 03 lesson (3-id supervision -> IoU 0.168).
#   R6 embedder  canon ckpt present; absent = TRAINS-IN-SLOT ([5c])
#
# Output: per-gate PASS/FIXED/RED/IN-SLOT lines + final READY/NOT-READY per
# survey + readiness.json in the state dir. Exit 0 iff NO RED anywhere.
# The week queue v2 refuses to start training unless this exits 0.
# Idempotent; nothing destructive (fixes are mv-to-experimental + rebuilds).
set -u
LOGS=/home/paperspace/logs
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
# sky/fg mask builders import sam3 — they need the SAM3 env, not nerf_new
# (first run failed with ModuleNotFoundError: iopath under the wrong env)
SAM3_PIXI=${SAM3_PIXI:-/home/paperspace/code/sam3/pixi.toml}
STATE=$LOGS/week_prod_20260814_state
CITRUS=/home/paperspace/data/citrus_all
KLAP=/home/paperspace/data/klapmuts
CFG=lio_row100
# Supervision bar is DERIVED, not hand-made (Paul 2026-08-15: "it must be
# derived from the sam3 assets that were made, farms look different"):
# a probe passes when its supervised-id count reaches RATIO_FLOOR% of the
# trees the survey's OWN registry places within reach of the block's
# trajectory. Dimensionless — transfers across farms.
RATIO_FLOOR=50   # percent of expected-from-registry
mkdir -p "$STATE"
RJSON=$STATE/readiness.json
mark() { echo "[$(date +%m-%d\ %H:%M:%S)] $1"; }

# dec_2025_ten_rows is OUT of the rotation (2026-08-18, Paul: "klapmuts
# ten rows needs to be looked at"). Open items: monos nest one level
# deeper (prod/monos/monolithics) unlike every other survey; two orphaned
# index files sit in prod/monos with no mono beside them; kf domain does
# not match (1790 kf_images vs 1792 mono entries), which [0c] now treats
# as a hard stop; and it has no markers_v2 or marker_hierarchy yet. Its
# root_of entry stays so the survey can still be driven by hand.
SURVEYS=(05_13D_Jackal 04_13D_Jackal apr_2026_zed 01_13B_Jackal 03_13B_Jackal 02_13B_Jackal)
[ $# -ge 1 ] && SURVEYS=("$1")
root_of() { case "$1" in
    apr_2026_zed) echo "$KLAP/apr_2026_zed";;
    dec_2025_ten_rows) echo "$KLAP/dec_2025_ten_rows";;
    *) echo "$CITRUS/$1";; esac; }

RED_TOTAL=0
echo "{" > "$RJSON.tmp"

for S in "${SURVEYS[@]}"; do
    R=$(root_of "$S"); RED=0; NOTES=""
    # PROD LAYOUT (2026-08-18): assets are addressed where they physically
    # live. The root shims are gone, so a stale path fails loudly.
    MONOS=$R/prod/monos
    [ -f "$MONOS/image_left_kf20cm.monolithic" ] || \
        { [ -d "$MONOS/monolithics" ] && MONOS=$MONOS/monolithics; }
    BATELEUR=$R/prod/bateleur
    TASSILI=$R/prod/tassili
    mark "=== $S readiness ==="

    # ---- R1 pose -----------------------------------------------------------
    # NO ALIASING (2026-08-18). This gate used to create root symlinks
    # (transform_lio -> monolithics/zed_transform, kf mono -> monolithics/...)
    # so root-addressed scripts kept resolving. All 3,222 survey symlinks were
    # removed because those shims let stale artifacts resolve silently; every
    # consumer now addresses prod directly, and a missing asset must fail here
    # rather than be papered over.
    if [ -f "$MONOS/transform_lio.monolithic" ] && [ -f "$MONOS/image_left_kf20cm.monolithic" ]; then
        mark "R1-PASS $S pose+kf monos in prod/monos"
    else
        [ -f "$MONOS/transform_lio.monolithic" ] || mark "R1-RED $S: no transform_lio.monolithic in $MONOS"
        [ -f "$MONOS/image_left_kf20cm.monolithic" ] || mark "R1-RED $S: no image_left_kf20cm.monolithic in $MONOS"
        RED=$((RED+1))
    fi

    # ---- R2 kdomain --------------------------------------------------------
    if [ -e "$MONOS/image_left_kf20cm.monolithic" ]; then
        # kf_domain is DEAD (2026-08-18): K = kf mono entry count, read by
        # the fixed C++ indexer; PNGs are regenerable renders in scratch_sam3
        KFC=$(cd /home/paperspace/code/nerf_new && pixi run python "$SRC/survey_paths.py" \
              --count "$R" 2>/dev/null | tail -1)
        PNGC=$(ls "$R"/prod/scratch_sam3/kf_*.png 2>/dev/null | wc -l)
        if [ -n "$KFC" ] && [ "$PNGC" -eq "$KFC" ]; then
            mark "R2-PASS $S K=$KFC pngs=$PNGC"
        elif [ -n "$KFC" ] && [ "$PNGC" -lt "$KFC" ]; then
            mark "R2-FIX $S extracting ($PNGC/$KFC)"
            (cd /home/paperspace/code/nerf_new && pixi run python "$SRC/extract_kf_pngs.py" \
                --data-dir "$R" --mono "$MONOS/image_left_kf20cm.monolithic" \
                --out-dir "$R/prod/scratch_sam3") \
                > "$LOGS/preprod_${S}_extract.log" 2>&1 \
                && mark "R2-FIXED $S ($(ls "$R"/prod/scratch_sam3/kf_*.png 2>/dev/null | wc -l) PNGs)" \
                || { mark "R2-RED $S extract failed"; RED=$((RED+1)); }
        else
            mark "R2-RED $S K probe failed (K='$KFC' pngs=$PNGC)"; RED=$((RED+1))
        fi
    fi

    # ---- R3 masks ----------------------------------------------------------
    K=${KFC:-0}
    for D in sky_masks fg_masks; do
        FIRST=$(ls "$TASSILI/$D/" 2>/dev/null | head -1)
        CNT=$(ls "$TASSILI/$D/"kf_*.png 2>/dev/null | wc -l)
        if [ -n "$FIRST" ] && ! echo "$FIRST" | grep -qE "^kf_|^\."; then
            mkdir -p "$R/experimental/masks_streamnamed_pre_kf_rebuild"
            T=$TASSILI/$D; [ -L "$TASSILI/$D" ] && T=$(readlink -f "$TASSILI/$D")
            mv "$T" "$R/experimental/masks_streamnamed_pre_kf_rebuild/$D" 2>/dev/null
            [ -L "$TASSILI/$D" ] && unlink "$TASSILI/$D"
            mark "R3-QUARANTINED $S $D (stream-named relic '$FIRST' -> experimental; prod holds only realisable assets)"
            CNT=0
        fi
        if [ "$CNT" -lt "$K" ] || [ "$K" -eq 0 ]; then
            # stale .done over an under-count must not survive: the pipeline
            # [4b] gate trusts it (a false FIXED stamped one on 0 PNGs once)
            [ -f "$TASSILI/$D/.done" ] && mv "$TASSILI/$D/.done" "$TASSILI/$D/.done.revoked_$(date +%s)"
            TOOL=build_sky_masks.py; [ "$D" = "fg_masks" ] && TOOL=build_fg_masks.py
            mark "R3-FIX $S rebuilding $D kf-named ($CNT/$K) — GPU"
            (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$SAM3_PIXI" \
                python "$SRC/$TOOL" --data-dir "$R") \
                > "$LOGS/preprod_${S}_${D}.log" 2>&1
            NEWCNT=$(ls "$TASSILI/$D/"kf_*.png 2>/dev/null | wc -l)
            # FIXED means the FILES exist — an exit code is not evidence
            if [ "$NEWCNT" -ge "$K" ] && [ "$K" -gt 0 ]; then
                touch "$TASSILI/$D/.done"
                mark "R3-FIXED $S $D ($NEWCNT PNGs)"
            else
                mark "R3-RED $S $D rebuild produced $NEWCNT/$K (see preprod_${S}_${D}.log)"
                RED=$((RED+1))
            fi
        else
            [ -f "$TASSILI/$D/.done" ] || touch "$TASSILI/$D/.done"
            mark "R3-PASS $S $D kf-named x$CNT"
        fi
    done

    # ---- R4 registry -------------------------------------------------------
    if [ -f "$BATELEUR/sam3_v2/global_ids.json" ] && ls "$BATELEUR"/scene_graph/marker_hierarchy.json >/dev/null 2>&1; then
        mark "R4-PASS $S registry present"
        HAS_REG=1
    else
        mark "R4-IN-SLOT $S registry absent — pipeline [3][4] builds it in the first slot"
        NOTES="$NOTES registry-in-slot;"
        HAS_REG=0
    fi

    # ---- R5a semantic-source freshness (Paul: derived artifacts must
    # POSTDATE their inputs — "if we generate new sam3 masks we have to
    # generate a new semantic monolithic"). DAG: sam3 registry ->
    # {markers_v2, instance_labels_v2, filtered_semantic_v2} monolithics ->
    # hierarchy -> per-block supervision. stat -L ALWAYS (a shim's own
    # date is migration day and lies). Proven live: 03's mono was
    # 2026-05-09 vs registry 2026-07-10 -> only 3 ids resolved.
    MONO=$MONOS/filtered_semantic_v2.monolithic
    GIDS=$BATELEUR/sam3_v2/global_ids.json
    if [ "$HAS_REG" = "1" ] && { [ ! -e "$MONO" ] || \
         [ "$(stat -Lc %Y "$MONO" 2>/dev/null || echo 0)" -lt "$(stat -Lc %Y "$GIDS")" ]; }; then
        mark "R5a-STALE $S semantic monolithics predate the registry — regenerating from sam3 assets"
        if (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$NS_PIXI" \
              python "$SRC/markers_v2_from_sam3.py" \
              --data-dir "$R" --use-sam3-frames \
              --depth-back 1.5 --min-observations 2 --drift-m 0) \
              > "$LOGS/preprod_${S}_markers_regen.log" 2>&1 \
           && (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$NS_PIXI" \
              python "$SRC/build_marker_hierarchy.py" \
              --semantic-monolithic "$MONOS/filtered_semantic_v2.monolithic" \
              --marker-monolithic "$BATELEUR/scene_graph/markers_v2.monolithic" \
              --dominant-direction-xz "1,0" \
              --out "$BATELEUR/scene_graph/marker_hierarchy.json") \
              > "$LOGS/preprod_${S}_hierarchy_regen.log" 2>&1; then
            # cascade: downstream markers older than the fresh mono are
            # stale too — mv them aside so supervision recompiles
            NMV=0
            while IFS= read -r MK; do
                [ "$(stat -Lc %Y "$MK")" -lt "$(stat -Lc %Y "$MONO")" ] || continue
                mv "$MK" "${MK}.stale_$(date +%s)" && NMV=$((NMV+1))
            done < <(find "$TASSILI/blocks_ns/$CFG/" \
                       \( -name ".palette_v2" -o -name "manifest.json" \) 2>/dev/null)
            # trailing slash: blocks_ns/<cfg> is a shim into prod and bare
            # find does not follow a symlink argument (found 0 on 03)
            mark "R5a-FIXED $S monolithics + hierarchy regenerated; $NMV stale supervision markers invalidated"
        else
            mark "R5a-STALE-UNFIXED $S regeneration failed (see preprod_${S}_markers_regen.log / _hierarchy_regen.log)"
            RED=$((RED+1))
        fi
    elif [ "$HAS_REG" = "1" ]; then
        mark "R5a-PASS $S semantic monolithics postdate the registry"
    fi

    # ---- R5b stage2 supersession (ALWAYS checked, not only on regen):
    # stage2 outputs older than the semantic mono are derived from a
    # previous supervision era — quarantine so the block's slot retrains
    # on the fresh chain (cache-miss = automatic requeue, never lingers)
    if [ "$HAS_REG" = "1" ] && [ -e "$MONO" ]; then
        NS2=0
        while IFS= read -r S2; do
            CK=$(ls -t "$S2"/high/*/nerfstudio_models*/*.ckpt 2>/dev/null | head -1)
            [ -n "$CK" ] && [ "$CK" -ot "$(readlink -f "$MONO")" ] || continue
            QD=$R/experimental/stale_stage2_pre_regen_$(date +%Y%m%d)
            mkdir -p "$QD/$(basename "$(dirname "$S2")")"
            mv "$S2" "$QD/$(basename "$(dirname "$S2")")/" && NS2=$((NS2+1))
        done < <(find "$TASSILI/blocks_ns/$CFG/" -maxdepth 3 -type d \
                   -name "stage2_censusinit_fw2" 2>/dev/null)
        if [ "$NS2" -gt 0 ]; then
            mark "R5b-REQUEUED $S $NS2 stage2 runs superseded by a fresher semantic mono -> experimental (slots retrain)"
            echo 0 > "$STATE/$S.next"
        fi
    fi

    # ---- R5 supervision viability (probe first + middle canonical block) --------
    if [ "$HAS_REG" = "1" ] && [ -d "$TASSILI/blocks_ns/$CFG" ]; then
        BLOCKS=($(ls -d "$TASSILI/blocks_ns/$CFG"/block_* 2>/dev/null | grep -E "block_[0-9]+$" | sort))
        NB=${#BLOCKS[@]}
        if [ "$NB" -gt 0 ]; then
            BEST_RATIO=-1; BEST_LINE=""
            for BD in "${BLOCKS[0]}" "${BLOCKS[$((NB/2))]}"; do
                # repaint when the marker is ABSENT or OLDER than the mono
                # (-ot, deref via readlink -f: a marker from a previous mono
                # era must not satisfy the gate — post-regen probes once
                # silently reused stale paint and measured the old era)
                if [ ! -f "$BD/semantic_v2_B/.palette_v2" ] \
                   || [ "$BD/semantic_v2_B/.palette_v2" -ot "$(readlink -f "$MONO")" ]; then
                    (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$NS_PIXI" \
                        python "$SRC/save_filtered_semantic_pngs.py" \
                        --block-dir "$BD" \
                        --semantic-monolithic "$MONOS/filtered_semantic_v2.monolithic" \
                        --marker-monolithic "$BATELEUR/scene_graph/markers_v2.monolithic" \
                        --global-ids "$BATELEUR/sam3_v2/global_ids.json") \
                        > "$LOGS/preprod_${S}_supervision_$(basename "$BD").log" 2>&1 \
                        && touch "$BD/semantic_v2_B/.palette_v2"
                fi
                # bar derived from the survey's OWN sam3 registry (expected
                # trees near this block's path); stderr into a probe log —
                # a silenced crash once hid a survey behind an empty readout
                READOUT=$(cd /home/paperspace/code/nerf_new && pixi run python \
                    "$SRC/audit_supervision_density.py" \
                    --supervision-dir "$BD/semantic_v2_B" \
                    --global-ids "$BATELEUR/sam3_v2/global_ids.json" \
                    --transforms "$BD/transforms.json" \
                    2>> "$LOGS/preprod_${S}_probe_$(basename "$BD").log" | tail -1)
                mark "R5-PROBE $S $(basename "$BD"): ${READOUT:-AUDIT-CRASHED (see probe log)}"
                RATIO=$(echo "$READOUT" | grep -oaE "ratio_pct=[0-9]+" | cut -d= -f2)
                if [ -n "${RATIO:-}" ] && [ "$RATIO" -gt "$BEST_RATIO" ]; then
                    BEST_RATIO=$RATIO; BEST_LINE=$READOUT
                fi
            done
            if [ "$BEST_RATIO" -ge "$RATIO_FLOOR" ]; then
                mark "R5-PASS $S supervision viable vs own registry ($BEST_LINE)"
            elif [ "$BEST_RATIO" -ge 0 ]; then
                mark "R5-RED $S supervision STARVED vs own registry (best: $BEST_LINE; floor ratio ${RATIO_FLOOR}%) — root-cause the semantic source before training"
                RED=$((RED+1))
            else
                mark "R5-RED $S supervision probes unreadable (see probe logs)"
                RED=$((RED+1))
            fi
        else
            mark "R5-IN-SLOT $S no $CFG partition yet (pipeline partitions in first slot; ratio floor enforced at compile)"
            NOTES="$NOTES partition-in-slot;"
        fi
    else
        mark "R5-IN-SLOT $S paint check deferred until registry exists (floor enforced at compile in-slot)"
        NOTES="$NOTES paint-in-slot;"
    fi

    # ---- R7 blocks live IN PROD -------------------------------------------
    # The canonical cfg is the realisable asset under construction, so the
    # slot is pre-placed and training writes land in prod from block one.
    # The root shim this gate used to leave behind is gone (2026-08-18):
    # blocks are addressed at prod/tassili/blocks_ns only.
    PB=$TASSILI/blocks_ns/$CFG
    # The check is on the ROOT copy, not the prod one — a blanket repoint of
    # $R/blocks_ns -> $TASSILI/blocks_ns rewrote this test too, so it fired RED
    # on all six surveys for having their blocks exactly where they belong.
    if [ -d "$R/blocks_ns/$CFG" ] && [ ! -L "$R/blocks_ns/$CFG" ]; then
        mark "R7-RED $S: $CFG still sits at the survey root — move it into $PB"
        RED=$((RED+1))
    else
        mkdir -p "$PB"
        mark "R7-PASS $S $CFG in prod/tassili ($(ls -d "$PB"/block_* 2>/dev/null | wc -l) blocks)"
    fi

    # R10 (auto-requeue on tree_iou_min < 0.10) DROPPED (Paul 2026-08-20).
    # It assumed near-zero containment was transient chain breakage that a
    # retrain repairs; apr proved it can be STRUCTURAL (broken row
    # hierarchy) — the rule then loops forever: drop entries, rewind to 0,
    # cache-hit resweep reproduces the same scores, repeat every launch
    # (and it silently overrode operator-set pointers twice on 08-20).
    # Near-zero verdicts now stay RECORDED for triage; requeue is a human
    # decision.

    if [ "$RED" -eq 0 ]; then
        mark ">>> $S READY${NOTES:+ (in-slot: $NOTES)}"
    else
        mark ">>> $S NOT-READY ($RED red gates)"
    fi
    echo " \"$S\": {\"red\": $RED, \"notes\": \"$NOTES\"}," >> "$RJSON.tmp"
    RED_TOTAL=$((RED_TOTAL + RED))
done

echo " \"_generated\": \"$(date -u +%FT%TZ)\", \"_stale_total\": $RED_TOTAL }" >> "$RJSON.tmp"
mv "$RJSON.tmp" "$RJSON"
mark "READINESS COMPLETE: $RED_TOTAL stale/unfixed items reported -> $RJSON"
# NO HALT (Paul 2026-08-15): readiness MAKES surveys ready and reports what
# stayed stale; surveys are self-contained and train on the information they
# have — per-block in-slot guards handle thin supervision. Always exit 0.
exit 0
