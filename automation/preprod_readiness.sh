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
#   R5 paint     probe-paint FIRST + MIDDLE canonical block (painter, CPU)
#                and audit density; floors: >=4 distinct ids AND >=40%
#                frames painted on at least one probe. Encodes the 03
#                lesson (3-id / 27%-frames paint starved census-init to
#                tree IoU 0.168 while the splat itself was clean).
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
SAM3_PIXI=${SAM3_PIXI:-/home/paperspace/code/nerf_new/pixi.toml}
STATE=$LOGS/week_prod_20260814_state
CITRUS=/home/paperspace/data/citrus_all
KLAP=/home/paperspace/data/klapmuts
CFG=lio_row100
ID_FLOOR=4
FRAME_FRAC_FLOOR=40   # percent
mkdir -p "$STATE"
RJSON=$STATE/readiness.json
mark() { echo "[$(date +%m-%d\ %H:%M:%S)] $1"; }

SURVEYS=(05_13D_Jackal 04_13D_Jackal apr_2026_zed 01_13B_Jackal 03_13B_Jackal 02_13B_Jackal dec_2025_ten_rows)
[ $# -ge 1 ] && SURVEYS=("$1")
root_of() { case "$1" in
    apr_2026_zed) echo "$KLAP/apr_2026_zed";;
    dec_2025_ten_rows) echo "$KLAP/dec_2025_ten_rows";;
    *) echo "$CITRUS/$1";; esac; }

RED_TOTAL=0
echo "{" > "$RJSON.tmp"

for S in "${SURVEYS[@]}"; do
    R=$(root_of "$S"); RED=0; NOTES=""
    mark "=== $S readiness ==="

    # ---- R1 pose -----------------------------------------------------------
    MD=$R; [ -d "$R/monolithics" ] && MD=$R/monolithics
    if [ ! -e "$R/transform_lio.monolithic" ]; then
        for CAND in zed_transform.monolithic transform_lio.monolithic; do
            if [ -f "$MD/$CAND" ] && [ "$MD" != "$R" ]; then
                ln -sfn "monolithics/$CAND" "$R/transform_lio.monolithic"
                mark "R1-FIXED $S: transform_lio -> monolithics/$CAND (camera-frame odom for training; INS stays georef-side)"
                break
            fi
        done
    fi
    if [ ! -e "$R/image_left_kf20cm.monolithic" ] && [ -f "$MD/image_left_kf20cm.monolithic" ]; then
        ln -sfn "monolithics/image_left_kf20cm.monolithic" "$R/image_left_kf20cm.monolithic"
        mark "R1-FIXED $S: kf mono root alias"
    fi
    if [ -e "$R/transform_lio.monolithic" ] && [ -e "$R/image_left_kf20cm.monolithic" ]; then
        mark "R1-PASS $S pose+kf monos resolve"
    else
        mark "R1-RED $S: no pose/kf mono (ingest incomplete)"; RED=$((RED+1))
    fi

    # ---- R2 kdomain --------------------------------------------------------
    if [ -e "$R/image_left_kf20cm.monolithic" ]; then
        KFC=$(cd /home/paperspace/code/nerf_new && pixi run python "$SRC/kf_domain.py" \
              --data-dir "$R" 2>/dev/null | grep -oaE "K=[0-9]+" | head -1 | cut -d= -f2)
        PNGC=$(ls "$R"/kf_images/kf_*.png 2>/dev/null | wc -l)
        if [ -n "$KFC" ] && [ "$PNGC" -eq "$KFC" ]; then
            mark "R2-PASS $S K=$KFC pngs=$PNGC"
        elif [ -n "$KFC" ] && [ "$PNGC" -lt "$KFC" ]; then
            mark "R2-FIX $S extracting ($PNGC/$KFC)"
            (cd /home/paperspace/code/nerf_new && pixi run python "$SRC/extract_kf_pngs.py" \
                --data-dir "$R" --mono "$R/image_left_kf20cm.monolithic") \
                > "$LOGS/preprod_${S}_extract.log" 2>&1 \
                && mark "R2-FIXED $S ($(ls "$R"/kf_images/kf_*.png 2>/dev/null | wc -l) PNGs)" \
                || { mark "R2-RED $S extract failed"; RED=$((RED+1)); }
        else
            mark "R2-RED $S K probe failed (K='$KFC' pngs=$PNGC)"; RED=$((RED+1))
        fi
    fi

    # ---- R3 masks ----------------------------------------------------------
    K=${KFC:-0}
    for D in sky_masks fg_masks; do
        FIRST=$(ls "$R/$D/" 2>/dev/null | head -1)
        CNT=$(ls "$R/$D/"kf_*.png 2>/dev/null | wc -l)
        if [ -n "$FIRST" ] && ! echo "$FIRST" | grep -qE "^kf_|^\."; then
            mkdir -p "$R/experimental/masks_streamnamed_pre_kf_rebuild"
            T=$R/$D; [ -L "$R/$D" ] && T=$(readlink -f "$R/$D")
            mv "$T" "$R/experimental/masks_streamnamed_pre_kf_rebuild/$D" 2>/dev/null
            [ -L "$R/$D" ] && unlink "$R/$D"
            mark "R3-QUARANTINED $S $D (stream-named relic '$FIRST' -> experimental; prod holds only realisable assets)"
            CNT=0
        fi
        if [ "$CNT" -lt "$K" ] || [ "$K" -eq 0 ]; then
            TOOL=build_sky_masks.py; [ "$D" = "fg_masks" ] && TOOL=build_fg_masks.py
            mark "R3-FIX $S rebuilding $D kf-named ($CNT/$K) — GPU"
            (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$SAM3_PIXI" \
                python "$SRC/$TOOL" --data-dir "$R") \
                > "$LOGS/preprod_${S}_${D}.log" 2>&1 \
                && { touch "$R/$D/.done"; mark "R3-FIXED $S $D ($(ls "$R/$D/"kf_*.png 2>/dev/null | wc -l) PNGs)"; } \
                || { mark "R3-RED $S $D rebuild failed (see preprod_${S}_${D}.log)"; RED=$((RED+1)); }
        else
            [ -f "$R/$D/.done" ] || touch "$R/$D/.done"
            mark "R3-PASS $S $D kf-named x$CNT"
        fi
    done

    # ---- R4 registry -------------------------------------------------------
    if [ -f "$R/sam3_v2/global_ids.json" ] && ls "$R"/scene_graph*/marker_hierarchy*.json >/dev/null 2>&1; then
        mark "R4-PASS $S registry present"
        HAS_REG=1
    else
        mark "R4-IN-SLOT $S registry absent — pipeline [3][4] builds it in the first slot"
        NOTES="$NOTES registry-in-slot;"
        HAS_REG=0
    fi

    # ---- R5 paint viability (probe first + middle canonical block) --------
    if [ "$HAS_REG" = "1" ] && [ -d "$R/blocks_ns/$CFG" ]; then
        BLOCKS=($(ls -d "$R/blocks_ns/$CFG"/block_* 2>/dev/null | grep -E "block_[0-9]+$" | sort))
        NB=${#BLOCKS[@]}
        if [ "$NB" -gt 0 ]; then
            BEST_IDS=0; BEST_FRAC=0
            for BD in "${BLOCKS[0]}" "${BLOCKS[$((NB/2))]}"; do
                if [ ! -f "$BD/semantic_v2_B/.palette_v2" ]; then
                    (cd /home/paperspace/code/nerf_new && pixi run --manifest-path "$NS_PIXI" \
                        python "$SRC/save_filtered_semantic_pngs.py" \
                        --block-dir "$BD" \
                        --semantic-monolithic "$R/filtered_semantic_v2.monolithic" \
                        --marker-monolithic "$(ls "$R"/scene_graph*/markers_v2*.monolithic | head -1)" \
                        --global-ids "$R/sam3_v2/global_ids.json") \
                        > "$LOGS/preprod_${S}_paint_$(basename "$BD").log" 2>&1 \
                        && touch "$BD/semantic_v2_B/.palette_v2"
                fi
                READOUT=$(cd /home/paperspace/code/nerf_new && pixi run python \
                    "$SRC/audit_paint_density.py" --paint-dir "$BD/semantic_v2_B" 2>/dev/null | tail -1)
                mark "R5-PROBE $S $(basename "$BD"): $READOUT"
                IDS=$(echo "$READOUT" | grep -oaE "ids=[0-9]+" | cut -d= -f2)
                FRAC=$(echo "$READOUT" | grep -oaE "painted_pct=[0-9]+" | cut -d= -f2)
                [ "${IDS:-0}" -gt "$BEST_IDS" ] && BEST_IDS=$IDS
                [ "${FRAC:-0}" -gt "$BEST_FRAC" ] && BEST_FRAC=$FRAC
            done
            if [ "$BEST_IDS" -ge "$ID_FLOOR" ] && [ "$BEST_FRAC" -ge "$FRAME_FRAC_FLOOR" ]; then
                mark "R5-PASS $S paint viable (best probe ids=$BEST_IDS painted_pct=$BEST_FRAC)"
            else
                mark "R5-RED $S paint STARVED (best probe ids=$BEST_IDS painted_pct=$BEST_FRAC; floors $ID_FLOOR/$FRAME_FRAC_FLOOR) — root-cause the survey semantic/filter before training"
                RED=$((RED+1))
            fi
        else
            mark "R5-IN-SLOT $S no $CFG partition yet (pipeline partitions in first slot; paint floor enforced at compile)"
            NOTES="$NOTES partition-in-slot;"
        fi
    else
        mark "R5-IN-SLOT $S paint check deferred until registry exists (floor enforced at compile in-slot)"
        NOTES="$NOTES paint-in-slot;"
    fi

    # ---- R7 blocks live IN PROD (Paul: "wouldnt the blocks be placed in
    # prod" — the canonical cfg IS the realisable asset under construction;
    # root keeps only a shim so training writes land in prod from block one)
    PB=$R/prod/tassili/blocks_ns/$CFG
    RB=$R/blocks_ns/$CFG
    if [ -d "$RB" ] && [ ! -L "$RB" ]; then
        mkdir -p "$R/prod/tassili/blocks_ns"
        if [ -d "$PB" ]; then
            mark "R7-RED $S: BOTH root and prod hold $CFG — manual merge needed"
            RED=$((RED+1))
        else
            mv "$RB" "$PB" && ln -sfn "../prod/tassili/blocks_ns/$CFG" "$RB" \
                && mark "R7-FIXED $S: $CFG moved into prod/tassili + root shim" \
                || { mark "R7-RED $S: cfg move failed"; RED=$((RED+1)); }
        fi
    elif [ -L "$RB" ] && [ -d "$PB" ]; then
        mark "R7-PASS $S $CFG already in prod (root shim)"
    else
        mkdir -p "$PB" "$R/blocks_ns"
        [ -e "$RB" ] || ln -sfn "../prod/tassili/blocks_ns/$CFG" "$RB"
        mark "R7-FIXED $S: empty $CFG slot pre-placed in prod + root shim (partition lands there)"
    fi

    # ---- R6 embedder -------------------------------------------------------
    case "$S" in
        01_13B_Jackal) TAG=01_13B;; 02_13B_Jackal) TAG=02_13B;; 03_13B_Jackal) TAG=03_13B;;
        04_13D_Jackal) TAG=04_13D;; 05_13D_Jackal) TAG=05_13D;; *) TAG=klap;;
    esac
    if ls -t /home/paperspace/data/high/nerf/${TAG}*/ckpts/model_best.pth >/dev/null 2>&1; then
        mark "R6-PASS $S embedder present"
    else
        mark "R6-IN-SLOT $S embedder absent — [5c] trains it in the first slot"
        NOTES="$NOTES embedder-in-slot;"
    fi

    if [ "$RED" -eq 0 ]; then
        mark ">>> $S READY${NOTES:+ (in-slot: $NOTES)}"
    else
        mark ">>> $S NOT-READY ($RED red gates)"
    fi
    echo " \"$S\": {\"red\": $RED, \"notes\": \"$NOTES\"}," >> "$RJSON.tmp"
    RED_TOTAL=$((RED_TOTAL + RED))
done

echo " \"_generated\": \"$(date -u +%FT%TZ)\", \"_red_total\": $RED_TOTAL }" >> "$RJSON.tmp"
mv "$RJSON.tmp" "$RJSON"
mark "READINESS ${RED_TOTAL} red gates total -> $RJSON"
exit $([ "$RED_TOTAL" -eq 0 ] && echo 0 || echo 1)
