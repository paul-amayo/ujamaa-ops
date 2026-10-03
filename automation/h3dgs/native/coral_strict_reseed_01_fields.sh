#!/bin/bash
# Re-seed every 01 field against the retrained embedder (Paul 2026-10-03: "yeah retrain reseed"; companion of
# coral_strict_swap_01.sh, which re-seeded chunk_3_1_sam3). Census weights are reused (embedder-independent); only the
# targets change. Old seeds MOVE to an era_census07_20260822/ folder beside them (they pair with the quarantined embedder
# in experimental/era_census07_20260822/embedder_01_13B_v1g) — nothing is deleted.
#   PART=sidecar : the live Citrus A identity side-car (h3dgs_sidecar_chunks/chunk_3_1, flags of sidecar_chunk_v2.sh)
#   PART=native  : experimental/h3dgs_native/chunk_3_1 features_census.bin (flags from its .json) + its text bank
#   PART=fleet   : prod lio_row100 block_NNN stage2_censusinit_glref seeds (flags of censusinit_block_glref.sh = defaults)
set -uo pipefail
PART=${1:?sidecar|native|fleet}
S=/home/paperspace/data/citrus_all/01_13B_Jackal; EMB=$S/prod/bateleur/embedder/01_13B_v1g/ckpts/model_best.pth; HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NAT=/home/paperspace/code/automation/h3dgs/native; ERA=era_census07_20260822
L=/home/paperspace/logs/coral_strict_reseed_01_$PART.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
cd /home/paperspace/code/nerf_new
reseed_ckpt() {   # <run nerfstudio_models dir> <W npz> <era dir> [census-init flags...]
    local MD=$1 W=$2 E=$3; shift 3
    local CK; CK=$(ls "$MD"/*.ckpt "$E"/*.ckpt 2>/dev/null | head -1); [ -n "$CK" ] || { say "  no seed ckpt in $MD"; return 1; }
    local NAME; NAME=$(basename "$CK"); CK=$MD/$NAME     # seeds keep stage 1's step name (step-000005000/10000/15000)
    mkdir -p "$E"
    if [ -f "$E/$NAME" ]; then say "  era copy already there ($E/$NAME) — re-using it as the source"; else mv "$CK" "$E/$NAME" || return 1; fi
    HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz "$W" --embedder $EMB --src-ckpt "$E/$NAME" --dst-dir "$MD" "$@" 2>&1 \
        | grep -aE 'background|floors|assigned|census-init checkpoint|Error|Traceback' | sed 's/^/    /' | tee -a $L
    [ -f "$MD/$NAME" ] && [ "$MD/$NAME" -nt "$EMB" ]
}
case $PART in
sidecar)
    O=$S/experimental/h3dgs_sidecar_chunks/chunk_3_1; RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ | head -1)
    say "side-car chunk_3_1: $RUN"
    reseed_ckpt "$RUN/nerfstudio_models" "$O/splat_runs_FEATFIX/interaction_W_glref_bg.npz" "$O/splat_runs_FEATFIX/$ERA" --tree-floor 1.0 --bg-competes --bg-ratio 2 \
        && say "side-car re-seeded" || say "side-car FAILED" ;;
native)
    N=$S/experimental/h3dgs_native/chunk_3_1; mkdir -p $N/$ERA
    for f in features_census.bin features_census.bin.json text_bank.npz; do [ -e $N/$f ] && [ ! -e $N/$ERA/$f ] && mv $N/$f $N/$ERA/$f; done
    HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_census.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'assigned|node features|Error' | sed 's/^/    /' | tee -a $L
    MAN=$(pixi run python -c "import numpy as np, json; print(json.loads(str(np.load('$N/$ERA/text_bank.npz', allow_pickle=True)['meta']))['manifest'])" 2>/dev/null | tail -1)
    pixi run python $NAT/build_text_bank.py --manifest "$MAN" --hierarchy-json $HJ --out $N/text_bank.npz 2>&1 | grep -a "text-bank\|Error" | tee -a $L
    say "native chunk_3_1 re-seeded" ;;
fleet)
    B=$S/prod/tassili/blocks_ns/lio_row100; ok=0; fail=0
    for BD in $B/block_[0-9][0-9][0-9]; do
        RUN=$(ls -d $BD/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/ 2>/dev/null | head -1)
        W=$BD/splat_runs_FEATFIX/interaction_W_glref.npz
        if [ -z "$RUN" ] || [ ! -f "$W" ]; then say "$(basename $BD): no seed or no census W — skipped"; continue; fi
        if [ -f $BD/splat_runs_FEATFIX/stage2_reseed_coral985.json ]; then say "$(basename $BD): already re-seeded"; ok=$((ok+1)); continue; fi
        say "$(basename $BD)"
        if reseed_ckpt "$RUN/nerfstudio_models" "$W" "$BD/splat_runs_FEATFIX/$ERA"; then
            printf '{"embedder": "%s", "embedder_mtime": "%s", "hierarchy": "%s", "row_solver": "coral 0.985", "census_w": "%s", "old_seed": "%s"}\n' \
              "$EMB" "$(date -r $EMB '+%F %T')" "$HJ" "$W" "$(ls $BD/splat_runs_FEATFIX/$ERA/*.ckpt | head -1)" > $BD/splat_runs_FEATFIX/stage2_reseed_coral985.json
            ok=$((ok+1))
        else fail=$((fail+1)); say "$(basename $BD) FAILED"; fi
    done
    say "fleet re-seed done: ok $ok, failed $fail" ;;
esac
