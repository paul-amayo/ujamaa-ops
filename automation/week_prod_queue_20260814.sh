#!/bin/bash
# WEEK PROD QUEUE 2026-08-14 (Paul: "a week queue through the gpu, get all
# the datasets to prod" + "do a block of a survey after each other, so we
# get survey coverage and big surveys dont hold us back").
#
# ROUND-ROBIN GPU rotation, one canonical block per survey per slot:
#   05_13D -> 04_13D -> klap apr -> 01_13B -> 03_13B -> 02_13B ->
#   klap ten-rows (gated on pose verification) -> (repeat)
# Per slot: unified pipeline (two-stage census-init, idempotent) -> machine
# verdict (containment -> verdicts_censusinit_fw2.json) -> export+register
# (splats.json). Every stage cache-hits, so kill/resume is free.
# CPU SIDECAR: klapmuts ten-rows ingest (kf cut + PNG extract) — the site
# is under active collection; ingest keeps pace with arriving data.
#
# Front-loaded phase A (cheap prod-cell flips): ledger refresh, bateleur
# topdown exports, 05 verdict backfill (b004 + measured b001), export smoke
# test, 05 block_000 raw-LIO quarantine (refined-consistent retrain).
#
# Markers: A*-DONE/FAIL, SLOT <survey> <block> OK/FAIL, ROUND-n-DONE,
# WEEK-DONE. Log: /home/paperspace/logs/week_prod_20260814.log
# Kill: ps -eo pid,pgid,args | grep "[w]eek_prod_queue" -> kill -TERM -<PGID>
# No rm, no git push, nothing destructive (quarantines are mv).
set -u
LOGS=/home/paperspace/logs
SRC=/home/paperspace/code/aru_sil_core/src/scripts
AUTO=/home/paperspace/code/automation
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
STATE=$LOGS/week_prod_20260814_state
CITRUS=/home/paperspace/data/citrus_all
KLAP=/home/paperspace/data/klapmuts
END_TS=$(( $(date +%s) + 6*86400 + 12*3600 ))   # 6.5 days
mkdir -p "$STATE"
mark() { echo "[$(date +%m-%d\ %H:%M:%S)] $1"; }
exec >> "$LOGS/week_prod_20260814.log" 2>&1
mark "WEEK QUEUE START (budget to $(date -d @$END_TS '+%m-%d %H:%M'))"

# refuse to double-book the GPU — and CLEAN UP ORPHANS from a previous run
# of THIS queue. A restart kills the parent but ns-train ignores SIGTERM
# mid-CUDA-op: orphans held 27G of 34G overnight, so every stage2 OOMed
# (masked as an NVML INTERNAL ASSERT) while returning success. Never
# pkill -f (matches this shell — the self-match trap): resolve PGIDs from
# the device holders and kill those groups explicitly.
ORPHAN_PGIDS=$(fuser /dev/nvidia0 2>/dev/null | tr ' ' '\n' | grep -E "^[0-9]+$" \
    | xargs -r -I{} ps -o pgid= -p {} 2>/dev/null | tr -d ' ' | sort -u)
for PG in $ORPHAN_PGIDS; do
    [ "$PG" = "$$" ] && continue
    ps -o args= -g "$PG" 2>/dev/null | grep -q "[n]s-train" || continue
    ps -o args= -g "$PG" 2>/dev/null | grep -q "splat_viewer" && continue
    mark "GPU-CLEAN killing orphaned trainer group $PG (held VRAM from a previous run)"
    kill -KILL -"$PG" 2>/dev/null
done
sleep 5
if ps -eo args | grep -qE "[n]s-train"; then
    mark "ABORT: ns-train still running after cleanup — GPU busy"; exit 1
fi
# hard VRAM floor: a slot needs ~10G; refuse to start blind
# A FILE, not `python -c`: pixi's deno_task_shell eats the -c quoting and
# returns nothing, which turned this hard floor into a no-op.
FREE_G=$(cd /home/paperspace/code/nerf_new && pixi run python \
    "$AUTO/free_vram_gb.py" 2>/dev/null | tail -1)
if [ -n "$FREE_G" ] && [ "$FREE_G" -lt 10 ]; then
    mark "ABORT: only ${FREE_G}G VRAM free (need >=10G) — investigate holders: fuser -v /dev/nvidia0"
    exit 1
fi
mark "GPU ready: ${FREE_G:-?}G free"

# ---------------- pre-prod readiness phase ---------------------------------
# Paul 2026-08-15: readiness MAKES surveys ready — it fixes/regenerates
# stale prod inputs (pose aliases, kf extraction, stream-mask quarantine +
# kf rebuild, supervision refresh + registry-derived probes) and REPORTS
# what stayed stale. THERE IS NO HALT: surveys are self-contained; if the
# information is there they train, and per-block in-slot guards handle the
# rest. readiness.json carries the stale report.
bash /home/paperspace/code/automation/preprod_readiness.sh
mark "READINESS PHASE COMPLETE (stale report in readiness.json) — entering training rotation"

export CENSUS_SEED_ONLY=${CENSUS_SEED_ONLY:-1}   # seed-only stage2 (2026-08-17: fw2 moved 02 b000 <0.01 over 5000 iters)

# ---------------- rotation config ------------------------------------------
# All splat-eligible surveys rotate — including 02 (Paul 2026-08-14: no
# reason to exclude the control epoch; stage2 is contingent on its painted
# semantics, compile fails loudly if absent) and ten-rows (GATED: joins the
# moment $STATE/tenrows_poses.ok appears — the INS odom is FP-ENU0-frame,
# not camera-frame LIO; wiring transform_lio without verifying the frame
# convention would train plausible-looking garbage, so that verification is
# daylight work, and the gate lets it join mid-week without a restart).
# dec_2025_ten_rows is OUT of the rotation (2026-08-18, Paul: "klapmuts
# ten rows needs to be looked at"). Open items: monos nest one level
# deeper (prod/monos/monolithics) unlike every other survey; two orphaned
# index files sit in prod/monos with no mono beside them; kf domain does
# not match (1790 kf_images vs 1792 mono entries), which [0c] now treats
# as a hard stop; and it has no markers_v2 or marker_hierarchy yet. Its
# root_of entry stays so the survey can still be driven by hand.
SURVEYS=(05_13D_Jackal 04_13D_Jackal apr_2026_zed 01_13B_Jackal 03_13B_Jackal 02_13B_Jackal)
root_of()  { case "$1" in
    apr_2026_zed) echo "$KLAP/apr_2026_zed";;
    dec_2025_ten_rows) echo "$KLAP/dec_2025_ten_rows";;
    *) echo "$CITRUS/$1";; esac; }
cfg_of()   { echo lio_row100; }   # pipeline ALWAYS partitions to its own CONFIG name — 04 lio_row6F assumption parked a finished 3h run
maxblk_of() { echo ""; }   # block counts derive from the partition ON DISK, never hardcoded — the 6F-era "11" would have silently stopped 04 at block_010 of 48
emb_of() {  # DERIVE exactly as the pipeline does (${SURVEY%_Jackal}_v1g),
    # never a hand-kept map: the old map scored apr with klapmuts_v1 and 04
    # with the retired v3vocab1k vocabulary -> containment 0.00. The verdict
    # itself now prefers stage2_provenance.json; this is the fallback.
    # The embedder lives with its survey in prod/bateleur (2026-08-18) — one
    # physical copy, never a second in a shared tree that can go stale.
    local exp="${1%_Jackal}_v1g"
    local p="$(root_of "$1")/prod/bateleur/embedder/$exp/ckpts/model_best.pth"
    [ -f "$p" ] && { echo "$p"; return; }
    ls -t "$(root_of "$1")/prod/bateleur/embedder/${1%_Jackal}"*/ckpts/model_best.pth 2>/dev/null | head -1
}
# The hierarchy lives in prod/bateleur/scene_graph (2026-08-18). The old
# "$ROOT"/scene_graph* glob resolved through a shim and would now silently
# return nothing, scoring every block against an empty hierarchy.
hier_of() { echo "$(root_of "$1")/prod/bateleur/scene_graph/marker_hierarchy.json"; }

# ---------------- phase A: cheap flips -------------------------------------
if [ ! -f "$STATE/A.done" ]; then
    mark "PHASE A start"
    bash "$AUTO/regression_harness.sh" --cpu-only > "$LOGS/week_A0_harness.log" 2>&1 \
        && mark "A0-DONE tier-A harness" || mark "A0-FAIL tier-A harness (continuing)"

    python3 /home/paperspace/code/ujamaa/sankofa/build_ledger_v2.py \
        > "$LOGS/week_A1_ledger.log" 2>&1 \
        && mark "A1-DONE ledger_v2 refresh" || mark "A1-FAIL ledger_v2 refresh"
    grep -q "05_13D" "$LOGS/week_A1_ledger.log" 2>/dev/null \
        && mark "A1-NOTE 05 present in ledger output" \
        || mark "A1-NOTE 05 NOT in ledger (builder needs assoc_04_05 support — daylight code work)"

    for S in 01_13B_Jackal 02_13B_Jackal 03_13B_Jackal 04_13D_Jackal 05_13D_Jackal; do
        cd /home/paperspace/code/nerf_new && pixi run python \
            /home/paperspace/code/ujamaa/project/export_bateleur_topdown.py "$CITRUS/$S" \
            > "$LOGS/week_A2_topdown_$S.log" 2>&1 \
            && mark "A2-DONE topdown $S" || mark "A2-FAIL topdown $S"
    done
    cd /home/paperspace/code/nerf_new && pixi run python \
        /home/paperspace/code/ujamaa/project/export_bateleur_topdown.py "$KLAP/apr_2026_zed" \
        > "$LOGS/week_A2_topdown_apr.log" 2>&1 \
        && mark "A2-DONE topdown apr_2026_zed" || mark "A2-FAIL topdown apr_2026_zed"

    # 05 verdict backfill: block_004 (sweep gap) + block_001 (measured
    # replaces the injected session record)
    CFG05=$CITRUS/05_13D_Jackal/blocks_ns/lio_row100
    for B in block_004 block_001; do
        CENSUS_EMBEDDER=$(emb_of 05_13D_Jackal) CENSUS_HIERARCHY=$(hier_of 05_13D_Jackal) \
        bash "$AUTO/verdict_block.sh" "$CFG05/$B" "$CFG05/$B/supervision/trees_only" \
            && mark "A3-DONE verdict $B" || mark "A3-FAIL verdict $B"
    done

    # splat exports RETIRED (Paul 2026-08-20): tassili renders directly from
    # checkpoints now — no .ply export/registration in rotation. The
    # export.ok flag is never created; both rotation-export call sites are
    # gated on it and stay dormant.
    mark "A4-SKIP splat exports retired (tassili renders from ckpts)"

    # 05 block_000: quarantine raw-LIO-stage1 runs -> refined-consistent retrain
    B0=$CFG05/block_000
    Q=$CITRUS/05_13D_Jackal/experimental/blocks_ns/lio_row100_block000_rawlio_runs
    if [ -d "$B0/splat_runs_STAGE1" ] && [ ! -d "$Q" ]; then
        mkdir -p "$Q"
        for D in splat_runs_STAGE1 splat_runs_FEATFIX stage2_init stage2_init_census; do
            [ -e "$B0/$D" ] && mv "$B0/$D" "$Q/$D"
        done
        # refined poses already swapped in transforms.json (mixed-pose was
        # stage1-vs-stage2); keep them and retrain both stages consistent
        mark "A5-DONE block_000 raw-LIO runs quarantined -> $Q"
    else
        mark "A5-SKIP block_000 quarantine ($([ -d "$Q" ] && echo already done || echo no stage1 present))"
    fi
    touch "$STATE/A.done"
    mark "PHASE A complete"
fi

# ---------------- apr one-time prep: kf-named mask rebuild -----------------
apr_prep() {
    local R; R=$(root_of apr_2026_zed)
    [ -f "$STATE/apr_masks.done" ] && return 0
    mark "APR-PREP: quarantining stream-named masks (kf-named rebuild follows in pipeline [4b])"
    mkdir -p "$R/experimental/masks_streamnamed_pre_kf_rebuild"
    local D
    for D in sky_masks fg_masks; do
        if [ -L "$R/$D" ]; then
            local TGT; TGT=$(readlink -f "$R/$D")
            mv "$TGT" "$R/experimental/masks_streamnamed_pre_kf_rebuild/$D" 2>/dev/null
            unlink "$R/$D" 2>/dev/null
        elif [ -d "$R/$D" ]; then
            mv "$R/$D" "$R/experimental/masks_streamnamed_pre_kf_rebuild/$D"
        fi
    done
    touch "$STATE/apr_masks.done"
    mark "APR-PREP done (pipeline will rebuild kf-named masks, ~40 GPU-min)"
}

# ---------------- CPU sidecar: klap ten-rows ingest ------------------------
# DISABLED 2026-08-18 with ten_rows out of the rotation. It also addressed
# $KLAP/dec_2025_ten_rows/monolithics and /kf_images, which are prod paths now
# (prod/monos/monolithics, prod/tassili/kf_images) — it would rebuild the kf
# cut at the survey root and recreate the split-brain we just cleared.
TENROWS_SIDECAR=${TENROWS_SIDECAR:-0}
[ "$TENROWS_SIDECAR" = "1" ] && (
    exec >> "$LOGS/week_cpu_sidecar.log" 2>&1
    mark "CPU sidecar start (ten-rows ingest)"
    TR=$KLAP/dec_2025_ten_rows/monolithics
    if [ ! -f "$TR/image_left_kf20cm.monolithic" ]; then
        /home/paperspace/code/aru_sil_core/build/bin/dump_keyframe_image_monolithic \
            --IMAGE_MONO "$TR/image_left.monolithic" \
            --TRANSFORM_MONO "$TR/zed_transform.monolithic" \
            --MIN_DIST 0.20 \
            --OUT_MONO "$TR/image_left_kf20cm.monolithic" \
            && mark "CPU-DONE ten-rows kf cut" || mark "CPU-FAIL ten-rows kf cut"
    else
        mark "CPU-SKIP ten-rows kf mono exists"
    fi
    if [ -f "$TR/image_left_kf20cm.monolithic" ] \
       && [ ! -d "$KLAP/dec_2025_ten_rows/kf_images" ]; then
        cd /home/paperspace/code/nerf_new && pixi run python "$SRC/extract_kf_pngs.py" \
            --data-dir "$KLAP/dec_2025_ten_rows" \
            --mono "$TR/image_left_kf20cm.monolithic" \
            && mark "CPU-DONE ten-rows kf_images ($(ls "$KLAP/dec_2025_ten_rows/kf_images" 2>/dev/null | wc -l) PNGs)" \
            || mark "CPU-FAIL ten-rows extract"
    fi
    mark "CPU sidecar done (gps.monolithic write + hierarchy = daylight code work)"
) &

# ---------------- rotation --------------------------------------------------
ROUND=$(cat "$STATE/round" 2>/dev/null || echo 0)
while [ "$(date +%s)" -lt "$END_TS" ]; do
    ROUND=$((ROUND + 1)); echo "$ROUND" > "$STATE/round"
    ACTIVE=0
    for S in "${SURVEYS[@]}"; do
        [ "$(date +%s)" -ge "$END_TS" ] && break
        # DISK FLOOR (2026-08-20): twice in one night ENOSPC killed this
        # queue mid-write — fast-failing every slot (1 s each, pointers
        # advancing past untried blocks) and truncating .next files to
        # empty. A full disk must stop the rotation LOUDLY and CLEANLY,
        # before any state write, not corrupt it.
        AVAIL_G=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
        if [ -n "$AVAIL_G" ] && [ "$AVAIL_G" -lt 15 ]; then
            mark "ABORT-DISK: only ${AVAIL_G}G free on / (floor 15G) — stopping cleanly before state corruption; clear space and relaunch"
            exit 1
        fi
        R=$(root_of "$S"); C=$(cfg_of "$S")
        NEXT=$(cat "$STATE/$S.next" 2>/dev/null || echo 0)
        MAX=$(maxblk_of "$S")
        # discovered max once the partition exists. ZERO blocks = the
        # partition hasn't been built yet (R7 pre-places an EMPTY cfg in
        # prod) — that is UNKNOWN, not complete: the first slot creates the
        # blocks. count-1=-1 once silently skipped 01 and ten-rows forever.
        if [ -z "$MAX" ] && [ -d "$R/prod/tassili/blocks_ns/$C" ]; then
            NBLK=$(ls -d "$R/prod/tassili/blocks_ns/$C"/block_* 2>/dev/null \
                     | grep -cE "block_[0-9]+$")
            if [ "$NBLK" -gt 0 ]; then
                MAX=$((NBLK - 1))
                echo "$MAX" > "$STATE/$S.max"
            fi
        fi
        [ -z "$MAX" ] && MAX=$(cat "$STATE/$S.max" 2>/dev/null || echo "")
        if [ -n "$MAX" ] && [ "$NEXT" -gt "$MAX" ]; then
            continue   # survey complete
        fi
        # ten-rows waits for the pose-domain verification marker
        if [ "$S" = "dec_2025_ten_rows" ] && [ ! -f "$STATE/tenrows_poses.ok" ]; then
            [ -f "$STATE/tenrows.waitnote" ] || {
                mark "TENROWS-WAIT: pose-domain unverified (INS is ENU0-frame);"\
" touch $STATE/tenrows_poses.ok after daylight wiring to admit it"
                touch "$STATE/tenrows.waitnote"; }
            continue
        fi
        ACTIVE=1
        [ "$S" = "apr_2026_zed" ] && apr_prep
        BID=$(printf '%03d' "$NEXT")
        mark "SLOT $S block_$BID (round $ROUND)"
        T0=$(date +%s)
        case "$R" in
            "$CITRUS"/*)
                bash "$SRC/run_unified_pipeline.sh" "$S" "$NEXT" "$NEXT" \
                    > "$LOGS/week_${S}_b${BID}.log" 2>&1 ;;
            *)
                SURVEY_ROOT=$R bash "$SRC/run_unified_pipeline.sh" "$S" "$NEXT" "$NEXT" \
                    > "$LOGS/week_${S}_b${BID}.log" 2>&1 ;;
        esac
        ELAPSED=$(( $(date +%s) - T0 ))
        BD=$R/prod/tassili/blocks_ns/$C/block_$BID
        if ls "$BD"/splat_runs_FEATFIX/stage2_censusinit_*/high/*/nerfstudio_models*/*.ckpt >/dev/null 2>&1; then
            SUP=$BD/supervision/trees_only
            [ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/strict_tree_v2
            EMB=$(emb_of "$S"); HJ=$(hier_of "$S")
            if [ -n "$EMB" ] && [ -n "$HJ" ] && [ -f "$SUP/manifest.json" ]; then
                CENSUS_EMBEDDER=$EMB CENSUS_HIERARCHY=$HJ \
                    bash "$AUTO/verdict_block.sh" "$BD" "$SUP" \
                    || mark "SLOT $S block_$BID verdict FAIL"
            fi
            [ -f "$STATE/export.ok" ] && { bash "$AUTO/export_register_stage2.sh" "$BD" \
                || mark "SLOT $S block_$BID export FAIL"; }
            # clip-cache eviction (Paul 2026-08-20): target tables cost 2.3G
            # per EMBEDDER GENERATION per block and were never cleaned —
            # audit found 103 dirs / 245.7G, ~85% keyed to retired
            # embedders. Keep the newest-built generation (the one this
            # successful run built or reused), evict older siblings. Pure
            # derived data: an over-eviction rebuilds itself in minutes at
            # the next train, so keep-newest is safe by construction.
            EVICT=$(ls -dt "$BD"/clip_cache_* 2>/dev/null | tail -n +2)
            if [ -n "$EVICT" ]; then
                EG=$(du -xsck $EVICT 2>/dev/null | tail -1 | cut -f1)
                rm -rf $EVICT
                mark "SLOT $S block_$BID clip-cache evict: $(( ${EG:-0} / 1048576 ))G freed (kept newest generation)"
            fi
            mark "SLOT $S block_$BID OK"
            echo 0 > "$STATE/consec_fails"   # any success resets the breaker
            echo $((NEXT + 1)) > "$STATE/$S.next"   # advance ONLY on OK
        elif AUD=$( (cd /home/paperspace/code/nerf_new && pixi run python \
                "$SRC/audit_supervision_density.py" --supervision-dir "$BD/semantic_v2_B") 2>/dev/null | tail -1) \
             && [ -n "$AUD" ] \
             && [ "$(echo "$AUD" | grep -oaE 'ids=[0-9]+' | cut -d= -f2)" -lt 4 ]; then
            # data-reality skip (not staleness): this block's paint is below
            # the census floor (03-entry-block class) — nothing to census
            # here; record and move on rather than halt
            mark "SLOT $S block_$BID SUP-SPARSE ($AUD) — advancing without stage2"
            echo $((NEXT + 1)) > "$STATE/$S.next"
        else
            # self-contained surveys, no halt, no parking: record the failed
            # block loudly, advance past it (each block is tried ONCE), and
            # keep every other survey training. FAILED markers surface in
            # the log + state dir for daylight triage.
            touch "$STATE/$S.block_$BID.FAILED"
            mark "SLOT $S block_$BID FAILED after ${ELAPSED}s — recorded"\
" ($STATE/$S.block_$BID.FAILED; see week_${S}_b${BID}.log) — advancing"
            echo $((NEXT + 1)) > "$STATE/$S.next"
            # CIRCUIT BREAKER (2026-08-20): fail-and-advance is right for
            # idiosyncratic bad blocks but catastrophic under a SYSTEMIC bug
            # — the kf-image regression fake-failed ~25 slots across all six
            # surveys over 7 h while pointers marched on. Four consecutive
            # failures with no OK in between is no longer block-specific:
            # stop loudly and leave state intact for triage.
            CF=$(( $(cat "$STATE/consec_fails" 2>/dev/null || echo 0) + 1 ))
            echo "$CF" > "$STATE/consec_fails"
            if [ "$CF" -ge 4 ]; then
                mark "CIRCUIT-BREAKER: $CF consecutive slot failures across surveys — systemic bug likely (cf. 2026-08-20 kf-image regression). PAUSING; run automation/audit_pointers.sh, fix the cause, rewind pointers, relaunch."
                exit 1
            fi
        fi
    done
    # round-end backfill: any block with a stage2 ckpt but a missing verdict
    # entry or registration gets swept here — GPU is quiet between rounds,
    # which is exactly what the verdict OOM diagnosis calls for
    for S in "${SURVEYS[@]}"; do
        [ -f "$STATE/$S.parked" ] && continue
        R=$(root_of "$S"); C=$(cfg_of "$S")
        VJ=$R/prod/tassili/blocks_ns/$C/verdicts_censusinit_fw2.json
        for BD in "$R/prod/tassili/blocks_ns/$C"/block_*; do
            [ -d "$BD" ] || continue
            BN=$(basename "$BD"); BNUM=${BN#block_}
            echo "$BN" | grep -qE "^block_[0-9]+$" || continue
            ls "$BD"/splat_runs_FEATFIX/stage2_censusinit_*/high/*/nerfstudio_models*/*.ckpt >/dev/null 2>&1 || continue
            HAS=$(python3 -c "import json,sys; d=json.load(open('$VJ')) if __import__('os').path.exists('$VJ') else {'blocks':{}}; print(int('$BNUM' in d.get('blocks',{})))" 2>/dev/null || echo 0)
            SUP=$BD/supervision/trees_only
            [ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/strict_tree_v2
            if [ "$HAS" != "1" ] && [ -f "$SUP/manifest.json" ]; then
                mark "BACKFILL verdict $S $BN"
                CENSUS_EMBEDDER=$(emb_of "$S") CENSUS_HIERARCHY=$(hier_of "$S") \
                    bash "$AUTO/verdict_block.sh" "$BD" "$SUP" || true
            fi
            if [ -f "$STATE/export.ok" ] \
               && [ ! -f "$BD/splats/splat_cropped_stage2_censusinit_fw2.ply" ]; then
                mark "BACKFILL export $S $BN"
                bash "$AUTO/export_register_stage2.sh" "$BD" || true
            fi
        done
    done
    python3 "$AUTO/build_prod_manifests.py" --migrate > "$LOGS/week_prodmd_r${ROUND}.log" 2>&1 \
        && mark "ROUND-$ROUND-DONE (PROD.md regenerated)" \
        || mark "ROUND-$ROUND-DONE (prod regen FAILED)"
    [ "$ACTIVE" = "0" ] && { mark "NO ACTIVE SURVEYS (parked:"\
" $(ls "$STATE"/*.parked 2>/dev/null | wc -l) — clear .parked markers to resume)"; break; }
done
python3 "$AUTO/build_prod_manifests.py" --migrate > "$LOGS/week_prodmd_final.log" 2>&1
mark "WEEK-DONE (rounds: $ROUND)"
