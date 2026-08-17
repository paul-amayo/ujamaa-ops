#!/bin/bash
# ESCALATION LADDER for a survey whose supervision chain is broken.
#   usage: diagnose_supervision_chain.sh <survey_id> [<block_dir>]
#
# Paul 2026-08-16: "when this fails twice this should launch the next
# diagnostic ... when all diagnostics fail then a human will be involved."
# Readiness calls this when the embedder repair fails twice OR a verdict
# lands ~0. Each tier FIXES and re-verifies, or narrows and escalates.
# Everything lands in a dossier; a human is called last, with evidence.
#
#   T3 oracle        can the embedder separate THIS block's own supervised
#                    trees on GT alone? splits the chain: upstream vs down
#   T2 hierarchy     structure the embedder trains on (PCA-trap rows,
#                    singletons, duplicate ids) -> rebuild + retrain + verify
#   T5 feature field trained ckpt uniform? (and the census seed, for
#                    "seeded flat" vs "stage2 washed it out")
#   T4 census        did supervision reach the rasterizer at all
#   T6 scoring       metric-side sanity (frame choice, threshold band)
#   T7 dossier       bundle + one honest unexplained-step line
# Exit 0 = resolved by a tier; 1 = ladder exhausted, human needed.
set -uo pipefail
S=$1
SRC=/home/paperspace/code/aru_sil_core/src/scripts
NS=/home/paperspace/code/nerf_new
CITRUS=/home/paperspace/data/citrus_all
KLAP=/home/paperspace/data/klapmuts
case "$S" in
    apr_2026_zed|dec_2025_*) R=$KLAP/$S;; *) R=$CITRUS/$S;;
esac
CFG=lio_row100
BD=${2:-$(ls -d "$R/blocks_ns/$CFG"/block_* 2>/dev/null | grep -E "block_[0-9]+$" | head -1)}
DOSS=$R/experimental/diagnosis_$(date +%Y%m%d_%H%M)
mkdir -p "$DOSS"
CK=/home/paperspace/data/high/nerf/${S%_Jackal}_v1g/ckpts/model_best.pth
HJ=$(ls -t "$R"/scene_graph*/marker_hierarchy*.json 2>/dev/null | head -1)
say() { echo "[ladder $S] $*" | tee -a "$DOSS/ladder.log"; }
run() { (cd "$NS" && pixi run python "$@") ; }

say "block=$(basename "$BD") embedder=$CK hierarchy=$HJ"

# ---- T3 oracle: upstream vs downstream ------------------------------------
run "$SRC/oracle_relevancy.py" --block-dir "$BD" --ckpt "$CK" \
    > "$DOSS/t3_oracle.log" 2>&1
T3=$?
say "T3 oracle: $(grep -aE 'ORACLE-(PASS|FAIL)|oracle_top1' "$DOSS/t3_oracle.log" | head -2 | tr '\n' ' ')"

if [ $T3 -ne 0 ]; then
    # ---- T2 hierarchy structure (upstream) --------------------------------
    run "$SRC/audit_hierarchy_structure.py" --hierarchy "$HJ" \
        > "$DOSS/t2_hierarchy.log" 2>&1
    T2=$?
    say "T2 hierarchy: $(grep -aE 'HIER-(OK|FAIL|WARN)' "$DOSS/t2_hierarchy.log" | head -2 | tr '\n' ' ')"
    if [ $T2 -ne 0 ]; then
        say "T2 FIX: rebuilding hierarchy (trajectory-derived rows) + retraining embedder"
        run "$SRC/build_marker_hierarchy.py" \
            --semantic-monolithic "$R/filtered_semantic_v2.monolithic" \
            --marker-monolithic "$(ls "$R"/scene_graph*/markers_v2*.monolithic | head -1)" \
            --data-dir "$R" --out "$R/scene_graph/marker_hierarchy.json" \
            > "$DOSS/t2_rebuild.log" 2>&1
        mv "$(dirname "$(dirname "$CK")")" "$R/experimental/collapsed_embedder_$(date +%s)" 2>/dev/null
        run "$SRC/../interfaces/rerun/HiGH/train_hyperembedder_graph.py" \
            --hierarchy-json "$R/scene_graph/marker_hierarchy.json" \
            --experiment-name "${S%_Jackal}_v1g" \
            --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 \
            --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
            --epochs 1500 --no-level-norms > "$DOSS/t2_retrain.log" 2>&1
        if run "$SRC/oracle_relevancy.py" --block-dir "$BD" --ckpt "$CK" \
             > "$DOSS/t2_verify.log" 2>&1; then
            say "T2 RESOLVED: hierarchy rebuild + retrain restored separation "
            say "$(grep -a oracle_top1 "$DOSS/t2_verify.log" | head -1)"
            exit 0
        fi
        say "T2 did not resolve — escalating"
    fi
    # upstream but hierarchy structurally fine -> the trainer itself
    say "T7 HUMAN: oracle fails with a structurally valid hierarchy — the "
    say "    embedder TRAINING is the unexplained step (see $DOSS/t2_retrain.log);"
    say "    next lever is the trainer's own objective/vocab, not the pipeline"
    exit 1
fi

# ---- oracle passed: fault is downstream -----------------------------------
S2CK=$(ls -t "$BD"/splat_runs_FEATFIX/stage2_censusinit_fw2/high/*/nerfstudio_models*/*.ckpt 2>/dev/null | head -1)
SEED=$(ls -t "$BD"/stage2_init_census/nerfstudio_models/*.ckpt 2>/dev/null | head -1)
if [ -n "$S2CK" ]; then
    run "$SRC/inspect_feature_field.py" --ckpt "$S2CK" > "$DOSS/t5_field.log" 2>&1
    say "T5 trained field: $(grep -aE 'FIELD-|pair_cos_mean' "$DOSS/t5_field.log" | head -2 | tr '\n' ' ')"
fi
if [ -n "$SEED" ]; then
    run "$SRC/inspect_feature_field.py" --ckpt "$SEED" > "$DOSS/t5_seed.log" 2>&1
    say "T5 census seed: $(grep -aE 'FIELD-|pair_cos_mean' "$DOSS/t5_seed.log" | head -2 | tr '\n' ' ')"
fi
W=$BD/splat_runs_FEATFIX/interaction_W.npz
if [ -f "$W" ]; then
    python3 "$SRC/inspect_census.py" --w-npz "$W" > "$DOSS/t4_census.log" 2>&1
    say "T4 census: $(grep -aE 'CENSUS-|labelled_frac' "$DOSS/t4_census.log" | head -2 | tr '\n' ' ')"
    if grep -qa "CENSUS-EMPTY\|CENSUS-DEGENERATE" "$DOSS/t4_census.log"; then
        say "T4 FIX: dropping census + init so the block's next slot rebuilds them"
        mv "$W" "$DOSS/interaction_W.stale.npz" 2>/dev/null
        mv "$BD/stage2_init_census" "$DOSS/stage2_init_census.stale" 2>/dev/null
        mv "$BD/splat_runs_FEATFIX/stage2_censusinit_fw2" "$DOSS/stage2_run.stale" 2>/dev/null
        say "T4 RESOLVED-PENDING: census artefacts quarantined; slot re-runs stage2"
        exit 0
    fi
fi
say "T6 scoring: re-score with the alternative formula / frame is the next lever"
say "T7 HUMAN: oracle PASSES (embedder fine) and census+field look populated —"
say "    the unexplained step is between a healthy field and the verdict"
say "    (scoring formula, frame choice, or the relevancy path). Dossier: $DOSS"
exit 1
