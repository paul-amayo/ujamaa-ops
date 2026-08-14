#!/bin/bash
# Per-block entry point for UCT HPC Slurm arrays — ONE block, ONE GPU.
#
#   hpc_block_chain.sh <SURVEY_ID> <BLOCK_INDEX> [stage]
#   hpc_block_chain.sh --sbatch <SURVEY_ID> [stage]   # print an array script
#
#   stage: all (default) | stage1 | stage2 | verdict
#
# This is the recipe of record (run_unified_pipeline.sh two-stage branch),
# unrolled to a single block so a Slurm array can run N blocks concurrently:
#   a. LiDAR init (metric, from laser.monolithic)
#   a2. per-pass GLOMAP pose refine (raw LIO kept at transforms_lio.json)
#   b0. compile supervision (order-proof palette)
#   b. stage1_bg00 — appearance only, features OFF, 15001 iters
#   c. stage2 census-init fw2 (censusinit_block.sh: census -> majority init
#      -> frozen refine -> verdict battery)
#   d. verdict distilled into blocks_ns/<cfg>/verdicts_censusinit_fw2.json
#      (--merge, per block) — which is what PROD.md reads, so the prod matrix
#      stays honest with no manual step.
#
# HPC-specific behaviour vs the VM pipeline:
#   * every cache that would otherwise land in $HOME (10 GB quota on UCT HPC)
#     is redirected to $UJAMAA_SCRATCH — torch extensions, HF, pip, pixi.
#   * STEPS_PER_SAVE defaults to 5000 (not 14998) so a preempted/requeued
#     array task resumes from the newest ckpt instead of losing the run.
#     Saving more often is numerically inert — it only costs disk, and the
#     intermediates are pruned once the stage lands.
#   * every stage is cache-guarded and idempotent: re-running a completed
#     block is a no-op, so `scontrol requeue` is always safe.
#   * DRY_RUN=1 prints the commands instead of running them.
#
# Paths: everything resolves under $UJAMAA_ROOT (default /home/paperspace) so
# the same script runs on the VM and on HPC via the /home/paperspace symlink
# or a singularity bind — see plans/hpc_migration.md §2.
set -uo pipefail

# ---- sbatch template mode -------------------------------------------------
if [ "${1:-}" = "--sbatch" ]; then
    SURVEY="${2:?usage: $0 --sbatch SURVEY_ID [stage]}"
    STAGE="${3:-all}"
    ROOT=${SURVEY_ROOT:-${UJAMAA_ROOT:-/home/paperspace}/data/citrus_all/$SURVEY}
    CFG=${CONFIG:-$(ls -d "$ROOT"/blocks_ns/*/ 2>/dev/null | head -1 | xargs -r basename)}
    N=$(ls -d "$ROOT/blocks_ns/$CFG"/block_[0-9][0-9][0-9] 2>/dev/null | wc -l)
    cat << SBATCH
#!/bin/sh
#SBATCH --account=${SLURM_ACCOUNT:-aru}
#SBATCH --partition=${SLURM_PARTITION:-a100}
#SBATCH --gres=gpu:ampere:1
#SBATCH --job-name="ujamaa-$SURVEY-$STAGE"
#SBATCH --array=0-$((N > 0 ? N - 1 : 0))%${ARRAY_THROTTLE:-6}
#SBATCH --ntasks=8
#SBATCH --time=${WALLTIME:-04:00:00}
#SBATCH --mail-user=paul.amayo@uct.ac.za
#SBATCH --mail-type=FAIL,END
#SBATCH --requeue
#SBATCH --output=${UJAMAA_SCRATCH:-/scratch/\$USER/ujamaa}/logs/$SURVEY-%A_%a.log

export UJAMAA_SCRATCH=${UJAMAA_SCRATCH:-/scratch/\$USER/ujamaa}
bash ${UJAMAA_ROOT:-/home/paperspace}/code/automation/hpc_block_chain.sh \\
     $SURVEY \$SLURM_ARRAY_TASK_ID $STAGE
SBATCH
    echo "# ^ $N canonical blocks in $CFG" >&2
    exit 0
fi

SURVEY="${1:?usage: $0 SURVEY_ID BLOCK_INDEX [all|stage1|stage2|verdict]}"
IDX="${2:?usage: $0 SURVEY_ID BLOCK_INDEX [all|stage1|stage2|verdict]}"
STAGE="${3:-all}"
BID=$(printf "%03d" "$IDX")

# ---- roots ---------------------------------------------------------------
UJAMAA_ROOT=${UJAMAA_ROOT:-/home/paperspace}
CODE=${UJAMAA_CODE:-$UJAMAA_ROOT/code}
SRC=$CODE/aru_sil_core/src/scripts
NS_PIXI=${NS_PIXI:-$CODE/nerf_new/pixi.toml}
# lidar_init needs plyfile, which nerf_new lacks — InstantSplat's env has it.
# On HPC, point LIDAR_PIXI at NS_PIXI once plyfile is added to that env.
LIDAR_PIXI=${LIDAR_PIXI:-$CODE/aru_sil_core/src/thirdparty/InstantSplat/pixi.toml}
ROOT=${SURVEY_ROOT:-$UJAMAA_ROOT/data/citrus_all/$SURVEY}

# ---- $HOME-quota guards (UCT HPC: /home is 10 GB) -------------------------
if [ -n "${UJAMAA_SCRATCH:-}" ]; then
    export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$UJAMAA_SCRATCH/.torch_ext}
    export HF_HOME=${HF_HOME:-$UJAMAA_SCRATCH/.hf}
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$UJAMAA_SCRATCH/.pipcache}
    export PIXI_HOME=${PIXI_HOME:-$UJAMAA_SCRATCH/.pixi}
    export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$UJAMAA_SCRATCH/.cache}
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$HF_HOME" "$UJAMAA_SCRATCH/logs"
fi

CFG=${CONFIG:-$(ls -d "$ROOT"/blocks_ns/*/ 2>/dev/null | head -1 | xargs -r basename)}
BD=$ROOT/blocks_ns/$CFG/block_$BID
LOGS=${UJAMAA_SCRATCH:-$ROOT/_logs}
mkdir -p "$LOGS"
HIER_JSON=${CENSUS_HIERARCHY:-$ROOT/scene_graph/marker_hierarchy.json}
HYPER_CKPT=${CENSUS_EMBEDDER:-}
SUP=$BD/supervision/trees_only
STEPS_PER_SAVE=${STEPS_PER_SAVE:-5000}
VERDICTS=$ROOT/blocks_ns/$CFG/verdicts_censusinit_fw2.json

say() { echo "[$(date +%H:%M:%S)] [$SURVEY/$BID] $*"; }
run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "    DRY: $*"; return 0; fi
    "$@"
}
die() { say "HPC-BLOCK-FAIL ($1)"; exit 1; }

[ -d "$BD" ] || die "no block dir $BD (array index past the block count?)"
[ -n "$CFG" ] || die "no blocks_ns config under $ROOT"
say "start stage=$STAGE cfg=$CFG bd=$BD"
if [ "${DRY_RUN:-0}" != "1" ] && command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
fi

# ---- a. LiDAR init --------------------------------------------------------
do_init() {
    if [ -f "$BD/init_lidar.ply" ]; then say "lidar init cache hit"; return 0; fi
    say "lidar init"
    run pixi run --manifest-path "$LIDAR_PIXI" python "$SRC/lidar_init_per_block.py" \
        --block-dir "$BD" --root "$ROOT" \
        > "$LOGS/${SURVEY}_block${BID}_lidar_init.log" 2>&1 \
        || die "lidar init"
}

# ---- a2. per-pass pose refine (transforms_lio.json = the swap marker) -----
do_refine() {
    [ "${REFINE_POSES:-1}" = "1" ] || { say "pose refine disabled"; return 0; }
    if [ -f "$BD/transforms_lio.json" ]; then say "poses already refined"; return 0; fi
    say "per-pass pose refine (GLOMAP)"
    if run pixi run --manifest-path "$NS_PIXI" python "$SRC/per_pass_colmap_refine.py" \
            --block-dir "$BD" --root "$ROOT" --log-dir "$LOGS" \
            > "$LOGS/${SURVEY}_block${BID}_ppref.log" 2>&1 \
       && [ -f "$BD/transforms_refined_per_pass.json" ]; then
        run cp "$BD/transforms.json" "$BD/transforms_lio.json"
        run cp "$BD/transforms_refined_per_pass.json" "$BD/transforms.json"
        say "poses refined (raw LIO kept at transforms_lio.json)"
    else
        # non-fatal by design: raw LIO is a valid pose source
        say "pose refine FAILED — continuing on raw LIO"
    fi
}

# ---- b0. compiled supervision --------------------------------------------
do_supervision() {
    if [ -f "$SUP/manifest.json" ]; then say "supervision cache hit"; return 0; fi
    say "compile supervision"
    run python3 "$SRC/compile_supervision.py" --block-dir "$BD" \
        --tree-source colour_png_bridge \
        --hierarchy "$HIER_JSON" \
        --fruit-ledger-glob "$ROOT/sam3_fruit/clip_*/frame_entries.json" \
        --filter strict_fruit_tree_v1 \
        --out-dir "$SUP" \
        > "$LOGS/${SURVEY}_block${BID}_compile_sup.log" 2>&1 \
        && say "supervision compiled" \
        || say "supervision compile FAILED — stage1 photometric, stage2 will skip"
}

# ---- b. stage1_bg00 (requeue-safe: resume from newest ckpt) ---------------
do_stage1() {
    local ck last
    ck=$(ls -t "$BD"/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt 2>/dev/null | head -1)
    if [ -n "$ck" ]; then
        last=$(basename "$ck" | tr -dc '0-9' | sed 's/^0*//')
        if [ "${last:-0}" -ge 14000 ]; then say "stage1 cache hit ($last steps)"; return 0; fi
        say "stage1 RESUME from $last steps (preempted run)"
        RESUME_ARGS=(--load-dir "$(dirname "$ck")")
    else
        RESUME_ARGS=()
    fi
    local TW=""
    [ -f "$SUP/manifest.json" ] && TW=$SUP
    [ -z "$TW" ] && [ -f "$BD/supervision/strict_tree_v2/manifest.json" ] \
        && TW=$BD/supervision/strict_tree_v2
    say "stage1_bg00 (15001 iters, treelod=${TW:-off}, save/${STEPS_PER_SAVE})"
    echo "n" | MAX_JOBS=4 CANARY_EVERY=2000 \
      TREE_WEIGHT_DIR=$TW TREE_WEIGHT_BG=0.0 \
      run pixi run --manifest-path "$NS_PIXI" ns-train high \
        --data "$BD" --output-dir "$BD/splat_runs_STAGE1" \
        --experiment-name stage1_bg00 \
        --pipeline.model.enable-high-features False \
        --pipeline.model.high-loss-weight 0.0 \
        --pipeline.datamanager.semantic-dir "${EMPTY_SEMANTIC:-$UJAMAA_ROOT/logs/empty_semantic}" \
        --pipeline.model.rasterize-mode antialiased \
        --pipeline.model.stop-split-at 6000 \
        --pipeline.model.sky-loss-lambda 1.0 \
        --pipeline.model.report-masked-metrics True \
        --max-num-iterations 15001 --steps-per-save "$STEPS_PER_SAVE" \
        --vis tensorboard nerfstudio-data \
        --eval-mode interval --eval-interval 10 \
        "${RESUME_ARGS[@]}" \
        > "$LOGS/${SURVEY}_block${BID}_stage1.log" 2>&1 \
        || die "stage1_bg00"
    say "stage1_bg00 done"
    # prune intermediate saves (scratch discipline — keep the newest only)
    if [ "${PRUNE_INTERMEDIATE:-1}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        ls -t "$BD"/splat_runs_STAGE1/stage1_bg00/high/*/nerfstudio_models/*.ckpt 2>/dev/null \
            | tail -n +2 | while read -r old; do rm -f "$old"; done
    fi
}

# ---- c. stage2 census-init (includes its own verdict battery) ------------
do_stage2() {
    [ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/strict_tree_v2
    [ -f "$SUP/manifest.json" ] || die "stage2 needs compiled supervision ($SUP/manifest.json)"
    [ -n "$HYPER_CKPT" ] && [ -f "$HYPER_CKPT" ] \
        || die "stage2 needs an embedder — set CENSUS_EMBEDDER"
    if [ -d "$BD/splat_runs_FEATFIX/stage2_censusinit_fw2" ]; then
        say "stage2 cache hit"; return 0
    fi
    say "stage2 census-init fw2"
    CENSUS_EMBEDDER=$HYPER_CKPT CENSUS_HIERARCHY=$HIER_JSON \
      run bash "$CODE/automation/censusinit_block.sh" "$BD" "$SUP" \
        > "$LOGS/${SURVEY}_block${BID}_stage2.log" 2>&1 \
        || die "stage2 census-init"
    say "stage2 census-init done"
    if [ "${PRUNE_INTERMEDIATE:-1}" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
        # stage2_init / stage2_init_census are rebuildable from stage1 + census
        rm -rf "$BD/stage2_init" "$BD/stage2_init_census"
        say "pruned rebuildable stage2 intermediates (~2 G)"
    fi
}

# ---- d. verdict -> machine-readable, merged per block --------------------
# censusinit_block.sh already ran containment_eval, but it prints WITHOUT the
# "[block frame]" prefix the distiller keys on (that prefix came from the
# sweep wrapper). Re-prefix this block's lines into a small log and merge.
do_verdict() {
    local s2log=$LOGS/${SURVEY}_block${BID}_stage2.log
    [ -f "$s2log" ] || die "no stage2 log to distil ($s2log)"
    local frame
    frame=$(grep -aoE "REPL-FRAME: kf_[0-9]+\.png" "$s2log" | tail -1 | awk '{print $2}')
    frame=${frame:-kf_unknown.png}
    local pre=$LOGS/${SURVEY}_block${BID}_verdict_prefixed.log
    if [ "${DRY_RUN:-0}" != "1" ]; then
        grep -aE '^(TREE|ROW) [0-9]+ +"' "$s2log" \
            | sed "s|^|[$BID $frame] |" > "$pre"
        [ -s "$pre" ] || { say "no containment lines in stage2 log — verdict SKIPPED"; return 0; }
        python3 "$CODE/automation/distill_containment_verdicts.py" \
            --log "$pre" --out "$VERDICTS" --merge \
            || die "verdict distil"
    fi
    say "verdict merged into $(basename "$VERDICTS")"
}

case "$STAGE" in
    stage1)  do_init; do_refine; do_supervision; do_stage1 ;;
    stage2)  do_stage2; do_verdict ;;
    verdict) do_verdict ;;
    all)     do_init; do_refine; do_supervision; do_stage1; do_stage2; do_verdict ;;
    *)       die "unknown stage '$STAGE'" ;;
esac

say "HPC-BLOCK-DONE stage=$STAGE"
