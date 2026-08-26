#!/bin/bash
# M2 fruit chain, prod-path port of chain_04_fruitfix.sh (2026-08-25).
#
#   usage: fruit_chain.sh <survey> [--steps 1|all] [--blocks N,M,..] [--floor GB]
#   e.g.   fruit_chain.sh 05_13D_Jackal --steps 1
#
# WHY A PORT: chain_04_fruitfix.sh encodes the canonical order (trees first,
# fruit WITHIN trees, 2x crop context + parent rejection) and is the recipe
# validated on 01 in July. But it addresses the PRE-MIGRATION layout —
# <survey>/blocks_ns/lio_row6F, <survey>/sam3_fruit_tree_bN, and an embedder
# under /home/paperspace/data/high/nerf/ — all superseded on 2026-08-18.
# Running it as-is would recreate the split-brain that migration cleared.
#
# LEDGER PLACEMENT: fruit_in_trees_ledger.py writes <data-dir>/<out-name>/clip_000.
# run_unified_pipeline.sh globs prod/bateleur/sam3_fruit/clip_*/frame_entries.json,
# so each block's clip_000 is MOVED to sam3_fruit/clip_<NNN>. That yields exactly
# what the existing glob expects — no shim, no second convention.
#
# DISK: unlike the fleet queue this chain has no built-in ABORT-DISK, so the floor
# is enforced here. Step 1 is ~200 MB/survey; the step-3 re-seed writes a NEW
# timestamped seed per block (~0.5 GB x n_blocks) and is what actually needs room.
#
# COST: the plan's '~8 h GPU each' for the re-seed is 3.5x pessimistic. Measured
# over 294 fleet blocks, census-init runs at a MEDIAN of 2.9 min (p90 3.4, max
# 4.3), so 04's 48 blocks is ~2.3 h and 05's 43 is ~2.1 h.
set -u

SURVEY=${1:?usage: fruit_chain.sh <survey> [--steps 1|all] [--blocks N,M] [--floor GB]}
shift
STEPS=1; ONLY=""; FLOOR=15   # bare ENOSPC catch only; raise with --floor if wanted
while [ $# -gt 0 ]; do
    case $1 in
        --steps)  STEPS=$2; shift 2;;
        --blocks) ONLY=$2;  shift 2;;
        --floor)  FLOOR=$2; shift 2;;
        *) echo "unknown arg $1"; exit 2;;
    esac
done

export PATH=/home/paperspace/.local/bin:/home/paperspace/.pixi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ARU=/home/paperspace/code/aru_sil_core
SCR=$ARU/src/scripts
NS_PIXI=/home/paperspace/code/nerf_new/pixi.toml
SAM3_PIXI=/home/paperspace/code/sam3/pixi.toml
LOGS=/home/paperspace/logs

case "$SURVEY" in
    apr_2026_zed|dec_2025_*) D=/home/paperspace/data/klapmuts/$SURVEY;;
    *)                       D=/home/paperspace/data/citrus_all/$SURVEY;;
esac
BATELEUR=$D/prod/bateleur
CFGDIR=$D/prod/tassili/blocks_ns/lio_row100
HIER=$BATELEUR/scene_graph/marker_hierarchy.json
FRUITDIR=$BATELEUR/sam3_fruit

mark() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
disk_ok() {
    local a; a=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc 0-9)
    [ -n "$a" ] && [ "$a" -lt "$FLOOR" ] && { mark "ABORT-DISK: ${a}G free < floor ${FLOOR}G"; return 1; }
    return 0
}

[ -f "$HIER" ] || { mark "no hierarchy at $HIER"; exit 1; }
[ -d "$CFGDIR" ] || { mark "no blocks at $CFGDIR"; exit 1; }
mkdir -p "$FRUITDIR"

BLOCKS=$(ls -d "$CFGDIR"/block_* 2>/dev/null | sort)
[ -n "$ONLY" ] && BLOCKS=$(for n in ${ONLY//,/ }; do printf '%s\n' "$CFGDIR/block_$(printf '%03d' "$n")"; done)

mark "FRUIT-CHAIN start survey=$SURVEY steps=$STEPS blocks=$(echo "$BLOCKS" | wc -l) floor=${FLOOR}G"
mark "  hierarchy: $HIER"
mark "  ledger out: $FRUITDIR/clip_<NNN>/frame_entries.json"

# ---- 2. fruit nodes into the hierarchy (survey-level, once) ----------------
# compile_supervision reads {tree_id: id} from H["fruits"], i.e. ONE node per
# tree, and paints FRUIT_ID_BASE + id. Ledgers must all exist first, so this runs
# after step 1 for every block; with --steps all on a fresh survey, run --steps 1
# first. Tree/row ids are untouched, so 04's WGS84 ledger entries stay valid.
if [ "$STEPS" = "all" ]; then
    disk_ok || exit 1
    python3 "$SCR/fruit_nodes_from_ledger.py" \
        --hierarchy "$HIER" \
        --ledger-glob "$FRUITDIR/clip_*/frame_entries.json" \
        2>&1 | sed 's/^/  /' | tee "$LOGS/fruit_${SURVEY}_nodes.log" \
        || { mark "fruit-nodes FAILED"; exit 1; }
    NN=$(python3 -c "import json;print(len(json.load(open('$HIER')).get('fruits') or []))")
    mark "hierarchy now carries $NN fruit nodes"

    # ---- 3b. embedder retrain against the fruit-bearing hierarchy ----------
    # fruit sits at the innermost hyperbolic radius, so the embedder must be
    # retrained before any block is re-seeded against it. ~3 min measured.
    EXP=${SURVEY%_Jackal}_v1g
    HYPER_OUT=$BATELEUR/embedder
    (cd /home/paperspace/code/nerf_new && pixi run python \
        "$ARU/src/interfaces/rerun/HiGH/train_hyperembedder_graph.py" \
        --hierarchy-json "$HIER" --experiment-name "$EXP" --output-dir "$HYPER_OUT" \
        --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 \
        --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
        --epochs 1500 --no-level-norms) > "$LOGS/fruit_${SURVEY}_hyper.log" 2>&1 \
        && mark "embedder retrained -> $HYPER_OUT/$EXP" \
        || { mark "embedder retrain FAILED (see fruit_${SURVEY}_hyper.log)"; exit 1; }
fi
EMB=$(ls -t "$BATELEUR"/embedder/*/ckpts/model_best.pth 2>/dev/null | head -1)

for BD in $BLOCKS; do
    [ -d "$BD" ] || { mark "SKIP $(basename "$BD") — no such block"; continue; }
    N=$(basename "$BD" | cut -d_ -f2)
    disk_ok || exit 1

    # ---- 1a. tree id maps (the crops fruit is searched WITHIN) --------------
    # chain_04_fruitfix ran r6_project_idmaps to build <block>/supervision_trees_r6.
    # That is unnecessary now and un-runnable here: r6_project_idmaps defaults
    # --masks-npz to /home/paperspace/logs/salvage_jul/r6_masks.npz, a July salvage
    # artifact that does not exist on this box. The prod supervision IS the tree id
    # map the migration made canonical — same kf_*.png shape, current ids — so point
    # the ledger straight at it.
    TIDMAP=$BD/supervision/trees_only
    [ -d "$TIDMAP" ] && ls "$TIDMAP"/kf_*.png >/dev/null 2>&1 \
        || { mark "b$N SKIP — no tree id maps at $TIDMAP"; continue; }

    # ---- 1b. fruit ledger, in-tree crops -----------------------------------
    if [ ! -f "$FRUITDIR/clip_$N/frame_entries.json" ]; then
        TMP="prod/bateleur/.fruit_stage_b$N"
        (cd "$ARU" && pixi run --manifest-path "$SAM3_PIXI" python "$SCR/fruit_in_trees_ledger.py" \
            --data-dir "$D" --block-dir "$BD" --out-name "$TMP" \
            --tree-idmap-dir "$TIDMAP") \
            > "$LOGS/fruit_${SURVEY}_b${N}_ledger.log" 2>&1 \
            || { mark "b$N ledger FAILED (see fruit_${SURVEY}_b${N}_ledger.log)"; continue; }
        if [ -d "$D/$TMP/clip_000" ]; then
            rm -rf "$FRUITDIR/clip_$N"
            mv "$D/$TMP/clip_000" "$FRUITDIR/clip_$N"
            rmdir "$D/$TMP" 2>/dev/null
        fi
    fi
    # count the way compile_supervision's load_fruit_ledger does: instances live
    # in frame_entries keyed by LOCAL index; "frames" is only the local->kf map.
    NF=$(python3 -c "
import json
try:
    d = json.load(open('$FRUITDIR/clip_$N/frame_entries.json'))
    fe = d.get('frame_entries', {})
    n = sum(len(v) for v in fe.values())
    print(f'{n} instances on {sum(1 for v in fe.values() if v)}/{len(d.get(\"frames\", []))} frames')
except Exception as e:
    print(f'unreadable ({type(e).__name__})')" 2>/dev/null)
    mark "b$N ledger ok — $NF"

    [ "$STEPS" = "all" ] || continue

    # ---- 3. supervision v3 (trees + fruit) ---------------------------------
    # Writes to trees_fruit_v3, NOT back over trees_only: trees_only is the
    # tree-source being read here AND the supervision the fleet's 318 blocks were
    # built against. Swapping it in is a deliberate later step, not a side effect
    # of this compile.
    disk_ok || exit 1
    python3 "$SCR/compile_supervision.py" --block-dir "$BD" \
        --tree-source idmap_dir --tree-src-dir "$TIDMAP" \
        --hierarchy "$HIER" \
        --fruit-ledger-glob "$FRUITDIR/clip_$N/frame_entries.json" \
        --filter strict_fruit_tree_v1 \
        --out-dir "$BD/supervision/trees_fruit_v3" \
        > "$LOGS/fruit_${SURVEY}_b${N}_compile.log" 2>&1 \
        && mark "b$N supervision recompiled (trees+fruit)" \
        || { mark "b$N compile FAILED"; continue; }

    NIDS=$(python3 -c "
import json
try: print(json.load(open('$BD/supervision/trees_fruit_v3/manifest.json')).get('n_ids','?'))
except Exception: print('?')" 2>/dev/null)

    # ---- 3c. densify + share-seed (fruit-bearing blocks ONLY) ---------------
    # Recipe validated on 05 b000 (2026-08-26), zero training end to end:
    #   share-seed alone           fruit IoU 0.500/0.412, trees intact
    #   densify + share-seed(0.1)  fruit IoU 0.681/0.486, tree 9 -0.08
    # The census argmax hands a fruit's pixels to the tree (fruit blend share
    # 0-10% at its own px), so majority init structurally cannot seed fruit;
    # share assignment can, and the mask-driven split pass sharpens it
    # (FRUIT-of-7 precision 0.557 -> 0.746). Fruit-free blocks keep their
    # fleet seed untouched.
    NFR=$(python3 -c "
import json
try:
    d=json.load(open('$FRUITDIR/clip_$N/frame_entries.json'))
    print(sum(len(v) for v in d.get('frame_entries',{}).values()))
except Exception: print(0)" 2>/dev/null)
    if [ "${NFR:-0}" = "0" ]; then
        mark "b$N no fruit — fleet seed kept, skipping densify"
        continue
    fi
    disk_ok || exit 1
    rm -rf "$BD/splat_runs_FEATFIX/fruit_densify" "$BD/stage2_init_densify" \
           "$BD/splat_runs_FEATFIX/interaction_W_densified.npz"
    CENSUS_EMBEDDER="$EMB" bash /home/paperspace/code/automation/densify_block.sh \
        "$BD" "$BD/supervision/trees_fruit_v3" "${DENSIFY_ITERS:-2000}" \
        > "$LOGS/fruit_${SURVEY}_b${N}_densify.log" 2>&1 \
        || { mark "b$N densify FAILED (see fruit_${SURVEY}_b${N}_densify.log)"; continue; }
    DRUN=$(ls -dt "$BD"/splat_runs_FEATFIX/fruit_densify/high/*/ | head -1)
    DCK=$(ls -t "$DRUN"/nerfstudio_models*/*.ckpt | head -1)

    WD=$BD/splat_runs_FEATFIX/interaction_W_densified.npz
    (cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
        "$SCR/gaussian_interaction_census.py" \
        --run-glob "$DRUN/config.yml" \
        --supervision-dir "$BD/supervision/trees_fruit_v3" --out-npz "$WD") \
        > "$LOGS/fruit_${SURVEY}_b${N}_census.log" 2>&1 \
        || { mark "b$N census FAILED"; continue; }
    python3 /home/paperspace/code/automation/fruit_diet_check.py "$WD" \
        | tee -a "$LOGS/fruit_${SURVEY}_b${N}_census.log" | grep DIET-MIN | sed "s/^/  b$N /"

    (cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
        "$SCR/build_census_init.py" \
        --w-npz "$WD" --embedder "$EMB" --src-ckpt "$DCK" \
        --dst-dir "$BD/stage2_init_densify/nerfstudio_models" \
        --fruit-share-assign "${FRUIT_SHARE_ASSIGN:-0.1}") \
        >> "$LOGS/fruit_${SURVEY}_b${N}_census.log" 2>&1 \
        || { mark "b$N share-seed init FAILED"; continue; }

    # stage under the CANONICAL seed name — verdict_block and every downstream
    # consumer glob stage2_censusinit_*; the old fleet seed is superseded.
    TS=$(basename "$DRUN")
    RUN=$BD/splat_runs_FEATFIX/stage2_censusinit_seed/high/$TS
    rm -rf "$BD/splat_runs_FEATFIX/stage2_censusinit_seed" "$BD/splat_runs_FEATFIX/stage2_censusinit_fw2"
    mkdir -p "$RUN/nerfstudio_models"
    sed 's|^experiment_name: .*$|experiment_name: stage2_censusinit_seed|' "$DRUN/config.yml" > "$RUN/config.yml"
    cp "$DRUN/dataparser_transforms.json" "$RUN/" 2>/dev/null
    cp "$BD"/stage2_init_densify/nerfstudio_models/*.ckpt "$RUN/nerfstudio_models/"
    mark "b$N densify+share-seed staged as stage2_censusinit_seed ($NFR fruit instances)"

    # ---- 4. verdicts: prod record (tree-mode frame) + fruit-frame containment
    CENSUS_EMBEDDER="$EMB" CENSUS_HIERARCHY="$HIER" \
        bash /home/paperspace/code/automation/verdict_block.sh \
            "$BD" "$BD/supervision/trees_fruit_v3" \
            > "$LOGS/fruit_${SURVEY}_b${N}_verdict.log" 2>&1 \
        || mark "b$N verdict FAILED"
    FFRAME=$(python3 -c "
import glob
import numpy as np
from PIL import Image
best=(0,None)
for f in sorted(glob.glob('$BD/supervision/trees_fruit_v3/kf_*.png')):
    a=np.array(Image.open(f)); n=int(((a>=10000)&(a!=65535)).sum())
    if n>best[0]: best=(n,f.split('/')[-1])
print(best[1] or '')" 2>/dev/null)
    if [ -n "$FFRAME" ]; then
        (cd /home/paperspace/code/nerf_new && HIGH_EMBEDDER_CKPT=$EMB pixi run python \
            "$SCR/containment_eval.py" \
            --config "$RUN/config.yml" --hyper-ckpt "$EMB" --hierarchy-json "$HIER" \
            --supervision-dir "$BD/supervision/trees_fruit_v3" --frame "$FFRAME" \
            --kf-images "$D/prod/scratch_sam3" \
            --out "/home/paperspace/code/lab_notebook/figs/fruit_${SURVEY}_b${N}_containment.png") \
            2>/dev/null | grep -aE '^(TREE|FRUIT|ROW) ' \
            | tee -a "$LOGS/fruit_${SURVEY}_b${N}_verdict.log" | grep -a FRUIT | sed "s/^/  b$N /"
    fi

    # reclaim the per-block intermediates (the densify run dir is the bulk)
    rm -rf "$BD/stage2_init_densify" "$BD/splat_runs_FEATFIX/fruit_densify" "$BD/stage2_init"
done

mark "FRUIT-CHAIN done"
