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
done

if [ "$STEPS" = "all" ]; then
    mark "steps 3b/4 (embedder retrain, re-seed sweep, verdicts) are NOT run by this script yet"
    mark "  re-seed writes ~0.5 G of NEW seed per block; check headroom against --floor first"
fi
mark "FRUIT-CHAIN done"
