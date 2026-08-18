#!/bin/bash
# Re-seed 05's STALE blocks against the CURRENT embedder + hierarchy.
#
# WHY: blocks 005-008 were seeded 2026-08-14 17:10-19:36; the embedder was
# retrained 08-17 19:40 (R6 firing on a healthy embedder, since deprecated) and
# the hierarchy rebuilt 19:36. Two embedders trained on identical data are
# latent-incompatible (cosine -0.12) though decode-equivalent (1.0), so features
# seeded under the old one read ~0 through the new one. Measured: these four are
# the fleet's ONLY stale blocks and its ONLY broken ones — rows 0.10-0.25 across
# all four, trees 0.001-0.013 on three of them. 009 has no stage2 ckpt at all.
#
# Runs blocks STRICTLY ONE AT A TIME: the week queue is training concurrently
# and two trainers can exhaust VRAM, which surfaces as an NVML INTERNAL ASSERT
# and costs the queue a slot.
#
# Absolute paths throughout — the previous attempt at 009 died on
# "bash: automation/censusinit_block.sh: No such file or directory" (wrong cwd).
set -uo pipefail

AUTO=/home/paperspace/code/automation
ROOT=/home/paperspace/data/citrus_all/05_13D_Jackal
BLK=$ROOT/prod/tassili/blocks_ns/lio_row100
LOG=/home/paperspace/logs/reseed_05_stale.log

export CENSUS_SEED_ONLY=1
export CENSUS_EMBEDDER=/home/paperspace/data/high/nerf/05_13D_v1g/ckpts/model_best.pth
export CENSUS_HIERARCHY=$ROOT/scene_graph/marker_hierarchy.json

BLOCKS=${1:-"005 006 007 008 009"}

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "RESEED start — embedder $(stat -c %y "$CENSUS_EMBEDDER" | cut -c1-19)"
say "            hierarchy $(stat -c %y "$CENSUS_HIERARCHY" | cut -c1-19)"

for b in $BLOCKS; do
    BD=$BLK/block_$b
    SUP=$BD/supervision/trees_only
    [ -f "$SUP/manifest.json" ] || SUP=$BD/supervision/strict_tree_v2
    if [ ! -f "$SUP/manifest.json" ]; then
        say "SKIP block_$b: no supervision manifest"; continue
    fi

    # Free VRAM check before each block, not just at start: the queue's slot
    # boundaries move while we run.
    FREE=$(cd /home/paperspace/code/nerf_new && timeout 300 pixi run python "$AUTO/free_vram_gb.py" 2>/dev/null | tail -1)
    say "block_$b: ${FREE:-?}G VRAM free"
    if [ -n "$FREE" ] && [ "$FREE" -lt 8 ]; then
        say "HOLD block_$b: only ${FREE}G free — waiting for the queue's slot to finish"
        for _ in $(seq 1 60); do
            sleep 60
            FREE=$(cd /home/paperspace/code/nerf_new && timeout 300 pixi run python "$AUTO/free_vram_gb.py" 2>/dev/null | tail -1)
            [ -n "$FREE" ] && [ "$FREE" -ge 8 ] && break
        done
        say "block_$b: resuming at ${FREE:-?}G free"
    fi

    say "SEED block_$b ..."
    if bash "$AUTO/censusinit_block.sh" "$BD" "$SUP" \
           >> "/home/paperspace/logs/reseed_05_b$b.log" 2>&1; then
        say "SEED block_$b OK"
    else
        say "SEED block_$b FAILED (see reseed_05_b$b.log) — continuing"
        continue
    fi

    say "VERDICT block_$b ..."
    if bash "$AUTO/verdict_block.sh" "$BD" "$SUP" \
           >> "/home/paperspace/logs/reseed_05_b$b.log" 2>&1; then
        say "VERDICT block_$b OK"
    else
        say "VERDICT block_$b FAILED (see reseed_05_b$b.log)"
    fi
done

say "RESEED done"
python3 - << 'PY' | tee -a "$LOG"
import json
p = ('/home/paperspace/data/citrus_all/05_13D_Jackal/prod/tassili/blocks_ns/'
     'lio_row100/verdicts_censusinit_fw2.json')
b = json.load(open(p))['blocks']
for k in sorted(b):
    v = b[k]
    print(f"  {k}: trees {v.get('trees', {})}  rows {v.get('rows', {})}")
PY
