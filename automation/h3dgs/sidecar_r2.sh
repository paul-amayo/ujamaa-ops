#!/bin/bash
# sidecar_r2.sh <survey id> <NNN> [margin m=8] — the settled containment side-car for one citrus block: H3DGS leaves ->
# stage-1 ckpt -> census-init seed (recipe of record, needed for the bootstrap config) -> void-row census + bg-competes
# ratio 2 seed (the final one). Skips blocks whose ratio-2 seed already exists.
set -uo pipefail
SV=${1:?survey}; NNN=${2:?block NNN}; MARGIN=${3:-8}
S=/home/paperspace/data/citrus_all/$SV; O=$S/experimental/h3dgs_sidecar/block_$NNN; L=/home/paperspace/logs/sidecar_${SV}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
if ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1; then say "block $NNN: ratio-2 seed exists, skipping"; exit 0; fi
cd /home/paperspace/logs; t0=$(date +%s)
bash /home/paperspace/logs/sidecar_block_glref.sh $SV $NNN $MARGIN > /home/paperspace/logs/sidecar_${SV}_b${NNN}_r2_1.out 2>&1 || { say "block $NNN: conversion/chain failed"; exit 1; }
bash /home/paperspace/logs/sidecar_bg_reseed.sh $SV $NNN sidecar 1.0 2 > /home/paperspace/logs/sidecar_${SV}_b${NNN}_r2_2.out 2>&1 || { say "block $NNN: void-row reseed failed"; exit 1; }
# disk-lean (fleet of 43 blocks, 2026-09-27): keep ONLY the ratio-2 seed (optimizer moments stripped), the bootstrap config, both census npz,
# supervision links and transforms; the converted stage-1 ckpt, zero-feature init, ratio-1 seed and the seed's init copy are rebuildable in ~2 min.
SEED=$(ls $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt | head -1)
/home/paperspace/miniconda3/envs/h3dgs/bin/python -c "
import torch,sys; p=sys.argv[1]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['optimizers']={}; ck['schedulers']={}; torch.save(ck,p)" "$SEED"
rm -rf $O/clip_cache_* $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/nerfstudio_models* $O/stage2_init_glref $O/stage2_init_census_glref $O/splat_runs_FEATFIX/stage2_censusinit_glref $O/stage2_init_census_glref_bg_f1.0_r2 $O/splat_runs_STAGE1
say "block $NNN ratio-2 side-car ready in $(( ($(date +%s)-t0)/60 )) min; kept $(du -sh $O | cut -f1) ($(df --output=avail -BG / | tail -1 | tr -dc 0-9)G free)"
