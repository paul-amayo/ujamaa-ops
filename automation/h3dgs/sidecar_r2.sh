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
say "block $NNN ratio-2 side-car ready in $(( ($(date +%s)-t0)/60 )) min"
