#!/bin/bash
# Phase 2 for ONE survey (arg 1, e.g. 01_13B_Jackal): pause the stage-1 fleet, census-seed stage-2 on all 43
# glref blocks (new run names beside prod), repoint the demo render service at the corrected era
# (RENDER_RUN_GLOB=stage2_censusinit_glref, RENDER_CKPT_POSES=opengl), then resume the fleet.
SV=${1:?survey id}; S=/home/paperspace/data/citrus_all/$SV; B=$S/prod/tassili/blocks_ns/lio_row100
EMB=$(ls -t $S/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json
LOG=/home/paperspace/logs/citrus_phase2_$SV.log; TSV=/home/paperspace/logs/citrus_phase2_$SV.tsv
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
echo -e "block\twall_s\tstatus" > $TSV
say "pausing the stage-1 fleet (resumable; skip-guards on relaunch)"
pkill -f "citrus_fleet_glref[.]sh"; sleep 2; pkill -f "ns-train hig[h]"; sleep 5
# The pause kills the block in flight; its run dir already has a config.yml, which the fleet's skip-guard would
# take as "trained" on relaunch. Quarantine every young glref stage-1 run without a final checkpoint (09-22:
# five 02 blocks were silently skipped this way).
for d in /home/paperspace/data/citrus_all/*/prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/splat_runs_STAGE1/stage1_bg00_glref/high/*/; do
  [ -e "$d/config.yml" ] || continue; ls "$d"/nerfstudio_models/*.ckpt >/dev/null 2>&1 && continue
  [ $(( $(date +%s) - $(stat -c %Y "$d") )) -lt 7200 ] || continue
  K=$(dirname $(dirname "$d"))_killed; mkdir -p "$K"; mv "$d" "$K"/ && say "quarantined killed partial run $d -> $K/"
done
NB=$(ls -d $B/block_[0-9][0-9][0-9] | wc -l); say "PHASE-2 START ($SV): $NB blocks, census seed on stage1_bg00_glref; embedder $(basename $(dirname $(dirname $EMB)))"
ok=0; bad=0
for BD in $B/block_[0-9][0-9][0-9]; do
  N=$(basename $BD); t0=$(date +%s)
  if ls $BD/splat_runs_FEATFIX/stage2_censusinit_glref/high/*/nerfstudio_models/*.ckpt >/dev/null 2>&1; then
    say "$N seed exists — skip"; echo -e "$N\t0\tskip" >> $TSV; ok=$((ok+1)); continue; fi
  CENSUS_EMBEDDER=$EMB CENSUS_HIERARCHY=$HJ bash /home/paperspace/logs/censusinit_block_glref.sh $BD > /home/paperspace/logs/cp2_${SV}_$N.log 2>&1
  rc=$?; w=$(( $(date +%s)-t0 ))
  if [ $rc -eq 0 ]; then ok=$((ok+1)); echo -e "$N\t$w\tok" >> $TSV; say "$N seeded in ${w}s ($(grep -c REPL-INIT0 /home/paperspace/logs/cp2_${SV}_$N.log) init)"
  else bad=$((bad+1)); echo -e "$N\t$w\tFAIL" >> $TSV; say "$N FAILED rc=$rc: $(grep -aE 'REPL-FAIL|Error' /home/paperspace/logs/cp2_${SV}_$N.log | tail -1 | cut -c1-120)"; fi
done
say "PHASE-2 SEEDS DONE: ok=$ok fail=$bad"
if [ "$SV" = "05_13D_Jackal" ] && [ $ok -ge 40 ]; then
  say "repointing demo render service (:8004) at the corrected era"
  pkill -f "render_service:ap[p] --host 127.0.0.1 --port 8004"; sleep 4
  ( cd /home/paperspace/code/aru_sil_core && RENDER_BLOCKS_ROOT=$B \
    RENDER_RUN_GLOB='splat_runs_FEATFIX/stage2_censusinit_glref/high/*/config.yml' RENDER_CKPT_POSES=opengl \
    HIGH_EMBEDDER_CKPT=$EMB RENDER_VRAM_BUDGET=20 RENDER_PRELOAD=3 \
    setsid nohup pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml python -m uvicorn \
      src.interfaces.splat_viewer.render_service:app --host 127.0.0.1 --port 8004 >> /home/paperspace/logs/render_service_8004.log 2>&1 < /dev/null & )
  for i in $(seq 1 30); do curl -sf -m3 http://127.0.0.1:8004/healthz >/dev/null 2>&1 && break; sleep 3; done
  say "render service: $(curl -s -m5 http://127.0.0.1:8004/healthz | cut -c1-120)"
else
  say "not the demo survey or too many failures ($bad) — render service untouched"
fi
say "resuming the stage-1 fleet (01→04)"
setsid nohup /home/paperspace/logs/citrus_fleet_glref.sh > /home/paperspace/logs/citrus_fleet_glref.out 2>&1 < /dev/null &
say "PHASE-2 DONE"
