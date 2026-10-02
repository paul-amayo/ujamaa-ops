#!/bin/bash
# fruit_wm_test_v2.sh — the CORRECTED fruit-densify re-test on 05 chunk_1_0_expo (10-01 21:4x). The 20:31 run (fruit_wm_test.sh)
# fixed only the scale-gate units and still boosted 0 gaussians: `fruit_densify_tail` (HiGH a5639f0, 2026-08-31 — after the
# 08-27/28 block runs that fired) makes the boost window `step < stop_split_at - tail` = 17001 - 2000 = 15001 on a pass resumed
# at 15000, i.e. the single resume step, before the protect tally exists. Every side-car densify since 09-27 ran that way.
# This run: FD_TAIL=0 (window = the whole 2000-step pass) AND the gate in metres (4 cm / 50 cm world), tagged run
# fruit_densify_wm_t0. Waits for the demo-chunk videos to finish (same card) and a quiet card; ~45 min; then the diet numbers
# and the fruit verdict, as before.
set -uo pipefail
L=/home/paperspace/logs/fruit_wm_test.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
S05=/home/paperspace/data/citrus_all/05_13D_Jackal; EXPO=$S05/experimental/h3dgs_expo; O=$S05/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*demo_chunks_run\.sh|^[^ ]*python[0-9.]* [^ ]*sidecar_(demo_overlay|row_backdrop)\.py' > /dev/null 2>&1; }
say "v2 (tail 0 + metres gate) queued: waiting for demo_chunks_run done, then no trainer/renderer alive and < 8 GB used for 2 consecutive minutes"
until grep -qE 'demo_chunks_run done' /home/paperspace/logs/demo_chunks_run.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
say "=== v2 start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
t0=$(date +%s)
FD_TAIL=0 FD_EXP=fruit_densify_wm_t0 FD_SCALE_M=0.04 FD_MAXSCALE_M=0.5 SIDECAR_PROJ=$EXPO bash /home/paperspace/code/automation/h3dgs/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_1_0_expo 1_0 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup|gate|fruit-densify|fruit-protect|floors|assigned' | tee -a $L
say "v2 densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
TL=/home/paperspace/logs/sidecar_05_13D_Jackal_chunk_1_0_expo_fruitdensify_fruit_densify_wm_t0.log
say "v2 boost lines: $(tr '\r' '\n' < $TL 2>/dev/null | grep -ac '^\[fruit-densify\] step=') | first/last: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aE '^\[fruit-densify\] step=' | sed -n '1p;$p' | paste -sd' ') | gaussians: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aoE 'Now having [0-9]+ GSs' | tail -1)"
cd /home/paperspace/code/nerf_new && pixi run python - "$O/splat_runs_FEATFIX/interaction_W_fruitdensify_fruit_densify_wm_t0_bg.npz" "$O/splat_runs_FEATFIX/interaction_W_fruitdensify_bg.npz" << 'PY' 2>&1 | grep -aE '^\[diet\]' | tee -a $L
import sys, numpy as np
def diet(tag, npz, share=0.1, fruit_floor=0.01):
    z=np.load(npz, allow_pickle=True); W=np.asarray(z['W'],np.float64); lab=[int(l) for l in z['labels']]
    if 65535 in lab:
        i=lab.index(65535); keep=[j for j in range(len(lab)) if j!=i]; W=W[keep]; lab=[lab[j] for j in keep]
    tot=W.sum(0); isfr=np.array([l>=10000 for l in lab]); frW=W[isfr]; fr_lab=np.array(lab)[isfr]
    sa=(frW.sum(0)/np.maximum(tot,1e-9)>share)&(tot>fruit_floor); best=frW.argmax(0); out=[]
    for k,l in enumerate(fr_lab):
        if l in (10000,10001,10002): out.append(f'{l}: share-assigned {int((sa&(best==k)).sum()):,} holding {frW[k][sa&(best==k)].sum()/max(frW[k].sum(),1e-9):.3f} of its mass')
    print(f'[diet] {tag}: {W.shape[1]:,} gaussians | '+' | '.join(out))
try: diet('v2 tail-0 + metres gate', sys.argv[1])
except Exception as e: print(f'[diet] v2: {e!r}'[:160])
diet('OLD (model-unit gate, tail 2000, IoU 0.027)', sys.argv[2])
PY
say "v2 fruit verdict: $(grep -aE '^\[chunk_1_0_expo sidecar fruitdensify_fruit_densify_wm_t0 ' /home/paperspace/logs/sidecar_05_13D_Jackal_fruit_verdicts.log | grep -aoE '(FRUIT of|TREE) [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "=== fruit_wm_test_v2 done"
