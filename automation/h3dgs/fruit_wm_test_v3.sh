#!/bin/bash
# fruit_wm_test_v3.sh — bounded fruit-densify pass on 05 chunk_1_0_expo with all three model-unit terms corrected (10-01 22:1x):
# tail 0 (window open), gate 0.04 / 0.5 m world, AND the gsplat split boundary at the gate (FD_SPLIT_M 0.04 -> densify-size-thresh
# 0.00142 model units) so boosted carriers SPLIT instead of cloning; 1500 iterations (boosts at 15500 and 16000, the pass ends
# before the 16500 cascade) so the count stays under the 40 GB OOM seen at 9.4 M. Tagged run fruit_densify_wm_t0s. Waits for
# the reels (demo_chunks_00 done) and a quiet card; ~40 min + census/reseed/verdict. Logs to fruit_wm_test.log as v2 did.
set -uo pipefail
L=/home/paperspace/logs/fruit_wm_test.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
S05=/home/paperspace/data/citrus_all/05_13D_Jackal; EXPO=$S05/experimental/h3dgs_expo; O=$S05/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
busy(){ pgrep -f '^[^ ]*python[0-9.]* -u train_(single|post)\.py|^[^ ]*/ns-train|^ns-train|^bash [^ ]*image_farm_h3dgs\.sh|^bash [^ ]*demo_chunks_(00|01_retry)\.sh|^[^ ]*python[0-9.]* [^ ]*sidecar_(demo_overlay|row_backdrop)\.py' > /dev/null 2>&1; }
say "v3 (tail 0 + metres gate + split boundary at the gate, 1500 iters) queued: waiting for demo_chunks_00 done, then a quiet card"
until grep -qE 'demo_chunks_00 done|05_0_0: backdrop FAILED' /home/paperspace/logs/demo_chunks_run.log 2>/dev/null; do sleep 120; done
quiet=0; while :; do if busy || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi; [ $quiet -ge 2 ] && break; sleep 60; done
say "=== v3 start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
t0=$(date +%s)
FD_TAIL=0 FD_EXP=fruit_densify_wm_t0s FD_SCALE_M=0.04 FD_MAXSCALE_M=0.5 FD_SPLIT_M=0.04 SIDECAR_PROJ=$EXPO bash /home/paperspace/code/automation/h3dgs/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_1_0_expo 1_0 1500 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup|gate|fruit-densify|fruit-protect|floors|assigned' | tee -a $L
say "v3 densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
TL=/home/paperspace/logs/sidecar_05_13D_Jackal_chunk_1_0_expo_fruitdensify_fruit_densify_wm_t0s.log
say "v3 boost lines: $(tr '\r' '\n' < $TL 2>/dev/null | grep -ac '^\[fruit-densify\] step=') | $(tr '\r' '\n' < $TL 2>/dev/null | grep -aE '^\[fruit-densify\] step=' | paste -sd' ') | refine: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aoE 'Step [0-9]+: [0-9]+ GSs duplicated, [0-9]+ GSs split' | sed -n '1p;$p' | paste -sd' | ') | gaussians: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aoE 'Now having [0-9]+ GSs' | tail -1)"
W=$O/splat_runs_FEATFIX/interaction_W_fruitdensify_fruit_densify_wm_t0s_bg.npz
if [ -e $W ]; then cd /home/paperspace/code/nerf_new && pixi run python - "$W" "$O/splat_runs_FEATFIX/interaction_W_fruitdensify_bg.npz" << 'PY' 2>&1 | grep -aE '^\[diet\]' | tee -a $L
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
diet('v3 tail-0 + metres gate + split at gate', sys.argv[1]); diet('OLD (model-unit gate, tail 2000, IoU 0.027)', sys.argv[2])
PY
else say "v3: no census (train failed) — no diet numbers"; fi
say "v3 fruit verdict: $(grep -aE '^\[chunk_1_0_expo sidecar fruitdensify_fruit_densify_wm_t0s ' /home/paperspace/logs/sidecar_05_13D_Jackal_fruit_verdicts.log 2>/dev/null | grep -aoE '(FRUIT of|TREE) [0-9]+ +"[a-z]+": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "=== fruit_wm_test_v3 done"
