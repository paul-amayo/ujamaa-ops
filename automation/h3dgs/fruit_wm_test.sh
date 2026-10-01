#!/bin/bash
# fruit_wm_test.sh — Paul (02:0x, 10-01): "Add it to the end": the world-unit fruit-densify test on 05 chunk_1_0_expo, queued
# BEHIND the dashboard's Kenya H3DGS runs. Starts only when no trainer is alive and the card has been < 8 GB used for two
# consecutive minutes; runs sidecar_fruit_densify_chunk.sh with the gate in metres (FD_SCALE_M 0.04 / FD_MAXSCALE_M 0.5 ->
# model units through the seed's dataparser scale) as a tagged run (fruit_densify_wm), then measures what changed:
# boost lines, gaussian count, the share-assigned mass fraction per fruit, and the fruit verdict.
set -uo pipefail
L=/home/paperspace/logs/fruit_wm_test.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
S05=/home/paperspace/data/citrus_all/05_13D_Jackal; EXPO=$S05/experimental/h3dgs_expo; O=$S05/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
say "queued behind the dashboard's runs: waiting for no train_single/train_post/ns-train and < 8 GB used (2 consecutive minutes)"
quiet=0
while :; do
  if pgrep -f 'train_single\.py|train_post\.py|ns-train|image_farm_h3dgs\.sh' > /dev/null 2>&1 || [ "$(gpu_used)" -ge 8000 ]; then quiet=0; else quiet=$((quiet+1)); fi
  [ $quiet -ge 2 ] && break; sleep 60
done
say "=== start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
t0=$(date +%s)
FD_EXP=fruit_densify_wm FD_SCALE_M=0.04 FD_MAXSCALE_M=0.5 SIDECAR_PROJ=$EXPO bash /home/paperspace/code/automation/h3dgs/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_1_0_expo 1_0 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup|gate|fruit-densify|fruit-protect|floors|assigned' | tee -a $L
say "densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
TL=/home/paperspace/logs/sidecar_05_13D_Jackal_chunk_1_0_expo_fruitdensify_fruit_densify_wm.log
say "boost lines: $(tr '\r' '\n' < $TL 2>/dev/null | grep -ac '^\[fruit-densify\] step=') | last: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aE '^\[fruit-densify\] step=' | tail -1) | gaussians: $(tr '\r' '\n' < $TL 2>/dev/null | grep -aoE 'Now having [0-9]+ GSs' | tail -1)"
cd /home/paperspace/code/nerf_new && pixi run python - "$O/splat_runs_FEATFIX/interaction_W_fruitdensify_fruit_densify_wm_bg.npz" "$O/splat_runs_FEATFIX/interaction_W_fruitdensify_bg.npz" << 'PY' 2>&1 | grep -aE '^\[diet\]' | tee -a $L
import sys, numpy as np
def diet(tag, npz, share=0.1, fruit_floor=0.01):
    z=np.load(npz, allow_pickle=True); W=np.asarray(z['W'],np.float64); lab=[int(l) for l in z['labels']]
    if 65535 in lab:
        i=lab.index(65535); keep=[j for j in range(len(lab)) if j!=i]; W=W[keep]; lab=[lab[j] for j in keep]
    tot=W.sum(0); isfr=np.array([l>=10000 for l in lab]); frW=W[isfr]; fr_lab=np.array(lab)[isfr]
    sa=(frW.sum(0)/np.maximum(tot,1e-9)>share)&(tot>fruit_floor); best=frW.argmax(0)
    out=[]
    for k,l in enumerate(fr_lab):
        if l in (10000,10001,10002): out.append(f'{l}: share-assigned {int((sa&(best==k)).sum()):,} holding {frW[k][sa&(best==k)].sum()/max(frW[k].sum(),1e-9):.3f} of its mass')
    print(f'[diet] {tag}: {W.shape[1]:,} gaussians | '+' | '.join(out))
try: diet('NEW world-unit gate', sys.argv[1])
except Exception as e: print(f'[diet] NEW: {e!r}'[:160])
diet('OLD (model-unit gate, IoU 0.027)', sys.argv[2])
PY
say "fruit verdict (new): $(grep -aE '^\[chunk_1_0_expo sidecar fruitdensify_fruit_densify_wm ' /home/paperspace/logs/sidecar_05_13D_Jackal_fruit_verdicts.log | grep -aoE '(FRUIT of|TREE) [0-9]+ +\"[a-z]+\": thr [0-9.]+ IoU [0-9.]+' | sed -E 's/ +/ /g' | tr '\n' ';')"
say "=== fruit_wm_test done"
