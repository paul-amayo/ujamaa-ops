#!/bin/bash
# sidecar_demo_queue_v3.sh — the Tuesday-demo GPU chain, v3 (= v2 + per-fruit containment cuts after the 1_0 fruit densify) (Paul's 1_0 decision relayed 13:5x, confirmed here). Order:
#   1. 05 h3dgs_expo chunk 0_0 identity side-car (tagged chunk_0_0_expo)
#   2. 01 chunk 3_1 identity side-car (template = 05 block_020)
#   3. 05 chunk 1_0 on the improved recipe (citrus_chunk_expo.sh: half views + per-image exposure, 12 M cap, 60k/45k/0.0075,
#      post-opt) + its IN-CELL score (training views inside the cell, exposure applied — the 01 scoring recipe)
#   4. 05 1_0 identity side-car (chunk_1_0_expo)      5. 05 1_0 fruit densify (trees 5 and 3 live in 1_0)
#   (0_0's fruit densify is dropped unless DEMO_FRUIT_00=1 — 1_0 is the fruit chunk.)
# Takes the GPU when the dashboard's queue hands it over (~/logs/demo_peer_turn.flag) or no image_farm_h3dgs.sh is alive,
# and hands it back ONLY at the very end by touching ~/logs/demo_peer_sidecars.done (trap on exit, success or not).
set -uo pipefail
L=/home/paperspace/logs/sidecar_demo_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
DONE=/home/paperspace/logs/demo_peer_sidecars.done; FLAG=/home/paperspace/logs/demo_peer_turn.flag
trap 'touch $DONE; say "handed the GPU back ($DONE)"' EXIT
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python
S05=/home/paperspace/data/citrus_all/05_13D_Jackal; EXPO=$S05/experimental/h3dgs_expo
say "v3 queued: waiting for $FLAG or no image_farm_h3dgs.sh on the box"
while [ ! -e $FLAG ] && pgrep -f 'image_farm_h3dgs\.sh' > /dev/null 2>&1; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 8000 ]; do say "GPU busy ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used), waiting"; sleep 60; done
say "=== v3 start: GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
# 1. 05 expo 0_0 identity
t0=$(date +%s); SIDECAR_PROJ=$EXPO SIDECAR_HIER=$EXPO/output/trained_chunks/0_0/hierarchy.hier_opt SIDECAR_TAG=_expo bash $A/sidecar_chunk_v2.sh 05_13D_Jackal 0_0 2>&1 | grep -aE 'ready in|failed|FAILED|missing|skipping|held-out' | tee -a $L
say "05 expo 0_0 side-car rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
if [ "${DEMO_FRUIT_00:-0}" = 1 ]; then
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_0_0_expo 0_0 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup' | tee -a $L
  say "05 expo 0_0 fruit densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
fi
# 2. 01 3_1 identity
TPL=$(ls $S05/experimental/h3dgs_sidecar/block_020/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt | head -1)
t0=$(date +%s); SIDECAR_TEMPLATE=$TPL bash $A/sidecar_chunk_v2.sh 01_13B_Jackal 3_1 2>&1 | grep -aE 'ready in|failed|FAILED|missing|skipping|held-out' | tee -a $L
say "01 3_1 side-car rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
# 3. 05 chunk 1_0 on the improved recipe (train + hierarchy + post-opt + paired score), then the in-cell score
t0=$(date +%s); bash /home/paperspace/logs/citrus_chunk_expo.sh 1_0 2>&1 | grep -aE 'train rc|post-opt rc|hierarchy|PAIRED expo|no point cloud' | tee -a $L
T=$EXPO/output/trained_chunks/1_0
if [ -e $T/hierarchy.hier_opt ]; then
  say "05 1_0 improved recipe done in $(( ($(date +%s)-t0)/60 )) min -> hierarchy.hier_opt $(du -h $T/hierarchy.hier_opt | cut -f1)"
  NAMES=$($PYH - "$EXPO" "1_0" << 'PY'
import sys, os, numpy as np
sys.path.insert(0,'/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary
P,c=sys.argv[1],sys.argv[2]; d=os.path.join(P,'camera_calibration/chunks',c)
ctr=np.array([float(x) for x in open(os.path.join(d,'center.txt')).read().split()]); ext=np.array([float(x) for x in open(os.path.join(d,'extent.txt')).read().split()])
lo,hi=ctr-ext/2,ctr+ext/2
tf=os.path.join(d,'sparse/0/test.txt'); test=set(x.strip() for x in open(tf)) if os.path.exists(tf) else set()
keep=[im.name for im in read_images_binary(os.path.join(d,'sparse/0/images.bin')).values()
      if im.name not in test and lo[0]<=(-im.qvec2rotmat().T@im.tvec)[0]<=hi[0] and lo[1]<=(-im.qvec2rotmat().T@im.tvec)[1]<=hi[1]]
print(','.join(sorted(keep)))
PY
)
  NK=$(echo "$NAMES" | tr ',' '\n' | wc -l)
  (cd $EXPO && CUDA_HOME=/home/paperspace/code/_cuda12 H3DGS_EVAL_NAMES="$NAMES" $PYH /home/paperspace/logs/h3dgs_eval_chunk.py $EXPO --hier output/trained_chunks/1_0/hierarchy.hier_opt \
    --only_chunk 1_0 --taus 0 --save 12 --exposure_json $T/exposure.json --out output/eval_incell_1_0 > /home/paperspace/logs/citrus_expo_1_0_incell.log 2>&1)
  say "05 1_0 IN-CELL ($NK training views inside the cell): $(grep -aoE 'tau 0: [0-9]+ views.*' /home/paperspace/logs/citrus_expo_1_0_incell.log | head -1 | sed -E 's/ in [0-9]+s//')   [0_0 expo reference: 27.54 / 28.07 sky-masked]"
  # 4. 1_0 identity side-car   5. 1_0 fruit densify
  t0=$(date +%s); SIDECAR_PROJ=$EXPO SIDECAR_HIER=$T/hierarchy.hier_opt SIDECAR_TAG=_expo bash $A/sidecar_chunk_v2.sh 05_13D_Jackal 1_0 2>&1 | grep -aE 'ready in|failed|FAILED|missing|skipping|held-out' | tee -a $L
  say "05 expo 1_0 side-car rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_1_0_expo 1_0 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup' | tee -a $L
  say "05 expo 1_0 fruit densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
  # 6. per-fruit containment verdicts -> fruit_cuts.json (the app lights fruit at its own verdict threshold)
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_fruit_cuts.sh 05_13D_Jackal chunk_1_0_expo 1_0 6 2>&1 | grep -aE 'fruit-cuts|scored in|skipped|no densified' | tee -a $L
  say "05 expo 1_0 fruit cuts rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min -> $EXPO/../h3dgs_sidecar_chunks/chunk_1_0_expo/fruit_cuts.json"
else
  say "05 1_0 improved recipe FAILED after $(( ($(date +%s)-t0)/60 )) min (no hierarchy.hier_opt) — side-car and fruit skipped"
fi
say "=== v3 queue done"
