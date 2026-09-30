#!/bin/bash
# sidecar_demo_queue_v4.sh — continuation of the Tuesday-demo GPU chain (v3 replaced 16:5x while its 01 3_1 side-car child was
# running; that child finishes on its own and this script waits for it). Adds per-TREE containment cuts (dashboard, 16:4x:
# tree queries through the per-frame Otsu split lit a neighbouring canopy; the app passes the fitted threshold as the query's
# `cut`). Order from here:
#   a. wait for the 3_1 side-car (orphaned child of v3) to finish;  b. tree cuts on chunk_0_0_expo and chunk_3_1;
#   c. wait until the demo render services release the card (<= 12 GB used: hier 9 GB + identity 7 GB are resident now and
#      the 1_0 post-opt alone peaks near 30 GB) then 05 chunk 1_0 on the improved recipe + in-cell score;
#   d. 1_0 identity side-car;  e. 1_0 fruit densify;  f. 1_0 fruit cuts;  g. 1_0 tree cuts.
# Hands the GPU back ONLY at the end by touching ~/logs/demo_peer_sidecars.done (trap on exit, success or not).
set -uo pipefail
L=/home/paperspace/logs/sidecar_demo_queue.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
DONE=/home/paperspace/logs/demo_peer_sidecars.done
trap 'touch $DONE; say "handed the GPU back ($DONE)"' EXIT
A=/home/paperspace/code/automation/h3dgs; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python
S05=/home/paperspace/data/citrus_all/05_13D_Jackal; EXPO=$S05/experimental/h3dgs_expo; S01=/home/paperspace/data/citrus_all/01_13B_Jackal
gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
say "v4 queued: waiting for the 01 3_1 side-car (started under v3) to finish"
while pgrep -f 'sidecar_chunk_v2\.sh 01_13B_Jackal 3_1' > /dev/null 2>&1; do sleep 60; done
say "01 3_1 side-car rc=$( ls $S01/experimental/h3dgs_sidecar_chunks/chunk_3_1/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/nerfstudio_models/*.ckpt > /dev/null 2>&1 && echo 0 || echo 1 ) (finished under v3): $(grep -aE 'chunk 3_1 side-car (ready|RGB)' /home/paperspace/logs/sidecar_01_13B_Jackal_chunks.log | tail -2 | sed -E 's/^\[[0-9 :-]+\] //' | cut -c1-120 | tr '\n' '|')"
# b. tree cuts on the two finished side-cars
t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_tree_cuts.sh 05_13D_Jackal chunk_0_0_expo 0_0 10 2>&1 | grep -aE 'tree-cuts|scored in|already|no seed' | tee -a $L
say "05 expo 0_0 tree cuts rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
t0=$(date +%s); bash $A/sidecar_tree_cuts.sh 01_13B_Jackal chunk_3_1 3_1 10 2>&1 | grep -aE 'tree-cuts|scored in|already|no seed' | tee -a $L
say "01 3_1 tree cuts rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
# c. 05 chunk 1_0 on the improved recipe — needs the card: wait for the demo services to be released
n=0; while [ "$(gpu_used)" -gt 12000 ]; do [ $((n % 10)) = 0 ] && say "waiting for the demo render services to release the card before the 1_0 training/post-opt ($(gpu_used) MiB used, need <= 12000)"; n=$((n+1)); sleep 60; done
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
  # d. identity side-car   e. fruit densify   f. fruit cuts   g. tree cuts
  t0=$(date +%s); SIDECAR_PROJ=$EXPO SIDECAR_HIER=$T/hierarchy.hier_opt SIDECAR_TAG=_expo bash $A/sidecar_chunk_v2.sh 05_13D_Jackal 1_0 2>&1 | grep -aE 'ready in|failed|FAILED|missing|skipping|held-out|chunk-manifest' | tee -a $L
  say "05 expo 1_0 side-car rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_fruit_densify_chunk.sh 05_13D_Jackal chunk_1_0_expo 1_0 2>&1 | grep -aE 'done in|FAILED|failed|missing|skipping|fruitsup' | tee -a $L
  say "05 expo 1_0 fruit densify rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_fruit_cuts.sh 05_13D_Jackal chunk_1_0_expo 1_0 6 2>&1 | grep -aE 'fruit-cuts|scored in|skipped|no densified' | tee -a $L
  say "05 expo 1_0 fruit cuts rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min -> $S05/experimental/h3dgs_sidecar_chunks/chunk_1_0_expo/fruit_cuts.json"
  t0=$(date +%s); SIDECAR_PROJ=$EXPO bash $A/sidecar_tree_cuts.sh 05_13D_Jackal chunk_1_0_expo 1_0 10 2>&1 | grep -aE 'tree-cuts|scored in|already|no seed' | tee -a $L
  say "05 expo 1_0 tree cuts rc=${PIPESTATUS[0]} in $(( ($(date +%s)-t0)/60 )) min"
else
  say "05 1_0 improved recipe FAILED after $(( ($(date +%s)-t0)/60 )) min (no hierarchy.hier_opt) — side-car, fruit and cuts skipped"
fi
say "=== v4 queue done"
