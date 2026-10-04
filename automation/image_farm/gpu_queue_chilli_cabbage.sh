#!/bin/bash
# GPU queue, 2026-10-04 evening. Waits for the 05 chunk 1_0 retrain chain (post-opt -> eval -> cell -> score) to exit, then:
#   1. the chilli render (automation/demo/chilli_render.py; Paul: "no I'll wait for the renders");
#   2. the whole-clip cabbage H3DGS at the 8 M recipe (image_farm_h3dgs.sh on IMG_7993_sall, from a frozen copy), only if
#      stage B passed (126/126 kept by the track check, pairing test 0 frames off).
# Each job also waits until no other process holds more than 2 GB of the GPU (another session's job).
set -u
L=/home/paperspace/logs/gpu_queue_chilli_cabbage.log; S=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_sall
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
gpu_free() { while [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)" -gt 0 ]; do sleep 60; done; }
say "waiting for the 05 retrain chain"
while ps -eo args | grep -q "^bash /home/paperspace/logs/retrain_05_1_0_01recipe.run.sh"; do sleep 60; done
gpu_free; say "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)); chilli render"
/home/paperspace/miniconda3/envs/h3dgs/bin/python /home/paperspace/code/automation/demo/chilli_render.py > /home/paperspace/logs/chilli_render.log 2>&1
say "chilli render rc=$?"; grep -a '^\[chilli\] \(trained\|held\|wrote\)' /home/paperspace/logs/chilli_render.log | tee -a $L
if [ -s $S/transforms.json ] && grep -q "frames off by > 1 step: 0" /home/paperspace/logs/cabbage_whole_clip.log && grep -q '"kept": 126' $S/gate.json; then
  gpu_free; cp /home/paperspace/code/automation/image_farm/image_farm_h3dgs.sh /home/paperspace/logs/if_h3dgs_IMG_7993_sall.run.sh
  say "cabbage whole-clip H3DGS (8 M recipe)"; bash /home/paperspace/logs/if_h3dgs_IMG_7993_sall.run.sh $S 60000 45000 0.0075 8000000
  say "cabbage H3DGS rc=$?"; grep -a "HELD-OUT\|TRAINING" /home/paperspace/logs/if_h3dgs_IMG_7993_sall.log | tail -2 | tee -a $L
else say "cabbage stage B not passed: H3DGS not started"; fi
say "queue done"
