#!/bin/bash
# demo_gwakungu_recut.sh <embedder experiment name> — re-cut the Gwakungu reel from the EXISTING identity maps after new verdict
# lines were appended (script with the honesty rule -> composite -> contact sheet -> coverage). No card, ~1 min.
# Same arguments as the reel steps of demo_gwakungu_final.sh; the maps/backdrop under $OUT are reused (--skip-maps).
set -uo pipefail
EMBN=${1:?embedder experiment name}; A=/home/paperspace/code/automation/h3dgs; NS=/home/paperspace/code/nerf_new
C=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage; S=$C/demo_root
SV=IMG_7993_s0_cabbage; L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
EMB=$S/prod/bateleur/embedder/$EMBN/ckpts/model_best.pth; TAG=glref_bg_f1.0_r2_$EMBN; R=$S; OUT=/home/paperspace/logs/demo_chunks/gwakungu_7993
say "=== gwakungu recut ($EMBN) from the existing maps"; t0=$(date +%s); cd $NS
pixi run python $A/sidecar_demo_script_chunk.py --survey $SV --survey-root $R --path $OUT/demo_path.json --maps $OUT/maps --models chunk_lane --noun cabbage --ask-count --hold 10 --tree-min-iou 0.5 --verdict-model chunk_lane --verdict-tag $TAG --out $OUT/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L
HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --survey-root $R --path $OUT/demo_path.json --models chunk_lane --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --seed-tag $TAG --verdict-tag $TAG --backdrop-dir $OUT/backdrop --out $OUT --fps 5 --max-dist 60 --noun cabbage --min-area 400 --tree-min-area 100 --tree-thr 0.4 --row-thr 0.4 --skip-maps --script $OUT/demo_script.json 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $OUT --cols 3 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
/home/paperspace/envs/match/bin/python - "$OUT" <<'PY' | tee -a $L
import json, numpy as np, sys
from pathlib import Path
O=Path(sys.argv[1]); pj=json.load(open(O/'demo_path.json')); cov=[]; ids=set()
for f in pj['frames']:
    z=np.load(O/'maps'/'chunk_lane'/(f['name']+'.npz')); m=z['tmg'].astype(np.float32)>=0; cov.append(m.mean()); ids|=set(np.unique(z['tid'][m]).tolist())
print(f'[cov] recut reel: cabbage-claimed {100*np.mean(cov):.1f}% of a frame (max {100*np.max(cov):.1f}%); heads decoded on the drive: {sorted(ids)} ({len(ids)} of 29)')
PY
say "=== gwakungu recut ($EMBN) done in $(( ($(date +%s)-t0)/60 )) min -> $OUT/demo.mp4"
