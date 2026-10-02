#!/bin/bash
# demo_gwakungu_final.sh <embedder experiment name> — reseed the cabbage side-car with the chosen embedder (census W reused;
# build_census_init encodes the targets with the embedder), verdicts on three frames (the tree thresholds and the honesty-rule
# IoUs), then the reel: thr 0.4 operating point, "which cabbage is this?" only on heads with verdict IoU >= 0.5.
set -uo pipefail
EMBN=${1:?embedder experiment name}; A=/home/paperspace/code/automation/h3dgs; NS=/home/paperspace/code/nerf_new; ARU=/home/paperspace/code/aru_sil_core/src/scripts
C=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage; S=$C/demo_root; O=$S/experimental/h3dgs_sidecar_chunks/chunk_lane; SUP=$O/supervision/trees_only; HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
SV=IMG_7993_s0_cabbage; L=/home/paperspace/logs/demo_chunks_run.log; VL=/home/paperspace/logs/sidecar_${SV}_verdicts.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
# point the shim's embedder at the chosen one (the reel/verdict read it through HIGH_EMBEDDER_CKPT)
mkdir -p $S/prod/bateleur/embedder/$EMBN/ckpts; ln -sfn $C/hyper/$EMBN/ckpts/model_best.pth $S/prod/bateleur/embedder/$EMBN/ckpts/model_best.pth; EMB=$S/prod/bateleur/embedder/$EMBN/ckpts/model_best.pth
TAG=glref_bg_f1.0_r2_$EMBN; SEEDRUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ | tail -1); TS=$(basename $SEEDRUN); W=$O/splat_runs_FEATFIX/interaction_W_glref_bg.npz
say "=== gwakungu final with embedder $EMBN: reseed"; t0=$(date +%s); cd $NS
pixi run python - "$SEEDRUN/nerfstudio_models/step-000015000.ckpt" "$O/src0_$EMBN.ckpt" <<'PY'
import sys, torch
ck=torch.load(sys.argv[1],map_location='cpu',weights_only=False); k=[k for k in ck['pipeline'] if k.endswith('gauss_params.high_features')][0]; ck['pipeline'][k]=torch.zeros_like(ck['pipeline'][k]); torch.save(ck,sys.argv[2])
PY
INIT=$O/stage2_init_census_$TAG/nerfstudio_models; RUN=$O/splat_runs_FEATFIX/stage2_censusinit_$TAG/high/$TS; mkdir -p $RUN/nerfstudio_models
HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $W --embedder $EMB --src-ckpt $O/src0_$EMBN.ckpt --dst-dir $INIT --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'assigned [0-9]+/' | tee -a $L
mv $INIT/*.ckpt $RUN/nerfstudio_models/step-000015000.ckpt; sed "s|^experiment_name: .*$|experiment_name: stage2_censusinit_$TAG|" $SEEDRUN/config.yml > $RUN/config.yml; cp $SEEDRUN/dataparser_transforms.json $RUN/ 2>/dev/null; rm -rf $O/stage2_init_census_$TAG $O/src0_$EMBN.ckpt
for FR in image_33.png image_40.png image_52.png; do
  HIGH_EMBEDDER_CKPT=$EMB timeout 500 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF --out /home/paperspace/logs/sidecar_figs_$SV/chunk_lane_${TAG}_$FR > /home/paperspace/logs/sidecar_${SV}_chunk_lane_verdict_${TAG}_$FR.log 2>&1
  grep -aE '^(TREE|ROW) ' /home/paperspace/logs/sidecar_${SV}_chunk_lane_verdict_${TAG}_$FR.log | sed "s/^/[chunk_lane sidecar $TAG $FR] /" >> $VL
  say "verdict $FR: $(grep -aE "^\[chunk_lane sidecar $TAG $FR\]" $VL | sed -E 's/ +/ /g' | grep -oE '(TREE|ROW) [0-9]+ "[a-z]+": thr [0-9.na]+ IoU [0-9.]+' | tr '\n' ';')"; rm -rf $O/clip_cache_*
done
say "reseed + verdicts in $(( ($(date +%s)-t0)/60 )) min; reel"; t0=$(date +%s); R=$S; OUT=/home/paperspace/logs/demo_chunks/gwakungu_7993; rm -rf $OUT/maps/chunk_lane
pixi run python $A/sidecar_demo_path.py --survey $SV --survey-root $R --blocks chunk_lane --orbit-block chunk_lane --orbit-seconds 0 --chunk-drive --chunk lane --min-pass 20 --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --fps 8 --scale 0.5 --seed-tag $TAG --out $OUT 2>&1 | grep -aE '^\[path\] drive|Error|Traceback' | tee -a $L
HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --survey-root $R --path $OUT/demo_path.json --models chunk_lane --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --seed-tag $TAG --verdict-tag $TAG --backdrop-dir $OUT/backdrop --out $OUT --fps 5 --max-dist 60 --noun cabbage --min-area 400 --tree-min-area 100 --tree-thr 0.4 --row-thr 0.4 2>&1 | grep -aE '^\[maps\] block|^\[demo\] wrote|Error|Traceback' | tee -a $L
pixi run python $A/sidecar_demo_script_chunk.py --survey $SV --survey-root $R --path $OUT/demo_path.json --maps $OUT/maps --models chunk_lane --noun cabbage --ask-count --hold 10 --tree-min-iou 0.5 --verdict-model chunk_lane --verdict-tag $TAG --out $OUT/demo_script.json 2>&1 | grep -aE '^\[script\]|Error|Traceback' | tee -a $L
HIGH_EMBEDDER_CKPT=$EMB pixi run python $A/sidecar_demo_overlay.py --survey $SV --survey-root $R --path $OUT/demo_path.json --models chunk_lane --sidecar-root experimental/h3dgs_sidecar_chunks --proj experimental/h3dgs --seed-tag $TAG --verdict-tag $TAG --backdrop-dir $OUT/backdrop --out $OUT --fps 5 --max-dist 60 --noun cabbage --min-area 400 --tree-min-area 100 --tree-thr 0.4 --row-thr 0.4 --skip-maps --script $OUT/demo_script.json 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L
/home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $OUT --cols 3 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
/home/paperspace/envs/match/bin/python - "$OUT" <<'PY' | tee -a $L
import json, numpy as np, sys
from pathlib import Path
O=Path(sys.argv[1]); pj=json.load(open(O/'demo_path.json')); cov=[]; ids=set()
for f in pj['frames']:
    z=np.load(O/'maps'/'chunk_lane'/(f['name']+'.npz')); m=z['tmg'].astype(np.float32)>=0; cov.append(m.mean()); ids|=set(np.unique(z['tid'][m]).tolist())
print(f'[cov] final reel: cabbage-claimed {100*np.mean(cov):.1f}% of a frame (max {100*np.max(cov):.1f}%); heads decoded on the drive: {sorted(ids)} ({len(ids)} of 29)')
PY
say "=== gwakungu final ($EMBN) done in $(( ($(date +%s)-t0)/60 )) min -> $OUT/demo.mp4"
