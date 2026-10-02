#!/bin/bash
# cabbage_oracle_drive.sh — per-head ORACLE (containment_eval --oracle = embedder ceiling) for v2_e400 on the frames that carry the
# reel drive's heads (image_64: head 30, image_50: head 23). Chained behind the second drive-head verdict run (card never shared).
set -uo pipefail
L=/home/paperspace/logs/cabbage_embedder_v2.log; RL=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
until [ "$(grep -caE '^\[[0-9 :-]+\] === gwakungu drive-head verdicts \(IMG_7993_s0_cabbage_v2_e400\) done' $RL)" -ge 2 ]; do sleep 15; done
C=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7993_s0_cabbage; S=$C/demo_root; O=$S/experimental/h3dgs_sidecar_chunks/chunk_lane; SUP=$O/supervision/trees_only; HJ=$S/prod/bateleur/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3; ARU=/home/paperspace/code/aru_sil_core/src/scripts
RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ | tail -1)
EMBN=IMG_7993_s0_cabbage_v2_e400; EMB=$C/hyper/$EMBN/ckpts/model_best.pth
say "=== cabbage oracle drive heads ($EMBN) on image_64 image_50"; cd /home/paperspace/code/nerf_new
for FR in image_64.png image_50.png; do
  HIGH_EMBEDDER_CKPT=$EMB timeout 500 pixi run python $ARU/containment_eval.py --config $RUN/config.yml --hyper-ckpt $EMB --hierarchy-json $HJ --supervision-dir $SUP --frame $FR --kf-images $KF --oracle --out /home/paperspace/logs/sidecar_figs_IMG_7993_s0_cabbage/oracle_${EMBN}_$FR > /home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_oracle_${EMBN}_$FR.log 2>&1
  say "ORACLE $EMBN $FR: $(grep -aE '^(TREE|ROW) ' /home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_oracle_${EMBN}_$FR.log | sed -E 's/ +/ /g' | grep -oE '(TREE|ROW) [0-9]+ "[a-z]+": thr [0-9.na]+ IoU [0-9.]+' | tr '\n' ';')$(grep -aE 'Error|no supervision frame' /home/paperspace/logs/sidecar_IMG_7993_s0_cabbage_oracle_${EMBN}_$FR.log | head -1)"
  rm -rf $O/clip_cache_*
done
say "=== cabbage oracle drive heads done"
