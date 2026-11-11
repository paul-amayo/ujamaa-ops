#!/bin/bash
# mi_chain.sh — MuST-C identity feasibility (JS-3, 2026-11-11): the HiGH new-cell recipe on one rig stop, after mi_dump_views.py,
# mi_sam3_masks.py, mi_project.py and mi_build_cell.py. Commands and flags transcribed from automation/lindendhof/anchor_identity_chain.sh
# (graph embedder), automation/h3dgs/v3/v3_identity_cell.sh (text bank, flat census with the unlabelled row, seed B) and
# automation/h3dgs/v3/chains/readout_chain.sh (void-free render, tangent readout). Tools = the ops origin/main copies in $W/bin, unmodified;
# build_census_init.py and train_hyperembedder_graph.py are run in place from aru_sil_core (read-only).
#   mi_chain.sh <stop, e.g. 230613_pos00> <stop dir name, e.g. pos00>      (box; setsid nohup; waits on nothing, writes <stop dir>/CHAIN_DONE)
set -uo pipefail
STOP=$1; SD=$2; W=/mnt/shared/must-c_bulk/work/identity_feas; B=$W/bin; D=$W/$SD; C=$D/cell; N=$D/identity; ID=$W/identity; mkdir -p $N $W/logs
NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi; ARU=/home/paperspace/code/aru_sil_core/src/scripts; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH
CFG=/mnt/shared/must-c_bulk/work/ws/$STOP/outputs/sf_30k_ba/splatfacto/run/config.yml; HJ=$ID/marker_hierarchy.json; EN=mustc198_${STOP%%_*}_v1; EMB=$ID/embedder/$EN/ckpts/model_best.pth
L=$W/logs/chain_$SD.log; say(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
gpuwait(){ while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt 12000 ]; do say "GPU under 12 GB free: waiting"; sleep 60; done; }
run(){ (cd $NS && nice -n 10 $PX run python "$@" 2>&1); }
say "=== mi_chain $STOP: cfg $CFG"
if [ ! -e $EMB ]; then gpuwait; t1=$(date +%s)
  run $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name $EN --output-dir $ID/embedder \
      --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row --epochs 1500 --no-level-norms > $W/logs/embedder_$EN.log 2>&1
  say "embedder rc=$? in $(( $(date +%s)-t1 )) s: $(grep -a 'Row (retrieval)' $W/logs/embedder_$EN.log | tail -1 | tr -s ' ') | $(grep -a 'Avg Loss' $W/logs/embedder_$EN.log | tail -1 | tr -s ' ')"
  find $ID/embedder/$EN/ckpts -name 'checkpoint-epoch*.pth' -delete 2>/dev/null   # intermediates; model_best.pth is the one the recipe uses
fi
[ -e $EMB ] || { say "embedder FAILED ($W/logs/embedder_$EN.log)"; exit 1; }
cp -n $C/frames.json $C/split_names.json $N/
[ -e $C/text_bank.npz ] || run $B/build_text_bank.py --manifest $C/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $C/text_bank.npz 2>&1 | grep -a 'text-bank\|Error\|Traceback' | tee -a $L
cp -n $C/text_bank.npz $N/
gpuwait; t1=$(date +%s)
[ -e $N/W.npz ] || run $B/splat_census.py --config $CFG --supervision-dir $C/supervision/trees_only --names-json $N/split_names.json --with-bg --out-npz $N/W.npz 2>&1 | grep -aE '^\[census\] (saved|splat|sf)|Error|Traceback' | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t1 )) s"
[ -e $N/features_B.bin ] || (cd $NS && HIGH_EMBEDDER_CKPT=$EMB $PX run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_B.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1) 2>&1 | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
[ -e $N/features_B.bin ] || { say "assignment FAILED"; exit 1; }
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
gpuwait; M=$D/maps_vf
run $B/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $M --void-free 0.5 2>&1 | grep -aE '^\[render\]|Error|Traceback' | sed -E 's/ -> .*//' | tee -a $L
for f in $FR; do [ -e $M/${f%????}.npz ] && mv $M/${f%????}.npz $M/$f.npz; done   # the renderer cuts 4 characters off the frame name, the scorer asks for <frame name>.npz (frames end .jpeg, the tools assume .png)
gpuwait
run $B/argmax_eval.py --identity $N --tag tangent_vf --space hyper_tangent --maps $M/%s.npz --out $N/protocol_tangent_vf.jsonl --dump-pred $D/pred_tangent_vf > $W/logs/argmax_$SD.log 2>&1
grep -aE '^\[(protocol|walk|vocab)\]|Error|Traceback' $W/logs/argmax_$SD.log | tee -a $L
say "=== done"; date -u > $D/CHAIN_DONE
