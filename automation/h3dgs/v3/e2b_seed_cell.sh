#!/bin/bash
# e2b_seed_cell.sh — the seeding ablation on one identity cell (paper E2b, 2026-10-13): Z = features learned from zero by gradient descent
# (stage-2 recipe, STEPS steps), R = the census seed refined the same way, P = projection-vote lift (centres, no alpha weights) with the
# production embedder's targets; each rendered on the cell's verdict frames and scored with the E1 protocol (hyperbolic space).
#   N=<ws>/identity [ARMS=Z,R,P] [STEPS=3000] [LR=0.01] e2b_seed_cell.sh
set -uo pipefail; : ${N:?}; ARMS=${ARMS:-Z,R,P}; STEPS=${STEPS:-3000}; LR=${LR:-0.01}
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; A=$N/arms; mkdir -p $A
CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); WS=$(dirname $N); SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
EMB_A=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])"); FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
L=/home/paperspace/logs/e2b_seed.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(echo $N | awk -F/ '{print $6"/"$8}'): $*" | tee -a $L; }; t0=$(date +%s)
run(){ (cd $NS && nice -n 10 pixi run python "$@" 2>&1) | grep -aE '^\[(train|projvote|reseed|render|protocol|vocab)\]|Error|Traceback' | tee -a $L; }
render(){ [ -e $2/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $V3/splat_feature_render.py --config $CFG --features $1 --frames $FR --out-dir $2 --colmap-sparse $SPARSE; }
score(){ [ -s $N/protocol_$1.jsonl ] || run $V3/argmax_eval.py --identity $N --tag $1 --space hyper --maps arms/$2/%s.npz; }
case ",$ARMS," in *,Z,*) [ -e $A/features_zero$STEPS.bin ] || run $V3/train_feature_field.py --identity $N --init zero --steps $STEPS --lr $LR --out $A/features_zero$STEPS.bin
  render $A/features_zero$STEPS.bin $A/maps_zero$STEPS; score zero$STEPS maps_zero$STEPS;; esac
case ",$ARMS," in *,R,*) [ -e $A/features_refine$STEPS.bin ] || run $V3/train_feature_field.py --identity $N --init $N/features_B.bin --steps $STEPS --lr $LR --out $A/features_refine$STEPS.bin
  render $A/features_refine$STEPS.bin $A/maps_refine$STEPS; score refine$STEPS maps_refine$STEPS;; esac
case ",$ARMS," in *,P,*) [ -e $A/projvote.labels.npy ] || run $V3/projection_vote_seed.py --identity $N --out $A/projvote.labels.npy
  [ -e $A/features_projvote.bin ] || run $V3/reseed_from_labels.py --labels $A/projvote.labels.npy --embedder $EMB_A --out $A/features_projvote.bin
  render $A/features_projvote.bin $A/maps_projvote; score projvote maps_projvote;; esac
say "seeding arms $ARMS done in $(( $(date +%s)-t0 )) s"
