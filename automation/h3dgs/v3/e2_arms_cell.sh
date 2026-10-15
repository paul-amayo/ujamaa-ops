#!/bin/bash
# e2_arms_cell.sh — the paper's geometry ablation on one identity cell (2026-10-13): same census W, frames, supervision and scorer as the
# v3 identity (arm A); B = a Euclidean embedder's seed (EMB_EUCLID), C = three flat PCA-32 fields tree|row|fruit (build_flat_seed.py,
# 96-d), D = the hyperbolic field scored at the pixel's own point (no ancestors). Outputs under <identity>/arms/ + protocol_<arm>.jsonl.
#   N=<ws>/identity [EMB_EUCLID=<ckpt>] [ARMS=B,C,D] [FLAGS=<seed flags>] e2_arms_cell.sh
set -uo pipefail
: ${N:?}; ARMS=${ARMS:-B,C,D}; FLAGS=${FLAGS:---tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1}
V3=/home/paperspace/code/automation/h3dgs/v3; ARU=/home/paperspace/code/aru_sil_core/src/scripts; NS=/home/paperspace/code/nerf_new
WS=$(dirname $N); CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); A=$N/arms; mkdir -p $A   # the census recorded the checkpoint it ran on (05 1_0 lives under splatfacto_1_0, not the chunk ws)
SPARSE=$WS/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
HJ=$(python3 -c "import numpy as np, json; print(json.loads(str(np.load('$N/text_bank.npz', allow_pickle=True)['meta']))['hierarchy_json'])")
EMB_A=$(python3 -c "import json; print(json.load(open('$N/features_B.bin.json'))['embedder'])")
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
L=/home/paperspace/logs/e2_arms.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $(echo $N | awk -F/ '{print $6"/"$8}'): $*" | tee -a $L; }; t0=$(date +%s)
run(){ (cd $NS && nice -n 10 pixi run python "$@" 2>&1) | grep -aE '^\[(census|flat-seed|reseed|labels|render|protocol|check|vocab)\]|assigned|Error|Traceback' | tee -a $L; }
seed(){ run $ARU/build_census_init.py --w-npz $N/W.npz --embedder $1 --out-features $2 $FLAGS; }
render(){ [ -e $2/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $V3/splat_feature_render.py --config $CFG --features $1 --frames $FR --out-dir $2 --colmap-sparse $SPARSE; }
[ -e $A/features_A.bin.labels.npy ] || { [ -e $N/features_B.bin.labels.npy ] || run $V3/labels_from_seed.py --identity $N; ln -sf $N/features_B.bin.labels.npy $A/features_A.bin.labels.npy; }   # arm A's assignment, read back from the existing seed (no census algebra: Dec's W is 772 x 1.47 M)
case ",$ARMS," in *,B,*) : ${EMB_EUCLID:?}; [ -e $A/features_euclid.bin ] || run $V3/reseed_from_labels.py --labels $A/features_A.bin.labels.npy --embedder $EMB_EUCLID --out $A/features_euclid.bin   # same assignment, Euclidean targets (no second W pass)
  render $A/features_euclid.bin $A/maps_euclid; [ -s $N/protocol_euclid.jsonl ] || run $V3/argmax_eval.py --identity $N --tag euclid --space euclid --emb $EMB_EUCLID --maps arms/maps_euclid/%s.npz;; esac
case ",$ARMS," in *,C,*) [ -e $A/features_flat96.bin ] || run $V3/build_flat_seed.py --labels $A/features_A.bin.labels.npy --hierarchy-json $HJ --out $A/features_flat96.bin
  render $A/features_flat96.bin $A/maps_flat96; [ -s $N/protocol_flat96.jsonl ] || run $V3/argmax_eval.py --identity $N --tag flat96 --space flat --basis $A/features_flat96.bin.basis.npz --maps arms/maps_flat96/%s.npz;; esac
case ",$ARMS," in *,D,*) [ -s $N/protocol_hyper_own.jsonl ] || run $V3/argmax_eval.py --identity $N --tag hyper_own --space hyper_own;; esac
case ",$ARMS," in *,C30,*) [ -e $A/features_flat30.bin ] || run $V3/build_flat_seed.py --labels $A/features_A.bin.labels.npy --hierarchy-json $HJ --out $A/features_flat30.bin --k 10   # equal-storage flat arm: 3 x 10 = 30 floats
  render $A/features_flat30.bin $A/maps_flat30; [ -s $N/protocol_flat30.jsonl ] || run $V3/argmax_eval.py --identity $N --tag flat30 --space flat --basis $A/features_flat30.bin.basis.npz --maps arms/maps_flat30/%s.npz;; esac
say "arms $ARMS done in $(( $(date +%s)-t0 )) s"
