#!/bin/bash
# joint_train_chain.sh (2026-11-10, Paul: "we went into census prematurely without considering the void mixing; show some joint training of
# features using geodesic distance, queued on the A100 for tomorrow"). On the fifteen fixed cells of the seeding ablation (cells with
# protocol_zero3000.jsonl), per-Gaussian features are trained against the census targets with the geodesic loss (train_feature_field.py,
# 3,000 steps, lr 0.01 as E2b) in four variants, then rendered with the void-free readout and scored exactly like the census (argmax_eval):
#   A  zero init, loss on the blend normalised by the pixel's total weight (--render-norm): transparency no longer shrinks the feature
#   B  A + a void term on unlabelled pixels (--void-weight 0.1): Gaussians seen against nothing labelled are pulled to the origin
#   C  B started from the census seed (refine)
#   D  OpenHype's lifting loss: zero init, geodesic + (d(f,O) - d(t,O))^2 (--norm-reg 1.0), plain blend
# Starts after 06:00 the next day, once the void-free chain has logged CHAIN DONE and the GPU has >= 12 GB free; the root disk is checked
# before each step (>= 5 GB). Feature bins on /mnt/shared2/joint_train (kept for 05 1_0 only), maps on /mnt/shared/voidfree_tmp (removed
# once scored). Tags jt_A..jt_D. Log: /home/paperspace/logs/joint_train/chain.log.  usage: setsid nohup joint_train_chain.sh &
set -uo pipefail
B=/home/paperspace/logs/joint_train/bin; LG=/home/paperspace/logs/joint_train; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi
INV=/home/paperspace/logs/identity_inventory_v2.jsonl; BINS=/mnt/shared2/joint_train; TMP=/mnt/shared/voidfree_tmp; mkdir -p $BINS $TMP
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(train|render|protocol)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_' | tail -4 | tee -a $LG/chain.log; }
diskwait(){ while [ $(df --output=avail -k / | tail -1) -lt 5000000 ]; do say "root disk under 5 GB free: waiting"; sleep 300; done; }
gpuwait(){ while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt 12000 ]; do say "GPU under 12 GB free: waiting"; sleep 300; done; }
cid(){ echo $1 | sed "s#/home/paperspace/data/##; s#/#__#g"; }
START=$(date -d 'tomorrow 06:00' +%s); say "queued: start after $(date -d @$START '+%m-%d %H:%M') and the void-free chain's CHAIN DONE"
while [ $(date +%s) -lt $START ] || ! grep -q "CHAIN DONE" /home/paperspace/logs/voidfree/chain.log; do sleep 600; done
say "joint training start"
CELLS=$(python3 -c "
import json, os
for l in open('$INV'):
    c = json.loads(l)
    if os.path.exists(c['identity'] + '/protocol_zero3000.jsonl'): print(c['identity'])")
say "$(echo "$CELLS" | wc -l) cells"
for N in $CELLS; do
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0; [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))"); C=$(cid $N); mkdir -p $BINS/$C
  for V in A B C D; do
    [ -s $N/protocol_jt_$V.jsonl ] && continue
    case $V in A) ARGS="--init zero --render-norm";; B) ARGS="--init zero --render-norm --void-weight 0.1";;
               C) ARGS="--init $N/features_B.bin --render-norm --void-weight 0.1";; D) ARGS="--init zero --norm-reg 1.0";; esac
    diskwait; gpuwait; F=$BINS/$C/jt_$V.bin
    [ -s $F ] || run $B/train_feature_field.py --identity $N $ARGS --steps 3000 --lr 0.01 --out $F
    M=$TMP/$C/jt_$V; run $B/splat_feature_render.py --config $CFG --features $F --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5
    run $B/argmax_eval.py --identity $N --tag jt_$V --space hyper --maps $M/%s.npz --out $N/protocol_jt_$V.jsonl
    rm -rf $M; case $C in *05_13D_Jackal*chunk_1_0*) ;; *) rm -f $F;; esac
    say "$C jt_$V done"
  done
done
say "tables"; cd $LG && python3 $B/protocol_table.py --inventory $INV --min-frac 0.002 --compare v3vf,jt_A,jt_B,jt_C,jt_D --json $LG/joint_train_compare.json --md $LG/joint_train_compare.md 2>&1 | tail -3 | tee -a $LG/chain.log
say "CHAIN DONE"
