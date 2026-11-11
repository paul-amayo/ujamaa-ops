#!/bin/bash
# sec4_chain.sh (2026-11-11, Paul: "regenerate Section IV from the new readout numbers", letter v39: containment = encoder path).
#   E2b  cells with protocol_zero3000.jsonl: the seeding arms (zero3000, refine3000, projvote) rendered void-free and scored with
#        --space hyper_tangent (tags <arm>_tvf); the seed column is tangent_vf from readout_chain.
#   E3   cells with protocol_cat_lite.jsonl: HiGH "tree" (and "orange") through the decoder path (--space hyper_walk --category), tag cat_walk_vf.
# Tools from origin/main. Maps on /mnt/shared/voidfree_tmp, removed once scored. Log: /home/paperspace/logs/sec4/chain.log
set -uo pipefail
B=/home/paperspace/logs/sec4/bin; LG=/home/paperspace/logs/sec4; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi
INV=/home/paperspace/logs/identity_inventory_v2.jsonl; TMP=/mnt/shared/voidfree_tmp; mkdir -p $TMP
git -C /home/paperspace/code fetch -q origin && for f in argmax_eval.py splat_feature_render.py protocol_table.py; do git -C /home/paperspace/code show origin/main:automation/h3dgs/v3/$f > $B/$f; done
diskwait(){ while [ $(df --output=avail -k / | tail -1) -lt 4000000 ]; do say "root disk under 4 GB free: waiting"; sleep 300; done; }
cid(){ echo $1 | sed "s#/home/paperspace/data/##; s#/#__#g"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol|walk)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_\|^\[render\] image' | tail -3 | tee -a $LG/chain.log; }
render(){ local N=$1 F=$2 M=$3 CFG SPARSE FR; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0
  [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  run $B/splat_feature_render.py --config $CFG --features $F --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5; }
CELLS=$(python3 -c "
import json
for l in open('$INV'): print(json.loads(l)['identity'])")
say "chain start"
for N in $CELLS; do   # E2b
  [ -s $N/protocol_zero3000.jsonl ] || continue; A=$N/arms
  for arm in zero3000 refine3000 projvote; do
    [ -e $A/features_$arm.bin ] || continue; [ -s $N/protocol_${arm}_tvf.jsonl ] && continue
    diskwait; M=$TMP/$(cid $N)/s4_$arm; render $N $A/features_$arm.bin $M
    run $B/argmax_eval.py --identity $N --tag ${arm}_tvf --space hyper_tangent --maps $M/%s.npz --out $N/protocol_${arm}_tvf.jsonl; rm -rf $M
  done; say "E2b $(echo $N | awk -F/ '{print $6"/"$8}')"
done
for N in $CELLS; do   # E3
  [ -s $N/protocol_cat_lite.jsonl ] || continue; [ -s $N/protocol_cat_walk_vf.jsonl ] && continue
  diskwait; M=$TMP/$(cid $N)/s4_cat; render $N $N/features_B.bin $M
  run $B/argmax_eval.py --identity $N --tag cat_walk_vf --space hyper_walk --category tree=ALL,orange=FRUIT --maps $M/%s.npz --out $N/protocol_cat_walk_vf.jsonl; rm -rf $M
  say "E3 $(echo $N | awk -F/ '{print $6"/"$8}')"
done
cd $LG
python3 $B/protocol_table.py --inventory $INV --tag tangent_vf --min-frac 0.002 --json $LG/e1_tangent.json --md $LG/e1_tangent.md 2>&1 | tail -1 | tee -a $LG/chain.log
python3 $B/protocol_table.py --inventory $INV --tag tangent_vf --min-frac 0.002 --compare tangent_vf,hyper_own_vf,flat96_vf,euclid_vf,flat30_vf --json $LG/e2_arms_vf.json --md $LG/e2_arms_vf.md 2>&1 | tail -1 | tee -a $LG/chain.log
python3 $B/protocol_table.py --inventory $INV --tag tangent_vf --min-frac 0.002 --compare tangent_vf,zero3000_tvf,refine3000_tvf,projvote_tvf --json $LG/e2b_seed_vf.json --md $LG/e2b_seed_vf.md 2>&1 | tail -1 | tee -a $LG/chain.log
python3 $B/protocol_table.py --inventory $INV --tag cat_walk_vf --min-frac 0.002 --compare cat_walk_vf,cat_lite --json $LG/e3_category_vf.json --md $LG/e3_category_vf.md 2>&1 | tail -1 | tee -a $LG/chain.log
say "CHAIN DONE"
