#!/bin/bash
# readout_chain.sh (2026-11-11, Paul: "phrase it both ways ... one from the encoder and the other from the decoder; rescore"). Every robot-survey
# cell of identity_inventory_v2 (phone walks excluded) is rendered void-free (as voidfree_chain phase 1) and scored by argmax_eval under the
# two readouts of draft 38: walk_vf (decoder path: the pixel walked to the queried level's radius, decoded) and tangent_vf (encoder path:
# cos(f, g(t_q))). Containment is v3vf from the void-free chain. Maps on /mnt/shared/voidfree_tmp, removed once scored. Resumable.
# Log: /home/paperspace/logs/readout/chain.log.   usage: setsid nohup readout_chain.sh &
set -uo pipefail
B=/home/paperspace/logs/readout/bin; LG=/home/paperspace/logs/readout; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi
INV=/home/paperspace/logs/identity_inventory_v2.jsonl; TMP=/mnt/shared/voidfree_tmp; mkdir -p $TMP
# the box checkout of ops diverges from origin (other sessions commit there): tools come from origin/main, not the working tree
git -C /home/paperspace/code fetch -q origin && for f in argmax_eval.py splat_feature_render.py protocol_table.py; do git -C /home/paperspace/code show origin/main:automation/h3dgs/v3/$f > $B/$f; done
diskwait(){ while [ $(df --output=avail -k / | tail -1) -lt 4000000 ]; do say "root disk under 4 GB free: waiting"; sleep 300; done; }
cid(){ echo $1 | sed "s#/home/paperspace/data/##; s#/#__#g"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol|walk)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_\|^\[render\] image' | tail -3 | tee -a $LG/chain.log; }
CELLS=$(python3 -c "
import json
for l in open('$INV'):
    c = json.loads(l)
    if not c['survey'].startswith('IMG_'): print(c['identity'])")
say "chain start: $(echo "$CELLS" | wc -l) cells"
for N in $CELLS; do
  if [ -s $N/protocol_walk_vf.jsonl ] && [ -s $N/protocol_tangent_vf.jsonl ]; then continue; fi
  diskwait; M=$TMP/$(cid $N)/readout
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0
  [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  [ -e $M/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $B/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5
  [ -s $N/protocol_walk_vf.jsonl ] || run $B/argmax_eval.py --identity $N --tag walk_vf --space hyper_walk --maps $M/%s.npz --out $N/protocol_walk_vf.jsonl
  [ -s $N/protocol_tangent_vf.jsonl ] || run $B/argmax_eval.py --identity $N --tag tangent_vf --space hyper_tangent --maps $M/%s.npz --out $N/protocol_tangent_vf.jsonl
  rm -rf $M; say "done $(echo $N | awk -F/ '{print $6"/"$8}')"
done
say "tables"; cd $LG && python3 $B/protocol_table.py --inventory $INV --min-frac 0.002 --compare v3vf,walk_vf,tangent_vf --json $LG/readout_compare.json --md $LG/readout_compare.md 2>&1 | tail -3 | tee -a $LG/chain.log
say "CHAIN DONE"
