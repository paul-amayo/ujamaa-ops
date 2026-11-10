#!/bin/bash
# voidfree_chain.sh (2026-11-10) — the HiGH letter's evaluation re-run under the void-free readout (Paul, draft 36: "never blend void
# pixels"): every feature map is rendered with splat_feature_render.py --void-free 0.5 (the blend over seeded Gaussians only, void where
# their weight < 0.5) and scored by argmax_eval.py exactly as before, under new tags, so nothing of the original evaluation is touched.
#   phase 1  E1   every robot-survey cell of identity_inventory_v2 (the phone walks excluded): v3vf (hyper) and hyper_own_vf (arm D)
#   phase 2  E2   cells that carry the geometry arms (protocol_euclid.jsonl): euclid_vf (arm B), flat96_vf (arm C), flat30_vf (equal storage)
#   phase 3  E2b  cells that carry the seeding arms (protocol_zero3000.jsonl): zero3000_vf, refine3000_vf, projvote_vf
# Resumable: a step is skipped when its protocol file exists. Tools are run from $B (copies of ops main taken at launch).
# Log: /home/paperspace/logs/voidfree/chain.log.   usage: setsid nohup voidfree_chain.sh &
set -uo pipefail
B=/home/paperspace/logs/voidfree/bin; LG=/home/paperspace/logs/voidfree; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi; VF=0.5
INV=/home/paperspace/logs/identity_inventory_v2.jsonl
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_\|^\[render\] image' | tail -3 | tee -a $LG/chain.log; }
CELLS=$(python3 -c "
import json
for l in open('$INV'):
    c = json.loads(l)
    if not c['survey'].startswith('IMG_'): print(c['identity'])")
render(){   # $1 features  $2 out dir  $3 identity
  local N=$3 CFG SPARSE FR; CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0
  [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  [ -e $2/$(echo $FR | cut -d' ' -f1 | sed 's/.png/.npz/') ] || run $B/splat_feature_render.py --config $CFG --features $1 --frames $FR --out-dir $2 --colmap-sparse $SPARSE --void-free $VF
}
score(){ local N=$1 TAG=$2; shift 2; [ -s $N/protocol_$TAG.jsonl ] || run $B/argmax_eval.py --identity $N --tag $TAG --out $N/protocol_$TAG.jsonl "$@"; }
say "chain start: $(echo "$CELLS" | wc -l) cells, void-free share $VF"
for N in $CELLS; do   # phase 1
  render $N/features_B.bin $N/maps_v3vf $N
  score $N v3vf --space hyper --maps $N/maps_v3vf/%s.npz
  score $N hyper_own_vf --space hyper_own --maps $N/maps_v3vf/%s.npz
  say "E1 $(echo $N | awk -F/ '{print $6"/"$8}')"
done
for N in $CELLS; do   # phase 2
  [ -s $N/protocol_euclid.jsonl ] || continue; A=$N/arms; SV=$(echo $N | awk -F/ '{print $6}')
  EMB=/home/paperspace/data/citrus_all/$SV/experimental/identity/embedder/${SV}_euclid_v1/ckpts/model_best.pth
  if [ -e $A/features_euclid.bin ] && [ -e $EMB ]; then render $A/features_euclid.bin $A/maps_euclidvf $N; score $N euclid_vf --space euclid --emb $EMB --maps $A/maps_euclidvf/%s.npz; fi
  if [ -e $A/features_flat96.bin ]; then render $A/features_flat96.bin $A/maps_flat96vf $N; score $N flat96_vf --space flat --basis $A/features_flat96.bin.basis.npz --maps $A/maps_flat96vf/%s.npz; fi
  if [ -e $A/features_flat30.bin ]; then render $A/features_flat30.bin $A/maps_flat30vf $N; score $N flat30_vf --space flat --basis $A/features_flat30.bin.basis.npz --maps $A/maps_flat30vf/%s.npz; fi
  say "E2 $(echo $N | awk -F/ '{print $6"/"$8}')"
done
for N in $CELLS; do   # phase 3
  [ -s $N/protocol_zero3000.jsonl ] || continue; A=$N/arms
  for arm in zero3000 refine3000 projvote; do
    [ -e $A/features_$arm.bin ] || continue; render $A/features_$arm.bin $A/maps_${arm}vf $N; score $N ${arm}_vf --space hyper --maps $A/maps_${arm}vf/%s.npz
  done
  say "E2b $(echo $N | awk -F/ '{print $6"/"$8}')"
done
say "CHAIN DONE"
