#!/bin/bash
# e2_rebuild_chain.sh (2026-11-11, Section IV regeneration for letter v39). The geometry-arm feature files of citrus 01, 04, most of 05,
# Klapmuts Apr and Lindendhof were removed in a cleanup, so the void-free re-evaluation of 11-10 scored the arms only on 02, 03 and 05 1_0.
# Rebuild them exactly as e2_arms_cell.sh did (no census pass: the seed's labels are read back and re-targeted): Euclidean (reseed_from_labels
# with the survey's Euclidean embedder), flat 3x32 and flat 3x10 (build_flat_seed), then render void-free and score as the void-free chain
# (tags euclid_vf, flat96_vf, flat30_vf). Cells: those with protocol_euclid.jsonl and no protocol_euclid_vf.jsonl. Tools from origin/main.
# Arm bins on /mnt/shared2/e2_rebuild/<cell> (removed once scored; the root disk is ~6 GB from full). Log: /home/paperspace/logs/e2_rebuild/chain.log
set -uo pipefail
B=/home/paperspace/logs/e2_rebuild/bin; LG=/home/paperspace/logs/e2_rebuild; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi
INV=/home/paperspace/logs/identity_inventory_v2.jsonl; TMP=/mnt/shared/voidfree_tmp; mkdir -p $TMP
git -C /home/paperspace/code fetch -q origin && for f in argmax_eval.py splat_feature_render.py protocol_table.py labels_from_seed.py reseed_from_labels.py build_flat_seed.py; do git -C /home/paperspace/code show origin/main:automation/h3dgs/v3/$f > $B/$f; done
diskwait(){ while [ $(df --output=avail -k / | tail -1) -lt 4000000 ]; do say "root disk under 4 GB free: waiting"; sleep 300; done; }
cid(){ echo $1 | sed "s#/home/paperspace/data/##; s#/#__#g"; }
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol|flat-seed|reseed|labels)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_\|^\[render\] image' | tail -3 | tee -a $LG/chain.log; }
CELLS=$(python3 -c "
import json
for l in open('$INV'): print(json.loads(l)['identity'])")
say "chain start"
for N in $CELLS; do
  [ -s $N/protocol_euclid.jsonl ] || continue
  [ -s $N/protocol_euclid_vf.jsonl ] && [ -s $N/protocol_flat96_vf.jsonl ] && [ -s $N/protocol_flat30_vf.jsonl ] && continue
  diskwait; A=/mnt/shared2/e2_rebuild/$(cid $N); mkdir -p $A;   # arm bins (96-288 MB each) off the root disk, removed once scored
  ROOT=${N%%/experimental/*}; SV=$(basename $ROOT)
  EMB=$(ls $ROOT/experimental/identity/embedder/${SV}_euclid_v1/ckpts/model_best.pth 2>/dev/null || ls $ROOT/experimental/identity/embedder/*euclid*/ckpts/model_best.pth 2>/dev/null | head -1)
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0
  [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  HJ=$(python3 -c "import numpy as np, json; print(json.loads(str(np.load('$N/text_bank.npz', allow_pickle=True)['meta']))['hierarchy_json'])")
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
  LAB=$N/features_B.bin.labels.npy; [ -e $LAB ] || run $B/labels_from_seed.py --identity $N
  if [ -n "$EMB" ] && [ ! -s $N/protocol_euclid_vf.jsonl ]; then
    [ -e $A/features_euclid.bin ] || run $B/reseed_from_labels.py --labels $LAB --embedder $EMB --out $A/features_euclid.bin
    M=$TMP/$(cid $N)/e2_euclid; run $B/splat_feature_render.py --config $CFG --features $A/features_euclid.bin --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5
    run $B/argmax_eval.py --identity $N --tag euclid_vf --space euclid --emb $EMB --maps $M/%s.npz --out $N/protocol_euclid_vf.jsonl; rm -rf $M; fi
  for K in 32 10; do T=$([ $K = 32 ] && echo flat96 || echo flat30); [ -s $N/protocol_${T}_vf.jsonl ] && continue
    [ -e $A/features_$T.bin ] || run $B/build_flat_seed.py --labels $LAB --hierarchy-json $HJ --out $A/features_$T.bin --k $K
    M=$TMP/$(cid $N)/e2_$T; run $B/splat_feature_render.py --config $CFG --features $A/features_$T.bin --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5
    run $B/argmax_eval.py --identity $N --tag ${T}_vf --space flat --basis $A/features_$T.bin.basis.npz --maps $M/%s.npz --out $N/protocol_${T}_vf.jsonl; rm -rf $M
  done
  rm -rf $A; say "done $(echo $N | awk -F/ '{print $6"/"$8}') emb=$(echo $EMB | awk -F/ '{print $(NF-2)}')"
done
cd $LG
python3 $B/protocol_table.py --inventory $INV --tag tangent_vf --min-frac 0.002 --compare tangent_vf,hyper_own_vf,flat96_vf,euclid_vf --json $LG/e2_arms_vf.json --md $LG/e2_arms_vf.md 2>&1 | tail -1 | tee -a $LG/chain.log
python3 $B/protocol_table.py --inventory $INV --tag tangent_vf --min-frac 0.002 --compare tangent_vf,flat96_vf,flat30_vf --json $LG/e2_arms_flat30_vf.json --md $LG/e2_arms_flat30_vf.md 2>&1 | tail -1 | tee -a $LG/chain.log
say "CHAIN DONE"
