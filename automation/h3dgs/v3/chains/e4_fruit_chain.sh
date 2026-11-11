#!/bin/bash
# e4_fruit_chain.sh (2026-11-11, Section IV regeneration for letter v39): the fruit level scored as a pixel level on the 12 fruit-bearing citrus
# cells (frames_fruit.json), void-free, under the encoder-path readout (--space hyper_tangent), mixed fruit+tree competition as the letter
# (tag fruit_tvf). Medians: e4_fruit_tvf.json. Tools from origin/main. Log: /home/paperspace/logs/e4_fruit/chain.log
set -uo pipefail
B=/home/paperspace/logs/e4_fruit/bin; LG=/home/paperspace/logs/e4_fruit; mkdir -p $B $LG
V3=/home/paperspace/code/automation/h3dgs/v3; NS=/home/paperspace/code/nerf_new; PX=/home/paperspace/.pixi/bin/pixi; TMP=/mnt/shared/voidfree_tmp
git -C /home/paperspace/code fetch -q origin && for f in argmax_eval.py splat_feature_render.py; do git -C /home/paperspace/code show origin/main:automation/h3dgs/v3/$f > $B/$f; done
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LG/chain.log; }
run(){ (cd $NS && env PYTHONPATH=$V3 nice -n 10 $PX run python "$@" 2>&1) | grep -aE '^\[(render|protocol)\]|Error|Traceback' | grep -av '^\[render\] kf\|^\[render\] f_' | tail -2 | tee -a $LG/chain.log; }
for J in /home/paperspace/data/citrus_all/*/experimental/v3/*/identity/frames_fruit.json; do
  N=$(dirname $J); [ -s $N/protocol_fruit_tvf.jsonl ] && continue
  CFG=$(python3 -c "import json; print(json.load(open('$N/W.json'))['config'])"); SPARSE=$(dirname $N)/sparse/0
  [ -e $SPARSE/images.bin ] || SPARSE=$(dirname $(dirname $(dirname $(dirname $CFG))))/sparse/0
  FR=$(python3 -c "import json; print(' '.join(json.load(open('$J'))['frames']))"); M=$TMP/$(echo $N | sed "s#/home/paperspace/data/##; s#/#__#g")/e4
  run $B/splat_feature_render.py --config $CFG --features $N/features_B.bin --frames $FR --out-dir $M --colmap-sparse $SPARSE --void-free 0.5
  run $B/argmax_eval.py --identity $N --tag fruit_tvf --space hyper_tangent --frames-json $J --maps $M/%s.npz --out $N/protocol_fruit_tvf.jsonl; rm -rf $M
  say "done $(echo $N | awk -F/ '{print $6"/"$8}')"
done
python3 - <<'PY' | tee -a $LG/chain.log
import json, glob, numpy as np
out = {}
for sv in ('04_13D_Jackal', '05_13D_Jackal'):
    for J in glob.glob(f'/home/paperspace/data/citrus_all/{sv}/experimental/v3/*/identity/frames_fruit.json'):
        p = J.replace('frames_fruit.json', 'protocol_fruit_tvf.jsonl')
        try: L = [json.loads(l) for l in open(p)]
        except FileNotFoundError: continue
        for l in L:
            if l.get('kind') in ('FRUIT', 'TREE'):
                k = ('census' if l['split'] == 'train' else 'held-out', l['kind']); out.setdefault(sv, {}).setdefault(k, []).append(l['argmax']['iou'])
res = {sv: {f'{s}/{k}': {'n': len(v), 'argmax_med': float(np.median(v))} for (s, k), v in d.items()} for sv, d in out.items()}
json.dump(res, open('/home/paperspace/logs/e4_fruit/e4_fruit_tvf.json', 'w'), indent=1); print(json.dumps(res))
PY
say "CHAIN DONE"
