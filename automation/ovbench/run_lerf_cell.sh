#!/bin/bash
# run_lerf_cell.sh <survey> <chunk> [method=lerf] — ovbench (2026-11-10): train published LERF (kerrj/lerf db08d57) on one identity
# cell's TRAINING views (the splatfacto run's own train_filenames), render its relevancy on the cell's verdict frames and score it
# with the HiGH protocol. Resumable by stage files under /home/paperspace/logs/ovbench/lerf/<survey>_<chunk>/. Bulky outputs
# (checkpoints, LERF's CLIP/DINO caches) live on /mnt/shared (root fs is ~100 % full) through a symlink.
#   setsid nohup nice -n 10 bash run_lerf_cell.sh 05_13D_Jackal chunk_1_0 > log 2>&1 &
set -uo pipefail
SURVEY=$1; CHUNK=$2; METHOD=${3:-lerf}
B=/home/paperspace/logs/ovbench; BIN=$B/bin; SRC=$B/src/lerf_kerrj_db08d57; CELL=${SURVEY}_${CHUNK#chunk_}
W=$B/$METHOD/$CELL; mkdir -p $W /mnt/shared/ovbench/$METHOD/$CELL/outputs; [ -e $W/outputs ] || ln -s /mnt/shared/ovbench/$METHOD/$CELL/outputs $W/outputs
cd $W || exit 1
PIXI="/home/paperspace/.pixi/bin/pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml"
PIX() { $PIXI bash -c "cd $W && export PYTHONPATH=$SRC && $*"; }   # pixi env, but in the cell dir (LERF caches are cwd-relative)
export PYTHONPATH=$SRC
INV=$(grep "\"survey\": \"$SURVEY\"" /home/paperspace/logs/identity_inventory_v2.jsonl | grep "\"chunk\": \"$CHUNK\"" | head -1)
ID=$(echo "$INV" | python3 -c 'import json,sys;print(json.load(sys.stdin)["identity"])'); SUP=$(echo "$INV" | python3 -c 'import json,sys;print(json.load(sys.stdin)["sup"])')
HJ=$(echo "$INV" | python3 -c 'import json,sys;print(json.load(sys.stdin)["hj"])')
SFCFG=$(python3 -c "import json;print(json.load(open('$ID/W.json'))['config'])")
DATA=$(python3 - "$SFCFG" <<'EOF'
import sys
L = open(sys.argv[1]).read().splitlines(); i = next(k for k, l in enumerate(L) if l.startswith('data:')); parts = []
for l in L[i + 1:]:
    if not l.startswith('- '): break
    parts.append(l[2:].strip())
print('/' + '/'.join(parts[1:]))
EOF
)
echo "[cell] $CELL identity=$ID sup=$SUP data=$DATA cfg=$SFCFG $(date -Is)"
python3 - "$DATA" "$ID" <<'EOF' || exit 1
import json, sys
t = json.load(open(sys.argv[1] + '/transforms.json')); fj = json.load(open(sys.argv[2] + '/frames.json')); tr = {p.split('/')[-1] for p in t['train_filenames']}
sn = json.load(open(sys.argv[2] + '/split_names.json')); assert len(tr) == len(sn['train']) and tr == set(sn['train']), 'LERF train set != the identity split train'
bad = [f for f in fj['frames'] if fj['info'][f]['split'] == 'eval' and f in tr]
print(f'[split] train {len(tr)} = identity train split; held-out verdict frames in train: {bad}'); sys.exit(1 if bad else 0)
EOF
RUN=$W/outputs/ovbench_$CELL/$METHOD/run
if [ ! -f $W/TRAIN_DONE ]; then
  T0=$(date +%s)
  PIX ns-train $METHOD --data $DATA --output-dir $W/outputs --experiment-name ovbench_$CELL --timestamp run --vis tensorboard > $W/train.log 2>&1
  RC=$?; T1=$(date +%s); echo "{\"rc\": $RC, \"seconds\": $((T1-T0)), \"start\": $T0}" > $W/train_time.json
  [ $RC -eq 0 ] && ls $RUN/nerfstudio_models/*.ckpt >/dev/null 2>&1 && touch $W/TRAIN_DONE || { echo "[train] failed rc=$RC"; tail -30 $W/train.log; exit 1; }
fi
echo "[train] done $(cat $W/train_time.json)"
PIX python $BIN/score_ov.py --write-queries $W/queries.json --identity $ID --sup $SUP --hier $HJ || exit 1
if [ ! -f $W/RENDER_DONE ]; then
  PIX python $BIN/lerf_relevancy.py --config $RUN/config.yml --identity $ID --out-dir $W/rel --queries-json $W/queries.json > $W/render.log 2>&1 && touch $W/RENDER_DONE || { tail -30 $W/render.log; exit 1; }
fi
PIX python $BIN/score_ov.py --rel-dir $W/rel --identity $ID --sup $SUP --hier $HJ --method $METHOD --cell $CELL --out $W/scores.jsonl --summary $W/summary.json > $W/score.log 2>&1 || { tail -30 $W/score.log; exit 1; }
tail -25 $W/score.log; touch $W/ALL_DONE; echo "[cell] $CELL done $(date -Is)"
