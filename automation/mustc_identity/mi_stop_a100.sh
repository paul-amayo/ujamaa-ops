#!/bin/bash
# mi_stop_a100.sh (A100 session, 11-11 20:35 UTC): one further rig stop of 230613 through the JS-3 pipeline, the steps and arguments the Mac
# agent used for pos00 (STATE.md, "Exact next step" item 5): mi_dump_views.py -> mi_sam3_masks.py (prompt plant, conf 0.3, min area 150) ->
# mi_project.py -> mi_build_cell.py -> mi_chain_a100.sh (embedder and hierarchy shared with pos00) -> mi_heights.py, mi_recount.py,
# mi_verify_heights.py, mi_fig.py. Resumable (each step checks its output); writes <stop dir>/STOP_DONE.
#   mi_stop_a100.sh <stop, e.g. 230613_pos01> <stop dir name, e.g. pos01>
set -uo pipefail
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
STOP=$1; SD=$2; W=/mnt/shared/must-c_bulk/work/identity_feas; B=$W/bin; D=$W/$SD; mkdir -p $D $W/logs
NS=/home/paperspace/code/nerf_new; S3=/home/paperspace/code/sam3; PX=/home/paperspace/.pixi/bin/pixi
CFG=/mnt/shared/must-c_bulk/work/ws/$STOP/outputs/sf_30k_ba/splatfacto/run/config.yml; SPLIT=/mnt/shared/must-c_bulk/work/ws/$STOP/split.json
L=$W/logs/stop_$SD.log; say(){ echo "[$(date -u '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
gpuwait(){ local n=0; while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -lt 16000 ]; do n=$((n+1)); [ $n -gt 60 ] && { say "GPU under 16 GB free for 60 min: stopping"; exit 3; }; say "GPU under 16 GB free: waiting"; sleep 60; done; }
say "=== stop $STOP -> $D"
if [ ! -e $D/cameras.json ]; then gpuwait; (cd $NS && nice -n 10 $PX run python $B/mi_dump_views.py --config $CFG --out $D) 2>&1 | grep -a '^\[dump\] [0-9]\+ views\|Error\|Traceback' | tee -a $L; fi
[ -e $D/cameras.json ] || { say "dump FAILED"; exit 1; }
if [ $(ls $D/masks/*.json 2>/dev/null | wc -l) -lt $(ls $D/views/*.png | wc -l) ]; then gpuwait; (cd $S3 && PROMPT='plant' nice -n 10 $PX run python $B/mi_sam3_masks.py $D/views $D/masks 0.3 150) 2>&1 | grep -a '^\[sam3\]\|Error\|Traceback' | tee -a $L; fi
[ $(ls $D/masks/*.json 2>/dev/null | wc -l) -ge $(ls $D/views/*.png | wc -l) ] || { say "masks FAILED"; exit 1; }
[ -e $D/plants.json ] || python3 $B/mi_project.py --stop $STOP --dir $D 2>&1 | tee -a $L
[ -e $D/plants.json ] || { say "project FAILED"; exit 1; }
[ -e $D/cell/build_summary.json ] || (cd $NS && $PX run python $B/mi_build_cell.py --dir $D --masks $D/masks --identity $W/identity --cell $D/cell --split $SPLIT) 2>&1 | grep -a '^\[build\] hierarchy\|Error\|Traceback\|Assertion' | tee -a $L
[ -e $D/cell/build_summary.json ] || { say "build FAILED"; exit 1; }
[ -e $D/CHAIN_DONE ] || bash $B/mi_chain_a100.sh $STOP $SD
[ -e $D/CHAIN_DONE ] || { say "chain FAILED (see logs/chain_$SD.log)"; exit 1; }
run(){ (cd $NS && nice -n 10 $PX run python "$@" 2>&1); }
run $B/mi_heights.py --stop $STOP --dir $D | grep -a '^\[heights\]\|Error\|Traceback' | tee -a $L
run $B/mi_recount.py --dir $D --identity-root $W/identity --out $D/recount.json > $W/logs/recount_$SD.log 2>&1; grep -a 'Error\|Traceback' $W/logs/recount_$SD.log | tee -a $L
run $B/mi_verify_heights.py --stop $STOP --dir $D --out $D/verify_heights.json > $W/logs/verify_heights_$SD.log 2>&1; grep -a 'Error\|Traceback' $W/logs/verify_heights_$SD.log | tee -a $L
mkdir -p $D/fig; run $B/mi_fig.py --dir $D --view nikon_5 --out $D/fig/addr_nikon_5.png | grep -a '^\[fig\]\|Error\|Traceback' | tee -a $L
run $B/mi_sup_overlay.py --dir $D --view nikon_5 --out $D/fig/sup_nikon_5.jpg | grep -a 'Error\|Traceback' | tee -a $L
say "=== stop done"; date -u > $D/STOP_DONE
