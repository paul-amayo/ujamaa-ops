#!/bin/bash
# Chilli H3DGS trainings v2: demo card windows (dashboard session for Paul, 2026-10-05): the card must be free from 18:15 UTC
# (20:15 SAST) until the dashboard says "rehearsal done" (-> touch $FLAG), and from 05:00 UTC Tue (07:00 SAST) until the demo is over.
# A phase starts only if its measured duration (+ margin) fits before the next blackout; a watchdog kills a phase still running when a
# blackout starts. Phases write their product only at the end (train_single: point cloud, the creator: hierarchy.hier, train_post:
# hierarchy.hier_opt), so a stop between or inside phases leaves nothing that needs the card to recover.
# Measured (8 M, 60k its): train 6,656-6,724 s, hierarchy ~150 s, post-opt 5,396-6,150 s, eval ~60 s, segment SfM ~10 min.
# Work: s1a (SfM + recipe 0-2 done by v1, step check OK) -> s0 (clip 0-150) -> s1b (clip 295-393, trained only if its step check passes).
set -u
G=/home/paperspace/data/image_farm/gwakungu/2026-05-16; IF=/home/paperspace/code/automation/image_farm; L=/home/paperspace/logs/chilli_trainings.log
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; REPO=/home/paperspace/code/hierarchical-3d-gaussians
PAUSE=$(date -u -d '2026-10-05 18:15' +%s); FREEZE=$(date -u -d '2026-10-06 05:00' +%s); FLAG=/home/paperspace/logs/rehearsal_done.flag
PIDF=/home/paperspace/logs/chilli_phase.pid; : > $PIDF
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
blackout() { local now; now=$(date -u +%s); { [ $now -ge $PAUSE ] && [ ! -e $FLAG ]; } || [ $now -ge $FREEZE ]; }
( while true; do   # watchdog: no phase on the card during a blackout
    p=$(cat $PIDF 2>/dev/null); if [ -n "$p" ] && kill -0 $p 2>/dev/null && blackout; then
      echo "[$(date '+%m-%d %H:%M:%S')] watchdog: blackout, stopping phase pid $p" >> $L; kill $p; sleep 30; kill -0 $p 2>/dev/null && kill -9 $p; fi
    sleep 20; done ) &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
gpu_free() { while [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)" -gt 0 ]; do sleep 60; done; }
slot() {   # <seconds needed>: 0 = start now, 1 = no window left before the freeze
  local need=$1 now
  while true; do
    now=$(date -u +%s)
    [ $((now + need)) -le $FREEZE ] || return 1
    if [ ! -e $FLAG ]; then
      if [ $now -lt $PAUSE ] && [ $((now + need)) -le $PAUSE ]; then gpu_free; now=$(date -u +%s); [ $((now + need)) -le $PAUSE ] && return 0; fi
      sleep 60; continue   # does not fit before the rehearsal, or the rehearsal is on: wait for "rehearsal done"
    fi
    gpu_free; now=$(date -u +%s); [ $((now + need)) -le $FREEZE ] && return 0 || return 1
  done
}
run() {   # <need s> <done-test file> <label> <cmd...>: start when a window fits, retry after a watchdog stop
  local need=$1 done=$2 label=$3; shift 3
  while [ ! -e "$done" ]; do
    slot $need || { say "$label: no window before 05:00 UTC (needs ${need}s) - not started"; return 1; }
    say "$label: start (needs ~${need}s)"; local t0; t0=$(date +%s)
    "$@" & local p=$!; echo $p > $PIDF; wait $p; local rc=$?; : > $PIDF
    say "$label: rc=$rc in $(( $(date +%s) - t0 ))s$([ -e "$done" ] && echo ', done' || echo ', NOT done (stopped or failed)')"
    [ -e "$done" ] || { [ $rc -eq 143 ] || [ $rc -eq 137 ] || return 1; }   # failed for real -> give up on this segment
  done
}
h3dgs_segment() {   # image_farm_h3dgs.sh, phase by phase (same commands and flags; project h3dgs, 8 M)
  local S=$1 N; N=$(basename $1); local P=$S/h3dgs; local CH=$P/camera_calibration/chunks/lane T=$P/output/trained_chunks/lane TL=/home/paperspace/logs/if_h3dgs_${N}_train.log
  [ -e $CH/center.txt ] || $PYH $IF/image_farm_h3dgs_prep.py $S --proj h3dgs 2>&1 | tail -1 | tee -a $L
  [ -e $CH/center.txt ] || { say "$N: H3DGS prep FAILED"; return 1; }
  mkdir -p $T; cd $REPO || return 1; export CUDA_HOME=/home/paperspace/code/_cuda12 H3DGS_MAX_GAUSSIANS=8000000
  run 7200 $T/point_cloud/iteration_60000/point_cloud.ply "$N train" bash -c "exec $PYH -u train_single.py --port $((6100 + RANDOM % 900)) --save_iterations -1 -i ../../rectified/images --iterations 60000 --position_lr_max_steps 60000 --densify_until_iter 45000 --densify_grad_threshold 0.0075 --exposure_lr_init 0.0 --eval -s $CH --model_path $T --bounds_file $CH > $TL 2>&1" || return 1
  run 600 $T/hierarchy.hier "$N hierarchy" bash -c "exec submodules/gaussianhierarchy/build/GaussianHierarchyCreator $T/point_cloud/iteration_60000/point_cloud.ply $CH $T >> $TL 2>&1" || return 1
  run 6600 $T/hierarchy.hier_opt "$N post-opt" bash -c "exec $PYH -u train_post.py --port $((6100 + RANDOM % 900)) --iterations 30000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i ../../rectified/images --exposure_lr_init 0.0 --eval -s $CH --model_path $T --hierarchy $T/hierarchy.hier >> $TL 2>&1" || return 1
  local EV=/home/paperspace/logs/h3dgs_eval_chunk.py
  run 900 $P/output/eval_lane_train/scores.json "$N eval" bash -c "export PATH=/home/paperspace/miniconda3/envs/h3dgs/bin:/home/paperspace/code/_cuda12/bin:\$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $PYH $EV $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 --save 8 --out output/eval_lane > /home/paperspace/logs/if_h3dgs_${N}_eval.log 2>&1 && exec $PYH $EV $P --hier output/trained_chunks/lane/hierarchy.hier_opt --only_chunk lane --taus 0 --train_sample 40 --save 8 --out output/eval_lane_train > /home/paperspace/logs/if_h3dgs_${N}_train_eval.log 2>&1" || return 1
  say "$N: HELD-OUT $(grep -aE '^\[eval\] tau' /home/paperspace/logs/if_h3dgs_${N}_eval.log | grep -oE 'PSNR mean [0-9.]+ median [0-9.]+' | head -1) | TRAINING $(grep -aE '^\[eval\] tau' /home/paperspace/logs/if_h3dgs_${N}_train_eval.log | grep -oE 'PSNR mean [0-9.]+ median [0-9.]+' | head -1)"
}
step_check() {
  $PYH - $1 <<'PY'
import re, sys, numpy as np
sys.path.insert(0, '/home/paperspace/code/hierarchical-3d-gaussians/preprocess')
from read_write_model import read_images_binary, qvec2rotmat
ims = read_images_binary(sys.argv[1] + '/sparse/0/images.bin')
d = sorted(((int(re.sub(r'\D', '', im.name)), -qvec2rotmat(im.qvec).T @ im.tvec) for im in ims.values()), key=lambda t: t[0])
C = np.array([c for _, c in d]); st = np.linalg.norm(np.diff(C, axis=0), axis=1); r = st / np.median(st)
ok = r.max() <= 5 and (r < 0.1).mean() < 0.05
print(f'[step-check] {len(d)} cameras: max step {r.max():.1f}x median at frame {d[int(r.argmax())][0]}, {int((r < 0.1).sum())} steps < 0.1x -> {"OK" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
PY
}
s1b_prep() {   # v1's prep_segment for the tail: frozen image_farm_prep.py with the segment override (GPU SfM), then recipe steps 0-2
  local NAME=IMG_7990_s1b PRE=/home/paperspace/logs/prep_override_IMG_7990_s1b.run.py REC=/home/paperspace/logs/recipe02_IMG_7990_s1b.run.sh
  python3 - $IF/image_farm_prep.py $PRE <<'PY' || return 1
import sys
s = open(sys.argv[1]).read()
for a, b in (('segs = json.load(open(segf))["segments"]', 'segs = [[295, 393]]   # chilli_trainings_queue_v2.sh override'),
             ('SD = ROOT / f"{NAME}_s{si}"', 'SD = ROOT / "IMG_7990_s1b"'),
             ('if not run_sfm(CLIP, LOGS / f"prep_{NAME}_sfm.log", glomap=False): sys.exit(3)', 'pass   # the clip database was reclaimed'),
             ('json.dump({"clip": NAME, "segment": si, "clip_frames"', 'json.dump({"clip": NAME, "segment": "IMG_7990_s1b", "clip_frames"'),
             ('meta = {"clip": NAME, "video": MOV.name, "segment": si,', 'meta = {"clip": NAME, "video": MOV.name, "segment": "IMG_7990_s1b",')):
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(sys.argv[2], 'w').write(s)
PY
  run 1800 $G/$NAME/gate.json "$NAME SfM" bash -c "exec $PYH $PRE $G/IMG_7990 >> /home/paperspace/logs/prep_override_IMG_7990_s1b.log 2>&1" || return 1
  CUT=$(grep -n '^# ---- 3\. SAM3' $IF/image_farm_recipe.sh | cut -d: -f1)
  { head -n $((CUT - 1)) $IF/image_farm_recipe.sh; echo 'say "steps 0-2 done (chilli_trainings_queue_v2.sh cut)"; exit 0'; } > $REC
  [ -s $G/$NAME/transforms.json ] || bash $REC $G/$NAME 2>&1 | tail -1 | tee -a $L
  [ -s $G/$NAME/transforms.json ]
}
say "=== chilli trainings v2 (windows: pause 18:15 UTC until $FLAG, freeze 05:00 UTC)"
h3dgs_segment $G/IMG_7990_s1a
say "$(step_check $G/IMG_7990_s0)"; h3dgs_segment $G/IMG_7990_s0
if s1b_prep; then say "$(step_check $G/IMG_7990_s1b)"; step_check $G/IMG_7990_s1b > /dev/null && h3dgs_segment $G/IMG_7990_s1b || say "IMG_7990_s1b: not trained (step check or prep)"; fi
say "=== chilli trainings v2 finished (or stopped for the windows)"
