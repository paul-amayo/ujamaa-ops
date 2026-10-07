#!/bin/bash
# April lane 2 over the FULL December lane (Paul 2026-10-06: "train but on the right lane"; the top-down showed the first April
# stretch kf 193-348 stops at ledger y -8 while December's lane 2 runs to +3; April keyframes kf 190-402 follow it within 0.8 m).
# extract kf 190-402 -> LO (dump + KISS-ICP, December settings) -> lane window (December rule, no start pad) -> frames outside the
# window moved to images_outside/ (December's SfM saw only its lane window) -> LO base + LiDAR init -> SfM -> place (warp v2) ->
# heading check. Stops before training (launched by hand after the check).
set -u
LD=/home/paperspace/data/klapmuts/apr_2026_zed/experimental/lane2_apr_full; A=/home/paperspace/code/automation/tenrows; L=/home/paperspace/logs/apr_lane2_full_chain.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
P310="env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib /home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10"
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python
say "=== April lane 2 full: kf 190-402 -> $LD"
[ -e $LD/stamps.json ] || $P310 $A/apr_lane_extract.py 190 402 $LD 2>&1 | grep -a '^\[apr-lane\]' | tee -a $L
[ -e $LD/laser_dump/n_scans.npy ] || $P310 $A/apr_lane_lo.py dump $LD 2>&1 | grep -a '^\[apr-lo\]' | tee -a $L
[ -e $LD/laser_dump/lo_poses.npz ] || /home/paperspace/envs/lo/bin/python $A/tenrows_kiss_icp.py --dump $LD/laser_dump 2>&1 | tail -1 | tee -a $L
PAD_START_S=0 $PYH $A/apr_lane_lo.py window $LD 2>&1 | tee -a $L
$PYH - $LD <<'PY' 2>&1 | tee -a $L
import json, sys, os
from pathlib import Path
LD = Path(sys.argv[1]); w = json.load(open(LD / "lane_window.json")); st = json.load(open(LD / "stamps.json")); (LD / "images_outside").mkdir(exist_ok=True); n = 0
for k, v in st.items():
    if not (w["t0_ms"] <= float(v) <= w["t1_ms"]) and (LD / "images" / k).exists(): os.rename(LD / "images" / k, LD / "images_outside" / k); n += 1
print(f"[apr-full] moved {n} frames outside the lane window to images_outside/; {len(os.listdir(LD / 'images'))} lane frames for the SfM", flush=True)
PY
$PYH $A/apr_lane_lo.py base $LD 2>&1 | tee -a $L
bash /home/paperspace/logs/apr_lane_run.sh $LD sfm place 2>&1 | grep -a 'SfM in\|FAILED\|lane-warp\|lane-prep\|REFUSED\|NO BASE' | cut -c1-500 | tee -a $L
$PYH $A/lane_heading_check.py $LD 2>&1 | tee -a $L
say "=== chain done (training not started)"
