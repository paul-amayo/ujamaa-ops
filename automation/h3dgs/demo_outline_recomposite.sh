#!/bin/bash
# demo_outline_recomposite.sh — re-composite the three 720p reels with the row-outline smoothing scaled for 720p
# (--outline-close 18 --outline-simplify 5; the 09-27 values 9 / 2.5 were tuned at 640x360). CPU only: identity maps,
# fruit maps, backdrops and scripts are reused (--skip-maps). Previous cuts kept as demo_outline9.mp4 (+ _crf24) for a diff.
set -uo pipefail
L=/home/paperspace/logs/demo_chunks_run.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
A=/home/paperspace/code/automation/h3dgs; NS=/home/paperspace/code/nerf_new; D=/home/paperspace/logs/demo_chunks
AREA="--min-area 3200 --tree-min-area 800 --fruit-min-area 120 --outline-close 18 --outline-simplify 5"
recomp(){ local tag=$1 sv=$2 model=$3 proj=$4 fa=$5; local O=$D/$tag; local EMB=$(ls -t /home/paperspace/data/citrus_all/$sv/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1); local t0=$(date +%s)
  [ -e $O/demo_outline9.mp4 ] || mv $O/demo.mp4 $O/demo_outline9.mp4; [ -e $O/demo_crf24.mp4 ] && [ ! -e $O/demo_outline9_crf24.mp4 ] && mv $O/demo_crf24.mp4 $O/demo_outline9_crf24.mp4
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB timeout 1500 pixi run python $A/sidecar_demo_overlay.py --survey $sv --path $O/demo_path.json --models $model --sidecar-root experimental/h3dgs_sidecar_chunks --proj $proj --backdrop-dir $O/backdrop --out $O --fps 8 --max-dist 60 $AREA --skip-maps --script $O/demo_script.json $fa 2>&1 | grep -aE '^\[demo\] wrote|Error|Traceback' | tee -a $L)
  /home/paperspace/envs/match/bin/python $A/demo_contact_sheet.py $O 2>&1 | grep -aE '^\[sheet\]' | tee -a $L
  ffmpeg -y -loglevel error -i $O/demo.mp4 -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart $O/demo_crf24.mp4
  say "$tag outline re-composite done in $(( ($(date +%s)-t0)/60 )) min -> demo.mp4 ($(du -h $O/demo.mp4 | cut -f1)), demo_crf24.mp4 ($(du -h $O/demo_crf24.mp4 | cut -f1)); previous kept as demo_outline9.mp4"
}
say "=== outline re-composite (720p, close 18 / simplify 5) start — CPU only"
recomp 01_3_1_720 01_13B_Jackal chunk_3_1 experimental/h3dgs ""
recomp 05_0_0_720 05_13D_Jackal chunk_0_0_expo experimental/h3dgs_expo ""
recomp 05_1_0_720 05_13D_Jackal chunk_1_0_expo experimental/h3dgs_expo "--fruit-models chunk_1_0_expo --fruit-seed-tag fruitdensify_fruit_densify_wm_t0s_r2 --fruit-verdict-tag fruitdensify_fruit_densify_wm_t0s --fruit-max-dist 30"
say "=== outline re-composite done"
