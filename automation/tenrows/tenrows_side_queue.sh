#!/bin/bash
# Side tests re-queued AFTER the odometry-pose fleet: (1) equal-epoch full-stream arm (40k iters),
# (2) refined-pose full-stream arm (COLMAP on all 246 frames of block_013's span, Sim3 into our world).
NS=/home/paperspace/code/nerf_new; T=/home/paperspace/data/klapmuts/dec_2025_ten_rows/prod/tassili; B=$T/blocks_ns/lio_row100
LOG=/home/paperspace/logs/tenrows_side_queue.log; : > $LOG; say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }
until grep -q "STAGE1-ALL DONE" /home/paperspace/logs/tenrows_stage1_driver.log 2>/dev/null; do sleep 120; done
say "odometry fleet done — side queue starts"
# (1) equal epochs
BD=$B/block_013_full; rm -rf $BD/splat_runs_STAGE1_40k; t1=$(date +%s)
( cd $NS && echo "n" | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1_40k" --experiment-name stage1_40k --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 16000 --pipeline.model.sky-loss-lambda 0.0 --pipeline.model.report-masked-metrics False --max-num-iterations 40001 --steps-per-save 10000 --vis tensorboard nerfstudio-data --eval-mode filename ) > /home/paperspace/logs/tenrows_stage1_block_013_full_40k.log 2>&1
say "[40k] rc=$? wall=$(( $(date +%s)-t1 ))s"; CFG=$(ls -t $BD/splat_runs_STAGE1_40k/stage1_40k/high/*/config.yml | head -1)
( cd $NS && pixi run ns-eval --load-config "$CFG" --output-path $BD/eval_40k.json ) > /home/paperspace/logs/tenrows_nseval_full40k.log 2>&1
python3 -c "
import json; m = json.load(open('$BD/eval_40k.json'))['results']; print('[40k] full frames @40k iters (equal epochs): ns-eval psnr %.2f ssim %.4f lpips %.4f  (refs @15k: kf 14.43/0.399/0.659, full 15.24/0.443/0.716)' % (m['psnr'], m['ssim'], m['lpips']))" | tee -a $LOG
# (2) refined full arm
BLK=block_013_full; W=$T/colmap_$BLK; CC=/home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap
GLD=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib
rm -rf $W; mkdir -p $W/images $W/sparse
python3 -c "
import json, shutil
from pathlib import Path
t = json.load(open('$B/$BLK/transforms.json'))
for f in t['frames']: shutil.copy(f['file_path'], '$W/images/' + Path(f['file_path']).name)
print('[full-ref] copied', len(t['frames']), 'frames')" | tee -a $LOG
export LD_LIBRARY_PATH=$GLD; t0=$(date +%s); SL=/home/paperspace/logs/tenrows_colmap_${BLK}_sfm.log
$CC feature_extractor --database_path $W/database.db --image_path $W/images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --ImageReader.camera_params "529.4046769303875,529.6521348953188,647.1984132364281,354.6428014457265,0,0,0,0" --FeatureExtraction.use_gpu 1 > $SL 2>&1
$CC exhaustive_matcher --database_path $W/database.db --FeatureMatching.use_gpu 1 >> $SL 2>&1
/home/paperspace/code/glomap/build_gpu/glomap/glomap mapper --database_path $W/database.db --image_path $W/images --output_path $W/sparse >> $SL 2>&1   # GLOMAP: the incremental mapper failed on 10/23 row blocks
unset LD_LIBRARY_PATH; say "[full-ref] SfM $(( $(date +%s)-t0 ))s"; python3 /home/paperspace/code/aru_sil_core/src/scripts/image_pipeline/colmap_to_nerfstudio.py $W > /dev/null 2>&1
python3 /home/paperspace/logs/tenrows_refine_block.py $BLK | tee -a $LOG
BD=$B/${BLK}_ref; rm -rf $BD/splat_runs_STAGE1; t1=$(date +%s)
( cd $NS && echo "n" | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True pixi run ns-train high --data "$BD" --output-dir "$BD/splat_runs_STAGE1" --experiment-name stage1 --pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0 --pipeline.datamanager.semantic-dir /home/paperspace/logs/empty_semantic --pipeline.model.rasterize-mode antialiased --pipeline.model.stop-split-at 6000 --pipeline.model.sky-loss-lambda 0.0 --pipeline.model.report-masked-metrics False --max-num-iterations 15001 --steps-per-save 5000 --vis tensorboard nerfstudio-data --eval-mode filename ) > /home/paperspace/logs/tenrows_stage1_${BLK}_ref.log 2>&1
say "[full-ref] train rc=$? wall=$(( $(date +%s)-t1 ))s"; /home/paperspace/logs/tenrows_ns_eval.sh $BD | tee -a $LOG
say "SIDE-QUEUE DONE (refs: refined kf arm 17.52 | odometry kf 14.43 / full 15.24)"
