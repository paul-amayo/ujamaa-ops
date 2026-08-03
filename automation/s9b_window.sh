#!/bin/bash
# s8_b001: fruit-aware densification test — s7 recipe + fruit-densify.
# Gates via ALIGNED template (own-supervision ruler).
set -x
CB=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001
EMB=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth
cd /home/paperspace/code/nerf_new
echo "n" | MAX_JOBS=4 HIGH_EMBEDDER_CKPT=$EMB \
  LIDAR_DEPTH_NPZ=$CB/lidar_depth_morph.npz \
  pixi run ns-train high \
    --data $CB --output-dir $CB/splat_runs_S9B --experiment-name fruit_s9b_window \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight 10.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.model.fruit-densify True \
    --pipeline.model.stop-split-at 19000 \
    --pipeline.model.sky-loss-lambda 1.0 \
    --pipeline.model.report-masked-metrics True \
    --pipeline.datamanager.semantic-dir $CB/supervision/strict_tree_v2 \
    --max-num-iterations 20001 --steps-per-save 9998 \
    --vis tensorboard nerfstudio-data \
    --eval-mode interval --eval-interval 10 || { echo "S9B-FAIL"; exit 1; }
echo "S9B-TRAINED"
CFG=$(ls -t $CB/splat_runs_S9B/fruit_s9b_window/high/*/config.yml | head -1)
cd /home/paperspace/code/aru_sil_core
HIGH_EMBEDDER_CKPT=$EMB pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml \
  python /home/paperspace/logs/aligned_gates.py "$CFG" 2>&1 \
  | grep -aE "ALIGNED|fruit pixels|tree  pixels|choosing" | sed "s/^/GATES s9b /"
echo "S9B-ALL-DONE"
