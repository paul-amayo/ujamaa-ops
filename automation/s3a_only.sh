#!/bin/bash
set -x
CB=/home/paperspace/data/citrus_all/04_13D_Jackal/blocks_ns/lio_row6F/block_001
cd /home/paperspace/code/nerf_new
echo "n" | \
  MAX_JOBS=4 \
  HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth \
  LIDAR_DEPTH_NPZ=$CB/lidar_depth_morph.npz \
  pixi run ns-train high \
    --data $CB --output-dir $CB/splat_runs_S3 --experiment-name fruit_s3a \
    --pipeline.model.rasterize-mode antialiased \
    --pipeline.model.output-depth-during-training True \
    --pipeline.model.high-loss-weight 1.0 \
    --pipeline.model.high-loss-fruit-weight 10.0 \
    --pipeline.model.fruit-protect True \
    --pipeline.datamanager.semantic-dir $CB/semantic_v2_C \
    --max-num-iterations 20001 --steps-per-save 9998 \
    --vis tensorboard nerfstudio-data || exit 1
echo "S3A-TRAINED"
S3CFG=$(ls -t $CB/splat_runs_S3/fruit_s3a/high/*/config.yml | head -1)
sed "s|CFG = Path('.*')|CFG = Path('$S3CFG')|" /home/paperspace/logs/s2_gates_template.py > /home/paperspace/logs/s3_gates.py
cd /home/paperspace/code/aru_sil_core
HIGH_EMBEDDER_CKPT=/home/paperspace/data/high/nerf/04_13D_v2F5/ckpts/model_best.pth \
pixi run --manifest-path /home/paperspace/code/nerf_new/pixi.toml \
  python /home/paperspace/logs/s3_gates.py 2>&1 | grep -aE "RENDERED|fruit pixels|tree  pixels|CROSS-LEVEL|choosing"
echo "S3A-ALL-DONE"
