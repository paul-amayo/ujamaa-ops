#!/bin/bash
# PROD IMAGE-ONLY RECIPE — bare frames -> splat + hierarchy + Bateleur inputs.
# Codified from the sukuma validation run (2026-09-02/03). The three-axis
# ingestion correction below is worth +6 dB (14.15 -> 20.11 train, measured):
#   * frames at WIDTH 1080 for portrait video (1080x1920) — NOT long-edge 1080
#   * camera model OPENCV (not SIMPLE_RADIAL)
#   * exhaustive matching (binds out-pass to back-pass; sequential-only leaves
#     the two passes of a down-and-back walk unregistered to each other)
# Init: DA3 window-fused all-views cloud (pose-conditioned, align_to_input_
# ext_scale) — beats 26-frame da3init and COLMAP-sparse on train AND eval
# (3-arm ablation 2026-09-02). DA3 scene mode also provides the survey's
# METRIC scale (pose-free pass: metres_per_unit).
#
#   usage: prod_image_recipe.sh <survey_dir> [--prompt planter] [--skip-train]
#   expects <survey_dir>/images_orig/ (or images/) holding the raw frames.
#
# STATUS: validated once end-to-end on gendia/sukuma. The twice-from-scratch
# repeatability demonstration (the acceptance bar prod_block_recipe met) is
# PENDING — run it before calling this recipe prod.
#
# Portability facts this script bakes in (each cost a debug cycle):
#   colmap needs ollama's CUDA-13 runtime on LD_LIBRARY_PATH on this box;
#   read_write_model.py staged at /home/paperspace/code/colmap/scripts/python;
#   ingest binding is cp310 -> run via nerf_new's python3.10;
#   SAM3 frames live at prod/scratch_sam3 as kf_%06d.png;
#   monolithics live at prod/monos; markers is rig-free under --no-lidar
#   with explicit intrinsics; kf_pose_records falls back to
#   lio_image_poses.json for image-only surveys.
set -uo pipefail
SURVEY=$(realpath "${1:?usage: prod_image_recipe.sh <survey_dir> [--prompt X] [--skip-train]}")
shift
PROMPT="planter"; SKIP_TRAIN=0
while [ $# -gt 0 ]; do case $1 in
  --prompt) PROMPT=$2; shift 2;;
  --skip-train) SKIP_TRAIN=1; shift;;
  *) echo "unknown arg $1"; exit 2;;
esac; done

ARU=/home/paperspace/code/aru_sil_core
IPL=$ARU/src/scripts/image_pipeline
SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
NS=/home/paperspace/code/nerf_new
PY310=$NS/.pixi/envs/default/bin/python3.10
CFG=lio_arc_size15.0_ov0.10_kf20cm_dedup
BD=$SURVEY/blocks_ns/$CFG/block_000
LOGS=/home/paperspace/logs
NAME=$(basename "$SURVEY")
say() { echo "[$(date '+%m-%d %H:%M:%S')] recipe($NAME) $*"; }
gate() { say "GATE FAILED: $*"; exit 3; }

# ---- 0. working frames: width 1080, sequential image_<i>.png --------------
if [ ! -d "$SURVEY/images" ] || [ -z "$(ls "$SURVEY/images" 2>/dev/null | head -1)" ]; then
  SRC=$SURVEY/images_orig
  [ -d "$SRC" ] || gate "no images_orig/ or images/"
  mkdir -p "$SURVEY/images"
  python3 - "$SRC" "$SURVEY/images" << 'PY'
import sys
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
frames = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
step = max(1, len(frames) // 240)
for i, p in enumerate(frames[::step][:300]):
    im = Image.open(p)
    w, h = im.size
    if w > h:            # landscape: height 1080
        nh = 1080; nw = int(round(w * 1080 / h / 2) * 2)
    else:                # portrait: WIDTH 1080 (the validated rule)
        nw = 1080; nh = int(round(h * 1080 / w / 2) * 2)
    im.resize((nw, nh), Image.LANCZOS).save(dst / f"image_{i}.png")
print(f"working set: {i+1} frames")
PY
fi
N=$(ls "$SURVEY/images" | wc -l)
say "working set: $N frames"

# ---- 1. SfM: OPENCV + exhaustive ------------------------------------------
if [ ! -f "$SURVEY/sparse/0/points3D.bin" ]; then
  export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so.0
  rm -f "$SURVEY/database.db"; mkdir -p "$SURVEY/sparse"
  colmap feature_extractor --database_path "$SURVEY/database.db" \
    --image_path "$SURVEY/images" --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 1 \
    > "$LOGS/recipe_${NAME}_sfm.log" 2>&1 || gate "feature_extractor"
  colmap exhaustive_matcher --database_path "$SURVEY/database.db" \
    --FeatureMatching.use_gpu 1 >> "$LOGS/recipe_${NAME}_sfm.log" 2>&1 \
    || gate "exhaustive_matcher"
  glomap mapper --database_path "$SURVEY/database.db" \
    --image_path "$SURVEY/images" --output_path "$SURVEY/sparse" \
    >> "$LOGS/recipe_${NAME}_sfm.log" 2>&1 || gate "glomap"
  unset LD_PRELOAD
fi
REG=$(grep -oE "Registered images: [0-9]+" "$LOGS/recipe_${NAME}_sfm.log" | tail -1 | tr -dc 0-9)
[ "${REG:-0}" -ge $((N * 95 / 100)) ] || gate "SfM registered $REG/$N (<95%)"
say "SfM ok: $REG/$N registered"

# ---- 2. nerfstudio conversion + ingest ------------------------------------
python3 "$IPL/colmap_to_nerfstudio.py" "$SURVEY" \
  > "$LOGS/recipe_${NAME}_convert.log" 2>&1 || gate "colmap_to_nerfstudio"
PYTHONPATH=$ARU/src/interfaces/build/temp.linux-x86_64-3.10/lib \
  "$PY310" "$IPL/image_to_monolithic.py" "$SURVEY" --fps 2 \
  > "$LOGS/recipe_${NAME}_ingest.log" 2>&1 || gate "image_to_monolithic"
mkdir -p "$SURVEY/prod/monos" "$SURVEY/prod/scratch_sam3"
for f in transform_lio.monolithic image_left.monolithic image_left.monolithic.index; do
  [ -f "$SURVEY/$f" ] && mv "$SURVEY/$f" "$SURVEY/prod/monos/"
done
python3 - "$SURVEY" << 'PY'
import shutil, sys
from pathlib import Path
S = Path(sys.argv[1])
for p in sorted((S/"images").glob("image_*.png")):
    i = int(p.stem.split("_")[1])
    dst = S/"prod/scratch_sam3"/f"kf_{i:06d}.png"
    if not dst.exists():
        shutil.copy2(p, dst)
print("scratch_sam3 staged")
PY

# ---- 3. SAM3 + DA3 windows (depths + fused init) --------------------------
"$SAM3_PY" "$ARU/src/scripts/build_tree_instances.py" \
  --data-dir "$SURVEY" --prompt "$PROMPT" \
  > "$LOGS/recipe_${NAME}_sam3.log" 2>&1 || gate "SAM3"
"$SAM3_PY" "$IPL/da3_windows_fuse.py" --survey "$SURVEY" --write-depth \
  > "$LOGS/recipe_${NAME}_da3.log" 2>&1 || gate "DA3 windows"
[ -f "$SURVEY/init_da3_fused.ply" ] || gate "no fused init ply"

# ---- 4. clustering + hierarchy --------------------------------------------
( cd "$NS" && pixi run python "$IPL/cluster_tree_instances_depth.py" \
  --data-dir "$SURVEY" --max-det-dist-m 5 --eps-m 0.2 --min-samples 3 ) \
  > "$LOGS/recipe_${NAME}_cluster.log" 2>&1 || gate "clustering"
python3 "$IPL/build_hierarchy_from_clusters.py" "$SURVEY" \
  --row-eps 0.6 --min-members 10 --max-bbox-diag 1.0 \
  > "$LOGS/recipe_${NAME}_hier.log" 2>&1 || gate "hierarchy"
grep -h "hierarchy" "$LOGS/recipe_${NAME}_hier.log" | tail -1

# ---- 5+ markers/semantic/embedder/train/export: see sukuma_complete_A/B ---
# (folded in after the repeatability run locks argument values)
say "structure pipeline complete — train stage: $([ $SKIP_TRAIN = 1 ] && echo skipped || echo 'run sukuma_complete_A/B pattern')"
say "DONE"
