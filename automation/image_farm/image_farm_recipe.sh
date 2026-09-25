#!/bin/bash
# IMAGE_FARM RECIPE — prod_image_recipe.sh (ratified gendia 09-03/04 version) with two changes for the vegetable-bed
# phone clips (2026-09-24):
#   * the hierarchy gate is a WARNING, not a stop: SAM3 'tree' finds almost nothing on bean/cabbage rows (IMG_7999:
#     81 masks in 347 frames -> 5 clusters -> "only 2 objects with >= 10 members"), and the structure stages still run;
#   * IF_MODE=rgb (default) trains the splat WITHOUT the HiGH feature chain (no markers / semantic PNGs / embedder):
#     enable-high-features False, empty semantic dir — the osuga_full_gpu.sh stage-1 pattern. IF_MODE=high = original.
# Everything else (working set, OPENCV+exhaustive SfM if sparse/0 is absent, COLMAP-sparse init, SAM3, DA3, masks,
# clustering, 5000-it train flags, export, manifests, PSNR readout) is unchanged from prod_image_recipe.sh.
# usage: image_farm_recipe.sh <survey_dir> [--prompt tree] [--eps 0.3] [--det-dist 30] [--sky 0] [--iters 5000] [--structure-only]
set -uo pipefail
SURVEY=$(realpath "${1:?usage: image_farm_recipe.sh <survey_dir> [--prompt X ...]}")
shift
PROMPT="tree"; EPS=0.3; DETDIST=30; SKY=0; ITERS=${IF_ITERS:-5000}; STRUCTURE_ONLY=0; MODE=${IF_MODE:-rgb}
SCHED=${IF_SCHED:-3000}; STOP_SPLIT=${IF_STOP_SPLIT:-15000}   # resolution_schedule / stop_split_at (nerfstudio defaults)
EVAL_INTERVAL=${IF_EVAL_INTERVAL:-10}   # held-out split: every N-th frame (HiGH's default train_split_fraction 0.99 leaves 0-1 eval images)
while [ $# -gt 0 ]; do case $1 in
  --prompt) PROMPT=$2; shift 2;;
  --eps) EPS=$2; shift 2;;
  --det-dist) DETDIST=$2; shift 2;;
  --sky) SKY=$2; shift 2;;
  --iters) ITERS=$2; shift 2;;
  --structure-only) STRUCTURE_ONLY=1; shift;;
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
  python3 - "$SRC" "$SURVEY/images" << 'PY' || gate "working-set build"
import sys
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
frames = sorted(list(src.glob("*.png")) + list(src.glob("*.jpg")))
step = max(1, len(frames) // 240)
picked = frames[::step][:300]
for i, p in enumerate(picked):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if w > h:            # landscape: height 1080
        nh = 1080; nw = int(round(w * 1080 / h / 2) * 2)
    else:                # portrait: WIDTH 1080 (the validated rule)
        nw = 1080; nh = int(round(h * 1080 / w / 2) * 2)
    im.resize((nw, nh), Image.LANCZOS).save(dst / f"image_{i}.png")
print(f"working set: {len(picked)} frames from {len(frames)} originals")
PY
fi
N=$(ls "$SURVEY/images" | wc -l)
FIRST=$(ls "$SURVEY/images" | sort -t_ -k2 -n | head -1)   # the track gate may have dropped image_0.png
read -r IW IH <<< "$(python3 -c "
from PIL import Image
im = Image.open('$SURVEY/images/$FIRST'); print(im.size[0], im.size[1])")"
[ -n "$IW" ] && [ -n "$IH" ] || gate "could not read frame dimensions from images/$FIRST"
say "working set: $N frames @ ${IW}x${IH} (mode=$MODE)"

# ---- 1. SfM: OPENCV + exhaustive (skipped when image_farm_prep.py already wrote sparse/0) ----
if [ ! -f "$SURVEY/sparse/0/points3D.bin" ]; then
  say "SfM (feature/exhaustive/glomap)"
  ( export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so.0
    rm -f "$SURVEY/database.db"; mkdir -p "$SURVEY/sparse"
    colmap feature_extractor --database_path "$SURVEY/database.db" \
      --image_path "$SURVEY/images" --ImageReader.single_camera 1 \
      --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 1 \
      > "$LOGS/recipe_${NAME}_sfm.log" 2>&1 &&
    colmap exhaustive_matcher --database_path "$SURVEY/database.db" \
      --FeatureMatching.use_gpu 1 >> "$LOGS/recipe_${NAME}_sfm.log" 2>&1 &&
    glomap mapper --database_path "$SURVEY/database.db" \
      --image_path "$SURVEY/images" --output_path "$SURVEY/sparse" \
      >> "$LOGS/recipe_${NAME}_sfm.log" 2>&1
  ) || gate "SfM"
fi
REG=$(python3 - "$SURVEY" << 'PY'
import sys, struct
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python")
from read_write_model import read_images_binary
print(len(read_images_binary(Path(sys.argv[1])/"sparse/0/images.bin")))
PY
)
[ "${REG:-0}" -ge $((N * 95 / 100)) ] || gate "SfM registered $REG/$N (<95%)"
say "SfM ok: $REG/$N registered"

# ---- 2. nerfstudio conversion (writes transforms.json + SPARSE init.ply) --
python3 "$IPL/colmap_to_nerfstudio.py" "$SURVEY" \
  > "$LOGS/recipe_${NAME}_convert.log" 2>&1 || gate "colmap_to_nerfstudio"
[ -f "$SURVEY/init.ply" ] || gate "no init.ply from conversion"
PYTHONPATH=$ARU/src/interfaces/build/temp.linux-x86_64-3.10/lib \
  "$PY310" "$IPL/image_to_monolithic.py" "$SURVEY" --fps 2 \
  > "$LOGS/recipe_${NAME}_ingest.log" 2>&1 || gate "image_to_monolithic"
mkdir -p "$SURVEY/prod/monos" "$SURVEY/prod/scratch_sam3"
for f in transform_lio.monolithic image_left.monolithic image_left.monolithic.index; do
  [ -f "$SURVEY/$f" ] && mv "$SURVEY/$f" "$SURVEY/prod/monos/"
done
# RGB mode never reads the image monolithic (only the HiGH marker/semantic chain does): drop the 0.8 GB it just wrote
[ "$MODE" = "rgb" ] && rm -f "$SURVEY/prod/monos/image_left.monolithic" "$SURVEY/prod/monos/image_left.monolithic.index"
python3 - "$SURVEY" << 'PY' || gate "scratch_sam3 staging"
import os, shutil, sys
from pathlib import Path
S = Path(sys.argv[1])
for p in sorted((S/"images").glob("image_*.png")):
    i = int(p.stem.split("_")[1])
    dst = S/"prod/scratch_sam3"/f"kf_{i:06d}.png"
    if not dst.exists():
        try: os.link(p, dst)          # same bytes, no second copy (24 GB of duplicates across the fleet on 09-25)
        except OSError: shutil.copy2(p, dst)
print("scratch_sam3 staged")
PY

# ---- 3. SAM3 + DA3 (depths for the mask lift; metric scale) ---------------
if [ ! -f "$SURVEY/prod/bateleur/sam3_v2/clip_000/frame_entries.json" ]; then
  say "SAM3 prompt='$PROMPT'"
  "$SAM3_PY" "$ARU/src/scripts/build_tree_instances.py" \
    --data-dir "$SURVEY" --prompt "$PROMPT" \
    > "$LOGS/recipe_${NAME}_sam3.log" 2>&1 || gate "SAM3"
else
  say "SAM3 outputs exist — skip"
fi
if [ ! -d "$SURVEY/depth_png" ] || [ -z "$(ls "$SURVEY/depth_png" 2>/dev/null | head -1)" ]; then
  say "DA3 windows (depth for clustering; fused ply kept for the record only)"
  ( cd /home/paperspace/code/sam3 && "$SAM3_PY" "$IPL/da3_windows_fuse.py" \
      --survey "$SURVEY" --write-depth --depth-size "${IW}x${IH}" ) \
    > "$LOGS/recipe_${NAME}_da3.log" 2>&1 || {
      # DA3 only feeds the clustering/marker chain; RGB mode does not need it (IMG_7970_s0: evo Umeyama
      # "Degenerate covariance rank" when a window of frames barely translates)
      say "DA3 WINDOWS FAILED: $(tail -1 "$LOGS/recipe_${NAME}_da3.log" | cut -c1-120)"
      [ "$MODE" = "high" ] && gate "DA3 windows (required for mode=high)"; }
else
  say "depth_png exists — skip DA3"
fi
if [ "$SKY" = "1" ]; then
  say "sky+fg masks"
  "$SAM3_PY" "$ARU/src/scripts/build_sky_masks.py" --data-dir "$SURVEY" \
    > "$LOGS/recipe_${NAME}_sky.log" 2>&1 || gate "sky masks"
  "$SAM3_PY" "$ARU/src/scripts/build_fg_masks.py" --data-dir "$SURVEY" \
    >> "$LOGS/recipe_${NAME}_sky.log" 2>&1 || gate "fg masks"
fi

# ---- 4. clustering + hierarchy (hierarchy is advisory on these crops) -----
( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python \
  "$IPL/cluster_tree_instances_depth.py" \
  --data-dir "$SURVEY" --max-det-dist-m "$DETDIST" --eps-m "$EPS" --min-samples 3 ) \
  > "$LOGS/recipe_${NAME}_cluster.log" 2>&1 && CLUSTER_OK=1 || CLUSTER_OK=0
grep -E "total_detections|global IDs" "$LOGS/recipe_${NAME}_cluster.log" | tail -2
if [ "$CLUSTER_OK" != "1" ]; then
  say "CLUSTERING WEAK (prompt='$PROMPT'): $(tail -1 "$LOGS/recipe_${NAME}_cluster.log")"
  [ "$MODE" = "high" ] && gate "clustering (required for mode=high)"
fi
if [ "$CLUSTER_OK" = "1" ] && python3 "$IPL/build_hierarchy_from_clusters.py" "$SURVEY" \
  --row-eps 0.6 --min-members 10 --max-bbox-diag 1.0 \
  > "$LOGS/recipe_${NAME}_hier.log" 2>&1; then
  tail -1 "$LOGS/recipe_${NAME}_hier.log"
else
  say "HIERARCHY WEAK (prompt='$PROMPT'): $(tail -1 "$LOGS/recipe_${NAME}_hier.log")"
  [ "$MODE" = "high" ] && gate "hierarchy (required for mode=high)"
fi
[ "$STRUCTURE_ONLY" = "1" ] && { say "STRUCTURE DONE (train skipped)"; exit 0; }

# ---- 6. block layout (COLMAP SPARSE init) ---------------------------------
mkdir -p "$BD"
python3 - "$SURVEY" "$BD" << 'PY' || gate "block layout"
import json, sys
from pathlib import Path
S, BD = Path(sys.argv[1]), Path(sys.argv[2])
tj = json.loads((S/"transforms.json").read_text())
for fr in tj["frames"]:
    p = Path(fr["file_path"])
    fr["file_path"] = str(p if p.is_absolute() else (S/p).resolve())
tj["ply_file_path"] = str(S/"init.ply")   # COLMAP SPARSE init — the measured default
(BD/"transforms.json").write_text(json.dumps(tj, indent=2))
print("block layout ok:", len(tj["frames"]), "frames, init =", tj["ply_file_path"])
PY

EMB=""; SEMDIR=/home/paperspace/logs/empty_semantic; FEATFLAGS="--pipeline.model.enable-high-features False --pipeline.model.high-loss-weight 0.0"
if [ "$MODE" = "high" ]; then
  # ---- 5. markers_v2 (rig-free; intrinsics = working space) ---------------
  read -r FX FY CX CY <<< "$(python3 -c "
import json; t=json.load(open('$SURVEY/transforms.json'))
print(t['fl_x'], t['fl_y'], t['cx'], t['cy'])")"
  ( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python \
    "$ARU/src/scripts/markers_v2_from_sam3.py" \
    --data-dir "$SURVEY" --no-lidar --depth-dir "$SURVEY/depth_png" \
    --use-sam3-frames --depth-back 1.5 --min-observations 2 \
    --markers-name v2_B --width "$IW" --height "$IH" \
    --fx "$FX" --fy "$FY" --cx "$CX" --cy "$CY" ) \
    > "$LOGS/recipe_${NAME}_markers.log" 2>&1 || gate "markers_v2"
  SEM_MONO=$(ls "$SURVEY/prod/monos/filtered_semantic_v2_B.monolithic" \
                "$SURVEY/filtered_semantic_v2_B.monolithic" 2>/dev/null | head -1)
  MARK_MONO=$(ls "$SURVEY/prod/bateleur/scene_graph/markers_v2_B.monolithic" \
                 "$SURVEY/scene_graph/markers_v2_B.monolithic" 2>/dev/null | head -1)
  [ -n "$SEM_MONO" ] && [ -n "$MARK_MONO" ] || gate "markers outputs not found"
  # ---- 6b. semantic PNGs --------------------------------------------------
  ( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python \
    "$ARU/src/scripts/save_filtered_semantic_pngs.py" \
    --block-dir "$BD" --semantic-monolithic "$SEM_MONO" \
    --marker-monolithic "$MARK_MONO" --n-monolithic-frames "$N" ) \
    > "$LOGS/recipe_${NAME}_sempng.log" 2>&1 || gate "semantic PNGs"
  NONEMPTY=$(grep -oE "non-empty=[0-9]+" "$LOGS/recipe_${NAME}_sempng.log" | tail -1 | tr -dc 0-9)
  [ "${NONEMPTY:-0}" -gt 0 ] || gate "semantic PNGs all empty (index misalignment?)"
  say "semantic PNGs non-empty=$NONEMPTY/$N"
  # ---- 7. embedder --------------------------------------------------------
  mkdir -p "$SURVEY/hyper"
  LIO_JSON=$(ls "$SURVEY/lio_image_poses_kf20cm.json" "$SURVEY/lio_image_poses.json" 2>/dev/null | head -1)
  ( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python \
    "$ARU/src/interfaces/rerun/HiGH/train_hyperembedder.py" \
    --semantic-monolithic "$SEM_MONO" --marker-monolithic "$MARK_MONO" \
    --lio-poses-json "$LIO_JSON" --output-dir "$SURVEY/hyper" \
    --experiment-name "${NAME}_v1c" --epochs 100 --batch-size 50 --run-train ) \
    > "$LOGS/recipe_${NAME}_embedder.log" 2>&1 || gate "embedder"
  EMB=$SURVEY/hyper/${NAME}_v1c/ckpts/model_best.pth
  [ -f "$EMB" ] || gate "no embedder ckpt"
  SEMDIR=$BD/semantic_v2_B; FEATFLAGS=""
fi

# ---- 8. train (+features when mode=high) + export + manifests -------------
SKYFLAGS=""
[ "$SKY" = "1" ] && SKYFLAGS="--pipeline.model.sky-loss-lambda 0.05"
rm -rf "$BD/splat_runs_high" "$NS/outputs/block_000" /home/paperspace/code/outputs/block_000
say "train ($ITERS iters, mode=$MODE, sky=$SKY)"
( cd "$NS" && echo n | MAX_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${EMB:+HIGH_EMBEDDER_CKPT=$EMB} env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run ns-train high \
  --max-num-iterations "$ITERS" --vis tensorboard \
  --output-dir "$BD/splat_runs_high" --experiment-name "${NAME}_block_000" \
  --pipeline.datamanager.semantic-dir "$SEMDIR" \
  $FEATFLAGS \
  --pipeline.model.resolution-schedule "$SCHED" --pipeline.model.stop-split-at "$STOP_SPLIT" \
  --pipeline.model.cull-alpha-thresh 0.01 \
  --pipeline.model.cull-scale-thresh 0.3 \
  --pipeline.model.densify-grad-thresh 0.0006 \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.background-color black \
  --pipeline.model.report-masked-metrics True \
  $SKYFLAGS \
  nerfstudio-data --data "$BD" --eval-mode interval --eval-interval "$EVAL_INTERVAL" ) \
  > "$LOGS/recipe_${NAME}_train.log" 2>&1 || gate "train"
if [ "$MODE" = "high" ]; then
  grep -q "HyperEmbedder ckpt: $EMB" "$LOGS/recipe_${NAME}_train.log" \
    || gate "train used WRONG embedder ckpt (env not honoured)"
fi
CFG_YML=$(ls "$BD"/splat_runs_high/*/*/*/config.yml | tail -1)
( cd "$NS" && ${EMB:+HIGH_EMBEDDER_CKPT=$EMB} env -u LD_LIBRARY_PATH -u LD_PRELOAD \
  pixi run ns-export gaussian-splat --load-config "$CFG_YML" \
  --output-dir "$BD/splats" ) \
  > "$LOGS/recipe_${NAME}_export.log" 2>&1 || gate "export"
[ -f "$BD/splats/splat.ply" ] || gate "no splat.ply"
# keep only the model weights in the checkpoint (the render service and ns-eval load `pipeline` only; the Adam state
# is 2x the weights and only serves resuming) — Paul's reclaim decision 2026-09-25
( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python - "$BD" << 'PY'
import sys, os, glob, torch
for p in glob.glob(os.path.join(sys.argv[1], "splat_runs_high/*/*/*/nerfstudio_models/*.ckpt")):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if set(ck) <= {"step", "pipeline"}: continue
    torch.save({"step": ck["step"], "pipeline": ck["pipeline"]}, p + ".tmp"); os.replace(p + ".tmp", p); print("stripped", os.path.basename(p))
PY
) 2>/dev/null | grep stripped
python3 - "$SURVEY" "$CFG" << 'PY' || gate "manifests"
import json, sys
import numpy as np
from pathlib import Path
S, cfg = Path(sys.argv[1]), sys.argv[2]
DEDUP = S/"blocks_ns"/cfg
BD = DEDUP/"block_000"
tj = json.loads((BD/"transforms.json").read_text())
T = np.array([fr["transform_matrix"] for fr in tj["frames"]], dtype=np.float64)
c_min, c_max = T[:, :3, 3].min(0).tolist(), T[:, :3, 3].max(0).tolist()
(DEDUP/"splats.json").write_text(json.dumps({"blocks": [{"id": 0, "dir": str(BD),
    "splat_path": str(BD/"splats/splat.ply"), "frame_count": len(tj["frames"]),
    "centre_min": c_min, "centre_max": c_max}]}, indent=2))
(DEDUP/"index.json").write_text(json.dumps({"blocks": [{"id": 0,
    "n_frames": len(tj["frames"]), "centre_min": c_min, "centre_max": c_max,
    "centre_extent": (np.array(c_max)-np.array(c_min)).tolist(),
    "dir": str(BD)}], "tree_to_block": []}, indent=2))
print("manifests written")
PY
# stage-1-comparable PSNR readout (tensorboard mean-last-1k)
( cd "$NS" && env -u LD_LIBRARY_PATH -u LD_PRELOAD pixi run python - "$BD" << 'PY'
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
run = sorted((Path(sys.argv[1])/"splat_runs_high").glob("*/*/*"))[-1]
ea = EventAccumulator(str(run)); ea.Reload()
for tag in ["Train Metrics Dict/psnr", "Eval Images Metrics/psnr"]:
    if tag in ea.Tags().get("scalars", []):
        s = ea.Scalars(tag); v = [x.value for x in s]; st = [x.step for x in s]
        lastk = [x for a, x in zip(st, v) if a >= st[-1] - 1000]
        print(f"PSNR {tag}: last={v[-1]:.2f} mean-last-1k={np.mean(lastk):.2f}")
PY
) 2>/dev/null | grep PSNR | tee -a "$LOGS/recipe_${NAME}_train.log"
say "DONE"
