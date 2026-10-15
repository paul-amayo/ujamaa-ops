#!/bin/bash
# fruitnerf_structure.sh <tree_0X> — our image-only pipeline (prod_image_recipe.sh stages 0-3) on one FruitNeRF real tree, fruit settings:
# ALL frames (the recipe's 300-frame cap would drop part of the orbit) at height 1080 (1620x1080), our own SfM (OPENCV + exhaustive + glomap,
# their poses unused), nerfstudio conversion + rosbag-parity ingest, SAM3 IMAGE mode prompt 'apple' (min mask 100 px — apples are ~40 px across
# at 1080), DA3 windowed depth (the mask lift). The registry (lift + 3-D clustering + count) is fruit_registry_count.py.
set -uo pipefail
T=${1:?tree_01|tree_02|tree_03}; SRC=/home/paperspace/data/public/fruitnerf/FruitNeRF_Dataset/$T/images; SURVEY=/home/paperspace/data/public/fruitnerf/high/$T
ARU=/home/paperspace/code/aru_sil_core; IPL=$ARU/src/scripts/image_pipeline; SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
NS=/home/paperspace/code/nerf_new; PY310=$NS/.pixi/envs/default/bin/python3.10; LOGS=/home/paperspace/logs; NAME=fruitnerf_$T
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $NAME: $*" | tee -a $LOGS/$NAME.log; }; gate(){ say "CHECK FAILED: $*"; exit 3; }
mkdir -p $SURVEY/images
if [ -z "$(ls $SURVEY/images 2>/dev/null | head -1)" ]; then
  python3 - "$SRC" "$SURVEY/images" <<'PY' || gate "working set"
import sys
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); frames = sorted(src.glob("*.JPG")) + sorted(src.glob("*.jpg"))
for i, p in enumerate(frames):
    im = Image.open(p).convert("RGB"); w, h = im.size; nh = 1080; nw = int(round(w * 1080 / h / 2) * 2)
    im.resize((nw, nh), Image.LANCZOS).save(dst / f"image_{i}.png")
(dst.parent / "source_names.txt").write_text("\n".join(p.name for p in frames) + "\n")
print(f"working set: {len(frames)} frames -> {nw}x{nh}")
PY
fi
N=$(ls $SURVEY/images | wc -l); read -r IW IH <<< "$(python3 -c "from PIL import Image; im=Image.open('$SURVEY/images/image_0.png'); print(im.size[0], im.size[1])")"
say "working set $N frames @ ${IW}x${IH}"
if [ ! -f $SURVEY/sparse/0/points3D.bin ]; then
  say "SfM (feature / exhaustive / glomap)"
  ( export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}; export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglog.so.0
    rm -f $SURVEY/database.db; mkdir -p $SURVEY/sparse
    colmap feature_extractor --database_path $SURVEY/database.db --image_path $SURVEY/images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 1 > $LOGS/${NAME}_sfm.log 2>&1 &&
    colmap exhaustive_matcher --database_path $SURVEY/database.db --FeatureMatching.use_gpu 1 >> $LOGS/${NAME}_sfm.log 2>&1 &&
    glomap mapper --database_path $SURVEY/database.db --image_path $SURVEY/images --output_path $SURVEY/sparse >> $LOGS/${NAME}_sfm.log 2>&1 ) || gate "SfM"
fi
REG=$(python3 - "$SURVEY" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/home/paperspace/code/colmap/scripts/python")
from read_write_model import read_images_binary
print(len(read_images_binary(Path(sys.argv[1]) / "sparse/0/images.bin")))
PY
)
say "SfM registered $REG/$N"; [ "${REG:-0}" -ge $((N * 90 / 100)) ] || gate "SfM registered $REG/$N (< 90 %)"
if [ ! -f $SURVEY/lio_image_poses.json ]; then
  python3 $IPL/colmap_to_nerfstudio.py $SURVEY > $LOGS/${NAME}_convert.log 2>&1 || gate "colmap_to_nerfstudio"
  PYTHONPATH=$ARU/src/interfaces/build/temp.linux-x86_64-3.10/lib $PY310 $IPL/image_to_monolithic.py $SURVEY --fps 2 > $LOGS/${NAME}_ingest.log 2>&1 || gate "image_to_monolithic"
fi
OUTD=$(ls -d $SURVEY/prod/bateleur/sam3_v2 $SURVEY/sam3_v2 2>/dev/null | head -1)
if [ -z "$OUTD" ] || [ -z "$(ls $OUTD/clip_*/frame_entries.json 2>/dev/null | head -1)" ]; then
  say "SAM3 image mode, prompt 'apple'"
  $SAM3_PY $ARU/src/scripts/build_tree_instances.py --data-dir $SURVEY --prompt apple --min-mask-area 100 > $LOGS/${NAME}_sam3.log 2>&1 || gate "SAM3"
fi
if [ -z "$(ls $SURVEY/depth_png 2>/dev/null | head -1)" ]; then
  say "DA3 windowed depth"
  ( cd /home/paperspace/code/sam3 && $SAM3_PY $IPL/da3_windows_fuse.py --survey $SURVEY --write-depth --depth-size ${IW}x${IH} ) > $LOGS/${NAME}_da3.log 2>&1 || gate "DA3"
fi
say "=== structure done: $(ls $SURVEY/depth_png | wc -l) depth maps, $(ls ${OUTD:-$SURVEY/sam3_v2}/clip_*/frame_entries.json 2>/dev/null | wc -l) SAM3 clips"
