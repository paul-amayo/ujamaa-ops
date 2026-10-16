#!/bin/bash
# fuji_structure.sh <east|west> — Fuji-SfM (Gené-Mola 2020) one row side through the FruitNeRF image-only pipeline (fruitnerf_structure.sh
# stages with path arguments): 291 raw images -> 1620x1080 working set -> our SfM (OPENCV, exhaustive, glomap) -> nerfstudio conversion +
# ingest -> SAM3 image mode 'apple' (staged kf_ symlinks) -> DA3 windowed depth. Survey root /home/paperspace/data/public/fuji_sfm/high/<side>.
set -uo pipefail
SIDE=${1:?east|west}; SRC=/home/paperspace/data/public/fuji_sfm/Fuji-SfM_dataset/Fuji-SfM_dataset/2-SfM-set/raw_images_${SIDE}_side; SURVEY=/home/paperspace/data/public/fuji_sfm/high/$SIDE
ARU=/home/paperspace/code/aru_sil_core; IPL=$ARU/src/scripts/image_pipeline; SAM3_PY=/home/paperspace/code/sam3/.pixi/envs/default/bin/python
NS=/home/paperspace/code/nerf_new; PY310=$NS/.pixi/envs/default/bin/python3.10; LOGS=/home/paperspace/logs; NAME=fuji_$SIDE
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $NAME: $*" | tee -a $LOGS/$NAME.log; }; gate(){ say "CHECK FAILED: $*"; exit 3; }
mkdir -p $SURVEY/images
if [ -z "$(ls $SURVEY/images 2>/dev/null | head -1)" ]; then
  python3 - "$SRC" "$SURVEY/images" <<'PY' || gate "working set"
import sys
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); frames = sorted(src.glob("*.JPG")) + sorted(src.glob("*.jpg"))
def one(a):
    i, p = a; im = Image.open(p).convert("RGB"); w, h = im.size; nh = 1080; nw = int(round(w * 1080 / h / 2) * 2); im.resize((nw, nh), Image.LANCZOS).save(dst / f"image_{i}.png"); return nw, nh
with Pool(16) as pool: sizes = pool.map(one, list(enumerate(frames)))
(dst.parent / "source_names.txt").write_text("\n".join(p.name for p in frames) + "\n"); print(f"working set: {len(frames)} frames -> {sizes[0]}")
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
mkdir -p $SURVEY/prod/scratch_sam3; for p in $SURVEY/images/image_*.png; do k=$(basename $p .png | cut -d_ -f2); ln -sf $p $SURVEY/prod/scratch_sam3/$(printf "kf_%06d.png" $k); done
OUTD=$(ls -d $SURVEY/prod/bateleur/sam3_v2 $SURVEY/sam3_v2 2>/dev/null | head -1)
if [ -z "$OUTD" ] || [ -z "$(ls $OUTD/clip_*/frame_entries.json 2>/dev/null | head -1)" ]; then
  say "SAM3 image mode, prompt 'apple'"
  $SAM3_PY $ARU/src/scripts/build_tree_instances.py --data-dir $SURVEY --images-dir $SURVEY/prod/scratch_sam3 --prompt apple --min-mask-area 100 --no-skip-existing > $LOGS/${NAME}_sam3.log 2>&1 || gate "SAM3"
fi
if [ -z "$(ls $SURVEY/depth_png 2>/dev/null | head -1)" ]; then
  say "DA3 windowed depth"
  ( cd /home/paperspace/code/sam3 && $SAM3_PY $IPL/da3_windows_fuse.py --survey $SURVEY --write-depth --depth-size ${IW}x${IH} ) > $LOGS/${NAME}_da3.log 2>&1 || gate "DA3"
fi
say "=== structure done: $(ls $SURVEY/depth_png | wc -l) depth maps, $(ls $SURVEY/prod/bateleur/sam3_v2/clip_*/frame_entries.json $SURVEY/sam3_v2/clip_*/frame_entries.json 2>/dev/null | wc -l) SAM3 clips"
