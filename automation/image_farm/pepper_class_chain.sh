#!/bin/bash
# Class-level PEPPER identity on the chilli splat (UJAMAA 2026-10-05; Paul: "go class-level pepper after the renders").
# v2: TILED=<tiled SAM3 dir> CELL=<cell name> select the mask set and the output cell ('red chili pepper' masks -> cell_red).
# Segment gwakungu IMG_7990_s1, H3DGS h3dgs/ (chunk lane, 8 M, RGB). Frames 142-240 have collapsed SfM poses (chilli crop entry,
# notebook 10-04), so the census uses training views 0-141 only.
#   maps     tiled SAM3 'pepper' class maps (sam3_tiled_class.py) -> uint16 supervision: pepper = fruit 10000, every other pixel =
#            plant/other 0 (the competitor word) -> undistorted OPENCV -> the splat's PINHOLE camera (outside -> 65535)
#   graph    minimal scene graph: one object (the chilli plants, tree 0) in row 0, one fruit (pepper = fruit 0 -> 10000)
#   embedder citrus graph recipe (c20cos20, 1500 ep, keep-super-row, no-level-norms) on that graph
#   census   hier_census.py over training views 0-141 (tau 3 at 1080 px, --with-bg)
#   seeds    build_census_init.py: majority / fruit share > 0.3 / > 0.1 (the 0.1 rule lit leaves on citrus - compare, don't assume)
#   bank     build_text_bank.py (words: asus = plant/other, bumper = pepper, oak = row)
#   score    pepper_bc_score.py per seed: best containment over {plant, pepper} vs the undistorted SAM3 pepper maps
set -uo pipefail
S=/home/paperspace/data/image_farm/gwakungu/2026-05-16/IMG_7990_s1; P=$S/h3dgs; C=$P/camera_calibration/chunks/lane; X=$S/experimental/pepper_class
NB=$X/${CELL:-cell}; SUP=$NB/supervision/pepper_class; RAW=$NB/supervision/pepper_raw; HJ=$NB/marker_hierarchy_graph.json; EN=IMG_7990_s1_pepper_v1
EMB=$X/embedder/$EN/ckpts/model_best.pth; L=/home/paperspace/logs/pepper_class_chain.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
H3=/home/paperspace/code/hierarchical-3d-gaussians; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NT=/home/paperspace/code/automation/h3dgs/native
ARU=/home/paperspace/code/aru_sil_core/src/scripts; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH; NS=/home/paperspace/code/nerf_new
HENV="env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH"; mkdir -p $NB $RAW
STAGES=${*:-maps graph embedder census seeds bank score}
for st in $STAGES; do say "=== $st"; case $st in
maps)
  $PYH - ${TILED:-$X/sam3_tiled}/class $RAW <<'PY' | tee -a $L
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); n = 0; px = 0
for p in sorted(src.glob('image_*.png')):
    m = np.array(Image.open(p)) > 0; Image.fromarray(np.where(m, 10000, 0).astype(np.uint16)).save(dst / p.name); n += 1; px += int(m.sum())
json.dump({'schema': 'supervision_manifest/v1', 'unlabelled_value': 65535, 'fruit_id_base': 10000,
           'filter_spec': {'name': 'sam3_tiled_class_pepper', 'grid': '3x4', 'overlap': 0.15, 'conf': 0.3, 'min_area': 20},
           'word_table': {'0': 'asus', '10000': 'bumper'}, 'level_table': {'0': 'tree', '10000': 'fruit'},
           'note': 'class level (2026-10-05): 10000 = every pepper pod SAM3 found (tiled), 0 = everything else (plant / ground / sky)'},
          open(dst / 'manifest.json', 'w'), indent=1)
print(f'[maps] {n} uint16 maps, {px:,} pepper px')
PY
  $PYH /home/paperspace/code/automation/image_farm/undistort_id_maps.py $S/sparse/0/cameras.bin $C/sparse/0/cameras.bin $RAW $SUP 2>&1 | tail -2 | tee -a $L ;;
graph)
  python3 - $HJ <<'PY' | tee -a $L
import json, sys
g = {'objects': [{'id': 0, 'name': 'chilli plants', 'xyz': [0.0, 0.0, 0.0], 'row_id': 0, 'rows_id': 0, 'super_row_id': 0, 'section_id': 0}],
     'rows': [{'id': 0, 'object_ids': [0]}], 'fruits': [{'id': 0, 'tree_id': 0, 'xyz': [0.0, 0.0, 0.0]}],
     'source': 'class-level pepper graph (pepper_class_chain.sh, 2026-10-05): one object = the chilli plants, one fruit = all pepper pods',
     'n_objects': 1}
json.dump(g, open(sys.argv[1], 'w'), indent=1); print('[graph] 1 object, 1 row, 1 fruit')
PY
  ;;
embedder)
  (cd $NS && pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name $EN --output-dir $X/embedder \
     --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
     --epochs 1500 --no-level-norms) > /home/paperspace/logs/pepper_embedder.log 2>&1; say "embedder rc=$? $(ls $EMB 2>/dev/null || echo 'NO model_best')"
  [ -e $EMB ] || exit 1 ;;
census)
  python3 - $C $NB <<'PY' | tee -a $L
import json, re, struct, sys
C, NB = sys.argv[1:3]
f = open(f'{C}/sparse/0/images.bin', 'rb'); n = struct.unpack('<Q', f.read(8))[0]; names = []
for _ in range(n):
    f.read(64); b = b''
    while (c := f.read(1)) != b'\x00': b += c
    k = struct.unpack('<Q', f.read(8))[0]; f.read(24 * k); names.append(b.decode())
test = {l.strip() for l in open(f'{C}/sparse/0/test.txt') if l.strip()}
train = sorted((x for x in names if x not in test and int(re.sub(r'\D', '', x)) <= 141), key=lambda x: int(re.sub(r'\D', '', x)))
json.dump({'source': f'{C}/sparse/0 minus test.txt, frames <= 141 (142-240 have collapsed poses)', 'train': train}, open(f'{NB}/split_names.json', 'w'), indent=1)
print(f'[census] {len(train)} training views 0-141')
PY
  (cd $H3 && $HENV $PYH $NT/hier_census.py -s $C -m $P/output/trained_chunks/lane --hierarchy $P/output/trained_chunks/lane/hierarchy.hier_opt -i ../../rectified/images \
     --eval --data_device cpu --supervision-dir $SUP --names-json $NB/split_names.json --taus 3 --cut-width 1080 --with-bg --out-npz $NB/W.npz) 2>&1 \
     | grep -aE '^\[census\] (saved|hierarchy)|Error|Traceback' | cut -c1-400 | tee -a $L ;;
seeds)
  for v in "maj:" "s03:--fruit-share-assign 0.3" "s01:--fruit-share-assign 0.1"; do t=${v%%:*}; fl=${v#*:}
    (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $NB/W.npz --embedder $EMB --out-features $NB/features_$t.bin --tree-floor 1.0 $fl) 2>&1 \
      | grep -aE 'target norms|floors|share|assigned|Error' | sed "s/^/[seed $t] /" | tee -a $L
  done ;;
bank)
  (cd $NS && pixi run python $NT/build_text_bank.py --manifest $SUP/manifest.json --hierarchy-json $HJ --out $NB/text_bank.npz) 2>&1 | grep -a "text-bank\|Error" | tee -a $L ;;
score)
  for t in maj s03 s01; do (cd /home/paperspace/code && CELL=${CELL:-cell} $HENV $PYH /home/paperspace/code/automation/image_farm/pepper_bc_score.py $NB/features_$t.bin $t) 2>&1 | grep -a "^\[pepper\|Error\|Traceback" | tee -a $L; done ;;
esac; done
say "=== done: $STAGES"
