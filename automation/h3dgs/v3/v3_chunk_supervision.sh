#!/bin/bash
# v3_chunk_supervision.sh — chunk-specific supervision for a v3 chunk (recipe v3, 2026-10-08) = steps 1-3 of native_sam3_cell.sh
# verbatim (Paul 10-03: "supervision needs to be chunk specific, the same process for the keyframes in the chunk"), plus the text bank
# the render service needs and the verdict-frame pick. No hierarchy, no GPU training.
#   1 folder  = transforms.json listing the chunk's keyframes (its COLMAP images) + the H3DGS train/test split (split_names.json)
#   2 paint   = save_filtered_semantic_pngs (filtered SAM3 semantic monolithic, markers, global ids -> semantic_v2_B + palette)
#   3 compile = compile_supervision (colour bridge + SAM3 fruit ledgers, strict_fruit_tree_v1) -> supervision/trees_only + manifest
#   4 text bank (build_text_bank.py) + frames.json (pick_frames.py, 4 train / 10 held-out)
# Env: SVN (survey id under citrus_all), PROJ (H3DGS project = chunk export), CN (chunk), OUT (cell dir); [FRUIT_GLOB] [MUST]
set -uo pipefail
: ${SVN:?} ${PROJ:?} ${CN:?} ${OUT:?}
S=/home/paperspace/data/citrus_all/$SVN; C=$PROJ/camera_calibration/chunks/$CN; N=$OUT; mkdir -p $N
MONOS=$S/prod/monos; B=$S/prod/bateleur; HJ=$B/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NT=/home/paperspace/code/automation/h3dgs/native; NS=/home/paperspace/code/nerf_new
L=/home/paperspace/logs/v3_supervision_${SVN}_${CN}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
t0=$(date +%s); say "=== $SVN chunk $CN: chunk supervision by the block process -> $N"
python3 - $C $KF $N <<'PY' | tee -a $L
import json, struct, sys
C, KF, N = sys.argv[1:4]
def names(p):
    out = []
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4 + 32 + 24 + 4); nm = b''
            while (c := f.read(1)) != b'\x00': nm += c
            k = struct.unpack('<Q', f.read(8))[0]; f.read(24 * k); out.append(nm.decode())
    return sorted(out)
im = names(f'{C}/sparse/0/images.bin'); T = {l.strip() for l in open(f'{C}/sparse/0/test.txt') if l.strip()}
json.dump({'frames': [{'file_path': f'{KF}/{n}'} for n in im]}, open(f'{N}/transforms.json', 'w'), indent=0)
json.dump({'source': f'{C}/sparse/0 (images.bin + test.txt)', 'train': [n for n in im if n not in T], 'eval': [n for n in im if n in T]}, open(f'{N}/split_names.json', 'w'), indent=0)
print(f'[folder] {len(im)} keyframes; H3DGS split: {len(im) - len(T & set(im))} train / {len(T & set(im))} held out')
PY
if [ ! -f $N/semantic_v2_B/palette.json ]; then
  (cd $NS && pixi run python $ARU/save_filtered_semantic_pngs.py --block-dir $N --semantic-monolithic $MONOS/filtered_semantic_v2.monolithic \
     --marker-monolithic $B/scene_graph/markers_v2.monolithic --global-ids $B/sam3_v2/global_ids.json) > $L.paint 2>&1
  grep -aE '^\[(palette|save)\]|Error|Traceback|wrote|written' $L.paint | tail -3 | tee -a $L
  [ -f $N/semantic_v2_B/palette.json ] || { say "paint FAILED (see $L.paint)"; exit 1; }
fi
if [ ! -f $N/supervision/trees_only/manifest.json ]; then
  (cd $NS && pixi run python $ARU/compile_supervision.py --block-dir $N --tree-source colour_png_bridge --hierarchy $HJ \
     --fruit-ledger-glob "${FRUIT_GLOB:-$B/sam3_fruit/clip_*/frame_entries.json}" --filter strict_fruit_tree_v1 --out-dir $N/supervision/trees_only) > $L.compile 2>&1
  grep -aE '^\[|Error|Traceback' $L.compile | tail -4 | cut -c1-250 | tee -a $L
  [ -f $N/supervision/trees_only/manifest.json ] || { say "compile FAILED (see $L.compile)"; exit 1; }
fi
[ -e $N/text_bank.npz ] || (cd $NS && pixi run python $NT/build_text_bank.py --manifest $N/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $N/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
[ -e $N/frames.json ] || python3 $NT/pick_frames.py --names $N/split_names.json --sup $N/supervision/trees_only --must ${MUST:-} --n-train 4 --n-eval 10 --out $N/frames.json | tee -a $L
say "supervision done in $(( $(date +%s)-t0 )) s: $(ls $N/supervision/trees_only | grep -c png) maps, $(python3 -c "import json; m=json.load(open('$N/supervision/trees_only/manifest.json')); print(len(m.get('word_table', {})))") ids"
