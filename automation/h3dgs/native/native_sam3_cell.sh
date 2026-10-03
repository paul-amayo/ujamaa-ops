#!/bin/bash
# native_sam3_cell.sh — HiGH identity in an H3DGS chunk with CHUNK-SPECIFIC supervision built by the BLOCK process
# (UJAMAA, 2026-10-03; Paul: "supervision needs to be chunk specific", "the same process for the keyframes in the chunk").
# 1 chunk folder = transforms.json listing the chunk's keyframes (its own COLMAP images) + its own train/test split
# 2 paint   = run_unified_pipeline step 6a verbatim (save_filtered_semantic_pngs: filtered SAM3 semantic monolithic,
#             markers, global ids -> semantic_v2_B + closed-form palette)
# 3 compile = step b0 verbatim (compile_supervision colour bridge + SAM3 fruit ledgers, strict_fruit_tree_v1)
# 4 check   = tree pixels vs the prod blocks' maps on shared keyframes
# 5 census on the chunk's TRAIN views (H3DGS split), survey embedder; 6 seeds: A block defaults, B + bg-competes/ratio 2
#   + fruit share 0.1; 7-9 render held-out + a few train frames, score against the new supervision (no side-car anywhere;
#   the earlier side-car-routed native seed is scored too, for reference only).
# Env: SVN (survey id), PROJ (H3DGS project), CN (chunk), OUT (cell dir)
set -uo pipefail
: ${SVN:?} ${PROJ:?} ${CN:?} ${OUT:?}
S=/home/paperspace/data/citrus_all/$SVN; C=$PROJ/camera_calibration/chunks/$CN; N=$OUT; mkdir -p $N
MONOS=$S/prod/monos; B=$S/prod/bateleur; HJ=$B/scene_graph/marker_hierarchy.json; KF=$S/prod/scratch_sam3
EMB=$(ls -t $B/embedder/*/ckpts/model_best.pth | head -1)
ARU=/home/paperspace/code/aru_sil_core/src/scripts; NT=/home/paperspace/code/automation/h3dgs/native; NS=/home/paperspace/code/nerf_new
H3=/home/paperspace/code/hierarchical-3d-gaussians; PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python
HENV="env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH"
L=/home/paperspace/logs/native_sam3_${SVN}_${CN}.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
CUTW=$(python3 -c "
import struct,sys; f=open(sys.argv[1],'rb'); n=struct.unpack('<Q',f.read(8))[0]; cid,model,w,h=struct.unpack('<iiQQ',f.read(24)); print(w)" $C/sparse/0/cameras.bin)
SCENE="-s $C -m $PROJ/output/trained_chunks/$CN --hierarchy $PROJ/output/trained_chunks/$CN/hierarchy.hier_opt -i ../../rectified/images --eval --data_device cpu"
t0=$(date +%s); say "=== $SVN chunk $CN: chunk supervision by the block process | embedder $EMB | cut width $CUTW"
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
(cd $NS && pixi run python $ARU/save_filtered_semantic_pngs.py --block-dir $N --semantic-monolithic $MONOS/filtered_semantic_v2.monolithic \
   --marker-monolithic $B/scene_graph/markers_v2.monolithic --global-ids $B/sam3_v2/global_ids.json) > $L.paint 2>&1
grep -aE '^\[(palette|save)\]|Error|Traceback|wrote|written' $L.paint | tail -4 | tee -a $L
[ -f $N/semantic_v2_B/palette.json ] || { say "paint FAILED (see $L.paint)"; exit 1; }
(cd $NS && pixi run python $ARU/compile_supervision.py --block-dir $N --tree-source colour_png_bridge --hierarchy $HJ \
   --fruit-ledger-glob "$B/sam3_fruit/clip_*/frame_entries.json" --filter strict_fruit_tree_v1 --out-dir $N/supervision/trees_only) > $L.compile 2>&1
grep -aE '^\[|Error|Traceback' $L.compile | tail -6 | cut -c1-250 | tee -a $L
[ -f $N/supervision/trees_only/manifest.json ] || { say "compile FAILED (see $L.compile)"; exit 1; }
python3 - $S $N <<'PY' | tee -a $L
import glob, os, sys, numpy as np
from PIL import Image
S, N = sys.argv[1:3]
blk = {}
for d in sorted(glob.glob(f'{S}/prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/supervision/trees_only')):
    for p in glob.glob(f'{d}/kf_*.png'): blk.setdefault(os.path.basename(p), p)
new = sorted(glob.glob(f'{N}/supervision/trees_only/kf_*.png')); shared = [p for p in new if os.path.basename(p) in blk]
agree = tot = fr = 0
for p in shared[::max(1, len(shared) // 60)]:
    a = np.array(Image.open(p), np.uint16); b = np.array(Image.open(blk[os.path.basename(p)]), np.uint16)
    m = (b < 10000) & (b != 65535); tot += int(m.sum()); agree += int((a[m] == b[m]).sum()); fr += int(((a >= 10000) & (a != 65535) & m).sum())
print(f'[check] new maps {len(new)}; shared with the prod blocks {len(shared)}; on block TREE pixels (sampled frames): same tree id {agree / max(tot, 1):.4f}, now fruit {fr / max(tot, 1):.4f}, other {1 - (agree + fr) / max(tot, 1):.4f}')
PY
SUP=$N/supervision/trees_only
t1=$(date +%s); (cd $H3 && $HENV $PYH $NT/hier_census.py $SCENE --supervision-dir $SUP --names-json $N/split_names.json --taus 3 --cut-width $CUTW --with-bg --out-npz $N/W.npz) 2>&1 | grep -aE '^\[census\] (saved|hierarchy)|Error|Traceback' | cut -c1-400 | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t1 )) s"
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_A_block.bin --tree-floor 1.0) 2>&1 | grep -aE 'floors|share|assigned|node features|Error' | sed 's/^/[seed A, block defaults] /' | tee -a $L
(cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_B_bg2share.bin --tree-floor 1.0 --bg-competes --bg-ratio 2 --fruit-share-assign 0.1) 2>&1 | grep -aE 'background|floors|share|assigned|node features|Error' | sed 's/^/[seed B, bg2 + share] /' | tee -a $L
python3 $NT/pick_frames.py --names $N/split_names.json --sup $SUP --must ${MUST:-} --n-train 4 --n-eval 10 --out $N/frames.json | tee -a $L
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
for v in A_block B_bg2share; do (cd $H3 && $HENV $PYH $NT/hier_feature_render.py $SCENE --features $N/features_$v.bin --frames $FR --taus 3 --cut-width $CUTW --out-dir $N/maps_$v) 2>&1 | grep -aE 'Error|Traceback' | tee -a $L; done
[ -n "${EARLIER:-}" ] && (cd $H3 && $HENV $PYH $NT/hier_feature_render.py $SCENE --features $EARLIER --frames $FR --taus 3 --cut-width $CUTW --out-dir $N/maps_earlier) 2>&1 | grep -aE 'Error|Traceback' | tee -a $L
for v in A_block B_bg2share ${EARLIER:+earlier}; do bash $NT/score_maps.sh $N $v maps_$v/%s_tau3.npz $SUP $EMB $HJ $KF 4 | tee -a $L; done
python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json --models A_block B_bg2share > $N/table_A_vs_B.md; sed -n '/| kind/,/^$/p' $N/table_A_vs_B.md | tee -a $L
[ -n "${EARLIER:-}" ] && { python3 $NT/native_table.py --verdicts $N/verdicts.log --frames $N/frames.json --models B_bg2share earlier > $N/table_B_vs_earlier.md; sed -n '/| kind/,/^$/p' $N/table_B_vs_earlier.md | tee -a $L; }
say "done in $(( ($(date +%s)-t0)/60 )) min"
