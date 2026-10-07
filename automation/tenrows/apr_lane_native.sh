#!/bin/bash
# April lane 2 relevancy (the regular H3DGS native-identity chain): kf map -> cell (supervision links, keyframe cameras in the lane
# model's frame) -> census (hier_census, tau 3, with bg) -> seed (build_census_init, survey embedder apr_2026_zed_v1g, the regular
# flags) -> text bank -> render the scoring frames -> containment verdicts (score_maps.sh / containment_eval). See apr_lane_native_prep.py.
set -uo pipefail
LD=/home/paperspace/data/klapmuts/apr_2026_zed/experimental/lane2_apr_full; PROJ=h3dgs_e2s; P=$LD/$PROJ; N=$P/native/cell_trees; T=$P/output/trained_chunks/lane
S=/home/paperspace/data/klapmuts/apr_2026_zed/prod; EMB=$S/bateleur/embedder/apr_2026_zed_v1g/ckpts/model_best.pth; HJ=$S/bateleur/scene_graph/marker_hierarchy.json
A=/home/paperspace/code/automation/tenrows; NT=/home/paperspace/code/automation/h3dgs/native; ARU=/home/paperspace/code/aru_sil_core/src/scripts; H3=/home/paperspace/code/hierarchical-3d-gaussians
PYH=/home/paperspace/miniconda3/envs/h3dgs/bin/python; NS=/home/paperspace/code/nerf_new; HENV="env CUDA_HOME=/home/paperspace/code/_cuda12 PATH=/home/paperspace/code/_cuda12/bin:$PATH"
P310="env -u LD_LIBRARY_PATH -u LD_PRELOAD PYTHONPATH=/home/paperspace/code/aru_sil_core/src/interfaces/build/temp.linux-x86_64-cpython-310/lib /home/paperspace/code/nerf_new/.pixi/envs/default/bin/python3.10"
L=/home/paperspace/logs/apr_lane2_native.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; t0=$(date +%s); mkdir -p $N
say "=== April lane 2 relevancy: $P | emb $EMB"
[ -e $LD/kf_map.json ] || $P310 $A/apr_lane_native_prep.py kfmap $LD 2>&1 | grep -a '^\[native-prep\]' | tee -a $L
$PYH $A/apr_lane_native_prep.py cell $LD $PROJ $N 2>&1 | grep -a '^\[native-prep\]\|Error\|Traceback\|assert' | tee -a $L
SCENE="-s $N/census_src -m $T --hierarchy $T/hierarchy.hier_opt -i $N/census_src/images --eval --data_device cpu"
t1=$(date +%s); (cd $H3 && $HENV $PYH $NT/hier_census.py $SCENE --supervision-dir $N/supervision/trees_only --names-json $N/split_names.json --taus 3 --cut-width 1280 --with-bg --out-npz $N/W.npz) 2>&1 | grep -aE '^\[census\] (saved|hierarchy)|Error|Traceback' | cut -c1-300 | tee -a $L
[ -e $N/W.npz ] || { say "census FAILED"; exit 1; }; say "census $(( $(date +%s)-t1 )) s"
t1=$(date +%s); (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $N/W.npz --embedder $EMB --out-features $N/features_census.bin --tree-floor 1.0 --bg-competes --bg-ratio 2) 2>&1 | grep -aE 'background|floors|assigned|share|node features|Error|Traceback' | tee -a $L
[ -e $N/features_census.bin ] || { say "seed FAILED"; exit 1; }; say "seed $(( $(date +%s)-t1 )) s"
(cd $NS && pixi run python $NT/build_text_bank.py --manifest $N/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $N/text_bank.npz) 2>&1 | grep -a 'text-bank\|Error' | tee -a $L
FR=$(python3 -c "import json; print(' '.join(json.load(open('$N/frames.json'))['frames']))")
t1=$(date +%s); (cd $H3 && $HENV $PYH $NT/hier_feature_render.py $SCENE --features $N/features_census.bin --frames $FR --taus 3 --cut-width 1280 --out-dir $N/maps_native) 2>&1 | grep -aE '^\[render\]|Error|Traceback' | sed -E 's/ -> .*//' | tail -3 | tee -a $L
bash $NT/score_maps.sh $N native maps_native/%s_tau3.npz $N/supervision/trees_only $EMB $HJ $N/census_src/images 4 | tee -a $L; say "render + score $(( $(date +%s)-t1 )) s"
python3 - $N <<'PY' | tee -a $L
import json, re, sys
from collections import defaultdict
N = sys.argv[1]; info = json.load(open(f'{N}/frames.json'))['info']; pat = re.compile(r'^\[native (\S+)\] (TREE|ROW|FRUIT of) +(\d+) +"([^"]*)": thr (\S+) IoU (\S+)')
agg = defaultdict(list)
for line in open(f'{N}/verdicts.log', errors='replace'):
    m = pat.match(line)
    if m: agg[(m.group(2).replace(' of', ''), info.get(m.group(1), {}).get('split', '?'))].append(float(m.group(6)))
for k in sorted(agg): v = agg[k]; print(f'[native-score] {k[0]:5s} {k[1]:5s}: {len(v)} heads, mean IoU {sum(v) / len(v):.3f}, median {sorted(v)[len(v) // 2]:.3f}')
PY
say "=== done in $(( ($(date +%s)-t0)/60 )) min"
