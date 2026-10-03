#!/bin/bash
# 01 swap to the MEDIAN-anchored global-id lifting (Paul 2026-10-03: "go for it"). Stages (run all, or name some):
#  era      copy/move the current prod survey artifacts into experimental/era_liftmin_20261003/
#  ids      median run (experimental/lifting_anchor_20261003/median) renumbered onto prod's tree ids (mutual-nearest centroid
#           <= 1.5 m; unmatched -> new ids above prod's) -> prod/bateleur/sam3_v2/global_ids.json
#  markers  markers_v2_from_sam3 (regen recipe: --use-sam3-frames --depth-back 1.5 --min-observations 2 --drift-m 0)
#  hier     CORAL strict hierarchy (dir 1,0 @ 0.985) + diff vs the previous one + rows-vs-lanes check
#  embedder retrain 01_13B_v1g in place (graph c20cos20, 1500 epochs)
#  paint    71 blocks: supervision rebuilt (block_rebuild.sh paint; block_000 first, then 4 at a time)
#  census   71 blocks: re-census + seed (block_rebuild.sh census; 3 GPU lanes)
#  chunk    chunk_3_1_sam3 native cell rebuilt fresh (native_sam3_cell.sh) + text bank; old cell kept beside it
#  sidecar  (DEPRECATED, Paul 2026-10-03 — not in the default stages) live Citrus A side-car re-link + re-census + seed
#  verdicts verdict_sweep_glref.sh 01_13B_Jackal
set -uo pipefail
S=/home/paperspace/data/citrus_all/01_13B_Jackal; B=$S/prod/bateleur; MONOS=$S/prod/monos; HJ=$B/scene_graph/marker_hierarchy.json
Q=$S/experimental/era_liftmin_20261003; X=$S/experimental/lifting_anchor_20261003; BL=$S/prod/tassili/blocks_ns/lio_row100
ARU=/home/paperspace/code/aru_sil_core/src/scripts; HIGH=/home/paperspace/code/aru_sil_core/src/interfaces/rerun/HiGH
W8=/home/paperspace/code/automation/lifting_median_swap; NAT=/home/paperspace/code/automation/h3dgs/native
L=/home/paperspace/logs/liftmed_swap_01.log; say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
NS=/home/paperspace/code/nerf_new; EMB=$B/embedder/01_13B_v1g/ckpts/model_best.pth
STAGES=${*:-era ids markers hier embedder paint census chunk verdicts}   # sidecar: deprecated (Paul 2026-10-03), not rebuilt
for ST in $STAGES; do say "=== stage $ST"; case $ST in
era)
  mkdir -p $Q/scene_graph $Q/monos $Q/sam3_v2
  cp -a $B/sam3_v2/global_ids.json $Q/sam3_v2/global_ids.json
  cp -a $HJ $Q/scene_graph/marker_hierarchy.json
  for f in markers_v2.monolithic instance_labels_v2.monolithic; do [ -e $B/scene_graph/$f ] && mv $B/scene_graph/$f $Q/scene_graph/$f; done
  for f in filtered_semantic_v2.monolithic filtered_semantic_v2.monolithic.index; do [ -e $MONOS/$f ] && mv $MONOS/$f $Q/monos/$f; done
  [ -d $Q/embedder_01_13B_v1g ] || cp -a $B/embedder/01_13B_v1g $Q/embedder_01_13B_v1g
  say "era copied/moved: $(ls $Q/* | tr '\n' ' ')" ;;
ids)
  python3 - $Q/sam3_v2/global_ids.json $X/median/global_ids.json $X/median/global_ids_prodids.json <<'PY' | tee -a $L
import json, sys, numpy as np
P0 = json.load(open(sys.argv[1])); M = json.load(open(sys.argv[2]))
C0 = {int(g): np.array(s['world_centroid']) for g, s in P0['stats'].items()}; C = {int(g): np.array(s['world_centroid']) for g, s in M['stats'].items()}
ga = list(C); gb = list(C0); d = np.linalg.norm(np.array([C[g] for g in ga])[:, None] - np.array([C0[g] for g in gb])[None], axis=2)
iab = d.argmin(1); iba = d.argmin(0)
mp = {ga[i]: gb[iab[i]] for i in range(len(ga)) if iba[iab[i]] == i and d[i, iab[i]] <= 1.5}
nxt = max(gb) + 1
for g in ga:
    if g not in mp: mp[g] = nxt; nxt += 1
assert len(set(mp.values())) == len(mp)
out = dict(M); out['member_to_global'] = {k: mp[int(v)] for k, v in M['member_to_global'].items()}
out['stats'] = {str(mp[int(g)]): s for g, s in M['stats'].items()}
out['id_remap'] = {'from': sys.argv[2], 'onto': 'prod 08-22 numbering (era_liftmin_20261003/sam3_v2/global_ids.json)', 'method': 'mutual-nearest world_centroid <= 1.5 m',
                   'renumbered_to_prod': sum(1 for a, b in mp.items() if b in C0), 'new_ids': {str(a): b for a, b in mp.items() if b not in C0},
                   'prod_ids_absent': sorted(set(gb) - set(mp.values()))}
json.dump(out, open(sys.argv[3], 'w'), indent=2)
print(f"[ids] {len(mp)} trees: {out['id_remap']['renumbered_to_prod']} on prod ids, new {out['id_remap']['new_ids']}, prod ids absent {out['id_remap']['prod_ids_absent']}; {len(out['member_to_global'])} masks with an id")
PY
  cp $X/median/global_ids_prodids.json $B/sam3_v2/global_ids.json && say "installed median global ids (prod numbering)" ;;
markers)
  (cd $NS && pixi run python $ARU/markers_v2_from_sam3.py --data-dir $S --use-sam3-frames --depth-back 1.5 --min-observations 2 --drift-m 0) > /home/paperspace/logs/liftmed_01_markers.log 2>&1 \
    || { say "markers FAILED"; exit 1; }
  grep -aE "^\[markers\] wrote|^\[semantic\] wrote" /home/paperspace/logs/liftmed_01_markers.log | tee -a $L
  ls -la $B/scene_graph/markers_v2.monolithic $MONOS/filtered_semantic_v2.monolithic | tee -a $L ;;
hier)
  (cd $NS && pixi run python $ARU/build_marker_hierarchy.py --row-solver coral --semantic-monolithic $MONOS/filtered_semantic_v2.monolithic \
     --marker-monolithic $B/scene_graph/markers_v2.monolithic --dominant-direction-xz "1,0" --dominant-direction-threshold 0.985 --out $HJ) \
     > /home/paperspace/logs/liftmed_01_hier.log 2>&1 || { say "hierarchy FAILED"; exit 1; }
  python3 - $Q/scene_graph/marker_hierarchy.json $HJ <<'PY' | tee -a $L
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ra = {o['id']: o['row_id'] for o in a['objects']}; rb = {o['id']: o['row_id'] for o in b['objects']}
ga = {}; gb = {}
for k, v in ra.items(): ga.setdefault(v, set()).add(k)
for k, v in rb.items(): gb.setdefault(v, set()).add(k)
same_rows = sum(1 for s in gb.values() if s in ga.values())
print(f"[hier] objects {len(ra)} -> {len(rb)} (removed {sorted(set(ra) - set(rb))}, added {sorted(set(rb) - set(ra))}); rows {len(ga)} -> {len(gb)}, "
      f"{same_rows} rows with identical membership; row ids changed for {sum(1 for k in rb if k in ra and ra[k] != rb[k])} trees")
PY
  (cd $NS && pixi run python /home/paperspace/code/automation/demo/supervision_void/rows_vs_lanes_survey.py $S 2>&1 | grep -a "objects," | tee -a $L) ;;
embedder)
  (cd $NS && pixi run python $HIGH/train_hyperembedder_graph.py --hierarchy-json $HJ --experiment-name 01_13B_v1g --output-dir $B/embedder \
     --contrastive-weight 2.0 --cosine-reconstruction-weight 2.0 --reconstruction-weight 1.0 --temperature 0.2 --keep-super-row \
     --epochs 1500 --no-level-norms) > /home/paperspace/logs/liftmed_01_embedder.log 2>&1 || { say "embedder FAILED"; exit 1; }
  ls -la --time-style=+%m-%d_%H:%M $EMB | tee -a $L ;;
paint)
  bash $W8/block_rebuild.sh $BL/block_000 paint
  ls -d $BL/block_[0-9][0-9][0-9] | grep -v block_000 | xargs -P 4 -I{} bash $W8/block_rebuild.sh {} paint
  say "paint: $(grep -c ' paint: ok' /home/paperspace/logs/liftmed_swap_01_blocks.log) ok, $(grep -c ' paint: .*FAIL\| paint: no manifest' /home/paperspace/logs/liftmed_swap_01_blocks.log) failed" ;;
census)
  ls -d $BL/block_[0-9][0-9][0-9] | xargs -P 3 -I{} bash $W8/block_rebuild.sh {} census
  say "census: $(grep -c ' census: ok' /home/paperspace/logs/liftmed_swap_01_blocks.log) ok, $(grep -c ' census: FAIL' /home/paperspace/logs/liftmed_swap_01_blocks.log) failed" ;;
chunk)
  C=$S/experimental/h3dgs_native/chunk_3_1_sam3; [ -d ${C}_era_liftmin_20261003 ] || mv $C ${C}_era_liftmin_20261003
  SVN=01_13B_Jackal PROJ=$S/experimental/h3dgs CN=3_1 OUT=$C MUST=kf_003426.png EARLIER=$S/experimental/h3dgs_native/chunk_3_1/features_census.bin \
    bash $NAT/native_sam3_cell.sh > /home/paperspace/logs/liftmed_01_chunk.log 2>&1
  (cd $NS && pixi run python $NAT/build_text_bank.py --manifest $C/supervision/trees_only/manifest.json --hierarchy-json $HJ --out $C/text_bank.npz 2>&1 | grep -a "text-bank\|Error" | tee -a $L)
  ls -la $C/features_B_bg2share.bin $C/features_A_block.bin $C/W.npz 2>&1 | tee -a $L ;;
sidecar)
  O=$S/experimental/h3dgs_sidecar_chunks/chunk_3_1; SUP=$O/supervision/trees_only; E=$O/era_liftmin_20261003; mkdir -p $E
  RUN=$(ls -d $O/splat_runs_FEATFIX/stage2_censusinit_glref_bg_f1.0_r2/high/*/ | head -1); BOOT=$(ls -t $O/splat_runs_FEATFIX/stage2_bootstrap_glref/high/*/config.yml | head -1)
  [ -d $E/supervision_trees_only ] || { mv $SUP $E/supervision_trees_only; mkdir -p $SUP; }
  [ -e $O/splat_runs_FEATFIX/interaction_W_glref_bg.npz ] && mv $O/splat_runs_FEATFIX/interaction_W_glref_bg.npz $E/
  for c in $RUN/nerfstudio_models/*.ckpt; do [ -e "$c" ] && mv "$c" $E/seed_$(basename $c); done
  python3 - $S $O $E/supervision_trees_only <<'PY' | tee -a $L
import glob, json, os, sys
from pathlib import Path
S, O, OLD = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); sup = {}; extra = {}
for tj in sorted(glob.glob(str(S / 'prod/tassili/blocks_ns/lio_row100/block_[0-9][0-9][0-9]/transforms.json'))):
    bd = Path(tj).parent; own = {Path(f['file_path']).name for f in json.load(open(tj))['frames']}
    for f in sorted(glob.glob(str(bd / 'supervision/trees_only/kf_*.png'))):
        n = os.path.basename(f); (sup.__setitem__(n, f) if n in own else extra.setdefault(n, f))
for n, f in extra.items(): sup.setdefault(n, f)
names = sorted(p.name for p in OLD.glob('kf_*.png')); n = 0
for nm in names:
    if nm in sup: os.link(sup[nm], O / 'supervision/trees_only' / nm); n += 1
print(f'[sidecar] relinked {n}/{len(names)} supervision maps to the rebuilt block maps')
PY
  python3 /home/paperspace/logs/sidecar_chunk_manifest.py 01_13B_Jackal $O 2>&1 | grep -a '^\[chunk-manifest\]' | tee -a $L
  (cd $NS && CENSUS_WITH_BG=1 HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/gaussian_interaction_census.py --run-glob "$BOOT" --supervision-dir $SUP \
     --out-npz $O/splat_runs_FEATFIX/interaction_W_glref_bg.npz > /home/paperspace/logs/liftmed_01_sidecar_census.log 2>&1)
  [ -e $O/splat_runs_FEATFIX/interaction_W_glref_bg.npz ] || { say "sidecar census FAILED"; exit 1; }
  INIT=$O/stage2_init_census_glref_bg_f1.0_r2/nerfstudio_models
  (cd $NS && HIGH_EMBEDDER_CKPT=$EMB pixi run python $ARU/build_census_init.py --w-npz $O/splat_runs_FEATFIX/interaction_W_glref_bg.npz --embedder $EMB \
     --src-ckpt $O/stage2_init_glref/nerfstudio_models/step-000015000.ckpt --dst-dir $INIT --tree-floor 1.0 --bg-competes --bg-ratio 2 2>&1 | grep -aE 'background|assigned' | tee -a $L)
  mv $INIT/*.ckpt $RUN/nerfstudio_models/
  /home/paperspace/miniconda3/envs/h3dgs/bin/python -c "
import torch,sys,glob; p=glob.glob(sys.argv[1]+'/*.ckpt')[0]; ck=torch.load(p,map_location='cpu',weights_only=False); ck['optimizers']={}; ck['schedulers']={}; torch.save(ck,p); print('[sidecar] staged', p)" "$RUN/nerfstudio_models" | tee -a $L ;;
verdicts)
  bash /home/paperspace/code/automation/citrus_glref/verdict_sweep_glref.sh 01_13B_Jackal > /home/paperspace/logs/liftmed_01_verdicts.log 2>&1
  tail -n 3 /home/paperspace/logs/verdict_sweep_01_13B_Jackal.log | tee -a $L ;;
*) say "unknown stage $ST"; exit 1 ;;
esac; done
say "=== chain done: $STAGES"
