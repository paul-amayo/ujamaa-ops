#!/bin/bash
# sidecar_cleanup_20260927b.sh — same rule Paul authorised at 09:55 (superseded side-car intermediates: the zero-feature
# stage2_init and the ratio-1 census-init seed + its run copy; every one rebuildable in ~2 min), applied to the five
# row blocks built afterwards (018/019/021/022/023), plus a byte-identical-copy -> hardlink dedup of the ratio-2 seed
# run dirs. Keeps per block: converted stage-1 ckpt, both census npz, bootstrap run, the ratio-2 seed, supervision.
set -u
SC=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs_sidecar; L=/home/paperspace/logs/sidecar_cleanup_20260927.log; say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }
before=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); say "=== cleanup b start: ${before}G free"; n=0
for b in block_018 block_019 block_021 block_022 block_023; do
  d=$SC/$b; [ -d $d ] || continue
  for init in $d/stage2_init_census_*/nerfstudio_models/*.ckpt; do [ -e "$init" ] || continue; x=$(basename $(dirname $(dirname $init))); x=${x#stage2_init_census_}
    for run in $d/splat_runs_FEATFIX/stage2_censusinit_${x}/high/*/nerfstudio_models/$(basename $init); do [ -e "$run" ] || continue
      if [ "$(stat -c %i $init)" != "$(stat -c %i $run)" ] && cmp -s "$init" "$run"; then ln -f "$init" "$run" && n=$((n+1)); fi; done; done
  for p in $d/stage2_init_glref $d/stage2_init_census_glref $d/splat_runs_FEATFIX/stage2_censusinit_glref; do [ -e "$p" ] && { s=$(du -sh "$p" | cut -f1); rm -rf "$p" && say "removed $s $p"; }; done
done
after=$(df --output=avail -BG / | tail -1 | tr -dc 0-9); say "=== cleanup b done: $n copies -> hardlinks; ${before}G -> ${after}G free"
