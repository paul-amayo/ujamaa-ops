#!/bin/bash
# v3_fleet.sh — recipe v3 over a queue of chunks, SEQUENTIAL (two per card measured 10-08 as no throughput gain). Per entry:
# workspace (build_chunk_ws.py) -> train + held-out + grid-applied training-view score (train_chunk.sh) -> chunk supervision by the
# block process (v3_chunk_supervision.sh, unless a native cell with supervision exists) -> identity (v3_identity_cell.sh: flat census,
# seed B, maps, verdicts). One JSON line per chunk in logs/v3_fleet_results.jsonl; the narrative log is logs/v3_fleet.log.
# queue lines: <survey id under citrus_all | survey root> <H3DGS project (chunk export)> <chunk> [identity 1|0 (default 1)]
#   usage: v3_fleet.sh <queue file>    env: MIN_FREE_GB (20), ITERS (60000), WAIT_GPU (1 = wait for running ns-train jobs first)
set -u; Q=$1; V3=/home/paperspace/code/automation/h3dgs/v3; L=/home/paperspace/logs/v3_fleet.log; R=/home/paperspace/logs/v3_fleet_results.jsonl
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $L; }; MIN_FREE=${MIN_FREE_GB:-20}; ITERS=${ITERS:-60000}; N=sf_${ITERS}_bilateral
say "=== fleet start: $(grep -vcE '^#|^$' $Q) entries in $Q, $ITERS iterations, stop below $MIN_FREE G free"
if [ "${WAIT_GPU:-1}" = 1 ]; then while pgrep -f 'ns-train splatfacto' > /dev/null; do sleep 60; done; say "card free of ns-train"; fi
while read -r SV PROJ CN IDENT; do
  [ -z "$SV" ] && continue; [[ "$SV" == \#* ]] && continue; IDENT=${IDENT:-1}
  case $SV in /*) ROOT=$SV; SVN=$(basename $SV);; *) ROOT=/home/paperspace/data/citrus_all/$SV; SVN=$SV;; esac
  free=$(df -BG --output=avail / | tail -1 | tr -d ' G'); [ "$free" -lt "$MIN_FREE" ] && { say "STOP: $free G free < $MIN_FREE G (before $SVN $CN) — clear space and rerun the fleet, it skips finished entries"; exit 2; }
  WS=$ROOT/experimental/v3/chunk_$CN; TL=/home/paperspace/logs/v3_${SVN}_$CN.log; t0=$(date +%s); say "--- $SVN $CN ($free G free)"
  [ -e $WS/split.json ] || python3 $V3/build_chunk_ws.py $PROJ $CN $WS 2>&1 | tee -a $L
  [ -e $WS/split.json ] || { say "$SVN $CN: workspace FAILED"; continue; }
  if [ -n "${PASSES:-}" ]; then   # PASSES=<n>: iterations = n passes per training view (min 30k, rounded to 1k) - the recipe-check H1 schedule
    ITERS=$(python3 -c "import json; n=len(json.load(open('$WS/split.json'))['train']); print(max(30000, int(round($PASSES * n / 1000.0)) * 1000))"); N=sf_${ITERS}_bilateral
  fi
  if [ ! -e $WS/outputs/$N/eval_heldout.json ]; then TAG=${SVN}_$CN bash $V3/train_chunk.sh $WS $ITERS > /dev/null 2>&1; fi
  [ -e $WS/outputs/$N/eval_heldout.json ] || { say "$SVN $CN: training FAILED (see $TL)"; continue; }
  t1=$(date +%s); IDS=""
  if [ "$IDENT" = 1 ] && [ -d $ROOT/prod/bateleur ]; then
    CELL=$ROOT/experimental/h3dgs_native/chunk_${CN}_sam3
    if [ ! -e $CELL/supervision/trees_only/manifest.json ]; then CELL=$WS/cell; [ -e $CELL/supervision/trees_only/manifest.json ] || SVN=$SVN PROJ=$PROJ CN=$CN OUT=$CELL bash $V3/v3_chunk_supervision.sh > /dev/null 2>&1; fi
    if [ -e $CELL/supervision/trees_only/manifest.json ]; then
      EMB=$(ls -t $ROOT/prod/bateleur/embedder/*/ckpts/model_best.pth | head -1)
      [ -e $WS/identity/verdicts.log ] || CFG=$WS/outputs/$N/splatfacto/run/config.yml CELL=$CELL EMB=$EMB HJ=$ROOT/prod/bateleur/scene_graph/marker_hierarchy.json KF=$ROOT/prod/scratch_sam3 OUT=$WS/identity SPARSE=$WS/sparse/0 bash $V3/v3_identity_cell.sh > /dev/null 2>&1
      IDS=$WS/identity
    else say "$SVN $CN: no supervision (cell build failed) — identity skipped"; fi
  fi
  python3 - "$SVN" "$CN" "$WS" "$TL" "$IDS" "$N" $t0 $t1 <<'PY' | tee -a $L >> $R
import json, re, sys, time, glob, os
svn, cn, ws, tl, ids, n, t0, t1 = sys.argv[1:9]; log = open(tl, errors='ignore').read() if os.path.exists(tl) else ''
r = {'survey': svn, 'chunk': cn, 'run': n, 'ws': ws, 'done': time.strftime('%Y-%m-%d %H:%M')}
m = re.search(r'train rc=0 in (\d+) s', log); r['train_s'] = int(m.group(1)) if m else None
m = re.search(r'held-out raw: psnr ([\d.]+)', log); r['held_out'] = float(m.group(1)) if m else None
m = re.search(r'WITH the trained bilateral grid ([\d.]+) \(mean ([\d.]+)\)', log); r['grid_median'], r['grid_mean'] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
m = re.search(r'gaussians (\d+)', log); r['gaussians'] = int(m.group(1)) if m else None
s = json.load(open(f'{ws}/split.json')); r['train_views'], r['held_views'] = len(s['train']), len(s['test'])
if ids and os.path.exists(f'{ids}/verdicts.log'):
    sp = json.load(open(f'{ids}/split_names.json')); ev = set(sp['eval']); acc = {}
    for line in open(f'{ids}/verdicts.log', errors='ignore'):
        mm = re.match(r'\[v3 (\S+)\] (TREE|ROW|FRUIT)\b.*IoU ([\d.]+)', line)
        if mm: acc.setdefault((mm.group(2), 'eval' if mm.group(1) in ev else 'train'), []).append(float(mm.group(3)))
    r['identity'] = {f'{k[0]}_{k[1]}': round(sum(v) / len(v), 3) for k, v in sorted(acc.items())}; r['identity_n'] = {f'{k[0]}_{k[1]}': len(v) for k, v in sorted(acc.items())}
r['minutes_total'] = round((time.time() - int(t0)) / 60, 1); r['minutes_train_stage'] = round((int(t1) - int(t0)) / 60, 1)
print(json.dumps(r))
PY
done < $Q
say "=== fleet done ($(wc -l < $R) results in $R)"
