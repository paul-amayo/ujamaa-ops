#!/bin/bash
# Stage watcher for the H3DGS training chain: emits one line when a chunk finishes training / hierarchy /
# post-opt, when the scaffold or merged hierarchy appears, a throughput sample every 20 min, and any error.
OUT=/home/paperspace/data/citrus_all/05_13D_Jackal/experimental/h3dgs/output
LOG=/home/paperspace/logs/h3dgs_full_train.log
seen=""; lastp=0; lasterr=0
while true; do
  for f in $OUT/scaffold/point_cloud/iteration_30000/point_cloud.ply $OUT/trained_chunks/*/point_cloud/iteration_30000/point_cloud.ply $OUT/trained_chunks/*/hierarchy.hier $OUT/trained_chunks/*/hierarchy.hier_opt $OUT/merged.hier; do
    [ -e "$f" ] || continue
    case "$seen" in *"|$f|"*) ;; *) seen="$seen|$f|"; echo "[$(date '+%H:%M')] READY ${f#$OUT/} ($(du -h "$f" | cut -f1))";; esac
  done
  now=$(date +%s)
  if [ $((now - lastp)) -ge 1200 ]; then lastp=$now; p=$(tr '\r' '\n' < $LOG | grep -a "Training progress" | tail -1 | cut -c1-120); [ -n "$p" ] && echo "[$(date '+%H:%M')] $p"; fi
  e=$(tr '\r' '\n' < $LOG | grep -caE "Traceback|CUDA out of memory|Error executing|RuntimeError"); if [ "$e" -gt "$lasterr" ]; then lasterr=$e; echo "[$(date '+%H:%M')] ERROR lines now $e: $(tr '\r' '\n' < $LOG | grep -aE "Traceback|CUDA out of memory|Error executing|RuntimeError" | tail -1 | cut -c1-160)"; fi
  grep -q "H3DGS TRAIN CHAIN DONE\|NO MERGED HIERARCHY" /home/paperspace/logs/h3dgs_train.log 2>/dev/null && { echo "[$(date '+%H:%M')] chain finished"; exit 0; }
  pgrep -f "[h]3dgs_train_05.sh" > /dev/null || { echo "[$(date '+%H:%M')] trainer script is no longer running"; exit 1; }
  sleep 60
done
